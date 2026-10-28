from dataclasses import replace
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from . import __version__
from .adapter import reconstruct_censored_demand, adapt_config_to_history
from .data import load_history, make_panel
from .evaluation import (make_splits, attach_actual, metric_table, calibrate,
                         apply_intervals, interval_metrics, cumulative_metrics)
from .models import Forecaster


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def run(history_path, cfg, output, synthetic=False, reconstruct_stockouts=False, allow_protocol_adaptation=False):
    out = Path(output)
    if out.exists():
        raise ValueError("Diretório de saída já existe; escolha outro para preservar a execução anterior")
    history = load_history(history_path)
    censoring_summary = None
    if reconstruct_stockouts:
        history, censoring_summary = reconstruct_censored_demand(history)
    panel = make_panel(history, cfg)
    cfg, protocol_report = adapt_config_to_history(panel, cfg, allow_adaptation=allow_protocol_adaptation)
    splits = make_splits(panel, cfg)
    out.mkdir(parents=True)
    write_json(out / "config.json", cfg.to_dict())
    write_json(out / "status.json", {"status": "running"})
    if censoring_summary is not None:
        write_json(out / "censoring_reconstruction.json", censoring_summary)
    versions = {}
    for package in ["numpy", "pandas", "scikit-learn", "xgboost", "prophet"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    manifest_data = {
        "project_version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": hashlib.sha256(Path(history_path).read_bytes()).hexdigest(),
        "input_rows": len(history), "sku_count": history.sku.nunique(),
        "start_date": str(history.date.min().date()), "end_date": str(history.date.max().date()),
        "target": "reconstructed_demand" if reconstruct_stockouts else "observed_outbound",
        "reconstruct_stockouts": bool(reconstruct_stockouts),
        "censoring_summary": censoring_summary,
        "protocol_adaptation": protocol_report,
        "synthetic": bool(synthetic),
        "stockout_rows": int(history.stockout.sum()), "versions": versions,
        "selection_rule": "mean MAE, equal weights per SKU; selection folds only",
        "interval_method": "absolute scaled errors, by horizon, separate calibration folds",
        "limitations": [
            "Reconstrução de censura usa histórico local sem variáveis exógenas." if reconstruct_stockouts
            else "Saídas observadas não recuperam demanda perdida.",
            "Bandas empíricas marginais não garantem cobertura; use joint_region_risk.csv na simulação para risco conjunto.",
            "Nenhuma prova de acurácia operacional real em dados sintéticos."
        ]}
    write_json(out / "manifest.json", manifest_data)
    pd.DataFrame(splits).to_csv(out / "splits.csv", index=False)

    def evaluate(name, split):
        train = panel[panel.ds <= split["origin"]].copy()
        split_horizon = int(split.get("horizon", cfg.base_horizon))
        eval_cfg = replace(cfg, forecast_horizon=split_horizon)
        print(f'{split["phase"]}: {name}, origem {split["origin"].date()}', flush=True)
        fitted = Forecaster(name, eval_cfg).fit(train)
        pred = attach_actual(fitted.predict(), panel)
        pred["phase"], pred["fold"] = split["phase"], split["fold"]
        return pred

    try:
        selection = pd.concat([evaluate(name, s) for s in splits if s["phase"] == "selection"
                               for name in cfg.models], ignore_index=True)
        selection.to_csv(out / "selection_predictions.csv", index=False)
        scores = metric_table(selection, ["model", "sku"])
        ranking = scores.groupby("model", as_index=False).mae.mean().sort_values(["mae", "model"])
        winner = str(ranking.iloc[0].model)
        ranking.to_csv(out / "ranking_selection.csv", index=False)
        scores.to_csv(out / "selection_metrics_by_sku.csv", index=False)
        metric_table(selection, ["model", "h"]).to_csv(out / "selection_metrics_by_horizon.csv", index=False)

        # Congela o vencedor da seleção, mas calibra, testa e refita TODOS os modelos configurados.
        write_json(out / "selected_model.json", {
            "model": winner,
            "all_models_trained": list(cfg.models),
            "selected_using": "selection_only",
            "forecast_horizon": cfg.horizon,
        })

        calibration_predictions = pd.concat([
            evaluate(name, s)
            for s in splits if s["phase"] == "calibration"
            for name in cfg.models
        ], ignore_index=True)
        calibration_predictions.to_csv(out / "calibration_predictions.csv", index=False)
        calibration = calibrate(calibration_predictions, cfg.minimum_calibration_samples)
        calibration.to_csv(out / "calibration.csv", index=False)

        test_raw = pd.concat([evaluate(name, splits[-1]) for name in cfg.models], ignore_index=True)
        test = apply_intervals(test_raw, calibration)
        test.to_csv(out / "test_predictions.csv", index=False)
        test_scores = metric_table(test, ["model", "sku"])
        test_scores.to_csv(out / "test_metrics_by_sku.csv", index=False)
        test_ranking = test_scores.groupby("model", as_index=False).mae.mean().sort_values(["mae", "model"])
        test_ranking.to_csv(out / "ranking_test.csv", index=False)
        metric_table(test, ["model", "h"]).to_csv(out / "test_metrics_by_horizon.csv", index=False)
        interval_metrics(test).to_csv(out / "test_interval_metrics.csv", index=False)
        cumulative_metrics(test, [3, 6, 12] if cfg.frequency == "monthly" else [7, 30]).to_csv(
            out / "test_cumulative_metrics.csv", index=False)

        # Refit final com 100% do histórico (ex: 2018 a 2025) para TODOS os modelos configurados.
        all_forecasts = []
        models_dir = out / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for name in cfg.models:
            print(f"refit_full_history: {name}, origem {panel.ds.max().date()}, horizonte={cfg.horizon}", flush=True)
            fitted_model = Forecaster(name, cfg).fit(panel)
            fitted_model.save(models_dir / name)
            if name == winner:
                fitted_model.save(out / "model")
            fc = apply_intervals(fitted_model.predict(), calibration)
            fc["frequency"] = cfg.frequency
            fc["synthetic"] = bool(synthetic)
            fc.to_csv(out / f"forecast_{name}.csv", index=False)
            if name == winner:
                fc.to_csv(out / "forecast.csv", index=False)
            all_forecasts.append(fc)

        forecast_all = pd.concat(all_forecasts, ignore_index=True)
        forecast_all.to_csv(out / "forecast_all_models.csv", index=False)

        # Resumo anual das previsões futuras (permite comparar 2026, 2027, 2028... entre todos os modelos)
        fc_year = forecast_all.assign(year=pd.to_datetime(forecast_all["ds"]).dt.year)
        by_year_sku = fc_year.groupby(["model", "year", "sku"], as_index=False)[
            ["yhat", "lower_80", "upper_80", "lower_95", "upper_95"]
        ].sum()
        by_year_sku.to_csv(out / "forecast_by_year_sku.csv", index=False)

        by_year_cd = fc_year.groupby(["model", "year"], as_index=False)[
            ["yhat", "lower_80", "upper_80", "lower_95", "upper_95"]
        ].sum()
        by_year_cd.to_csv(out / "forecast_by_year_total_cd.csv", index=False)

        write_json(out / "status.json", {
            "status": "completed",
            "selected_model": winner,
            "trained_models": list(cfg.models),
            "forecast_horizon": cfg.horizon,
            "note": "Todos os modelos foram ajustados com 100% do histórico em models/<nome> e forecast_<nome>.csv; forecast.csv e model/ contêm o vencedor.",
        })
        print(
            f"Concluído: {out.resolve()} | modelos treinados (100% da base): {', '.join(cfg.models)} | "
            f"vencedor: {winner} | horizonte futuro: {cfg.horizon} ({cfg.frequency})",
            flush=True,
        )
        return out
    except Exception as exc:
        write_json(out / "status.json", {"status": "failed", "error": str(exc)})
        raise
