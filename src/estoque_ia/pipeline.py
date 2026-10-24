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
        print(f'{split["phase"]}: {name}, origem {split["origin"].date()}', flush=True)
        fitted = Forecaster(name, cfg).fit(train)
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
        # Freeze the winner before calibration and the final holdout.
        write_json(out / "selected_model.json", {"model": winner, "selected_using": "selection_only"})
        calibration_predictions = pd.concat([evaluate(winner, s) for s in splits
                                             if s["phase"] == "calibration"], ignore_index=True)
        calibration_predictions.to_csv(out / "calibration_predictions.csv", index=False)
        calibration = calibrate(calibration_predictions, cfg.minimum_calibration_samples)
        calibration.to_csv(out / "calibration.csv", index=False)
        test = apply_intervals(evaluate(winner, splits[-1]), calibration)
        test.to_csv(out / "test_predictions.csv", index=False)
        metric_table(test, ["model", "sku"]).to_csv(out / "test_metrics_by_sku.csv", index=False)
        metric_table(test, ["model", "h"]).to_csv(out / "test_metrics_by_horizon.csv", index=False)
        interval_metrics(test).to_csv(out / "test_interval_metrics.csv", index=False)
        cumulative_metrics(test, [3, 6, 12] if cfg.frequency == "monthly" else [7, 30]).to_csv(
            out / "test_cumulative_metrics.csv", index=False)
        # Final production model refits all observed history after holdout reporting.
        final = Forecaster(winner, cfg).fit(panel)
        final.save(out / "model")
        forecast = apply_intervals(final.predict(), calibration)
        forecast["frequency"] = cfg.frequency
        forecast["synthetic"] = bool(synthetic)
        forecast.to_csv(out / "forecast.csv", index=False)
        write_json(out / "status.json", {"status": "completed", "selected_model": winner,
                   "note": "Teste final avalia o modelo ajustado até a origem do teste; model/ contém o refit posterior."})
        print(f"Concluído: {out.resolve()} | modelo selecionado: {winner}", flush=True)
        return out
    except Exception as exc:
        write_json(out / "status.json", {"status": "failed", "error": str(exc)})
        raise
