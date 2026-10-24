import math
import numpy as np
import pandas as pd


def metrics(frame):
    y, yhat = frame.y.to_numpy(float), frame.yhat.to_numpy(float)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(yhat).all():
        raise ValueError("Métricas exigem observações finitas")
    error = y - yhat
    denominator = np.abs(y).sum()
    return {"n": len(y), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "bias_prediction_minus_actual": float(-error.mean()),
            "wape_pct": float(100 * np.abs(error).sum()/denominator) if denominator else None}


def metric_table(frame, keys):
    rows = []
    for identity, group in frame.groupby(keys, sort=True, dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        rows.append(dict(zip(keys, identity)) | metrics(group))
    return pd.DataFrame(rows)


def make_splits(panel, cfg):
    dates = pd.DatetimeIndex(sorted(panel.ds.unique()))
    blocks = cfg.selection_folds + cfg.calibration_folds + 1
    first_train = len(dates) - blocks * cfg.horizon
    minimum = cfg.lookback + cfg.horizon
    if first_train < minimum:
        raise ValueError(
            f"Histórico insuficiente: {len(dates)} períodos; precisa de pelo menos "
            f"{minimum + blocks * cfg.horizon} para este protocolo. "
            "Reduza folds explicitamente apenas com justificativa; teste não será encurtado.")
    splits = []
    for i in range(blocks):
        end = first_train + i * cfg.horizon
        phase = ("selection" if i < cfg.selection_folds else
                 "calibration" if i < blocks-1 else "test")
        splits.append({"phase": phase, "fold": i, "origin": dates[end-1],
                       "target_start": dates[end], "target_end": dates[end+cfg.horizon-1]})
    return splits


def attach_actual(prediction, panel):
    result = prediction.merge(panel[["sku", "ds", "y"]], on=["sku", "ds"], how="left", validate="one_to_one")
    if result.y.isna().any() or not (result.ds > result.origin).all():
        raise ValueError("Previsões sem alvos válidos ou com vazamento temporal")
    return result


def calibrate(frame, minimum=20):
    """Quantis empíricos de erro absoluto escalado, separados por horizonte.
    Não implica cobertura garantida sob dependência entre SKUs ou no tempo.
    """
    rows = []
    for h, g in frame.groupby("h", sort=True):
        scores = np.sort(np.abs(g.y.to_numpy() - g.yhat.to_numpy()) / g.scale.to_numpy())
        row = {"h": int(h), "n_calibration": len(scores)}
        for level in (80, 95):
            rank = math.ceil((len(scores)+1)*level/100)
            row[f"q{level}"] = (float(scores[rank-1])
                                if len(scores) >= minimum and rank <= len(scores) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def apply_intervals(frame, calibration):
    result = frame.merge(calibration, on="h", how="left", validate="many_to_one")
    for level in (80, 95):
        width = result[f"q{level}"] * result.scale
        result[f"lower_{level}"] = (result.yhat-width).clip(lower=0)
        result[f"upper_{level}"] = result.yhat+width
    result["interval_status"] = np.where(
        result[["q80", "q95"]].notna().all(axis=1),
        "empirical_not_guaranteed", "insufficient_calibration")
    return result


def interval_metrics(frame):
    rows = []
    for h, group in frame.groupby("h"):
        for level in (80, 95):
            valid = group.dropna(subset=[f"lower_{level}", f"upper_{level}"])
            rows.append({"h": int(h), "nominal_pct": level, "n": len(valid),
                         "coverage_pct": float(100*((valid.y >= valid[f"lower_{level}"]) &
                                                   (valid.y <= valid[f"upper_{level}"])).mean()) if len(valid) else None,
                         "mean_width": float((valid[f"upper_{level}"]-valid[f"lower_{level}"]).mean()) if len(valid) else None})
    return pd.DataFrame(rows)


def cumulative_metrics(frame, endpoints):
    rows = []
    for end in endpoints:
        sums = frame[frame.h <= end].groupby(["model", "fold", "sku"], as_index=False)[["y", "yhat"]].sum()
        for model, g in sums.groupby("model"):
            rows.append({"model": model, "through_horizon": end} | metrics(g))
    return pd.DataFrame(rows)
