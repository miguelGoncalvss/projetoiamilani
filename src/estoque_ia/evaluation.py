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
    split_horizon = cfg.horizon
    if cfg.forecast_horizon is not None and len(dates) < (cfg.lookback + split_horizon + blocks * split_horizon):
        split_horizon = cfg.base_horizon
    first_train = len(dates) - blocks * split_horizon
    minimum = cfg.lookback + split_horizon
    if first_train < minimum:
        raise ValueError(
            f"Histórico insuficiente: {len(dates)} períodos; precisa de pelo menos "
            f"{minimum + blocks * split_horizon} para este protocolo. "
            "Reduza folds explicitamente apenas com justificativa; teste não será encurtado.")
    splits = []
    for i in range(blocks):
        end = first_train + i * split_horizon
        phase = ("selection" if i < cfg.selection_folds else
                 "calibration" if i < blocks-1 else "test")
        splits.append({"phase": phase, "fold": i, "origin": dates[end-1],
                       "target_start": dates[end], "target_end": dates[end+split_horizon-1],
                       "horizon": split_horizon})
    return splits


def attach_actual(prediction, panel):
    result = prediction.merge(panel[["sku", "ds", "y"]], on=["sku", "ds"], how="left", validate="one_to_one")
    if result.y.isna().any() or not (result.ds > result.origin).all():
        raise ValueError("Previsões sem alvos válidos ou com vazamento temporal")
    return result


def calibrate(frame, minimum=20):
    """Quantis empíricos de erro absoluto escalado, separados por horizonte (e por modelo quando presente).
    Não implica cobertura garantida sob dependência entre SKUs ou no tempo.
    """
    rows = []
    has_model = "model" in frame.columns
    group_keys = ["model", "h"] if has_model else ["h"]
    for keys, g in frame.groupby(group_keys, sort=True):
        if has_model:
            model_name, h = keys
        else:
            h = keys[0] if isinstance(keys, tuple) else keys
        scores = np.sort(np.abs(g.y.to_numpy() - g.yhat.to_numpy()) / g.scale.to_numpy())
        row = ({"model": str(model_name)} if has_model else {}) | {"h": int(h), "n_calibration": len(scores)}
        for level in (80, 95):
            rank = math.ceil((len(scores)+1)*level/100)
            row[f"q{level}"] = (float(scores[rank-1])
                                if len(scores) >= minimum and rank <= len(scores) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _expand_calibration_for_multiyear(calibration, max_h):
    max_cal_h = int(calibration["h"].max()) if not calibration.empty else 0
    if max_cal_h < 1 or max_h <= max_cal_h:
        return calibration
    extra_rows = []
    for h in range(max_cal_h + 1, int(max_h) + 1):
        h_ref = ((h - 1) % max_cal_h) + 1
        cycle_idx = (h - 1) // max_cal_h
        growth = math.sqrt(1.0 + 0.15 * cycle_idx)
        ref_slice = calibration[calibration["h"] == h_ref].copy()
        ref_slice["h"] = int(h)
        for level in (80, 95):
            if f"q{level}" in ref_slice.columns:
                ref_slice[f"q{level}"] = ref_slice[f"q{level}"] * growth
        extra_rows.append(ref_slice)
    return pd.concat([calibration] + extra_rows, ignore_index=True)


def apply_intervals(frame, calibration):
    cal = _expand_calibration_for_multiyear(calibration, int(frame["h"].max()) if len(frame) else 0)
    join_keys = ["model", "h"] if ("model" in frame.columns and "model" in cal.columns) else ["h"]
    result = frame.merge(cal, on=join_keys, how="left", validate="many_to_one")
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
    has_multi_model = "model" in frame.columns and frame["model"].nunique() > 1
    group_keys = ["model", "h"] if has_multi_model else ["h"]
    for keys, group in frame.groupby(group_keys, sort=True):
        if has_multi_model:
            model_name, h = keys
            prefix = {"model": str(model_name), "h": int(h)}
        else:
            h = keys[0] if isinstance(keys, tuple) else keys
            prefix = {"h": int(h)}
        for level in (80, 95):
            valid = group.dropna(subset=[f"lower_{level}", f"upper_{level}"])
            rows.append(prefix | {"nominal_pct": level, "n": len(valid),
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
