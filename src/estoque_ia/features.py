"""Atributos gerados na origem; nenhum lag consulta o período-alvo."""
import numpy as np
import pandas as pd


def origin_features(values, dates, origin_index, target_date, horizon, sku, cfg):
    past = np.asarray(values[:origin_index + 1], dtype=float)
    if len(past) < cfg.lookback:
        raise ValueError("Histórico insuficiente para os atributos")

    local_scale = max(float(past[-cfg.lookback:].mean()), 1.0)
    lags = [1, 3, 6, 12] if cfg.frequency == "monthly" else [1, 7, 14, 28]

    # Lag sazonal alinhado exatamente ao horizonte futuro h (mesmo mês do ano anterior ou mesmo dia da semana)
    seasonal_offset = cfg.season - ((horizon - 1) % cfg.season)
    seasonal_target_lag = float(past[-seasonal_offset])
    seasonal_cycles = [
        float(past[-(seasonal_offset + k * cfg.season)])
        for k in range(3)
        if len(past) >= (seasonal_offset + k * cfg.season)
    ]
    seasonal_target_mean = float(np.mean(seasonal_cycles))

    # Inclinação de tendência linear estimada exclusivamente até a origem (normalizada por local_scale)
    t_idx = np.arange(len(past), dtype=float)
    t_centered = t_idx - t_idx.mean()
    denom = float(np.dot(t_centered, t_centered))
    slope_per_step = float(np.dot(t_centered, past - past.mean()) / denom) if denom > 0 else 0.0
    slope_norm = slope_per_step / local_scale

    # Distância temporal entre o último ciclo sazonal observado daquela fase e o período-alvo
    steps_from_seasonal_lag = float(seasonal_offset + horizon - 1)
    trend_growth_from_lag = max(0.25, 1.0 + slope_norm * steps_from_seasonal_lag)
    seasonal_trend_anchor_norm = (seasonal_target_lag / local_scale) * trend_growth_from_lag

    row = {
        "sku": str(sku),
        "h": horizon,
        "last": float(past[-1]),
        "scale": max(float(past.mean()), 1.0),
        "local_scale": local_scale,
        "last_norm": float(past[-1]) / local_scale,
        "seasonal_target_lag": seasonal_target_lag,
        "seasonal_target_norm": seasonal_target_lag / local_scale,
        "seasonal_target_mean_norm": seasonal_target_mean / local_scale,
        "slope_norm": slope_norm,
        "trend_horizon_factor": max(0.25, 1.0 + slope_norm * horizon),
        "seasonal_trend_anchor_norm": seasonal_trend_anchor_norm,
    }
    # lag_1 é a última observação disponível, não o último valor do alvo futuro.
    for lag in lags:
        val = float(past[-lag])
        row[f"lag_{lag}"] = val
        row[f"lag_{lag}_norm"] = val / local_scale

    windows = [3, 6, 12] if cfg.frequency == "monthly" else [7, 14, 28]
    for w in windows:
        m_val = float(past[-w:].mean())
        s_val = float(past[-w:].std())
        row[f"mean_{w}"] = m_val
        row[f"std_{w}"] = s_val
        row[f"mean_{w}_norm"] = m_val / local_scale
        row[f"std_{w}_norm"] = s_val / local_scale

    row.update(
        month_sin=np.sin(2 * np.pi * target_date.month / 12),
        month_cos=np.cos(2 * np.pi * target_date.month / 12),
        weekday_sin=np.sin(2 * np.pi * target_date.dayofweek / 7),
        weekday_cos=np.cos(2 * np.pi * target_date.dayofweek / 7),
        target_month=float(target_date.month),
        target_dayofweek=float(target_date.dayofweek),
        elapsed=origin_index,
    )
    return row


def training_rows(panel, cfg):
    rows, targets, audit = [], [], []
    for sku, g in panel.groupby("sku", sort=True):
        g = g.sort_values("ds")
        values, dates = g.y.to_numpy(), pd.DatetimeIndex(g.ds)
        stride = 1 if cfg.frequency == "monthly" else 7
        for i in range(cfg.lookback - 1, len(g) - 1, stride):
            for h in range(1, min(cfg.horizon, len(g) - 1 - i) + 1):
                rows.append(origin_features(values, dates, i, dates[i+h], h, sku, cfg))
                targets.append(values[i+h])
                audit.append({"sku": str(sku), "origin": dates[i], "target": dates[i+h], "h": h})
    if not rows or max(r["h"] for r in rows) < cfg.horizon:
        raise ValueError("Treino insuficiente para aprender o horizonte completo")
    return pd.DataFrame(rows), np.asarray(targets), pd.DataFrame(audit)


def prediction_rows(panel, cfg):
    rows, keys = [], []
    for sku, g in panel.groupby("sku", sort=True):
        g = g.sort_values("ds")
        values, dates = g.y.to_numpy(), pd.DatetimeIndex(g.ds)
        for h, target in enumerate(cfg.dates_after(dates[-1]), 1):
            rows.append(origin_features(values, dates, len(g)-1, target, h, sku, cfg))
            keys.append({"sku": str(sku), "ds": target, "h": h, "origin": dates[-1],
                         "scale": max(float(values.mean()), 1.0)})
    return pd.DataFrame(rows), pd.DataFrame(keys)
