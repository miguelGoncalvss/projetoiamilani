"""Atributos gerados na origem; nenhum lag consulta o período-alvo."""
import numpy as np
import pandas as pd


def origin_features(values, dates, origin_index, target_date, horizon, sku, cfg):
    past = np.asarray(values[:origin_index + 1], dtype=float)
    if len(past) < cfg.lookback:
        raise ValueError("Histórico insuficiente para os atributos")
    lags = [1, 3, 6, 12] if cfg.frequency == "monthly" else [1, 7, 14, 28]
    row = {"sku": str(sku), "h": horizon, "last": past[-1], "scale": max(past.mean(), 1.0)}
    # lag_1 é a última observação disponível, não o último valor do alvo futuro.
    row.update({f"lag_{lag}": past[-lag] for lag in lags})
    windows = [3, 6, 12] if cfg.frequency == "monthly" else [7, 14, 28]
    for w in windows:
        row[f"mean_{w}"] = past[-w:].mean()
        row[f"std_{w}"] = past[-w:].std()
    row.update(month_sin=np.sin(2 * np.pi * target_date.month / 12),
               month_cos=np.cos(2 * np.pi * target_date.month / 12),
               weekday_sin=np.sin(2 * np.pi * target_date.dayofweek / 7),
               weekday_cos=np.cos(2 * np.pi * target_date.dayofweek / 7),
               elapsed=origin_index)
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
