"""Atributos gerados na origem; nenhum lag consulta o período-alvo."""
import numpy as np
import pandas as pd


def month_calendar_factors(dt_index):
    """Calcula dias no mês, proporção de fins de semana e peso efetivo de calendário (100% determinístico)."""
    dt_index = pd.DatetimeIndex(dt_index)
    days = np.asarray(dt_index.days_in_month, dtype=float)
    first_dow = np.asarray(dt_index.dayofweek, dtype=int)
    extra = days.astype(int) - 28
    wknd = np.full(len(days), 8.0, dtype=float)
    for idx in range(len(days)):
        for d in range(extra[idx]):
            if (first_dow[idx] + d) % 7 >= 5:
                wknd[idx] += 1.0
    wknd_share = wknd / days
    cal_weight = days * (1.0 - 0.35 * (wknd_share - (2.0 / 7.0)))
    return days, wknd_share, cal_weight


def origin_features(values, dates, origin_index, target_date, horizon, sku, cfg, precomputed_cal=None):
    if isinstance(values, np.ndarray) and values.dtype == np.float64:
        past = values[:origin_index + 1]
    else:
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

    season = cfg.season
    n = len(past)

    if cfg.frequency == "monthly":
        if precomputed_cal is not None:
            p_days = precomputed_cal["days"][:origin_index + 1]
            p_wknd = precomputed_cal["wknd_share"][:origin_index + 1]
            cal_weight_past = precomputed_cal["cal_weight"][:origin_index + 1]
        else:
            p_days, p_wknd, cal_weight_past = month_calendar_factors(dates[:origin_index + 1])
        t_days_arr, t_wknd_arr, t_cal_arr = month_calendar_factors([target_date])
        target_days = float(t_days_arr[0])
        target_wknd_share = float(t_wknd_arr[0])
        cal_weight_target = float(t_cal_arr[0])
        rate = past / np.maximum(cal_weight_past, 1.0)
    else:
        target_days = 1.0
        target_wknd_share = 1.0 if target_date.dayofweek >= 5 else 0.0
        cal_weight_target = 1.0
        rate = past

    # Inclinação de tendência dessazonalizada usando ciclos completos (evita viés sazonal Jan->Dez)
    trend_period = 12 if cfg.frequency == "monthly" else 365
    n_full = (n // trend_period) * trend_period
    if n_full >= 2 * trend_period:
        reshaped = rate[-n_full:].reshape(-1, trend_period)
        cycle_means = reshaped.mean(axis=1)
        c_idx = np.arange(len(cycle_means), dtype=float) * trend_period
        c_cent = c_idx - c_idx.mean()
        denom = float(np.dot(c_cent, c_cent))
        slope_rate = float(np.dot(c_cent, cycle_means - cycle_means.mean()) / denom) if denom > 0 else 0.0
        base_rate = max(float(cycle_means[-1]), 1e-6)
        slope_norm = slope_rate / base_rate
    else:
        t_idx = np.arange(n, dtype=float)
        t_centered = t_idx - t_idx.mean()
        denom = float(np.dot(t_centered, t_centered))
        slope_per_step = float(np.dot(t_centered, rate - rate.mean()) / denom) if denom > 0 else 0.0
        slope_norm = slope_per_step / max(float(rate[-cfg.lookback:].mean()), 1e-6)

    if cfg.frequency == "daily" and n >= 365:
        # Na frequência diária, incorpora o fator semanal e o fator anual do mesmo período do calendário
        dow_vals = [float(rate[-lag]) for lag in range(seasonal_offset, min(n, seasonal_offset + 8 * 7) + 1, 7)]
        recent_28_mean = max(float(rate[-28:].mean()), 1e-6)
        dow_factor = float(np.mean(dow_vals)) / recent_28_mean if dow_vals else 1.0

        csum = precomputed_cal["csum"] if (precomputed_cal is not None and "csum" in precomputed_cal) else np.concatenate(([0.0], np.cumsum(rate)))
        annual_ratios = []
        max_yrs = min(7, n // 365)
        for yr in range(1, max_yrs + 1):
            center_idx = n - 1 + horizon - int(round(yr * 365.25))
            if 14 <= center_idx < n - 14:
                w_lo, w_hi = center_idx - 7, center_idx + 7
                win_mean = float(csum[w_hi] - csum[w_lo]) / 14.0
                b_lo, b_hi = max(0, center_idx - 180), min(n, center_idx + 180)
                yr_base = float(csum[b_hi] - csum[b_lo]) / float(b_hi - b_lo)
                if yr_base > 0:
                    annual_ratios.append(win_mean / yr_base)
        annual_factor = float(np.mean(annual_ratios)) if annual_ratios else 1.0
        overall_year_mean = float(csum[n] - csum[n - 365]) / 365.0
        trend_from_year_center = max(0.25, 1.0 + slope_norm * (182.5 + horizon))
        seasonal_trend_anchor = max(overall_year_mean * trend_from_year_center * annual_factor * dow_factor, 1e-3)
    else:
        # Projeção multi-ciclo corrigida por tendência e calendário para ocorrências passadas da mesma fase
        proj_rates = []
        k = 0
        max_k = 12
        while k < max_k and (seasonal_offset + k * season <= n):
            lag_dist = seasonal_offset + k * season
            obs_rate = float(rate[-lag_dist])
            obs_trend = max(0.25, 1.0 - slope_norm * (lag_dist - (season - 1) / 2.0))
            targ_trend = max(0.25, 1.0 + slope_norm * (horizon + (season - 1) / 2.0))
            proj_rates.append(obs_rate * (targ_trend / obs_trend))
            k += 1
        anchor_rate = float(np.mean(proj_rates)) if proj_rates else float(rate[-1])
        seasonal_trend_anchor = max(anchor_rate * cal_weight_target, 1e-3)

    seasonal_trend_anchor_norm = seasonal_trend_anchor / local_scale

    row = {
        "sku": str(sku),
        "h": horizon,
        "last": float(past[-1]),
        "scale": max(float(past.mean()), 1.0),
        "local_scale": local_scale,
        "seasonal_trend_anchor": seasonal_trend_anchor,
        "last_norm": float(past[-1]) / local_scale,
        "seasonal_target_lag": seasonal_target_lag,
        "seasonal_target_norm": seasonal_target_lag / local_scale,
        "seasonal_target_mean_norm": seasonal_target_mean / local_scale,
        "slope_norm": slope_norm,
        "trend_horizon_factor": max(0.25, 1.0 + slope_norm * horizon),
        "seasonal_trend_anchor_norm": seasonal_trend_anchor_norm,
        "target_days_in_month": target_days,
        "target_wknd_share": target_wknd_share,
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
        values, dates = g.y.to_numpy(dtype=float), pd.DatetimeIndex(g.ds)
        if cfg.frequency == "monthly":
            d_arr, w_arr, c_arr = month_calendar_factors(dates)
            pre_cal = {"days": d_arr, "wknd_share": w_arr, "cal_weight": c_arr}
            stride = 1
        else:
            pre_cal = {"csum": np.concatenate(([0.0], np.cumsum(values)))}
            stride = 7 if cfg.horizon <= 60 else (14 if cfg.horizon <= 180 else 28)
        for i in range(cfg.lookback - 1, len(g) - 1, stride):
            for h in range(1, min(cfg.horizon, len(g) - 1 - i) + 1):
                rows.append(origin_features(values, dates, i, dates[i+h], h, sku, cfg, precomputed_cal=pre_cal))
                targets.append(values[i+h])
                audit.append({"sku": str(sku), "origin": dates[i], "target": dates[i+h], "h": h})
    min_required_h = min(cfg.horizon, cfg.base_horizon)
    if not rows or max(r["h"] for r in rows) < min_required_h:
        raise ValueError("Treino insuficiente para aprender o horizonte completo")
    return pd.DataFrame(rows), np.asarray(targets), pd.DataFrame(audit)


def prediction_rows(panel, cfg):
    rows, keys = [], []
    for sku, g in panel.groupby("sku", sort=True):
        g = g.sort_values("ds")
        values, dates = g.y.to_numpy(dtype=float), pd.DatetimeIndex(g.ds)
        if cfg.frequency == "monthly":
            d_arr, w_arr, c_arr = month_calendar_factors(dates)
            pre_cal = {"days": d_arr, "wknd_share": w_arr, "cal_weight": c_arr}
        else:
            pre_cal = {"csum": np.concatenate(([0.0], np.cumsum(values)))}
        for h, target in enumerate(cfg.dates_after(dates[-1]), 1):
            rows.append(origin_features(values, dates, len(g)-1, target, h, sku, cfg, precomputed_cal=pre_cal))
            keys.append({"sku": str(sku), "ds": target, "h": h, "origin": dates[-1],
                         "scale": max(float(values.mean()), 1.0)})
    return pd.DataFrame(rows), pd.DataFrame(keys)
