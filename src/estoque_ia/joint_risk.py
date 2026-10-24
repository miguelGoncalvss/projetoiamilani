"""Risco conjunto de demanda, ruptura e ocupação via simulação de Monte Carlo / Bootstrap Multivariado."""
import numpy as np
import pandas as pd


def sample_joint_demand_trajectories(forecast_df, calibration_predictions_df=None, n_sims=300, seed=42):
    """Gera matriz 3D (n_sims, n_skus, n_horizons) de trajetórias conjuntas de demanda preservando correlação entre SKUs."""
    rng = np.random.default_rng(seed)
    fc = forecast_df.sort_values(["sku", "h"]).copy()
    skus = sorted(fc["sku"].astype(str).unique())
    horizons = sorted(fc["h"].astype(int).unique())
    n_skus = len(skus)
    n_h = len(horizons)

    yhat_mat = np.zeros((n_skus, n_h), dtype=float)
    scale_mat = np.ones((n_skus, n_h), dtype=float)
    q95_mat = np.full((n_skus, n_h), np.nan, dtype=float)

    sku_to_idx = {s: i for i, s in enumerate(skus)}
    h_to_idx = {h: j for j, h in enumerate(horizons)}

    for row in fc.itertuples():
        i = sku_to_idx[str(row.sku)]
        j = h_to_idx[int(row.h)]
        yhat_mat[i, j] = float(row.yhat)
        if hasattr(row, "scale") and pd.notna(row.scale):
            scale_mat[i, j] = max(float(row.scale), 1.0)
        else:
            scale_mat[i, j] = max(float(row.yhat), 1.0)
        if hasattr(row, "q95") and pd.notna(row.q95):
            q95_mat[i, j] = float(row.q95)

    # Tenta extrair matriz empírica de resíduos escalados alinhados no tempo (fold, h) x sku
    residual_vectors = None
    if calibration_predictions_df is not None and not calibration_predictions_df.empty:
        cal = calibration_predictions_df.copy()
        cal["sku"] = cal["sku"].astype(str)
        cal = cal[cal["sku"].isin(sku_to_idx)]
        cal["scaled_err"] = (cal["y"].astype(float) - cal["yhat"].astype(float)) / np.maximum(cal["scale"].astype(float), 1.0)
        piv = cal.pivot_table(index=["fold", "h"], columns="sku", values="scaled_err", aggfunc="mean")
        piv = piv.reindex(columns=skus).dropna(how="any")
        if len(piv) >= 4:
            residual_vectors = piv.to_numpy(dtype=float)

    trajectories = np.zeros((n_sims, n_skus, n_h), dtype=float)

    if residual_vectors is not None:
        # Combina bootstrap em bloco de vetores multivariados de erro com jitter correlacionado
        cov = np.cov(residual_vectors, rowvar=False)
        if n_skus == 1:
            cov = np.array([[float(cov)]])
        cov = cov + np.eye(n_skus) * 1e-6
        n_pool = len(residual_vectors)

        for j in range(n_h):
            boot_idx = rng.integers(0, n_pool, size=n_sims)
            sampled_err = residual_vectors[boot_idx]  # (n_sims, n_skus)
            jitter = rng.multivariate_normal(np.zeros(n_skus), 0.15 * cov, size=n_sims)
            total_scaled_err = sampled_err + jitter

            # Ajusta escala pelo q95 do horizonte quando disponível
            emp_q95 = np.quantile(np.abs(total_scaled_err), 0.95, axis=0)
            target_q95 = np.where(np.isfinite(q95_mat[:, j]), q95_mat[:, j], emp_q95)
            ratio = np.where(emp_q95 > 1e-6, target_q95 / emp_q95, 1.0)

            sim_y = yhat_mat[:, j][None, :] + total_scaled_err * ratio[None, :] * scale_mat[:, j][None, :]
            trajectories[:, :, j] = np.maximum(0.0, sim_y)
    else:
        # Fallback paramétrico correlacionado quando calibration_predictions não é fornecido
        corr = np.full((n_skus, n_skus), 0.35, dtype=float)
        np.fill_diagonal(corr, 1.0)
        for j in range(n_h):
            half_width_95 = np.where(
                np.isfinite(q95_mat[:, j]),
                q95_mat[:, j] * scale_mat[:, j],
                0.25 * np.maximum(yhat_mat[:, j], 1.0),
            )
            sigma = half_width_95 / 1.96
            cov = np.outer(sigma, sigma) * corr
            noise = rng.multivariate_normal(np.zeros(n_skus), cov + np.eye(n_skus) * 1e-6, size=n_sims)
            trajectories[:, :, j] = np.maximum(0.0, yhat_mat[:, j][None, :] + noise)

    return trajectories, skus, horizons


def simulate_joint_risk(
    forecast_df,
    stock_df,
    receipts_df,
    regions_df,
    calibration_predictions_df=None,
    n_sims=300,
    seed=42,
):
    """Propaga trajetórias conjuntas de demanda pelo modelo de estoque e restrições físicas."""
    trajectories, skus, horizons = sample_joint_demand_trajectories(
        forecast_df, calibration_predictions_df, n_sims=n_sims, seed=seed
    )
    n_skus = len(skus)
    n_h = len(horizons)

    fc = forecast_df.sort_values(["sku", "ds"]).copy()
    dates = sorted(pd.to_datetime(fc["ds"].unique()))

    stock_indexed = stock_df.set_index("sku").reindex(skus)
    init_stock = stock_indexed["initial_stock"].astype(float).to_numpy()
    unit_vol = stock_indexed["unit_volume_m3"].astype(float).to_numpy()
    unit_wt = (
        pd.to_numeric(stock_indexed["unit_weight_kg"], errors="coerce").fillna(10.0).to_numpy()
        if "unit_weight_kg" in stock_indexed.columns
        else np.full(n_skus, 10.0)
    )
    units_pallet = (
        pd.to_numeric(stock_indexed["units_per_pallet"], errors="coerce").fillna(25.0).to_numpy()
        if "units_per_pallet" in stock_indexed.columns
        else np.full(n_skus, 25.0)
    )
    stack_levels = (
        np.maximum(1, pd.to_numeric(stock_indexed["stackable_levels"], errors="coerce").fillna(1).astype(int).to_numpy())
        if "stackable_levels" in stock_indexed.columns
        else np.ones(n_skus, dtype=int)
    )
    sku_regions = stock_indexed["region"].astype(str).to_numpy()

    rec = receipts_df.copy()
    rec["date"] = pd.to_datetime(rec["date"])
    rec_piv = rec.pivot(index="sku", columns="date", values="inbound").reindex(index=skus, columns=dates).fillna(0.0)
    inbound_mat = rec_piv.to_numpy(dtype=float)  # (n_skus, n_h)

    # Simulação vetorizada nas n_sims trajetórias
    opening_arr = np.zeros((n_sims, n_skus, n_h), dtype=float)
    available_arr = np.zeros((n_sims, n_skus, n_h), dtype=float)
    shortage_arr = np.zeros((n_sims, n_skus, n_h), dtype=float)
    closing_arr = np.zeros((n_sims, n_skus, n_h), dtype=float)

    balance = np.repeat(init_stock[None, :], n_sims, axis=0)  # (n_sims, n_skus)
    for t in range(n_h):
        opening_arr[:, :, t] = balance
        avail = balance + inbound_mat[None, :, t]
        available_arr[:, :, t] = avail
        dem = trajectories[:, :, t]
        shortage_arr[:, :, t] = np.maximum(0.0, dem - avail)
        balance = np.maximum(0.0, avail - dem)
        closing_arr[:, :, t] = balance

    # 1. Tabela de risco por SKU e horizonte
    sku_risk_rows = []
    for i, sku in enumerate(skus):
        for t, h in enumerate(horizons):
            sh = shortage_arr[:, i, t]
            cl = closing_arr[:, i, t]
            dem = trajectories[:, i, t]
            sku_risk_rows.append({
                "sku": sku,
                "region": str(sku_regions[i]),
                "date": dates[t],
                "h": int(h),
                "demand_mean": float(dem.mean()),
                "demand_p10": float(np.quantile(dem, 0.10)),
                "demand_p90": float(np.quantile(dem, 0.90)),
                "stockout_probability_pct": float(100.0 * (sh > 1e-6).mean()),
                "expected_unmet_units": float(sh.mean()),
                "unmet_p95_units": float(np.quantile(sh, 0.95)),
                "closing_stock_p05": float(np.quantile(cl, 0.05)),
                "closing_stock_p50": float(np.quantile(cl, 0.50)),
                "closing_stock_p95": float(np.quantile(cl, 0.95)),
            })
    sku_risk_df = pd.DataFrame(sku_risk_rows)

    # 2. Tabela de risco conjunto por Região e Total do CD
    reg_df = regions_df.set_index("region")
    region_list = sorted(regions_df["region"].astype(str).unique())
    groups_to_eval = [(r, np.where(sku_regions == r)[0]) for r in region_list]
    if len(region_list) > 1:
        groups_to_eval.append(("TOTAL_CD", np.arange(n_skus)))

    region_risk_rows = []
    for reg_name, idx in groups_to_eval:
        if len(idx) == 0:
            continue
        if reg_name == "TOTAL_CD":
            cap_m3 = float(regions_df["capacity_m3"].astype(float).sum())
            max_pallets = (
                float(pd.to_numeric(regions_df["max_pallet_positions"], errors="coerce").sum())
                if "max_pallet_positions" in regions_df.columns
                else np.nan
            )
            max_weight = (
                float(pd.to_numeric(regions_df["max_floor_weight_kg"], errors="coerce").sum())
                if "max_floor_weight_kg" in regions_df.columns
                else np.nan
            )
        else:
            r_row = reg_df.loc[reg_name]
            cap_m3 = float(r_row["capacity_m3"])
            max_pallets = (
                float(pd.to_numeric(r_row.get("max_pallet_positions", np.nan), errors="coerce"))
                if "max_pallet_positions" in reg_df.columns
                else np.nan
            )
            max_weight = (
                float(pd.to_numeric(r_row.get("max_floor_weight_kg", np.nan), errors="coerce"))
                if "max_floor_weight_kg" in reg_df.columns
                else np.nan
            )

        for t, h in enumerate(horizons):
            joint_dem = trajectories[:, idx, t].sum(axis=1)
            peak_vol = (available_arr[:, idx, t] * unit_vol[None, idx]).sum(axis=1)
            end_vol = (closing_arr[:, idx, t] * unit_vol[None, idx]).sum(axis=1)

            peak_pallets = np.ceil(
                np.ceil(available_arr[:, idx, t] / units_pallet[None, idx]) / stack_levels[None, idx]
            ).sum(axis=1)
            peak_weight = (available_arr[:, idx, t] * unit_wt[None, idx]).sum(axis=1)

            any_stockout = (shortage_arr[:, idx, t] > 1e-6).any(axis=1)
            over_vol = peak_vol > cap_m3
            over_pal = (peak_pallets > max_pallets) if (np.isfinite(max_pallets) and max_pallets > 0) else np.zeros(n_sims, dtype=bool)
            over_wt = (peak_weight > max_weight) if (np.isfinite(max_weight) and max_weight > 0) else np.zeros(n_sims, dtype=bool)
            over_any = over_vol | over_pal | over_wt

            region_risk_rows.append({
                "region": reg_name,
                "date": dates[t],
                "h": int(h),
                "capacity_m3": cap_m3,
                "joint_demand_mean": float(joint_dem.mean()),
                "joint_demand_lower_80": float(np.quantile(joint_dem, 0.10)),
                "joint_demand_upper_80": float(np.quantile(joint_dem, 0.90)),
                "joint_demand_lower_95": float(np.quantile(joint_dem, 0.025)),
                "joint_demand_upper_95": float(np.quantile(joint_dem, 0.975)),
                "peak_volume_p50_m3": float(np.quantile(peak_vol, 0.50)),
                "peak_volume_p90_m3": float(np.quantile(peak_vol, 0.90)),
                "peak_volume_p95_m3": float(np.quantile(peak_vol, 0.95)),
                "closing_volume_p10_m3": float(np.quantile(end_vol, 0.10)),
                "closing_volume_p50_m3": float(np.quantile(end_vol, 0.50)),
                "closing_volume_p90_m3": float(np.quantile(end_vol, 0.90)),
                "closing_volume_p95_m3": float(np.quantile(end_vol, 0.95)),
                "peak_pallet_positions_p95": float(np.quantile(peak_pallets, 0.95)),
                "peak_weight_p95_kg": float(np.quantile(peak_weight, 0.95)),
                "any_sku_stockout_prob_pct": float(100.0 * any_stockout.mean()),
                "overcapacity_volume_prob_pct": float(100.0 * over_vol.mean()),
                "overcapacity_pallets_prob_pct": float(100.0 * over_pal.mean()) if np.isfinite(max_pallets) else np.nan,
                "overcapacity_any_constraint_prob_pct": float(100.0 * over_any.mean()),
            })

    region_risk_df = pd.DataFrame(region_risk_rows)
    return sku_risk_df, region_risk_df
