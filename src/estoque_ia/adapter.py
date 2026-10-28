"""Adaptador de base bruta (ERP), reconstrução causal de demanda censurada e ajuste de protocolo."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import Config
from .data import load_history


ERP_FAMILIES = [
    {
        "region": "R_SECO",
        "compatibility_class": "SECO",
        "capacity_m3": 420.0,
        "max_pallet_positions": 320,
        "max_floor_weight_kg": 195000.0,
        "max_rack_height_m": 2.20,
        "allowed_classes": "SECO|EMBALAGEM",
        "sku_count": 6,
        "base_demand": 28.0,
        "season_amp": 0.32,
        "phase": 0.15,
        "unit_volume_m3": 0.035,
        "unit_weight_kg": 8.5,
        "units_per_pallet": 40,
        "pallet_height_m": 1.65,
        "stackable_levels": 2,
        "lead_time_monthly": 1,
        "lead_time_daily": 5,
    },
    {
        "region": "R_REFRIG",
        "compatibility_class": "REFRIGERADO",
        "capacity_m3": 260.0,
        "max_pallet_positions": 190,
        "max_floor_weight_kg": 115000.0,
        "max_rack_height_m": 1.90,
        "allowed_classes": "REFRIGERADO",
        "sku_count": 5,
        "base_demand": 22.0,
        "season_amp": 0.38,
        "phase": 0.85,
        "unit_volume_m3": 0.042,
        "unit_weight_kg": 11.0,
        "units_per_pallet": 32,
        "pallet_height_m": 1.55,
        "stackable_levels": 1,
        "lead_time_monthly": 1,
        "lead_time_daily": 4,
    },
    {
        "region": "R_QUIMICO",
        "compatibility_class": "QUIMICO",
        "capacity_m3": 190.0,
        "max_pallet_positions": 140,
        "max_floor_weight_kg": 98000.0,
        "max_rack_height_m": 1.80,
        "allowed_classes": "QUIMICO",
        "sku_count": 4,
        "base_demand": 16.0,
        "season_amp": 0.26,
        "phase": 1.55,
        "unit_volume_m3": 0.050,
        "unit_weight_kg": 16.0,
        "units_per_pallet": 24,
        "pallet_height_m": 1.50,
        "stackable_levels": 1,
        "lead_time_monthly": 2,
        "lead_time_daily": 7,
    },
]


def generate_erp_raw(directory, skus=15, seed=42, start_date="2018-01-01", end_date="2025-12-31"):
    """Gera logs transacionais brutos de ERP com padrões fortes, correlação por família e rupturas."""
    out = Path(directory)
    if out.exists():
        raise ValueError("Diretório de dados brutos já existe")
    if skus < 1:
        raise ValueError("Quantidade de SKUs deve ser positiva")
    out.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    n_days = len(dates)

    day_of_year = dates.dayofyear.to_numpy(dtype=float)
    day_of_week = dates.dayofweek.to_numpy(dtype=int)
    month_arr = dates.month.to_numpy(dtype=int)
    time_idx = np.arange(n_days, dtype=float)

    # Perfil mensal e semanal compartilhado (sinal estruturado forte para alta previsibilidade)
    monthly_multipliers = np.array([0.92, 0.88, 0.96, 1.00, 1.04, 0.98, 0.95, 1.02, 1.06, 1.10, 1.18, 1.24])
    weekly_multipliers = np.array([1.14, 1.18, 1.12, 1.08, 1.04, 0.74, 0.70])

    # Choques macro/familiares suaves para criar correlação realista entre SKUs da mesma região
    family_shocks = {
        fam["region"]: rng.normal(0.0, 0.004, size=n_days)
        for fam in ERP_FAMILIES
    }

    movements = []
    catalog_rows = []
    stock_rows = []
    monthly_receipts = []
    daily_receipts = []

    sku_idx = 0
    for fam_i, fam in enumerate(ERP_FAMILIES):
        count = fam["sku_count"] if skus == 15 else max(1, round(skus * fam["sku_count"] / 15))
        for local_i in range(count):
            if sku_idx >= skus:
                break
            sku_idx += 1
            sku = f"SKU_{sku_idx:03d}"
            base = fam["base_demand"] + 1.5 * local_i
            amp = fam["season_amp"] * (1.0 + 0.03 * (local_i % 3))
            phase = fam["phase"] + 0.08 * local_i
            trend_slope = 0.000095 * (1.0 + 0.05 * (local_i % 3))

            season_wave = 1.0 + amp * np.sin(2.0 * np.pi * (day_of_year / 365.25) + phase)
            month_factor = monthly_multipliers[month_arr - 1]
            week_factor = weekly_multipliers[day_of_week]
            trend_factor = 1.0 + trend_slope * time_idx
            shared_shock = family_shocks[fam["region"]]
            sku_noise = rng.normal(0.0, 0.007, size=n_days)

            # Taxa determinística de CD B2B com tendência + sazonalidade + ruído operacional controlado
            expected_rate = base * season_wave * month_factor * week_factor * trend_factor * (1.0 + shared_shock + sku_noise)
            true_demand = np.maximum(1, np.rint(expected_rate)).astype(int)

            # Janelas de ruptura histórica (concentradas antes de 2023 para testar reconstrução no treino)
            stockout_flags = np.zeros(n_days, dtype=int)
            for yr_offset in range(0, min(5, max(1, n_days // 365))):
                center = yr_offset * 365 + 90 + (sku_idx * 19) % 180
                if center + 4 < n_days:
                    stockout_flags[center:center + 4] = 1

            unit_vol = round(fam["unit_volume_m3"] + 0.003 * local_i, 4)
            unit_wt = round(fam["unit_weight_kg"] + 0.8 * local_i, 2)
            units_pallet = int(fam["units_per_pallet"])
            order_mult = units_pallet

            catalog_rows.append({
                "sku": sku,
                "region": fam["region"],
                "compatibility_class": fam["compatibility_class"],
                "unit_volume_m3": unit_vol,
                "unit_weight_kg": unit_wt,
                "units_per_pallet": units_pallet,
                "pallet_height_m": fam["pallet_height_m"],
                "stackable_levels": fam["stackable_levels"],
                "lead_time_periods_monthly": fam["lead_time_monthly"],
                "lead_time_periods_daily": fam["lead_time_daily"],
                "order_multiple": order_mult,
                "min_order_qty": order_mult * 2,
                "safety_factor_z": 1.65,
            })

            for j, day in enumerate(dates):
                day_str = str(day.date())
                dem = int(true_demand[j])
                is_stockout = int(stockout_flags[j])
                if is_stockout == 1 and dem > 2:
                    observed_out = max(1, int(round(dem * 0.42)))
                else:
                    observed_out = dem
                    is_stockout = 0

                # Simula sujeira de formatação em parte dos registros do ERP
                raw_sku = f" {sku} " if (j + sku_idx) % 47 == 0 else sku

                if observed_out > 0:
                    # Divide em 2 transações em alguns dias para testar agregação do adaptador
                    if j % 11 == 0 and observed_out > 4:
                        part1 = observed_out // 2
                        part2 = observed_out - part1
                        movements.append({
                            "date": day_str, "sku": raw_sku,
                            "movement_type": "SAIDA_EXPEDICAO", "quantity": part1,
                        })
                        movements.append({
                            "date": day_str, "sku": sku,
                            "movement_type": "SAIDA_EXPEDICAO", "quantity": part2,
                        })
                    else:
                        movements.append({
                            "date": day_str, "sku": raw_sku,
                            "movement_type": "SAIDA_EXPEDICAO", "quantity": observed_out,
                        })

                # Entradas regulares a cada 7 dias
                if j % 7 == (sku_idx % 7):
                    in_qty = int(max(1, round(expected_rate[max(0, j - 6):j + 1].sum())))
                    movements.append({
                        "date": day_str, "sku": sku,
                        "movement_type": "ENTRADA_FORNECEDOR", "quantity": in_qty,
                    })

                # Devoluções e ajustes esporádicos (devem ser separados pelo adaptador)
                if j % 53 == (sku_idx % 13):
                    movements.append({
                        "date": day_str, "sku": sku,
                        "movement_type": "DEVOLUCAO_CLIENTE", "quantity": int(max(1, round(base * 0.1))),
                    })
                if j % 89 == (sku_idx % 17):
                    movements.append({
                        "date": day_str, "sku": sku,
                        "movement_type": "AJUSTE_INVENTARIO", "quantity": -2,
                    })
                if is_stockout == 1:
                    movements.append({
                        "date": day_str, "sku": sku,
                        "movement_type": "RUPTURA_REGISTRADA", "quantity": int(dem - observed_out),
                    })

            # Saldo na data de corte (último dia do histórico)
            last_day_str = str(dates[-1].date())
            mean_last_30 = float(expected_rate[-30:].mean())
            initial_stock = int(round(mean_last_30 * 24))
            stock_rows.append({
                "sku": sku,
                "snapshot_date": last_day_str,
                "initial_stock": initial_stock,
                "unit_volume_m3": unit_vol,
                "region": fam["region"],
                "unit_weight_kg": unit_wt,
                "units_per_pallet": units_pallet,
                "pallet_height_m": fam["pallet_height_m"],
                "stackable_levels": fam["stackable_levels"],
                "compatibility_class": fam["compatibility_class"],
                "lead_time_periods": fam["lead_time_monthly"],
                "lead_time_days": fam["lead_time_daily"],
                "order_multiple": order_mult,
                "min_order_qty": order_mult * 2,
                "safety_factor_z": 1.65,
            })

            next_month_start = dates[-1] + pd.Timedelta(days=1)
            for m_date in pd.date_range(next_month_start, periods=12, freq="MS"):
                m_mult = monthly_multipliers[m_date.month - 1]
                planned_in = int(round(mean_last_30 * 30.0 * m_mult / order_mult) * order_mult)
                monthly_receipts.append({
                    "sku": sku, "date": str(m_date.date()), "inbound": planned_in,
                })
            for d_i, d_date in enumerate(pd.date_range(next_month_start, periods=30, freq="D")):
                planned_day = int(round(mean_last_30 * 7.0 / order_mult) * order_mult) if d_i % 7 == (sku_idx % 7) else 0
                daily_receipts.append({
                    "sku": sku, "date": str(d_date.date()), "inbound": planned_day,
                })

    region_rows = [
        {
            "region": fam["region"],
            "capacity_m3": fam["capacity_m3"],
            "max_pallet_positions": fam["max_pallet_positions"],
            "max_floor_weight_kg": fam["max_floor_weight_kg"],
            "max_rack_height_m": fam["max_rack_height_m"],
            "allowed_classes": fam["allowed_classes"],
        }
        for fam in ERP_FAMILIES
    ]

    pd.DataFrame(movements).to_csv(out / "erp_movements.csv", index=False)
    pd.DataFrame(catalog_rows).to_csv(out / "erp_sku_catalog.csv", index=False)
    pd.DataFrame(stock_rows).to_csv(out / "erp_stock_snapshot.csv", index=False)
    pd.DataFrame(region_rows).to_csv(out / "erp_regions.csv", index=False)
    pd.DataFrame(monthly_receipts).to_csv(out / "erp_receipts_monthly.csv", index=False)
    pd.DataFrame(daily_receipts).to_csv(out / "erp_receipts_daily.csv", index=False)
    (out / "ERP_RAW_README.txt").write_text(
        "DADOS BRUTOS SIMULADOS DE ERP. Contém múltiplas transações por dia, espaços em SKUs, "
        "devoluções de clientes, ajustes de inventário e eventos de ruptura. "
        "Use o comando adapt-erp para converter ao contrato validado.\n",
        encoding="utf-8",
    )
    return out


def adapt_erp_to_contract(raw_directory, output_directory, calendar_start=None, calendar_end=None):
    """Converte logs transacionais brutos de ERP para o contrato estrito do pipeline."""
    raw = Path(raw_directory)
    out = Path(output_directory)
    if out.exists():
        raise ValueError("Diretório de saída do adaptador já existe")

    mov = pd.read_csv(raw / "erp_movements.csv", dtype={"sku": "string", "movement_type": "string"})
    if mov.empty:
        raise ValueError("Arquivo de movimentações do ERP está vazio")

    raw_rows = len(mov)
    mov["sku_clean"] = mov.sku.astype(str).str.strip()
    dirty_skus = int((mov.sku.astype(str) != mov["sku_clean"]).sum())
    mov["sku"] = mov["sku_clean"]
    mov["date"] = pd.to_datetime(mov["date"], format="%Y-%m-%d", errors="raise")
    mov["quantity"] = pd.to_numeric(mov["quantity"], errors="raise")

    # Separa transações operacionais de devoluções e ajustes para não contaminar saídas
    returns_df = mov[mov.movement_type == "DEVOLUCAO_CLIENTE"]
    adjustments_df = mov[mov.movement_type == "AJUSTE_INVENTARIO"]
    outbound_df = mov[mov.movement_type == "SAIDA_EXPEDICAO"]
    inbound_df = mov[mov.movement_type == "ENTRADA_FORNECEDOR"]
    stockout_df = mov[mov.movement_type == "RUPTURA_REGISTRADA"]

    if (outbound_df.quantity < 0).any() or (inbound_df.quantity < 0).any():
        raise ValueError("Saídas de expedição e entradas de fornecedor no ERP não podem ser negativas")

    start = pd.Timestamp(calendar_start) if calendar_start else mov.date.min()
    end = pd.Timestamp(calendar_end) if calendar_end else mov.date.max()

    # Cobertura por SKU: mantém os SKUs do catálogo ou observados e reconstrói grade diária completa
    if (raw / "erp_sku_catalog.csv").exists():
        catalog = pd.read_csv(raw / "erp_sku_catalog.csv", dtype={"sku": str})
        catalog["sku"] = catalog.sku.str.strip()
        skus = sorted(catalog.sku.unique())
    else:
        skus = sorted(mov.sku.unique())

    full_dates = pd.date_range(start, end, freq="D")
    grid = pd.MultiIndex.from_product([skus, full_dates], names=["sku", "date"]).to_frame().reset_index(drop=True)

    out_agg = outbound_df.groupby(["sku", "date"], as_index=False)["quantity"].sum().rename(columns={"quantity": "outbound"})
    in_agg = inbound_df.groupby(["sku", "date"], as_index=False)["quantity"].sum().rename(columns={"quantity": "inbound"})
    so_agg = (
        stockout_df.assign(stockout=1)
        .groupby(["sku", "date"], as_index=False)["stockout"]
        .max()
    )

    history = (
        grid.merge(out_agg, on=["sku", "date"], how="left")
        .merge(in_agg, on=["sku", "date"], how="left")
        .merge(so_agg, on=["sku", "date"], how="left")
    )
    history["outbound"] = history["outbound"].fillna(0.0)
    history["inbound"] = history["inbound"].fillna(0.0)
    history["stockout"] = history["stockout"].fillna(0).astype(int)
    history["date"] = history["date"].dt.strftime("%Y-%m-%d")
    history = history[["date", "sku", "outbound", "inbound", "stockout"]].sort_values(["sku", "date"]).reset_index(drop=True)

    out.mkdir(parents=True)
    history.to_csv(out / "history.csv", index=False)

    # Valida imediatamente contra o contrato estrito
    validated = load_history(out / "history.csv")

    for src_name, dst_name in [
        ("erp_stock_snapshot.csv", "stock.csv"),
        ("erp_regions.csv", "regions.csv"),
        ("erp_receipts_monthly.csv", "receipts_monthly.csv"),
        ("erp_receipts_daily.csv", "receipts_daily.csv"),
        ("erp_sku_catalog.csv", "sku_catalog.csv"),
    ]:
        if (raw / src_name).exists():
            pd.read_csv(raw / src_name, dtype=str).to_csv(out / dst_name, index=False)

    report = {
        "raw_movement_rows": raw_rows,
        "cleaned_sku_whitespace_rows": dirty_skus,
        "separated_client_returns_rows": int(len(returns_df)),
        "separated_client_returns_units": float(returns_df.quantity.sum()) if len(returns_df) else 0.0,
        "separated_inventory_adjustments_rows": int(len(adjustments_df)),
        "separated_inventory_adjustments_net_units": float(adjustments_df.quantity.sum()) if len(adjustments_df) else 0.0,
        "adapted_daily_panel_rows": int(len(validated)),
        "sku_count": int(validated.sku.nunique()),
        "start_date": str(validated.date.min().date()),
        "end_date": str(validated.date.max().date()),
        "stockout_days_flagged": int(validated.stockout.sum()),
    }
    (out / "adaptation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "SYNTHETIC.txt").write_text(
        "DADOS SIMULADOS DE ERP ADAPTADOS AO CONTRATO DO PIPELINE.\n",
        encoding="utf-8",
    )
    return validated, report


def reconstruct_censored_demand(history):
    """Reconstrói demanda censurada em dias de stockout=1 usando apenas observações passadas (causal)."""
    df = history.sort_values(["sku", "date"]).copy()
    df["outbound_observed"] = df["outbound"].astype(float)
    reconstructed = []
    recovered_total = 0.0
    reconstructed_days = 0

    for _, group in df.groupby("sku", sort=False):
        vals = group["outbound_observed"].to_numpy(dtype=float)
        stockouts = group["stockout"].to_numpy(dtype=int)
        dates = pd.DatetimeIndex(group["date"])
        dow = dates.dayofweek.to_numpy()
        clean_vals = vals.copy()

        for i in range(len(vals)):
            if stockouts[i] == 1:
                # Busca até 8 observações anteriores do mesmo dia da semana sem ruptura
                past_idx = [
                    k for k in range(max(0, i - 63), i)
                    if dow[k] == dow[i] and stockouts[k] == 0
                ][-8:]
                # Fallback: últimas 14 observações sem ruptura
                recent_idx = [
                    k for k in range(max(0, i - 21), i)
                    if stockouts[k] == 0
                ][-14:]

                estimates = []
                if past_idx:
                    estimates.append(float(np.median(clean_vals[past_idx])))
                    estimates.append(float(np.mean(clean_vals[past_idx])))
                elif recent_idx:
                    estimates.append(float(np.mean(clean_vals[recent_idx])))

                est = float(np.mean(estimates)) if estimates else vals[i]
                new_val = max(vals[i], round(est, 2))
                if new_val > vals[i]:
                    recovered_total += (new_val - vals[i])
                    reconstructed_days += 1
                clean_vals[i] = new_val
        reconstructed.append(pd.Series(clean_vals, index=group.index))

    df["demand_reconstructed"] = pd.concat(reconstructed).sort_index()
    df["censored_recovered"] = df["demand_reconstructed"] - df["outbound_observed"]
    df["outbound"] = df["demand_reconstructed"]

    summary = {
        "stockout_rows": int((df["stockout"] == 1).sum()),
        "reconstructed_rows": int(reconstructed_days),
        "recovered_units": round(float(recovered_total), 2),
        "observed_total_outbound": round(float(df["outbound_observed"].sum()), 2),
        "reconstructed_total_demand": round(float(df["demand_reconstructed"].sum()), 2),
    }
    return df, summary


def adapt_config_to_history(panel, cfg: Config, allow_adaptation=False):
    """Ajusta folds de seleção e calibração caso o histórico seja menor que 84 meses (ex: 48 ou 60 meses),
    preservando obrigatoriamente o horizonte completo de teste final.
    """
    dates = pd.DatetimeIndex(sorted(panel.ds.unique()))
    eval_horizon = cfg.base_horizon if (cfg.forecast_horizon is not None and cfg.forecast_horizon > cfg.base_horizon) else cfg.horizon
    minimum_train = cfg.lookback + eval_horizon
    required_standard = minimum_train + (cfg.selection_folds + cfg.calibration_folds + 1) * eval_horizon

    if len(dates) >= required_standard or not allow_adaptation:
        return cfg, {
            "protocol_adapted": False,
            "available_periods": int(len(dates)),
            "selection_folds": cfg.selection_folds,
            "calibration_folds": cfg.calibration_folds,
            "forecast_horizon": cfg.horizon,
        }

    max_blocks = (len(dates) - minimum_train) // eval_horizon
    if max_blocks < 3:
        raise ValueError(
            f"Histórico insuficiente mesmo para protocolo adaptado: {len(dates)} períodos; "
            f"mínimo absoluto para 1 seleção + 1 calibração + 1 teste é {minimum_train + 3 * eval_horizon}."
        )

    # Reserva sempre 1 bloco para teste; divide os demais entre seleção e calibração
    remaining = max_blocks - 1
    new_cal = min(cfg.calibration_folds, max(1, remaining // 2))
    new_sel = min(cfg.selection_folds, max(1, remaining - new_cal))

    adapted_cfg = replace(cfg, selection_folds=new_sel, calibration_folds=new_cal)
    return adapted_cfg, {
        "protocol_adapted": True,
        "available_periods": int(len(dates)),
        "original_selection_folds": cfg.selection_folds,
        "original_calibration_folds": cfg.calibration_folds,
        "selection_folds": new_sel,
        "calibration_folds": new_cal,
        "note": "Protocolo ajustado ao tamanho do histórico preservando o bloco de teste no horizonte completo.",
    }
