import numpy as np
import pandas as pd
import pytest
from estoque_ia.config import Config
from estoque_ia.adapter import (
    generate_erp_raw,
    adapt_erp_to_contract,
    reconstruct_censored_demand,
    adapt_config_to_history,
)
from estoque_ia.replenishment import plan_replenishment
from estoque_ia.physical import enrich_inventory_physical, enrich_occupancy_physical
from estoque_ia.joint_risk import simulate_joint_risk
from estoque_ia.inventory import simulate


def test_erp_generator_and_adapter_clean_and_validate(tmp_path):
    raw_dir = tmp_path / "erp_raw"
    adapted_dir = tmp_path / "erp_adapted"
    generate_erp_raw(raw_dir, skus=3, seed=7, start_date="2024-01-01", end_date="2024-03-31")
    history, report = adapt_erp_to_contract(raw_dir, adapted_dir)

    assert report["sku_count"] == 3
    assert report["cleaned_sku_whitespace_rows"] > 0
    assert report["separated_client_returns_rows"] > 0
    assert report["separated_inventory_adjustments_rows"] > 0
    assert len(history) == 3 * 91
    assert set(history.columns) == {"date", "sku", "outbound", "inbound", "stockout"}
    assert (adapted_dir / "stock.csv").exists()
    assert (adapted_dir / "regions.csv").exists()


def test_reconstruct_censored_demand_is_causal_and_recovers_lost_units():
    dates = pd.date_range("2025-01-01", periods=35, freq="D")
    rows = []
    for i, d in enumerate(dates):
        is_so = 1 if i == 28 else 0
        out = 4.0 if is_so else 20.0
        rows.append({"date": d, "sku": "SKU_001", "outbound": out, "inbound": 20.0, "stockout": is_so})
    hist = pd.DataFrame(rows)

    rec1, summary = reconstruct_censored_demand(hist)
    assert summary["stockout_rows"] == 1
    assert summary["recovered_units"] > 10.0
    assert rec1.loc[rec1.stockout == 1, "outbound"].iloc[0] == pytest.approx(20.0, rel=1e-2)

    # Verifica invariância causal: alterar valores após o dia 28 não muda a reconstrução do dia 28
    hist_future_changed = hist.copy()
    hist_future_changed.loc[30:, "outbound"] = 9999.0
    rec2, _ = reconstruct_censored_demand(hist_future_changed)
    assert rec1.loc[28, "outbound"] == rec2.loc[28, "outbound"]


def test_protocol_adaptation_for_60_months_preserves_full_test_horizon():
    dates = pd.date_range("2021-01-01", periods=60, freq="MS")
    panel = pd.DataFrame([{"sku": "S1", "ds": d, "y": 15.0} for d in dates])
    cfg = Config(frequency="monthly", selection_folds=2, calibration_folds=2)

    adapted_cfg, report = adapt_config_to_history(panel, cfg, allow_adaptation=True)
    assert report["protocol_adapted"] is True
    assert adapted_cfg.selection_folds == 1
    assert adapted_cfg.calibration_folds == 1
    assert adapted_cfg.horizon == 12


def test_replenishment_policy_sQ_respects_lot_multiples_and_lead_time(tmp_path):
    ds = pd.date_range("2026-01-01", periods=12, freq="MS")
    forecast = pd.DataFrame({
        "sku": ["SKU_A"] * 12,
        "ds": ds,
        "origin": [pd.Timestamp("2025-12-01")] * 12,
        "h": range(1, 13),
        "yhat": [50.0] * 12,
        "upper_95": [60.0] * 12,
        "scale": [50.0] * 12,
        "q95": [0.2] * 12,
        "frequency": ["monthly"] * 12,
        "model": ["prophet"] * 12,
    })
    stock = pd.DataFrame([{
        "sku": "SKU_A",
        "snapshot_date": "2025-12-31",
        "initial_stock": 80.0,
        "unit_volume_m3": 0.05,
        "region": "R_1",
        "unit_weight_kg": 12.0,
        "units_per_pallet": 25,
        "pallet_height_m": 1.6,
        "stackable_levels": 2,
        "compatibility_class": "SECO",
        "lead_time_periods": 1,
        "order_multiple": 25,
        "min_order_qty": 50,
        "safety_factor_z": 1.65,
    }])
    regions = pd.DataFrame([{
        "region": "R_1",
        "capacity_m3": 50.0,
        "max_pallet_positions": 10,
        "max_floor_weight_kg": 5000.0,
        "max_rack_height_m": 2.0,
        "allowed_classes": "SECO",
    }])

    fc_path = tmp_path / "forecast.csv"
    st_path = tmp_path / "stock.csv"
    rg_path = tmp_path / "regions.csv"
    forecast.to_csv(fc_path, index=False)
    stock.to_csv(st_path, index=False)
    regions.to_csv(rg_path, index=False)

    inv, occ = simulate(
        fc_path, st_path, "auto", rg_path, tmp_path / "sim_sq", "cenario_sQ", policy="sQ", n_sims=50, seed=42
    )
    orders = pd.read_csv(tmp_path / "sim_sq" / "replenishment_orders.csv")
    placed = orders[orders["order_placed_Q"] > 0]["order_placed_Q"]
    assert len(placed) > 0
    assert ((placed % 25) == 0).all()
    assert (placed >= 50).all()
    assert (tmp_path / "sim_sq" / "joint_sku_risk.csv").exists()
    assert (tmp_path / "sim_sq" / "joint_region_risk.csv").exists()
    assert "required_pallet_positions_after_receipts" in occ.columns


def test_physical_constraints_detect_incompatibility_and_height_violation():
    inv = pd.DataFrame([{
        "sku": "SKU_Q",
        "date": pd.Timestamp("2026-01-01"),
        "region": "R_SECO",
        "opening_stock": 100.0,
        "inbound": 0.0,
        "closing_stock": 80.0,
        "required_volume_after_receipts_m3": 10.0,
        "required_volume_end_m3": 8.0,
    }])
    stock = pd.DataFrame([{
        "sku": "SKU_Q",
        "unit_weight_kg": 20.0,
        "units_per_pallet": 20,
        "pallet_height_m": 2.10,
        "stackable_levels": 2,
        "compatibility_class": "QUIMICO",
    }])
    regions = pd.DataFrame([{
        "region": "R_SECO",
        "capacity_m3": 100.0,
        "max_pallet_positions": 4,
        "max_floor_weight_kg": 1500.0,
        "max_rack_height_m": 1.80,
        "allowed_classes": "SECO",
    }])

    enriched_inv = enrich_inventory_physical(inv, stock, regions)
    assert bool(enriched_inv["compatible_region"].iloc[0]) is False
    assert bool(enriched_inv["height_violation"].iloc[0]) is True
    # Como houve violação de altura, empilhamento efetivo cai para 1 -> 5 pallets = 5 posições
    assert enriched_inv["pallet_positions_after_receipts"].iloc[0] == 5

    base_occ = pd.DataFrame([{
        "region": "R_SECO",
        "date": pd.Timestamp("2026-01-01"),
        "required_volume_after_receipts_m3": 10.0,
        "required_volume_end_m3": 8.0,
        "capacity_m3": 100.0,
        "required_occupancy_after_receipts_pct": 10.0,
        "required_occupancy_end_pct": 8.0,
        "excess_after_receipts_m3": 0.0,
        "excess_end_m3": 0.0,
    }])
    occ = enrich_occupancy_physical(base_occ, enriched_inv, regions)
    assert occ["excess_pallet_positions_after_receipts"].iloc[0] == 1.0
    assert occ["excess_weight_after_receipts_kg"].iloc[0] == 500.0
    assert occ["dominant_constraint_after_receipts"].iloc[0] == "PHYSICAL_COMPATIBILITY_OR_HEIGHT"
