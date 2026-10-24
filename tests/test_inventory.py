import pandas as pd
import pytest
from estoque_ia.inventory import simulate


def inputs(tmp_path):
    ds = pd.date_range("2026-01-01", periods=12, freq="MS")
    forecast = pd.DataFrame({"sku": ["A"]*12, "ds": ds, "origin": [pd.Timestamp("2025-12-01")]*12,
                             "h": range(1, 13), "yhat": [8.]*12, "frequency": ["monthly"]*12,
                             "model": ["seasonal"]*12})
    stock = pd.DataFrame([{"sku": "A", "snapshot_date": "2025-12-31", "initial_stock": 10,
                           "unit_volume_m3": 1, "region": "R"}])
    receipts = pd.DataFrame({"sku": ["A"]*12, "date": ds, "inbound": [5.]*12})
    regions = pd.DataFrame([{"region": "R", "capacity_m3": 12}])
    paths = []
    for name, frame in [("forecast", forecast), ("stock", stock), ("receipts", receipts), ("regions", regions)]:
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def test_conservation_and_receipt_peak_over_capacity(tmp_path):
    rows, occupancy = simulate(*inputs(tmp_path), tmp_path / "out", "test")
    assert rows.closing_stock.iloc[0] == 7
    assert rows.unmet_scenario.iloc[3] == 2
    assert (rows.closing_stock >= 0).all()
    assert ((rows.opening_stock + rows.inbound - rows.fulfilled - rows.closing_stock).abs() < 1e-8).all()
    assert occupancy.required_occupancy_after_receipts_pct.iloc[0] == 125
    assert occupancy.excess_after_receipts_m3.iloc[0] == 3


def test_missing_receipt_cannot_silently_mean_zero(tmp_path):
    paths = inputs(tmp_path)
    receipts = pd.read_csv(paths[2]).iloc[1:]
    receipts.to_csv(paths[2], index=False)
    with pytest.raises(ValueError, match="Recebimentos"):
        simulate(*paths, tmp_path / "out", "test")


def test_snapshot_must_match_forecast_origin(tmp_path):
    paths = inputs(tmp_path)
    stock = pd.read_csv(paths[1])
    stock["snapshot_date"] = "2024-12-31"
    stock.to_csv(paths[1], index=False)
    with pytest.raises(ValueError, match="Saldo inicial"):
        simulate(*paths, tmp_path / "out", "test")
