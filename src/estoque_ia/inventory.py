"""Simulação de cenário: recebimentos fixos ou política (s, Q), restrições físicas e risco conjunto."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import read_numeric_csv
from .replenishment import plan_replenishment
from .physical import enrich_inventory_physical, enrich_occupancy_physical
from .joint_risk import simulate_joint_risk


def simulate(
    forecast_path,
    stock_path,
    receipts_path,
    regions_path,
    output,
    scenario,
    policy="fixed_receipts",
    calibration_predictions_path=None,
    n_sims=300,
    seed=42,
):
    out = Path(output)
    if out.exists():
        raise ValueError("Saída já existe; escolha outro diretório")
    forecast = pd.read_csv(forecast_path, dtype={"sku": str}, parse_dates=["ds", "origin"])
    required = {"sku", "ds", "origin", "h", "yhat", "frequency", "model"}
    if not required <= set(forecast) or forecast.empty or forecast[list(required)].isna().any().any():
        raise ValueError("Forecast inválido")
    if forecast.duplicated(["sku", "ds"]).any() or forecast.frequency.nunique() != 1 or forecast.origin.nunique() != 1:
        raise ValueError("Use uma execução e uma frequência por simulação")
    if forecast.model.nunique() != 1:
        raise ValueError("Use apenas um modelo no cenário")
    if not np.isfinite(forecast.yhat).all() or (forecast.yhat < 0).any():
        raise ValueError("Previsões devem ser finitas e não negativas")
    frequency = forecast.frequency.iloc[0]
    if frequency not in {"monthly", "daily"}:
        raise ValueError("Frequência inválida")
    origin = forecast.origin.iloc[0]
    snapshot = origin + pd.offsets.MonthEnd(0) if frequency == "monthly" else origin
    expected = pd.date_range(origin + (pd.offsets.MonthBegin(1) if frequency == "monthly" else pd.Timedelta(days=1)),
                             periods=12 if frequency == "monthly" else 30, freq="MS" if frequency == "monthly" else "D")
    for sku, g in forecast.groupby("sku"):
        if not pd.DatetimeIndex(g.ds.sort_values()).equals(expected):
            raise ValueError(f"Horizonte incompleto para {sku}")
    stock = read_numeric_csv(stock_path, ["sku"], ["initial_stock", "unit_volume_m3"], ["unit_volume_m3"])
    if not {"region", "snapshot_date"} <= set(stock) or stock[["region", "snapshot_date"]].isna().any().any():
        raise ValueError("Stock exige region e snapshot_date")
    stock["region"] = stock.region.astype(str)
    if not (pd.to_datetime(stock.snapshot_date, format="%Y-%m-%d") == snapshot).all():
        raise ValueError(f"Saldo inicial deve ser o fechamento de {snapshot.date()}")
    regions = read_numeric_csv(regions_path, ["region"], ["capacity_m3"], ["capacity_m3"])
    if set(stock.sku) != set(forecast.sku):
        raise ValueError("SKUs do saldo e da previsão devem coincidir")
    if not set(stock.region) <= set(regions.region):
        raise ValueError("Região desconhecida")

    orders_df = None
    if policy == "sQ":
        init_rec = None
        if receipts_path and str(receipts_path).lower() not in ("none", "auto") and Path(receipts_path).exists():
            init_rec = read_numeric_csv(receipts_path, ["sku", "date"], ["inbound"])
        receipts, orders_df = plan_replenishment(forecast, stock, init_rec)
    elif policy == "fixed_receipts":
        receipts = read_numeric_csv(receipts_path, ["sku", "date"], ["inbound"])
        receipts["date"] = pd.to_datetime(receipts.date, format="%Y-%m-%d")
        keys = set(zip(forecast.sku, forecast.ds))
        if keys != set(zip(receipts.sku, receipts.date)):
            raise ValueError("Recebimentos devem explicitar todo SKU/período, inclusive zeros, sem extras")
    else:
        raise ValueError(f"Política de reposição desconhecida: {policy}")

    frame = forecast.merge(stock[["sku", "initial_stock", "unit_volume_m3", "region"]], on="sku", validate="many_to_one").merge(
        receipts.rename(columns={"date": "ds"}), on=["sku", "ds"], validate="one_to_one")
    records = []
    for sku, group in frame.groupby("sku", sort=True):
        balance = float(group.initial_stock.iloc[0])
        for row in group.sort_values("ds").itertuples():
            start = balance
            available = start + row.inbound
            shortage = max(0.0, row.yhat - available)
            balance = max(0.0, available - row.yhat)
            records.append({"sku": str(sku), "date": row.ds, "region": row.region,
                            "scenario": scenario, "policy": policy, "model": row.model,
                            "opening_stock": start, "inbound": row.inbound,
                            "forecast_outbound": row.yhat, "fulfilled": min(available, row.yhat),
                            "unmet_scenario": shortage, "closing_stock": balance,
                            "required_volume_after_receipts_m3": available * row.unit_volume_m3,
                            "required_volume_end_m3": balance * row.unit_volume_m3})
    result = pd.DataFrame(records)
    result = enrich_inventory_physical(result, stock, regions)

    occupation = result.groupby(["region", "date"], as_index=False)[
        ["required_volume_after_receipts_m3", "required_volume_end_m3"]].sum().merge(
            regions[["region", "capacity_m3"]], on="region", validate="many_to_one")
    for point in ("after_receipts", "end"):
        occupation[f"required_occupancy_{point}_pct"] = 100*occupation[f"required_volume_{point}_m3"]/occupation.capacity_m3
        occupation[f"excess_{point}_m3"] = (occupation[f"required_volume_{point}_m3"]-occupation.capacity_m3).clip(lower=0)
    occupation["scenario"] = scenario
    occupation["policy"] = policy
    occupation = enrich_occupancy_physical(occupation, result, regions)

    # Descoberta automática de calibration_predictions.csv se estiver no mesmo diretório do forecast.csv
    cal_df = None
    cal_path = Path(calibration_predictions_path) if calibration_predictions_path else (Path(forecast_path).parent / "calibration_predictions.csv")
    if cal_path and cal_path.exists():
        cal_df = pd.read_csv(cal_path, dtype={"sku": str})

    sku_risk_df, region_risk_df = simulate_joint_risk(
        forecast, stock, receipts, regions, calibration_predictions_df=cal_df, n_sims=n_sims, seed=seed
    )

    out.mkdir(parents=True)
    result.to_csv(out / "inventory.csv", index=False)
    occupation.to_csv(out / "occupancy.csv", index=False)
    sku_risk_df.to_csv(out / "joint_sku_risk.csv", index=False)
    region_risk_df.to_csv(out / "joint_region_risk.csv", index=False)
    if orders_df is not None:
        orders_df.to_csv(out / "replenishment_orders.csv", index=False)
    return result, occupation
