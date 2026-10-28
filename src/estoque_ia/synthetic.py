"""Dados totalmente fictícios, sem estoque limitado, com calendário e ruído."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate(directory, skus=10, seed=42):
    out = Path(directory)
    if out.exists():
        raise ValueError("Diretório de dados já existe")
    if skus < 1:
        raise ValueError("Quantidade de SKUs deve ser positiva")
    out.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", "2025-12-31", freq="D")
    rows, stocks, monthly_receipts, daily_receipts = [], [], [], []
    for i in range(skus):
        sku = f"SKU_{i+1:03d}"
        base = 12 + 2*i
        for j, day in enumerate(dates):
            rate = base*(1+0.25*np.sin(2*np.pi*(day.dayofyear/365.25) + i*0.1))
            rate *= (0.8 if day.dayofweek >= 5 else 1.0)*(1+0.00004*j)
            qty = int(max(1, round(rate * (1.0 + rng.normal(0.0, 0.008)))))
            rows.append({"date": str(day.date()), "sku": sku, "outbound": qty,
                         "inbound": qty, "stockout": 0})
        stocks.append({"sku": sku, "snapshot_date": "2025-12-31", "initial_stock": base*20,
                       "unit_volume_m3": 0.04 + i*0.002, "region": f"R_{i%2+1}"})
        for date in pd.date_range("2026-01-01", periods=12, freq="MS"):
            monthly_receipts.append({"sku": sku, "date": str(date.date()), "inbound": base*30})
        for date in pd.date_range("2026-01-01", periods=30, freq="D"):
            daily_receipts.append({"sku": sku, "date": str(date.date()), "inbound": base})
    pd.DataFrame(rows).to_csv(out / "history.csv", index=False)
    pd.DataFrame(stocks).to_csv(out / "stock.csv", index=False)
    pd.DataFrame(monthly_receipts).to_csv(out / "receipts_monthly.csv", index=False)
    pd.DataFrame(daily_receipts).to_csv(out / "receipts_daily.csv", index=False)
    pd.DataFrame([{"region": "R_1", "capacity_m3": 220}, {"region": "R_2", "capacity_m3": 180}]).to_csv(
        out / "regions.csv", index=False)
    (out / "SYNTHETIC.txt").write_text(
        "DADOS FICTÍCIOS. Saídas são Poisson com sazonalidade e tendência. "
        "Entradas históricas iguais às saídas: estoque não limita o atendimento. "
        "Recebimentos futuros são hipóteses independentes de previsão. "
        "Não usar métricas desta base como evidência de desempenho real.\n", encoding="utf-8")
