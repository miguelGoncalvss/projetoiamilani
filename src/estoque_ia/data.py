"""Contrato estrito: uma linha por SKU/dia, inclusive dias explicitamente zerados."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_history(path):
    df = pd.read_csv(path, dtype={"sku": "string"})
    required = {"date", "sku", "outbound", "inbound", "stockout"}
    if not required <= set(df):
        raise ValueError(f"Colunas ausentes: {sorted(required - set(df))}")
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError("Histórico vazio ou com valores ausentes")
    if df.sku.str.strip().eq("").any() or (df.sku != df.sku.str.strip()).any():
        raise ValueError("SKU vazio ou com espaços nas extremidades")
    df["date"] = pd.to_datetime(df.date, format="%Y-%m-%d", errors="raise")
    for c in ["outbound", "inbound", "stockout"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
        if not np.isfinite(df[c]).all() or (df[c] < 0).any():
            raise ValueError(f"{c} precisa ser finito e não negativo")
    if not df.stockout.isin([0, 1]).all():
        raise ValueError("stockout deve ser 0 ou 1")
    if df.duplicated(["sku", "date"]).any():
        raise ValueError("Duplicidade SKU/data: agregue transações antes de importar")
    # Nesta primeira versão, todos os SKUs pertencem ao mesmo painel completo.
    dates = pd.date_range(df.date.min(), df.date.max(), freq="D")
    for sku, group in df.groupby("sku"):
        if not pd.DatetimeIndex(group.date.sort_values()).equals(dates):
            raise ValueError(f"Lacunas ou cobertura diferente no SKU {sku}; não serão preenchidas com zero")
    return df.sort_values(["sku", "date"]).reset_index(drop=True)


def make_panel(history, cfg):
    if cfg.frequency == "daily":
        return history[["sku", "date", "outbound"]].rename(columns={"date": "ds", "outbound": "y"})
    start, end = history.date.min(), history.date.max()
    if not start.is_month_start or not end.is_month_end:
        raise ValueError("Previsão mensal exige meses completos, do primeiro ao último dia")
    df = history.assign(ds=history.date.dt.to_period("M").dt.to_timestamp())
    return df.groupby(["sku", "ds"], as_index=False).outbound.sum().rename(columns={"outbound": "y"})


def read_numeric_csv(path, keys, numeric, positive=()):
    df = pd.read_csv(Path(path), dtype={k: "string" for k in keys if k != "date"})
    required = set(keys) | set(numeric)
    if not required <= set(df) or df[list(required)].isna().any().any():
        raise ValueError(f"CSV precisa conter valores para {sorted(required)}")
    if df.duplicated(keys).any():
        raise ValueError(f"Chaves duplicadas: {keys}")
    for k in keys:
        if k != "date" and (df[k].str.strip().eq("").any() or (df[k] != df[k].str.strip()).any()):
            raise ValueError(f"Chave inválida em {k}")
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="raise")
        if not np.isfinite(df[c]).all() or (df[c] < 0).any() or (c in positive and (df[c] <= 0).any()):
            raise ValueError(f"Valores inválidos em {c}")
    return df
