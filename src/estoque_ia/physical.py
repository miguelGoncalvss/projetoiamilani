"""Restrições físicas de armazenagem: posições-pallet, peso, altura de rack, empilhamento e compatibilidade."""
import numpy as np
import pandas as pd


def enrich_inventory_physical(inventory_df, stock_df, regions_df):
    """Adiciona cálculos de pallets, posições-pallet, peso, altura e compatibilidade por SKU/período."""
    inv = inventory_df.copy()
    stock = stock_df.copy()
    regions = regions_df.copy()

    # Defaults seguros para retrocompatibilidade com CSVs mínimos da v0.1
    defaults_stock = {
        "unit_weight_kg": 10.0,
        "units_per_pallet": 25.0,
        "pallet_height_m": 1.50,
        "stackable_levels": 1,
        "compatibility_class": "GERAL",
    }
    for col, val in defaults_stock.items():
        if col not in stock.columns:
            stock[col] = val

    defaults_region = {
        "max_pallet_positions": np.nan,
        "max_floor_weight_kg": np.nan,
        "max_rack_height_m": 2.50,
        "allowed_classes": "*",
    }
    for col, val in defaults_region.items():
        if col not in regions.columns:
            regions[col] = val

    for col in ("unit_weight_kg", "units_per_pallet", "pallet_height_m", "stackable_levels"):
        stock[col] = pd.to_numeric(stock[col], errors="raise")
    regions["max_rack_height_m"] = pd.to_numeric(regions["max_rack_height_m"], errors="coerce").fillna(2.50)

    phys_cols = ["sku", "unit_weight_kg", "units_per_pallet", "pallet_height_m", "stackable_levels", "compatibility_class"]
    reg_cols = ["region", "max_rack_height_m", "allowed_classes"]

    inv = inv.merge(stock[phys_cols], on="sku", how="left", validate="many_to_one")
    inv = inv.merge(regions[reg_cols], on="region", how="left", validate="many_to_one")

    available_units = inv["opening_stock"] + inv["inbound"]
    closing_units = inv["closing_stock"]

    # Compatibilidade entre classe do SKU e classes permitidas na região
    def _check_compat(row):
        allowed = str(row["allowed_classes"]).strip()
        if allowed in ("*", "", "nan", "None"):
            return True
        allowed_set = {c.strip().upper() for c in allowed.split("|") if c.strip()}
        return str(row["compatibility_class"]).strip().upper() in allowed_set

    inv["compatible_region"] = inv.apply(_check_compat, axis=1)
    inv["height_violation"] = inv["pallet_height_m"] > inv["max_rack_height_m"]

    eff_stack = np.maximum(1, inv["stackable_levels"].astype(int))
    inv["effective_stack_levels"] = np.where(inv["height_violation"], 1, eff_stack)

    inv["pallets_after_receipts"] = np.ceil(available_units / inv["units_per_pallet"]).astype(int)
    inv["pallets_end"] = np.ceil(closing_units / inv["units_per_pallet"]).astype(int)

    inv["pallet_positions_after_receipts"] = np.ceil(
        inv["pallets_after_receipts"] / inv["effective_stack_levels"]
    ).astype(int)
    inv["pallet_positions_end"] = np.ceil(
        inv["pallets_end"] / inv["effective_stack_levels"]
    ).astype(int)

    inv["weight_after_receipts_kg"] = available_units * inv["unit_weight_kg"]
    inv["weight_end_kg"] = closing_units * inv["unit_weight_kg"]

    return inv


def enrich_occupancy_physical(occupancy_df, enriched_inventory_df, regions_df):
    """Agrega posições-pallet, peso e violações físicas por região/período e identifica gargalos."""
    occ = occupancy_df.copy()
    reg = regions_df.copy()

    agg = enriched_inventory_df.groupby(["region", "date"], as_index=False).agg(
        required_pallets_after_receipts=("pallets_after_receipts", "sum"),
        required_pallets_end=("pallets_end", "sum"),
        required_pallet_positions_after_receipts=("pallet_positions_after_receipts", "sum"),
        required_pallet_positions_end=("pallet_positions_end", "sum"),
        required_weight_after_receipts_kg=("weight_after_receipts_kg", "sum"),
        required_weight_end_kg=("weight_end_kg", "sum"),
        incompatible_skus=("compatible_region", lambda s: int((~s).sum())),
        height_violation_skus=("height_violation", lambda s: int(s.sum())),
    )

    occ = occ.merge(agg, on=["region", "date"], how="left", validate="one_to_one")

    if "max_pallet_positions" in reg.columns:
        reg_pallets = reg[["region", "max_pallet_positions"]].copy()
        reg_pallets["max_pallet_positions"] = pd.to_numeric(reg_pallets["max_pallet_positions"], errors="coerce")
        if "max_pallet_positions" not in occ.columns:
            occ = occ.merge(reg_pallets, on="region", how="left")
    else:
        occ["max_pallet_positions"] = np.nan

    if "max_floor_weight_kg" in reg.columns:
        reg_wt = reg[["region", "max_floor_weight_kg"]].copy()
        reg_wt["max_floor_weight_kg"] = pd.to_numeric(reg_wt["max_floor_weight_kg"], errors="coerce")
        if "max_floor_weight_kg" not in occ.columns:
            occ = occ.merge(reg_wt, on="region", how="left")
    else:
        occ["max_floor_weight_kg"] = np.nan

    for point in ("after_receipts", "end"):
        pos_col = f"required_pallet_positions_{point}"
        wt_col = f"required_weight_{point}_kg"

        occ[f"required_pallet_occupancy_{point}_pct"] = np.where(
            occ["max_pallet_positions"].notna() & (occ["max_pallet_positions"] > 0),
            100.0 * occ[pos_col] / occ["max_pallet_positions"],
            np.nan,
        )
        occ[f"excess_pallet_positions_{point}"] = np.where(
            occ["max_pallet_positions"].notna() & (occ["max_pallet_positions"] > 0),
            np.maximum(0.0, occ[pos_col] - occ["max_pallet_positions"]),
            0.0,
        )

        occ[f"required_weight_occupancy_{point}_pct"] = np.where(
            occ["max_floor_weight_kg"].notna() & (occ["max_floor_weight_kg"] > 0),
            100.0 * occ[wt_col] / occ["max_floor_weight_kg"],
            np.nan,
        )
        occ[f"excess_weight_{point}_kg"] = np.where(
            occ["max_floor_weight_kg"].notna() & (occ["max_floor_weight_kg"] > 0),
            np.maximum(0.0, occ[wt_col] - occ["max_floor_weight_kg"]),
            0.0,
        )

    def _bottleneck(row):
        if row["incompatible_skus"] > 0 or row["height_violation_skus"] > 0:
            return "PHYSICAL_COMPATIBILITY_OR_HEIGHT"
        candidates = {
            "VOLUME": float(row.get("required_occupancy_after_receipts_pct", 0.0) or 0.0),
            "PALLET_POSITIONS": float(row.get("required_pallet_occupancy_after_receipts_pct", 0.0) or 0.0),
            "WEIGHT": float(row.get("required_weight_occupancy_after_receipts_pct", 0.0) or 0.0),
        }
        top_name, top_val = max(candidates.items(), key=lambda kv: kv[1] if np.isfinite(kv[1]) else -1.0)
        return f"OVER_{top_name}" if top_val > 100.0 else f"OK_LEADING_{top_name}"

    occ["dominant_constraint_after_receipts"] = occ.apply(_bottleneck, axis=1)
    return occ
