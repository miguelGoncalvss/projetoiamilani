"""Política de reposição (s, Q) com Lead Time, Estoque de Segurança e Lotes Mínimo/Múltiplo."""
import math
import numpy as np
import pandas as pd


def _round_up_multiple(qty, min_qty, multiple):
    if qty <= 0:
        return 0.0
    target = max(float(qty), float(min_qty))
    mult = float(multiple) if multiple and multiple > 0 else 1.0
    return float(math.ceil((target - 1e-9) / mult) * mult)


def plan_replenishment(forecast_df, stock_df, initial_receipts_df=None):
    """Simula política de ponto de pedido (s, Q) por SKU ao longo do horizonte previsto.

    Retorna:
      - receipts_df: DataFrame (sku, date, inbound) compatível com o contrato de simulate()
      - orders_df: DataFrame detalhado com ponto de pedido, estoque de segurança, pedidos emitidos e entregas
    """
    freq = str(forecast_df["frequency"].iloc[0])
    stock_map = stock_df.set_index("sku").to_dict(orient="index")

    in_transit_Lookup = {}
    if initial_receipts_df is not None and not initial_receipts_df.empty:
        tmp = initial_receipts_df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        for row in tmp.itertuples():
            in_transit_Lookup[(str(row.sku), pd.Timestamp(row.date))] = float(row.inbound)

    receipt_rows = []
    order_rows = []

    for sku, group in forecast_df.groupby("sku", sort=True):
        group = group.sort_values("ds").reset_index(drop=True)
        s_info = stock_map[str(sku)]
        if freq == "daily" and "lead_time_days" in s_info and pd.notna(s_info["lead_time_days"]):
            lead_time = max(1, int(float(s_info["lead_time_days"])))
        elif "lead_time_periods" in s_info and pd.notna(s_info["lead_time_periods"]):
            lead_time = max(1, int(float(s_info["lead_time_periods"])))
        else:
            lead_time = 1 if freq == "monthly" else 5

        order_mult = float(s_info.get("order_multiple", 1.0) or 1.0)
        min_order = float(s_info.get("min_order_qty", 0.0) or 0.0)
        z_factor = float(s_info.get("safety_factor_z", 1.65) or 1.65)

        n_steps = len(group)
        dates = pd.DatetimeIndex(group["ds"])
        yhats = group["yhat"].to_numpy(dtype=float)

        has_upper95 = "upper_95" in group.columns and group["upper_95"].notna().all()
        if has_upper95:
            half_widths_95 = np.maximum(0.0, group["upper_95"].to_numpy(dtype=float) - yhats)
        else:
            half_widths_95 = 0.20 * yhats

        scheduled_arrivals = np.zeros(n_steps, dtype=float)
        # Pedidos já em trânsito na data de corte (apenas dentro do lead time inicial)
        for t in range(min(lead_time, n_steps)):
            scheduled_arrivals[t] = in_transit_Lookup.get((str(sku), dates[t]), 0.0)

        balance = float(s_info["initial_stock"])

        # Avaliação na origem (t=0, antes do passo 1): se a posição inicial já estiver abaixo do ponto de pedido,
        # emite pedido na origem que chega após `lead_time` períodos (índice lead_time - 1)
        ltd_0 = float(yhats[:min(lead_time, n_steps)].sum())
        ss_0 = float((z_factor / 1.65) * half_widths_95[0] * math.sqrt(lead_time))
        rop_0 = ltd_0 + ss_0
        target_0 = rop_0 + max(min_order, float(yhats[0]))
        ip_0 = balance + float(scheduled_arrivals.sum())
        if ip_0 <= rop_0:
            origin_order = _round_up_multiple(target_0 - ip_0, min_order, order_mult)
            arrival_idx = min(lead_time - 1, n_steps - 1)
            scheduled_arrivals[arrival_idx] += origin_order

        for t in range(n_steps):
            arriving = float(scheduled_arrivals[t])
            available = balance + arriving
            fulfilled = min(available, float(yhats[t]))
            closing = max(0.0, available - float(yhats[t]))

            # Demanda esperada nos próximos `lead_time` períodos
            future_slice = yhats[t + 1: min(t + 1 + lead_time, n_steps)]
            if len(future_slice) < lead_time:
                pad_val = float(yhats[-1]) if len(future_slice) == 0 else float(future_slice.mean())
                ltd = float(future_slice.sum() + (lead_time - len(future_slice)) * pad_val)
            else:
                ltd = float(future_slice.sum())

            ss = float((z_factor / 1.65) * half_widths_95[t] * math.sqrt(lead_time))
            reorder_point = ltd + ss
            target_level = reorder_point + max(min_order, float(yhats[t]))

            in_transit_future = float(scheduled_arrivals[t + 1:].sum()) if t + 1 < n_steps else 0.0
            inventory_position = closing + in_transit_future

            order_qty = 0.0
            delivery_date = None
            if inventory_position <= reorder_point:
                raw_needed = target_level - inventory_position
                order_qty = _round_up_multiple(raw_needed, min_order, order_mult)
                arrival_t = t + lead_time
                if arrival_t < n_steps:
                    scheduled_arrivals[arrival_t] += order_qty
                    delivery_date = str(dates[arrival_t].date())
                else:
                    delivery_date = "after_horizon"

            balance = closing
            receipt_rows.append({
                "sku": str(sku),
                "date": dates[t],
                "inbound": arriving,
            })
            order_rows.append({
                "sku": str(sku),
                "date": dates[t],
                "h": int(group["h"].iloc[t]),
                "lead_time_periods": lead_time,
                "safety_stock": round(ss, 3),
                "lead_time_demand": round(ltd, 3),
                "reorder_point_s": round(reorder_point, 3),
                "target_level_S": round(target_level, 3),
                "inventory_position_before_order": round(inventory_position, 3),
                "order_placed_Q": round(order_qty, 3),
                "inbound_arriving": round(arriving, 3),
                "scheduled_delivery_date": delivery_date,
            })

    return pd.DataFrame(receipt_rows), pd.DataFrame(order_rows)
