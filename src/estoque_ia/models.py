"""Baselines, XGBoost direto global normalizado por escala/tendência e Prophet por SKU; artefatos JSON sem pickle."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import Config
from .features import training_rows, prediction_rows, month_calendar_factors


class Forecaster:
    def __init__(self, name, cfg):
        self.name, self.cfg = name, cfg
        self.models = {}
        self.target_scaled = True
        self.scale_column = "seasonal_trend_anchor"
        self.prophet_rate_scaled = True

    def fit(self, panel):
        self.panel = panel.sort_values(["sku", "ds"]).copy()
        self.skus = sorted(self.panel.sku.astype(str).unique())
        if self.name == "xgboost":
            from xgboost import XGBRegressor
            x, y, _ = training_rows(self.panel, self.cfg)
            if self.cfg.frequency == "monthly" and "seasonal_trend_anchor" in x.columns:
                scale_col = "seasonal_trend_anchor"
                max_depth, lr, mcw, reg_lam = 2, 0.02, 15, 25.0
            else:
                scale_col = "local_scale"
                max_depth, lr, mcw, reg_lam = 3, 0.05, 10, 5.0
            base_scale = np.maximum(x[scale_col].to_numpy(dtype=float), 1e-3)
            x_enc = self._encode(x)
            self.columns = x_enc.columns.tolist()
            self.target_scaled = True
            self.scale_column = scale_col
            self.regressor = XGBRegressor(
                objective="reg:squarederror", n_estimators=self.cfg.xgb_estimators,
                max_depth=max_depth, learning_rate=lr, min_child_weight=mcw, reg_lambda=reg_lam,
                subsample=1.0, colsample_bytree=1.0, n_jobs=2,
                random_state=self.cfg.seed, tree_method="hist")
            self.regressor.fit(x_enc, y / base_scale)
        elif self.name == "prophet":
            from prophet import Prophet
            self.prophet_rate_scaled = (self.cfg.frequency == "monthly")
            for sku, g in self.panel.groupby("sku", sort=True):
                m = Prophet(
                    yearly_seasonality=(False if self.cfg.frequency == "monthly" else True),
                    weekly_seasonality=self.cfg.frequency == "daily",
                    daily_seasonality=False,
                    seasonality_mode="multiplicative",
                    uncertainty_samples=0,
                    changepoint_prior_scale=0.01)
                data = self._prophet_frame(g[["ds", "y"]], rate_scaled=self.prophet_rate_scaled)
                for month in range(2, 13):
                    m.add_regressor(f"month_{month}", standardize=False, mode="multiplicative")
                if self.cfg.frequency == "monthly" and self.prophet_rate_scaled:
                    m.add_regressor("wknd_share", standardize=False, mode="multiplicative")
                m.fit(data, seed=self.cfg.seed)
                self.models[str(sku)] = m
        elif self.name not in {"naive", "seasonal"}:
            raise ValueError(f"Modelo desconhecido: {self.name}")
        return self

    def _prophet_frame(self, frame, rate_scaled=False):
        frame = frame.copy()
        for month in range(2, 13):
            frame[f"month_{month}"] = (frame.ds.dt.month == month).astype(float)
        if self.cfg.frequency == "monthly" and rate_scaled:
            days, wknd_share, _ = month_calendar_factors(frame.ds)
            frame["wknd_share"] = wknd_share
            if "y" in frame.columns:
                frame["y"] = frame["y"].astype(float) / days
        return frame

    def _encode(self, frame):
        frame = frame.copy()
        sku = frame.pop("sku").astype(str)
        if not set(sku) <= set(self.skus):
            raise ValueError("SKU desconhecido pelo modelo")
        for i, value in enumerate(self.skus):
            frame[f"sku_{i}"] = (sku == value).astype(float)
        return frame.astype(float)

    def predict(self):
        x, keys = prediction_rows(self.panel, self.cfg)
        if self.name == "xgboost":
            raw_pred = self.regressor.predict(self._encode(x)[self.columns])
            if getattr(self, "target_scaled", True):
                scale_col = getattr(self, "scale_column", "local_scale")
                if scale_col not in x.columns:
                    scale_col = "local_scale"
                raw_pred = raw_pred * x[scale_col].to_numpy(dtype=float)
            keys["yhat"] = raw_pred
        else:
            predictions = []
            for sku, g in self.panel.groupby("sku", sort=True):
                vals = g.sort_values("ds").y.to_numpy()
                if self.name == "naive":
                    pred = np.repeat(vals[-1], self.cfg.horizon)
                elif self.name == "seasonal":
                    if len(vals) < self.cfg.season:
                        raise ValueError("Sem ciclo completo para baseline sazonal")
                    min_trend_len = 60 if self.cfg.frequency == "monthly" else 365
                    if len(vals) >= min_trend_len:
                        t_period = 12 if self.cfg.frequency == "monthly" else 365
                        n_full = (len(vals) // t_period) * t_period
                        reshaped = vals[-n_full:].astype(float).reshape(-1, t_period)
                        c_means = reshaped.mean(axis=1)
                        c_idx = np.arange(len(c_means), dtype=float) * t_period
                        c_cent = c_idx - c_idx.mean()
                        denom = float(np.dot(c_cent, c_cent))
                        slope = float(np.dot(c_cent, c_means - c_means.mean()) / denom) if denom > 0 else 0.0
                        slope_norm = slope / max(float(c_means[-1]), 1.0)
                        pred = [
                            float(vals[-self.cfg.season + ((h - 1) % self.cfg.season)]) * max(0.25, 1.0 + slope_norm * (self.cfg.season + ((h - 1) // self.cfg.season) * self.cfg.season))
                            for h in range(1, self.cfg.horizon + 1)
                        ]
                    else:
                        pred = [vals[-self.cfg.season + ((h-1) % self.cfg.season)]
                                for h in range(1, self.cfg.horizon+1)]
                else:
                    future = pd.DataFrame({"ds": self.cfg.dates_after(g.ds.max())})
                    rate_scaled = bool(getattr(self, "prophet_rate_scaled", False)) and (self.cfg.frequency == "monthly")
                    pred = self.models[str(sku)].predict(self._prophet_frame(future, rate_scaled=rate_scaled)).yhat.to_numpy()
                    if rate_scaled:
                        f_days, _, _ = month_calendar_factors(future.ds)
                        pred = pred * f_days
                predictions.extend(pred)
            keys["yhat"] = predictions
        keys["yhat"] = keys.yhat.clip(lower=0)
        if not np.isfinite(keys.yhat).all():
            raise ValueError("Modelo produziu previsão não finita")
        keys["model"] = self.name
        return keys

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        meta = {"name": self.name, "config": self.cfg.to_dict(), "skus": self.skus}
        self.panel.to_csv(directory / "training_panel.csv", index=False)
        if self.name == "xgboost":
            self.regressor.save_model(directory / "xgboost.json")
            meta["columns"] = self.columns
            meta["target_scaled"] = bool(getattr(self, "target_scaled", True))
            meta["scale_column"] = getattr(self, "scale_column", "local_scale")
        elif self.name == "prophet":
            from prophet.serialize import model_to_json
            meta["prophet_rate_scaled"] = bool(getattr(self, "prophet_rate_scaled", False))
            for i, sku in enumerate(self.skus):
                (directory / f"prophet_{i}.json").write_text(model_to_json(self.models[sku]), encoding="utf-8")
        (directory / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        obj = cls(meta["name"], Config(**meta["config"]))
        obj.skus = meta["skus"]
        obj.panel = pd.read_csv(directory / "training_panel.csv", dtype={"sku": str}, parse_dates=["ds"])
        if obj.name == "xgboost":
            from xgboost import XGBRegressor
            obj.regressor = XGBRegressor()
            obj.regressor.load_model(directory / "xgboost.json")
            obj.columns = meta["columns"]
            obj.target_scaled = bool(meta.get("target_scaled", False))
            obj.scale_column = meta.get("scale_column", "local_scale")
        elif obj.name == "prophet":
            from prophet.serialize import model_from_json
            obj.prophet_rate_scaled = bool(meta.get("prophet_rate_scaled", False))
            for i, sku in enumerate(obj.skus):
                obj.models[sku] = model_from_json((directory / f"prophet_{i}.json").read_text(encoding="utf-8"))
        return obj
