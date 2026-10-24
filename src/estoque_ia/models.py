"""Baselines, XGBoost direto global e Prophet por SKU; artefatos JSON sem pickle."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import Config
from .features import training_rows, prediction_rows


class Forecaster:
    def __init__(self, name, cfg):
        self.name, self.cfg = name, cfg
        self.models = {}

    def fit(self, panel):
        self.panel = panel.sort_values(["sku", "ds"]).copy()
        self.skus = sorted(self.panel.sku.astype(str).unique())
        if self.name == "xgboost":
            from xgboost import XGBRegressor
            x, y, _ = training_rows(self.panel, self.cfg)
            x = self._encode(x)
            self.columns = x.columns.tolist()
            self.regressor = XGBRegressor(
                objective="reg:squarederror", n_estimators=self.cfg.xgb_estimators,
                max_depth=3, learning_rate=0.05, min_child_weight=5,
                subsample=1.0, colsample_bytree=1.0, n_jobs=2,
                random_state=self.cfg.seed, tree_method="hist")
            self.regressor.fit(x, y)
        elif self.name == "prophet":
            from prophet import Prophet
            for sku, g in self.panel.groupby("sku", sort=True):
                m = Prophet(
                    yearly_seasonality=(False if self.cfg.frequency == "monthly" else "auto"),
                    weekly_seasonality=self.cfg.frequency == "daily",
                    daily_seasonality=False, uncertainty_samples=0,
                    changepoint_prior_scale=0.05)
                data = self._prophet_frame(g[["ds", "y"]])
                if self.cfg.frequency == "monthly":
                    for month in range(2, 13):
                        m.add_regressor(f"month_{month}", standardize=False)
                m.fit(data, seed=self.cfg.seed)
                self.models[str(sku)] = m
        elif self.name not in {"naive", "seasonal"}:
            raise ValueError(f"Modelo desconhecido: {self.name}")
        return self

    def _prophet_frame(self, frame):
        frame = frame.copy()
        if self.cfg.frequency == "monthly":
            for month in range(2, 13):
                frame[f"month_{month}"] = (frame.ds.dt.month == month).astype(float)
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
            keys["yhat"] = self.regressor.predict(self._encode(x)[self.columns])
        else:
            predictions = []
            for sku, g in self.panel.groupby("sku", sort=True):
                vals = g.sort_values("ds").y.to_numpy()
                if self.name == "naive":
                    pred = np.repeat(vals[-1], self.cfg.horizon)
                elif self.name == "seasonal":
                    if len(vals) < self.cfg.season:
                        raise ValueError("Sem ciclo completo para baseline sazonal")
                    pred = [vals[-self.cfg.season + ((h-1) % self.cfg.season)]
                            for h in range(1, self.cfg.horizon+1)]
                else:
                    future = pd.DataFrame({"ds": self.cfg.dates_after(g.ds.max())})
                    pred = self.models[str(sku)].predict(self._prophet_frame(future)).yhat.to_numpy()
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
        elif self.name == "prophet":
            from prophet.serialize import model_to_json
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
        elif obj.name == "prophet":
            from prophet.serialize import model_from_json
            for i, sku in enumerate(obj.skus):
                obj.models[sku] = model_from_json((directory / f"prophet_{i}.json").read_text(encoding="utf-8"))
        return obj
