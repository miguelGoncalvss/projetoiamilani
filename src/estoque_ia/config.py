from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class Config:
    frequency: str = "monthly"
    models: list[str] = field(default_factory=lambda: ["naive", "seasonal", "xgboost", "prophet"])
    selection_folds: int = 2
    calibration_folds: int = 2
    minimum_calibration_samples: int = 20
    seed: int = 42
    xgb_estimators: int = 100
    forecast_horizon: int | None = None

    def __post_init__(self):
        if self.frequency not in {"monthly", "daily"}:
            raise ValueError("frequency deve ser monthly ou daily")
        allowed = {"naive", "seasonal", "xgboost", "prophet"}
        if not self.models or not set(self.models) <= allowed or len(set(self.models)) != len(self.models):
            raise ValueError("Lista de modelos inválida ou duplicada")
        for name in ("selection_folds", "calibration_folds", "minimum_calibration_samples", "xgb_estimators"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} deve ser inteiro positivo")
        if self.forecast_horizon is not None:
            if isinstance(self.forecast_horizon, bool) or not isinstance(self.forecast_horizon, int) or self.forecast_horizon < 1:
                raise ValueError("forecast_horizon deve ser inteiro positivo")

    @property
    def base_horizon(self):
        return 12 if self.frequency == "monthly" else 30

    @property
    def horizon(self):
        return self.forecast_horizon if self.forecast_horizon is not None else self.base_horizon

    @property
    def lookback(self):
        return 12 if self.frequency == "monthly" else 28

    @property
    def season(self):
        return 12 if self.frequency == "monthly" else 7

    @property
    def freq(self):
        return "MS" if self.frequency == "monthly" else "D"

    def dates_after(self, origin, horizon=None):
        import pandas as pd
        periods = int(horizon) if horizon is not None else self.horizon
        offset = pd.offsets.MonthBegin(1) if self.frequency == "monthly" else pd.Timedelta(days=1)
        return pd.date_range(origin + offset, periods=periods, freq=self.freq)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def read(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "horizon" in raw and "forecast_horizon" not in raw:
            raw["forecast_horizon"] = raw.pop("horizon")
        if "forecast_years" in raw:
            years = raw.pop("forecast_years")
            if "forecast_horizon" not in raw and years is not None:
                freq = raw.get("frequency", "monthly")
                raw["forecast_horizon"] = int(years) * (12 if freq == "monthly" else 365)
        return cls(**raw)
