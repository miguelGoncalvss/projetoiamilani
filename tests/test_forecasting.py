import numpy as np
import pandas as pd
import pytest
from estoque_ia.config import Config
from estoque_ia.data import load_history, make_panel
from estoque_ia.features import training_rows
from estoque_ia.models import Forecaster
from estoque_ia.evaluation import make_splits, metrics, calibrate, apply_intervals


def panel(n=96, skus=2, frequency="monthly"):
    dates = pd.date_range("2018-01-01", periods=n, freq="MS" if frequency == "monthly" else "D")
    return pd.DataFrame([{"sku": f"S{i}", "ds": d, "y": float(10+i+j%12)}
                         for i in range(skus) for j, d in enumerate(dates)])


def test_temporal_blocks_are_separate_and_full_horizon():
    cfg = Config()
    data = panel()
    splits = make_splits(data, cfg)
    assert [s["phase"] for s in splits] == ["selection", "selection", "calibration", "calibration", "test"]
    for s in splits:
        assert s["origin"] < s["target_start"] <= s["target_end"]
        assert len(pd.date_range(s["target_start"], s["target_end"], freq="MS")) == 12
    assert splits[1]["target_end"] < splits[2]["target_start"]
    assert splits[-2]["target_end"] < splits[-1]["target_start"]


def test_short_history_is_not_silently_backtested_on_shorter_horizon():
    with pytest.raises(ValueError, match="Histórico insuficiente"):
        make_splits(panel(36), Config())


def test_features_do_not_use_future_observations():
    cfg = Config()
    original = panel(36, 1)
    changed = original.copy()
    changed.loc[changed.ds > pd.Timestamp("2019-01-01"), "y"] = 99999
    x1, y1, a1 = training_rows(original, cfg)
    x2, y2, a2 = training_rows(changed, cfg)
    keep = a1.origin <= pd.Timestamp("2019-01-01")
    pd.testing.assert_frame_equal(x1[keep], x2[keep])
    assert np.any(y1[keep] != y2[keep])  # Change reaches targets, not features.
    assert (a1.target > a1.origin).all()
    assert a2.target.max() <= changed.ds.max()


def test_seasonal_repeats_last_cycle_for_all_steps():
    data = panel(24, 1)
    pred = Forecaster("seasonal", Config()).fit(data).predict()
    np.testing.assert_array_equal(pred.yhat, data.y.iloc[-12:])
    daily = panel(58, 1, "daily")
    result = Forecaster("seasonal", Config(frequency="daily")).fit(daily).predict()
    np.testing.assert_array_equal(result.yhat, np.resize(daily.y.iloc[-7:].to_numpy(), 30))


def test_zero_actual_has_no_undefined_percentage():
    result = metrics(pd.DataFrame({"y": [0, 0], "yhat": [1, 2]}))
    assert result["wape_pct"] is None
    assert result["mae"] == 1.5


def test_insufficient_calibration_does_not_invent_intervals():
    data = pd.DataFrame({"h": [1, 1], "y": [2., 4.], "yhat": [1., 2.], "scale": [1., 1.]})
    bands = apply_intervals(data, calibrate(data, minimum=20))
    assert bands.lower_95.isna().all()
    assert (bands.interval_status == "insufficient_calibration").all()


def test_empirical_interval_widths_and_nonnegative_lower():
    data = pd.DataFrame({"h": [1]*25, "y": np.arange(25), "yhat": [0.]*25, "scale": [1.]*25})
    bands = apply_intervals(data, calibrate(data, minimum=20))
    assert (bands.upper_95 >= bands.upper_80).all()
    assert (bands.lower_95 >= 0).all()


@pytest.mark.parametrize("bad", ["gap", "duplicate", "negative", "infinity"])
def test_history_validation(tmp_path, bad):
    data = pd.DataFrame({"sku": ["001"]*3, "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                         "outbound": [1., 2., 3.], "inbound": [0, 0, 0], "stockout": [0, 0, 0]})
    if bad == "gap":
        data = data.drop(1)
    elif bad == "duplicate":
        data = pd.concat([data, data.iloc[[0]]])
    else:
        data.loc[0, "outbound"] = -1 if bad == "negative" else np.inf
    path = tmp_path / "data.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_history(path)


def test_monthly_partial_month_is_rejected(tmp_path):
    data = pd.DataFrame({"sku": ["001"]*2, "date": ["2024-01-01", "2024-01-02"],
                         "outbound": [1, 2], "inbound": [1, 2], "stockout": [0, 0]})
    path = tmp_path / "data.csv"
    data.to_csv(path, index=False)
    hist = load_history(path)
    assert hist.sku.iloc[0] == "001"
    with pytest.raises(ValueError, match="meses completos"):
        make_panel(hist, Config())


@pytest.mark.parametrize("name", ["naive", "seasonal", "xgboost", "prophet"])
def test_all_models_produce_full_finite_horizon_and_reload(tmp_path, name):
    cfg = Config(xgb_estimators=15)
    model = Forecaster(name, cfg).fit(panel(36, 2))
    result = model.predict()
    assert len(result) == 24
    assert np.isfinite(result.yhat).all() and (result.yhat >= 0).all()
    model.save(tmp_path / name)
    loaded = Forecaster.load(tmp_path / name).predict()
    np.testing.assert_allclose(result.yhat, loaded.yhat, rtol=1e-6, atol=1e-6)


def test_daily_xgboost_full_direct_horizon():
    model = Forecaster("xgboost", Config(frequency="daily", xgb_estimators=15)).fit(panel(120, 2, "daily"))
    result = model.predict()
    assert len(result) == 60 and result.h.max() == 30
    assert (result.ds > result.origin).all()


def test_daily_prophet_full_horizon():
    result = Forecaster("prophet", Config(frequency="daily")).fit(panel(90, 1, "daily")).predict()
    assert len(result) == 30 and np.isfinite(result.yhat).all()
