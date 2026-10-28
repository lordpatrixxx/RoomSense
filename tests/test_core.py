import joblib
import numpy as np
import pandas as pd
import pytest

from src.core import (ROOT, FEATURES, load_data, split_data, validate_features,
                      predict_frame, simulate_energy)


@pytest.fixture(scope="module")
def frame():
    return load_data()


def test_chronology_and_target(frame):
    train, val, test = split_data(frame)
    assert len(frame) == 10129
    assert train.timestamp.max() < val.timestamp.min() < test.timestamp.min()
    assert val.timestamp.max() < test.timestamp.min()
    assert len(train) + len(val) + len(test) == len(frame)
    assert set(train.occupied) == set(val.occupied) == set(test.occupied) == {0, 1}
    assert (frame.occupied == (frame.Room_Occupancy_Count > 0)).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, "not a number"])
def test_invalid_numeric_rejected(frame, bad):
    row = frame[FEATURES].head(1).astype(object).copy()
    row.loc[0, "S5_CO2"] = bad
    with pytest.raises(ValueError, match="readings"):
        validate_features(row)


def test_missing_column_and_invalid_pir(frame):
    with pytest.raises(ValueError, match="Missing sensor"):
        validate_features(frame.drop(columns="S5_CO2"))
    row = frame.head(1).copy()
    row.loc[0, "S6_PIR"] = 2
    with pytest.raises(ValueError, match="0 or 1"):
        validate_features(row)


def test_model_input_order_and_target_exclusion(frame):
    bundle = joblib.load(ROOT / "artifacts" / "occupancy_model.joblib")
    assert "occupied" not in bundle["features"]
    assert "Room_Occupancy_Count" not in bundle["features"]
    x = frame.head(15).copy()
    a = predict_frame(bundle, x)
    x["Room_Occupancy_Count"] = 99
    b = predict_frame(bundle, x[x.columns[::-1]])
    np.testing.assert_allclose(a.occupied_score, b.occupied_score)
    assert a.occupied_score.between(0, 1).all()


def test_delay_and_energy_units():
    times = pd.date_range("2026-01-01", periods=7, freq="30s")
    result = simulate_energy(times, [0]*7, watts=1000, tariff=10, delay_minutes=1)
    assert result["observed_hours"] == pytest.approx(3/60)
    assert result["eligible_hours"] == pytest.approx(2/60)
    assert result["potential_kwh"] == pytest.approx(2/60)
    assert result["potential_cost"] == pytest.approx(1/3)


def test_gap_resets_timer_and_never_counts_missing_hours():
    times = pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:00:30",
                            "2026-01-02 00:00:00", "2026-01-02 00:00:30"])
    result = simulate_energy(times, [0]*4, delay_minutes=1)
    assert result["eligible_hours"] == 0
    assert result["observed_hours"] == pytest.approx(1/60)
    assert result["excluded_gaps"] == 1


def test_false_empty_and_occupied_reset():
    times = pd.date_range("2026-01-01", periods=6, freq="30s")
    result = simulate_energy(times, [0, 0, 1, 0, 0, 0], delay_minutes=.5,
                             truth=[0, 1, 1, 0, 1, 0])
    assert result["eligible_hours"] == pytest.approx(1/60)
    assert result["false_empty_alert_minutes"] == pytest.approx(1)


def test_single_row_empty_input_and_unsorted_time():
    result = simulate_energy(pd.to_datetime(["2026-01-01"]), [0], delay_minutes=0)
    assert result["potential_kwh"] == 0
    with pytest.raises(ValueError):
        simulate_energy(pd.to_datetime(["2026-01-02", "2026-01-01"]), [0, 0])
    with pytest.raises(ValueError):
        simulate_energy(pd.to_datetime(["2026-01-01"]), [0], watts=-10)


def test_baseline_is_not_selected_on_test():
    import json
    summary = json.loads((ROOT / "artifacts" / "summary.json").read_text())
    validation = pd.read_csv(ROOT / "reports" / "validation_comparison.csv")
    assert summary["selected_algorithm"] == validation.iloc[0].algorithm
    assert summary["selected_sensor_set"] == validation.iloc[0].sensor_set
