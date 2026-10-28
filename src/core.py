"""Shared data validation, inference and elapsed-time simulation logic."""
from pathlib import Path
import hashlib
import io
import zipfile
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Occupancy_Estimation.csv"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/864/room+occupancy+estimation.zip"
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation"
FEATURES = ([f"S{i}_Temp" for i in range(1, 5)]
            + [f"S{i}_Light" for i in range(1, 5)]
            + [f"S{i}_Sound" for i in range(1, 5)]
            + ["S5_CO2", "S6_PIR", "S7_PIR"])
FEATURE_SETS = {
    "All sensors": FEATURES,
    "Ambient only": [c for c in FEATURES if "Sound" not in c and "PIR" not in c],
    "Three readings": ["S1_Temp", "S1_Light", "S5_CO2"],
    "Motion only": ["S6_PIR", "S7_PIR"],
}
UNITS = {"Temp": "°C", "Light": "lux", "Sound": "V", "CO2": "ppm", "PIR": "0 / 1"}


def download_data():
    """Keep an existing source CSV; download only when absent."""
    if DATA_PATH.exists():
        return DATA_PATH
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(SOURCE_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    members = [n for n in archive.namelist() if Path(n).name == DATA_PATH.name]
    if len(members) != 1:
        raise ValueError("The UCI archive does not contain the expected CSV.")
    # Read the exact member, avoiding arbitrary archive path extraction.
    DATA_PATH.write_bytes(archive.read(members[0]))
    return DATA_PATH


def validate_features(frame, columns=FEATURES):
    """Reject invalid input explicitly instead of making silent predictions."""
    if frame.empty:
        raise ValueError("The CSV has no data rows.")
    if frame.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError("Missing sensor columns: " + ", ".join(missing))
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    bad = numeric.columns[~np.isfinite(numeric.to_numpy(dtype=float)).all(axis=0)].tolist()
    if bad:
        raise ValueError("Missing, nonnumeric or infinite readings in: " + ", ".join(bad))
    for col in columns:
        if "PIR" in col and not numeric[col].isin([0, 1]).all():
            raise ValueError(f"{col} must contain only 0 or 1.")
        if any(k in col for k in ("Light", "Sound", "CO2")) and (numeric[col] < 0).any():
            raise ValueError(f"{col} cannot contain negative readings.")
    return numeric


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    validate_features(df)
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric)
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="raise")
    if df["timestamp"].duplicated().any():
        raise ValueError("The source contains duplicate timestamps; inspect before training.")
    target = pd.to_numeric(df["Room_Occupancy_Count"], errors="raise")
    if not target.isin([0, 1, 2, 3]).all():
        raise ValueError("Expected occupancy counts from 0 to 3.")
    df["occupied"] = (target > 0).astype(int)
    return df.sort_values("timestamp").reset_index(drop=True)


def split_data(df):
    """First observed day = training; second = validation; later days = test."""
    dates = df.timestamp.dt.normalize()
    days = sorted(dates.unique())
    if len(days) < 3:
        raise ValueError("At least three observation dates are required.")
    train, val, test = (df.loc[dates == days[0]].copy(),
                        df.loc[dates == days[1]].copy(),
                        df.loc[dates > days[1]].copy())
    for name, part in [("training", train), ("validation", val), ("test", test)]:
        if part.occupied.nunique() != 2:
            raise ValueError(f"The {name} split must contain both target classes.")
    return train, val, test


def predict_frame(bundle, frame):
    columns = bundle["features"]
    x = validate_features(frame, columns)
    estimator = bundle["pipeline"]
    positive_index = list(estimator.classes_).index(1)
    probability = estimator.predict_proba(x)[:, positive_index]
    result = frame.copy()
    result["occupied_score"] = probability
    result["prediction"] = np.where(probability >= 0.5, "Occupied", "Empty")
    result["outside_training_range"] = [False] * len(result)
    for col in columns:
        limits = bundle["ranges"][col]
        result["outside_training_range"] |= ~x[col].between(limits["min"], limits["max"])
    return result


def simulate_energy(timestamps, predictions, watts=300.0, tariff=8.0,
                    delay_minutes=5.0, max_gap_seconds=90, truth=None):
    """Integrate adjacent valid intervals only; never bridge missing data.

    A review alert starts after the specified continuous predicted-empty time.
    Energy assumes the user-specified equipment otherwise stays on throughout
    those intervals. It is hypothetical potential, not measured consumption.
    """
    if watts < 0 or tariff < 0 or delay_minutes < 0:
        raise ValueError("Power, tariff and delay must be non-negative.")
    times = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="raise"))
    pred = np.asarray(predictions)
    if len(times) != len(pred) or not np.isin(pred, [0, 1]).all():
        raise ValueError("Provide one binary prediction for each timestamp.")
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Timestamps must be valid, unique and increasing.")
    actual = None if truth is None else np.asarray(truth)
    if actual is not None and (len(actual) != len(pred) or not np.isin(actual, [0, 1]).all()):
        raise ValueError("Provide one binary ground-truth value per timestamp.")
    durations = np.zeros(len(times))
    if len(times) > 1:
        # pandas may store datetimes as microseconds or nanoseconds. Use the
        # public timedelta conversion rather than assuming integer units.
        durations[:-1] = (times[1:] - times[:-1]).total_seconds()
    valid = (durations > 0) & (durations <= max_gap_seconds)
    eligible = np.zeros(len(times))
    empty_run = 0.0
    for i in range(len(times)):
        if pred[i] == 1 or not valid[i]:
            empty_run = 0.0
            continue
        before = empty_run
        empty_run += durations[i]
        eligible[i] = max(0.0, empty_run - delay_minutes * 60) - max(0.0, before - delay_minutes * 60)
    seconds = float(eligible.sum())
    kwh = seconds / 3600 * watts / 1000
    unsafe_seconds = None if actual is None else float(eligible[actual == 1].sum())
    return {
        "observed_hours": float(durations[valid].sum() / 3600),
        "eligible_hours": seconds / 3600,
        "potential_kwh": kwh,
        "potential_cost": kwh * tariff,
        "false_empty_alert_minutes": None if unsafe_seconds is None else unsafe_seconds / 60,
        "excluded_gaps": int((durations > max_gap_seconds).sum()),
        "intervals": pd.DataFrame({"timestamp": times, "eligible_seconds": eligible}),
    }


def dataset_hash():
    return hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
