"""Cached app assets and accessible chart helpers."""
import json
import altair as alt
import joblib
import pandas as pd
import streamlit as st
from src.core import ROOT, load_data

alt.data_transformers.disable_max_rows()


@st.cache_data(max_entries=2, show_spinner=False)
def data_assets(version):
    return {
        "data": load_data(),
        "summary": json.loads((ROOT / "artifacts/summary.json").read_text(encoding="utf-8")),
        "validation": pd.read_csv(ROOT / "reports/validation_comparison.csv"),
        "test": pd.read_csv(ROOT / "reports/test_comparison.csv"),
        "predictions": pd.read_csv(ROOT / "reports/test_predictions.csv", parse_dates=["timestamp"]),
        "importance": pd.read_csv(ROOT / "reports/sensor_importance.csv"),
        "daily": pd.read_csv(ROOT / "reports/daily_test_metrics.csv"),
    }


@st.cache_resource(max_entries=2, show_spinner=False)
def model_asset(version):
    # Only the model generated locally by train.py is loaded; never uploaded files.
    return joblib.load(ROOT / "artifacts/occupancy_model.joblib")


def assets():
    version = (ROOT / "artifacts/summary.json").stat().st_mtime_ns
    return data_assets(version), model_asset(version)


def metric_row(items):
    with st.container(horizontal=True):
        for label, value, help_text in items:
            st.metric(label, value, help=help_text, border=True)


def timeline(frame, y, title, label, color="#007f73", height=260):
    """Separate line groups at recording gaps; never imply unobserved continuity."""
    plot = frame[["timestamp", y]].copy()
    plot["segment"] = (plot.timestamp.diff().dt.total_seconds().fillna(0) > 90).cumsum().astype(str)
    # Keep all observations; the source dataset is small and dates are filterable.
    chart = alt.Chart(plot).mark_line(color=color, strokeWidth=1.8).encode(
        x=alt.X("timestamp:T", title="Recorded time"),
        y=alt.Y(f"{y}:Q", title=label), detail="segment:N",
        tooltip=[alt.Tooltip("timestamp:T", title="Recorded time", format="%d %b %H:%M:%S"),
                 alt.Tooltip(f"{y}:Q", title=label, format=".3f")],
    ).properties(height=height, title=title)
    st.altair_chart(chart, alt=f"{title}. Lines break at missing recording intervals.")


def metric_table(frame):
    cols = ["sensor_set", "algorithm", "readings", "accuracy", "precision", "recall", "f1", "false_empty"]
    display = frame[cols].rename(columns={"sensor_set":"Sensor set", "algorithm":"Algorithm",
        "readings":"Readings", "accuracy":"Accuracy", "precision":"Occupied precision",
        "recall":"Occupied recall", "f1":"Occupied F1", "false_empty":"False empty"})
    st.dataframe(display, hide_index=True, alt="Algorithm and sensor comparison with occupied-class metrics",
        column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                       ["Accuracy", "Occupied precision", "Occupied recall", "Occupied F1"]})
