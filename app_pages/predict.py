import csv
import io
import numpy as np
import pandas as pd
import streamlit as st
from src.core import ROOT, FEATURES, UNITS, predict_frame
from src.ui import assets, metric_row

a, bundle = assets()
st.title("Predict occupancy", icon=":material/sensors:")
st.markdown("Use a recorded example, enter your own sensor readings, or classify a CSV in one step.")
st.caption(f"Model: {bundle['algorithm']} · {bundle['sensor_set']} · decision threshold 0.50")
st.caption(f"Later-period occupied recall: {a['summary']['test_metrics']['recall']:.1%}. An empty prediction does not confirm that the room is vacant.")
mode = st.segmented_control("Input method", ["Single reading", "Upload CSV"], default="Single reading", key="predict_mode")

if mode == "Single reading":
    preset = st.selectbox("Start with", ["Recorded occupied example", "Recorded empty example", "Training medians"], key="preset")
    if preset == "Training medians":
        defaults = {c: bundle["ranges"][c]["median"] for c in bundle["features"]}
    else:
        label = "occupied" if preset == "Recorded occupied example" else "empty"
        defaults = pd.read_csv(ROOT / "data" / f"example_{label}.csv").iloc[0].to_dict()
    st.caption("Examples contain actual source readings. Changing a value creates an exploratory scenario.")
    with st.form("single_prediction", border=True):
        cols = st.columns(3)
        values = {}
        for i, c in enumerate(bundle["features"]):
            family = next(f for f in UNITS if f in c)
            reference = bundle["ranges"][c]
            with cols[i % 3]:
                if family == "PIR":
                    values[c] = st.selectbox(f"{c} · motion", [0, 1], index=int(defaults[c]), key=f"reading_{preset}_{c}")
                else:
                    values[c] = st.number_input(f"{c} ({UNITS[family]})", value=float(defaults[c]),
                        min_value=-50.0 if family == "Temp" else 0.0,
                        max_value=100.0 if family == "Temp" else (1000000.0 if family != "Sound" else 1000.0),
                        step=0.01 if family in ["Temp", "Sound"] else 1.0,
                        help=f"Development range: {reference['min']:.2f} to {reference['max']:.2f}",
                        key=f"reading_{preset}_{c}")
        submitted = st.form_submit_button("Predict room status", type="primary", icon=":material/play_arrow:")
    if submitted:
        result = predict_frame(bundle, pd.DataFrame([values])).iloc[0]
        with st.container(border=True):
            if result.prediction == "Occupied":
                st.success("Room predicted occupied", icon=":material/person:")
                st.markdown("Occupancy evidence is present. This prototype does not recommend switching equipment off.")
            else:
                st.info("Room predicted empty", icon=":material/meeting_room:")
                st.markdown("Check the room manually. A single empty prediction is not enough to establish a sustained vacancy.")
            st.metric("Occupied model score", f"{result.occupied_score:.3f}")
            st.progress(float(result.occupied_score), text="0 = empty end of scale · 1 = occupied end of scale")
            if result.outside_training_range:
                st.warning("Some readings fall outside the development-data range. This prediction may be unreliable.")
            st.caption("A model score is not calibrated certainty. Results are based on a single-room experiment.")
else:
    st.markdown("Upload a UTF-8 CSV with one sensor reading per row. Column names must match the template; extra columns are preserved but not used by the model.")
    st.download_button("Download example CSV", (ROOT / "data/sample_upload.csv").read_bytes(),
                       file_name="roomsense_example.csv", mime="text/csv", icon=":material/download:")
    upload = st.file_uploader("Sensor readings CSV", type=["csv"], key="sensor_csv")
    if upload is not None:
        try:
            if upload.size > 10 * 1024 * 1024:
                raise ValueError("Please use a CSV smaller than 10 MB.")
            text = upload.getvalue().decode("utf-8-sig")
            headers = next(csv.reader(io.StringIO(text)), [])
            if len(headers) != len(set(headers)):
                raise ValueError("Column names must be unique; remove duplicated headers.")
            frame = pd.read_csv(io.StringIO(text))
            if len(frame) > 100000:
                raise ValueError("Please upload at most 100,000 rows at a time.")
            result = predict_frame(bundle, frame)
            metric_row([("Rows classified", f"{len(result):,}", "All rows in the upload."),
                        ("Predicted occupied", f"{(result.prediction == 'Occupied').sum():,}", "Threshold is fixed at 0.50."),
                        ("Outside reference range", f"{result.outside_training_range.sum():,}", "At least one input is outside the development range.")])
            st.dataframe(result.head(1000), hide_index=True, alt="Uploaded sensor readings and model predictions")
            if len(result) > 1000:
                st.caption("Preview shows the first 1,000 rows. The download contains every prediction.")
            if result.outside_training_range.any():
                st.warning("Some uploaded readings lie outside training ranges. Review them before interpreting the predictions.")
            # Escape spreadsheet-formula prefixes in arbitrary uploaded text on export.
            exported = result.copy()
            for col in exported.select_dtypes(include=["object", "str", "string"]).columns:
                exported[col] = exported[col].map(lambda v: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v)
            st.download_button("Download all predictions", exported.to_csv(index=False).encode("utf-8"),
                               file_name="roomsense_predictions.csv", mime="text/csv", icon=":material/download:", type="primary")
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError, StopIteration) as exc:
            st.error(f"Cannot predict this file: {exc}")
    with st.expander("Required columns and units"):
        st.code(",".join(bundle["features"]), language=None)
        st.markdown("Temperature: °C · Light: lux · Sound amplitude: volts · CO₂: ppm · PIR: 0 or 1. Missing, infinite and nonnumeric inputs are rejected.")
