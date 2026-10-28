import altair as alt
import pandas as pd
import streamlit as st
from src.ui import assets, metric_row, metric_table

a, _ = assets()
s, m = a["summary"], a["summary"]["test_metrics"]
st.title("Model lab", icon=":material/experiment:")
st.markdown("Three algorithms. Four sensor sets. One choice frozen before test evaluation.")
st.success(f"Validation-selected model: {s['selected_algorithm']} / {s['selected_sensor_set']}", icon=":material/check_circle:")
metric_row([("Test F1", f"{m['f1']:.3f}", "Occupied-class F1."),
            ("Test recall", f"{m['recall']:.1%}", "Fraction of occupied readings detected."),
            ("Balanced accuracy", f"{m['balanced_accuracy']:.1%}", "Average empty and occupied recall."),
            ("Baseline accuracy", f"{s['baseline_metrics']['accuracy']:.1%}", "Predicting only the majority class can produce a misleading accuracy.")])
view = st.segmented_control("Experiment view", ["Evaluation", "Sensor experiment", "Methodology"], default="Evaluation", key="model_view")
if view == "Evaluation":
    left, right = st.columns([1, 1.2])
    with left, st.container(border=True):
        st.subheader("Test confusion matrix")
        cm = pd.DataFrame({"Actual":["Empty","Empty","Occupied","Occupied"],
                           "Predicted":["Empty","Occupied","Empty","Occupied"],
                           "Readings":[m["true_empty"],m["false_occupied"],m["false_empty"],m["true_occupied"]]})
        base = alt.Chart(cm).encode(x=alt.X("Predicted:N",sort=["Empty","Occupied"]),
                                   y=alt.Y("Actual:N",sort=["Empty","Occupied"]))
        chart = base.mark_rect().encode(color=alt.Color("Readings:Q", scale=alt.Scale(scheme="tealblues"), legend=None))
        labels = base.mark_text(fontSize=22, fontWeight="bold").encode(text="Readings:Q",
            color=alt.condition(alt.datum.Readings > cm.Readings.max()/2, alt.value("white"),alt.value("#183142")))
        st.altair_chart((chart+labels).properties(height=270), alt="Confusion matrix with actual and predicted empty or occupied counts")
        st.caption("False empty = occupied in reality, predicted empty.")
    with right, st.container(border=True):
        st.subheader("Performance by recorded date")
        st.dataframe(a["daily"], hide_index=True, alt="Test performance by date, including occupied counts and errors")
        st.caption("Several dates contain only empty readings. Check occupied dates when interpreting performance.")
    partition = st.selectbox("Comparison partition", ["Validation — used for selection", "Test — diagnostic only"], key="comparison_partition")
    metric_table(a["validation"] if partition.startswith("Validation") else a["test"])
    st.caption("Decision threshold: 0.50. No claim of calibrated probability. Never tune repeatedly against these test scores.")
elif view == "Sensor experiment":
    best = a["validation"].groupby("sensor_set", sort=False).first().reset_index()
    st.subheader("How few readings are enough?")
    chart = alt.Chart(best).mark_bar(cornerRadiusEnd=4, color="#007f73").encode(
        y=alt.Y("sensor_set:N", title=None, sort="-x"), x=alt.X("f1:Q", title="Best validation F1", scale=alt.Scale(domain=[0,1])),
        tooltip=["sensor_set", "algorithm", "readings", alt.Tooltip("f1:Q",format=".3f")],
    ).properties(height=270)
    st.altair_chart(chart, alt="Best validation F1 for four predefined sensor sets")
    metric_table(best)
    st.caption("Readings count input columns. This is not a hardware price or physical device-count comparison.")
    st.subheader("Which readings matter to the selected model?")
    importance = a["importance"]
    chart = alt.Chart(importance).mark_bar(color="#4c7290").encode(
        y=alt.Y("sensor:N", title=None, sort="-x"), x=alt.X("f1_decrease:Q", title="Validation F1 decrease after permutation"),
        tooltip=["sensor", alt.Tooltip("f1_decrease:Q",format=".4f"), alt.Tooltip("std:Q",format=".4f")],
    ).properties(height=max(200, len(importance)*25))
    st.altair_chart(chart, alt="Validation permutation importance for the selected model configuration")
    st.caption("Computed on validation with a train-only model. Correlated sensors can share importance; this does not establish causation.")
else:
    st.subheader("Time stays in order")
    splits = pd.DataFrame(s["splits"]).T.reset_index(names="Partition")
    st.dataframe(splits, hide_index=True, alt="Training, validation and test date ranges and class counts")
    st.markdown("""1. Fit each model and any scaling on the first recorded date.
2. Compare configurations on the second date using occupied F1.
3. Break ties by occupied recall, fewer readings, then algorithm name.
4. Freeze the choice and refit using the first two dates.
5. Evaluate on all later dates. Compare against a majority baseline.

Date, Time, occupancy count and CO₂ slope are excluded from model inputs. Random seed **42** and a fixed threshold of **0.50** make the experiment reproducible.""")
    st.warning("This is one room with 0–3 people and a strongly imbalanced later period. The result does not establish accuracy in new rooms or at PCCoE.")
    st.caption(f"Dataset SHA256: {s['dataset_sha256']}")
