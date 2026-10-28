import altair as alt
import pandas as pd
import streamlit as st
from src.ui import assets, metric_row, timeline

a, bundle = assets()
df, summary = a["data"], a["summary"]
m = summary["test_metrics"]
st.caption("THE RESEARCH QUESTION")
st.title("Is anyone there?")
st.markdown("Understand room occupancy through environmental data. Explore the evidence, test a reading, and examine what fewer sensors can tell us.")
with st.container(horizontal=True):
    st.badge("Real UCI dataset", color="green", icon=":material/database:")
    st.badge("Chronological evaluation", color="blue", icon=":material/schedule:")
    st.badge("Energy simulation", color="orange", icon=":material/bolt:")

metric_row([
    ("Recorded observations", f"{len(df):,}", "Complete source dataset; no synthetic training records."),
    ("Test occupied F1", f"{m['f1']:.1%}", "F1 for the occupied class on dates after validation."),
    ("Test occupied recall", f"{m['recall']:.1%}", "Share of actual occupied readings detected."),
    ("False-empty readings", f"{m['false_empty']:,}", "Occupied test readings incorrectly labelled empty."),
])
if m["recall"] < .8:
    st.warning(f"The selected model detects only {m['recall']:.1%} of occupied readings in the later test period. This generalization gap is a key research finding; empty predictions require manual verification.")

left, right = st.columns([1.65, 1], gap="large")
with left:
    with st.container(border=True):
        st.subheader("A day in the room")
        dates = sorted(df.timestamp.dt.date.unique())
        date = st.selectbox("Recording date", dates, format_func=lambda x:x.strftime("%d %b %Y"), key="overview_date")
        day = df[df.timestamp.dt.date == date]
        timeline(day, "Room_Occupancy_Count", "Recorded occupancy", "People", height=230)
        st.caption("Ground truth from the source experiment. Gaps in the record remain gaps.")
with right:
    with st.container(border=True):
        st.subheader("The selected model", icon=":material/model_training:")
        st.markdown(f"### {summary['selected_algorithm']}")
        st.markdown(f"**{summary['selected_sensor_set']}** · {len(summary['features'])} input readings")
        st.caption("Chosen by validation F1, then refit on the first two recorded dates. Test results did not choose this model.")
        st.metric("Validation F1", f"{summary['validation_metrics']['f1']:.1%}")
        if st.button("Try an occupancy prediction", icon=":material/arrow_forward:", type="primary", width="stretch"):
            st.switch_page("app_pages/predict.py")

st.subheader("From readings to a useful question")
cols = st.columns(3)
for col, title, text in zip(cols,
    ["01 · Understand", "02 · Compare", "03 · Explore"],
    ["Inspect temperature, light, sound, CO₂ and motion alongside recorded occupancy.",
     "Compare three algorithms and four sensor sets. Could fewer readings be enough?",
     "Replay predictions and estimate hypothetical energy under your own assumptions."]):
    with col, st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(text)
st.caption("Scope: one controlled room with 0–3 people. No live monitoring or automated equipment control. The later test period contains mostly empty readings, so accuracy alone is insufficient.")
