import altair as alt
import pandas as pd
import streamlit as st
from src.core import simulate_energy
from src.ui import assets, metric_row

a, _ = assets()
predictions = a["predictions"]
st.title("Energy sandbox", icon=":material/bolt:")
st.markdown("What could a sustained empty-room prediction mean for equipment use?")
st.warning("Simulation only. The dataset contains no electricity readings or equipment state. Results assume your specified equipment would otherwise remain on.", icon=":material/science:")
left, right = st.columns([1, 2], gap="large")
with left, st.container(border=True):
    st.subheader("Set your assumptions")
    watts = st.number_input("Equipment power (W)", min_value=0.0, max_value=50000.0, value=300.0, step=50.0, key="energy_watts")
    tariff = st.number_input("Illustrative tariff (₹ / kWh)", min_value=0.0, max_value=100.0, value=8.0, step=0.5, key="energy_tariff")
    delay = st.slider("Continuous empty time before review (minutes)", 0, 30, 5, key="energy_delay")
    days = sorted(predictions.timestamp.dt.date.unique())
    date = st.selectbox("Replay period", ["All test dates"] + days, format_func=lambda x:x if isinstance(x,str) else x.strftime("%d %b %Y"), key="energy_date")
    st.caption("Intervals over 90 seconds reset the timer and are excluded. No duration is invented after the final reading.")
selected = predictions if date == "All test dates" else predictions[predictions.timestamp.dt.date == date]
result = simulate_energy(selected.timestamp, selected.predicted_occupied, watts, tariff, delay, truth=selected.occupied)
with right:
    metric_row([("Potential energy", f"{result['potential_kwh']:.2f} kWh", "Conditional scenario, not measured savings."),
                ("Potential cost", f"₹{result['potential_cost']:.2f}", "Calculated from your illustrative tariff.")])
    metric_row([("Eligible review time", f"{result['eligible_hours']:.2f} h", "Predicted empty after the waiting period."),
                ("False-empty alert time", f"{result['false_empty_alert_minutes']:.1f} min", "Eligible intervals whose recorded ground truth indicates occupants.")])
    st.caption(f"Valid observed time: {result['observed_hours']:.2f} h · Excluded recording gaps: {result['excluded_gaps']}")
    st.markdown("**Calculation:** eligible hours × equipment watts ÷ 1,000 × tariff.")
    if result["false_empty_alert_minutes"] > 0:
        st.error("Some review alerts occur while the room is occupied. These intervals are not confirmed safe opportunities; this prototype should not switch equipment automatically.")
st.subheader("Replay recorded predictions")
row = st.slider("Recorded observation", 0, max(1, len(selected)-1), 0, key=f"replay_{date}")
current = selected.iloc[min(row,len(selected)-1)]
with st.container(border=True):
    st.markdown(f"**{current.timestamp:%d %b %Y · %H:%M:%S}**")
    metric_row([("Predicted status", "Occupied" if current.predicted_occupied else "Empty", "Frozen model prediction."),
                ("Recorded status", "Occupied" if current.occupied else "Empty", "Ground truth from the dataset."),
                ("Occupied score", f"{current.occupied_score:.3f}", "Uncalibrated model output.")])
st.caption("This is a recorded-data replay. Moving the slider does not connect to live room sensors.")
export = {k:v for k,v in result.items() if k != "intervals"}
export.update({"assumed_watts": watts, "assumed_tariff_inr_kwh":tariff, "delay_minutes":delay, "scenario_only":True})
st.download_button("Download scenario summary", pd.DataFrame([export]).to_csv(index=False).encode(), file_name="roomsense_energy_scenario.csv", mime="text/csv", icon=":material/download:")
