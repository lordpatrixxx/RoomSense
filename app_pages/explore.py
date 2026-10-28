import altair as alt
import pandas as pd
import streamlit as st
from src.core import FEATURES, UNITS
from src.ui import assets, metric_row, timeline

a, _ = assets()
df = a["data"]
st.title("Explore the evidence", icon=":material/query_stats:")
st.markdown("Look at the recorded room conditions before interpreting the predictions.")
days = sorted(df.timestamp.dt.date.unique())
selected = st.multiselect("Observation dates", days, default=days, format_func=lambda x:x.strftime("%d %b %Y"), key="explore_days")
filtered = df[df.timestamp.dt.date.isin(selected)]
if filtered.empty:
    st.info("Select at least one observation date to explore the data.")
    st.stop()
metric_row([("Observations", f"{len(filtered):,}", "Rows within your date selection."),
            ("Occupied share", f"{filtered.occupied.mean():.1%}", "Fraction of readings with one or more occupants."),
            ("Recorded dates", str(len(selected)), "These dates are separated by recording gaps."),
            ("Missing sensor values", str(int(filtered[FEATURES].isna().sum().sum())), "Null values in model-eligible sensor columns.")])
sensor = st.selectbox("Sensor to inspect", FEATURES, index=FEATURES.index("S5_CO2"), key="explore_sensor")
unit = next(v for k,v in UNITS.items() if k in sensor)
with st.container(border=True):
    timeline(filtered, sensor, f"{sensor} across recorded time", f"{sensor} ({unit})")
left, right = st.columns(2)
plot = filtered.assign(Status=filtered.occupied.map({0:"Empty",1:"Occupied"}))
with left, st.container(border=True):
    st.subheader("Reading distribution")
    chart = alt.Chart(plot).mark_boxplot(extent="min-max", color="#007f73").encode(
        x=alt.X("Status:N", title=None), y=alt.Y(f"{sensor}:Q", title=f"{sensor} ({unit})")
    ).properties(height=250)
    st.altair_chart(chart, alt=f"{sensor} distribution for empty and occupied readings")
with right, st.container(border=True):
    st.subheader("Occupancy balance")
    balance = plot.groupby("Status").size().reset_index(name="Observations")
    chart = alt.Chart(balance).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("Status:N", title=None), y="Observations:Q",
        color=alt.Color("Status:N", scale=alt.Scale(domain=["Empty","Occupied"], range=["#adbdc9","#007f73"]), legend=None),
        tooltip=["Status", "Observations"],
    ).properties(height=250)
    st.altair_chart(chart, alt="Count of empty and occupied readings in selected dates")
if st.toggle("Show correlation heatmap", key="show_correlation"):
    corr = filtered[FEATURES + ["occupied"]].corr().rename_axis("Sensor").reset_index().melt(id_vars="Sensor", var_name="Compared with", value_name="Correlation")
    chart = alt.Chart(corr).mark_rect().encode(
        x=alt.X("Sensor:N", title=None), y=alt.Y("Compared with:N", title=None),
        color=alt.Color("Correlation:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue")),
        tooltip=["Sensor", "Compared with", alt.Tooltip("Correlation:Q",format=".2f")]
    ).properties(height=450)
    st.altair_chart(chart, alt="Pairwise sensor and occupancy correlation; descriptive, not causal")
    st.caption("Correlation is descriptive. Adjacent readings are related in time; this is not evidence of causation.")
with st.expander("Inspect and download the selected records"):
    st.dataframe(filtered, hide_index=True, alt="Source records for the selected observation dates")
    st.download_button("Download selected records", filtered.to_csv(index=False).encode(), file_name="roomsense_selected_data.csv", mime="text/csv")
