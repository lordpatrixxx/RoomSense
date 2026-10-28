"""RoomSense GUI entry point. Run with: python -m streamlit run app.py"""
import streamlit as st
from src.core import ROOT

st.set_page_config(page_title="RoomSense · Occupancy intelligence", page_icon=":material/sensors:", layout="wide")

with st.sidebar:
    st.title("RoomSense", icon=":material/sensors:")
    st.caption("ROOM OCCUPANCY INTELLIGENCE")
    st.badge("Local research prototype", icon=":material/science:", color="green")

pages = {
    "Workspace": [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("app_pages/predict.py", title="Predict occupancy", icon=":material/sensors:", url_path="predict"),
        st.Page("app_pages/explore.py", title="Explore data", icon=":material/query_stats:", url_path="explore"),
    ],
    "Research": [
        st.Page("app_pages/models.py", title="Model lab", icon=":material/experiment:", url_path="models"),
        st.Page("app_pages/energy.py", title="Energy sandbox", icon=":material/bolt:", url_path="energy"),
        st.Page("app_pages/guide.py", title="Project guide", icon=":material/menu_book:", url_path="guide"),
    ],
}
page = st.navigation(pages)
with st.sidebar:
    st.caption("PCCoE Pune · Information Technology")
    st.markdown("**VSEC Data Science Laboratory**  \nSY B.Tech · Semester III · B3-4")
    st.caption("Recorded UCI data · No live sensor connection")

if not (ROOT / "artifacts/occupancy_model.joblib").exists():
    st.error("The trained model is missing. Run SETUP.bat or train.py from the project folder, then restart the app.")
    st.stop()
try:
    page.run()
except (FileNotFoundError, OSError) as exc:
    st.error(f"A project file could not be read: {exc}. Run train.py to regenerate project outputs.")
