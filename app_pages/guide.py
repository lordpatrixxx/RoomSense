import pandas as pd
import streamlit as st
from src.core import ROOT, SOURCE_PAGE

st.title("Project guide", icon=":material/menu_book:")
st.markdown("### Is Anyone There? Predicting Room Occupancy for Smarter Energy Use")
st.markdown("Pimpri Chinchwad College of Engineering, Pune · Department of Information Technology  \nSY B.Tech · Semester III · Division B · Group B3-4")
st.markdown("**Course:** VSEC Data Science Laboratory · **Faculty:** JYOTI MANOORKAR")
st.table(pd.DataFrame({"Student":["Pratik Deepak Phapale","Swastik Sandipkumar Phapale","Nirjar Rupesh Patil"],
                       "PRN":["125B1F210","125B1F212","125B1F206"]}), hide_index=True, alt="Mini project group members and PRNs")
st.subheader("Submission materials")
files = [("Printable report", "reports/PROJECT_REPORT.html", "text/html"),
         ("Editable report", "reports/PROJECT_REPORT.md", "text/markdown"),
         ("Experimental results", "reports/RESULTS.md", "text/markdown"),
         ("Executed notebook", "RoomSense_Analysis.ipynb", "application/x-ipynb+json"),
         ("Viva guide", "docs/VIVA_GUIDE.md", "text/markdown")]
with st.container(horizontal=True):
    for title, path, mime in files:
        file = ROOT / path
        if file.exists():
            st.download_button(title, file.read_bytes(), file_name=file.name, mime=mime, icon=":material/download:")
st.subheader("A clear five-step demonstration")
st.markdown("""1. **Explore:** show recorded occupancy and environmental readings.
2. **Explain:** describe the chronological split and occupied-class metrics.
3. **Compare:** examine the four sensor sets in Model lab.
4. **Predict:** enter a reading or upload the supplied example CSV.
5. **Simulate:** change power and vacancy delay, and explain the limits of the energy estimate.""")
with st.expander("Read the project report"):
    st.markdown((ROOT / "reports/PROJECT_REPORT.md").read_text(encoding="utf-8"))
with st.expander("Prepare for the viva"):
    st.markdown((ROOT / "docs/VIVA_GUIDE.md").read_text(encoding="utf-8"))
st.subheader("Data credit and scope")
st.markdown(f"[Room Occupancy Estimation · UCI]({SOURCE_PAGE}) — Singh, A. & Chaudhari, S. (2018). [DOI](https://doi.org/10.24432/C5P605) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).")
st.caption("Source CSV preserved. Binary target derived from occupancy count. One controlled room; no campus validation, live sensing or measured energy savings.")
