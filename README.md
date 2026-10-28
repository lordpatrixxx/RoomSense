# RoomSense

## Is Anyone There? Predicting Room Occupancy for Smarter Energy Use

Data Science Laboratory mini project, Second Year Information Technology, Semester III, Division B, PCCoE Pune.

**Faculty:** JYOTI MANOORKAR  
**Group:** B3-4

| Student | PRN |
|---|---|
| Pratik Deepak Phapale | 125B1F210 |
| Swastik Sandipkumar Phapale | 125B1F212 |
| Nirjar Rupesh Patil | 125B1F206 |

## Run the GUI

### 1-Click Launch (Windows)
1. Double-click `SETUP.bat` once. It locates Python (3.10+), creates `.venv`, and installs dependencies. Pre-trained model weights and dataset are already included.
2. Double-click `START_PROJECT.bat` to launch the Streamlit GUI in your browser.
3. Use `STOP_PROJECT.bat` to stop the background server.

### Terminal Launch (Cross-Platform)
```bash
# 1. Clone repository
git clone https://github.com/lordpatrixxx/RoomSense.git
cd RoomSense

# 2. Set up virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# 3. Launch RoomSense GUI
streamlit run app.py
```

## What is included

- Six GUI views: overview, prediction, exploration, model lab, energy sandbox and project guide.
- Real UCI sensor data and sample CSV files in `data/`.
- Three classification algorithms, a majority-class baseline and four sensor configurations.
- Chronological train/validation/test partitions and documented selection rules.
- A saved inference pipeline in `artifacts/occupancy_model.joblib`.
- An executed notebook, `RoomSense_Analysis.ipynb`, with code, figures and results.
- `reports/PROJECT_REPORT.md`, `reports/RESULTS.md`, metric tables and figures.
- A self-contained printable report, `reports/PROJECT_REPORT.html`, with the actual results and embedded figures. Open it in a browser and use Print to save a PDF.
- `docs/VIVA_GUIDE.md`, methodology notes and data attribution.
- Automated validation, time-integration and GUI smoke tests.

## Reproduce

```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe scripts\build_notebook.py
.\.venv\Scripts\python.exe scripts\build_report.py
.\.venv\Scripts\python.exe -m pytest -q
```

`train.py` does not overwrite the raw CSV. It regenerates project-owned models, metrics, sample inputs and result figures. Restart the app after retraining so the cached model is refreshed. Only load the model supplied with this project or one you trained yourself: joblib files are executable Python serialization, so the app does not accept model uploads.

To open the executed notebook, use VS Code's Jupyter extension or JupyterLab. Select `.venv/Scripts/python.exe` as the kernel. Notebook cells assume the project root is the working directory (or its immediate parent can be located).

## Scientific scope

The target is **empty / occupied**, derived from the recorded count of 0–3 people. It does not identify people or estimate classroom-sized crowds. The timestamp column is used for splitting and plotting, not as a predictive feature. The supplied CO2 slope is excluded because its window construction is not verified. Training and feature selection never use the final test period.

The public source description mentions four days; the downloaded CSV spans seven calendar dates from 22 December 2017 to 11 January 2018, with gaps. This project uses the actual timestamps. Missing recording intervals are excluded from the energy calculation and reset the empty-room timer. Model scores are not calibrated certainty. Reported accuracy is limited to this one-room experiment.

Energy results are a **what-if simulation** based on user-entered power and tariff. The CSV contains no electricity-meter data or device on/off state. An empty-room prediction recommends a manual check; it never triggers equipment automatically.

## Dataset credit

Singh, A. & Chaudhari, S. (2018). *Room Occupancy Estimation*. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5P605. Licensed CC BY 4.0: https://creativecommons.org/licenses/by/4.0/.

Source: https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation

Keep this attribution with redistributed data and project copies. `data/DATA_CARD.md` documents transformations and boundaries.
