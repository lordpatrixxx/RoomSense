"""Build and execute the submission notebook with this environment's Python."""
import os
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def build():
    nb = nbf.v4.new_notebook()
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text))
    md("""# Is Anyone There? Predicting Room Occupancy for Smarter Energy Use

**PCCoE Pune · Information Technology · SY B.Tech · Semester III · Division B**  
**VSEC Data Science Laboratory · Faculty: JYOTI MANOORKAR · Group B3-4**

| Student | PRN |
|---|---|
| Pratik Deepak Phapale | 125B1F210 |
| Swastik Sandipkumar Phapale | 125B1F212 |
| Nirjar Rupesh Patil | 125B1F206 |

## Aim
Classify a room as empty or occupied from environmental readings, compare four sensor configurations, and demonstrate a transparent hypothetical energy scenario. This notebook contains executed results. It is a single-room research prototype, not a verified campus automation system.

Dataset: Singh, A. & Chaudhari, S. (2018), [Room Occupancy Estimation, UCI](https://doi.org/10.24432/C5P605), CC BY 4.0.""")
    code("""from pathlib import Path
import sys, json
ROOT = Path.cwd()
if not (ROOT / 'src').exists() and (ROOT.parent / 'src').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from src.core import load_data, split_data, FEATURES, FEATURE_SETS, simulate_energy
sns.set_theme(style='whitegrid', palette=['#007f73', '#71869b'])
df = load_data()
print('Shape:', df.shape, '| Original source rows:', len(df))
display(df.head())""")
    md("""## 1. Data audit and preparation

The target `occupied` is 1 when the recorded count is greater than zero. Date and Time are excluded from model inputs. The supplied CO2 slope is also excluded because its sliding-window construction is not verified. The shared loader rejects nonnumeric values, invalid PIR values, missing readings and duplicate timestamps. No source rows are synthetically added or randomly shuffled.""")
    code("""audit = pd.DataFrame({'dtype': df.dtypes.astype(str), 'missing': df.isna().sum(),
                      'unique_values': df.nunique()})
display(audit)
print('Exact duplicate rows:', df.duplicated().sum())
print('Timestamp span:', df.timestamp.min(), 'to', df.timestamp.max())
display(df.groupby(df.timestamp.dt.date).agg(rows=('occupied','size'), occupied=('occupied','sum')))
display(df[FEATURES].describe().round(2))""")
    md("""## 2. Exploratory analysis

The public dataset description mentions four collection days, while the CSV contains seven calendar dates with gaps. We use the actual timestamps. Class imbalance and correlated measurements matter when evaluating performance.""")
    code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(data=df, x='occupied', ax=axes[0], color='#007f73')
axes[0].set(xticks=[0,1], xticklabels=['Empty','Occupied'], title='Target balance', xlabel='Room status')
sns.boxplot(data=df, x='occupied', y='S5_CO2', ax=axes[1], color='#9bc9c3')
axes[1].set(xticks=[0,1], xticklabels=['Empty','Occupied'], title='CO2 by recorded occupancy', ylabel='CO2 (ppm)')
plt.tight_layout(); plt.show()
fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(df[FEATURES + ['occupied']].corr(), cmap='BrBG', center=0, ax=ax)
ax.set_title('Descriptive correlation — not causal evidence')
plt.tight_layout(); plt.show()""")
    md("""## 3. A statistical question

Is recorded motion associated with occupancy in this sample? We show a contingency table and conditional occupied proportions. A naive chi-square p-value would treat adjacent readings as independent; because the data is temporally correlated, we avoid claiming independent-observation significance.""")
    code("""any_motion = ((df.S6_PIR > 0) | (df.S7_PIR > 0)).astype(int)
table = pd.crosstab(any_motion.rename('Any motion'), df.occupied.rename('Occupied'))
display(table)
display(table.div(table.sum(axis=1), axis=0).round(4))
display(df.groupby('occupied')[['S1_Temp','S1_Light','S5_CO2']].agg(['mean','median','std']).round(2))""")
    md("""## 4. Chronological experimental design

First observed date: training. Second date: validation. Remaining dates: held-out test. All partitions contain both target classes, although later periods are strongly imbalanced. This tests a later period in the same room; it does not establish external-room generalization. No random row split is used.""")
    code("""train, val, test = split_data(df)
splits = pd.DataFrame([{'split': name, 'rows':len(part), 'occupied':int(part.occupied.sum()),
                       'start':part.timestamp.min(), 'end':part.timestamp.max()}
                      for name, part in [('Training',train),('Validation',val),('Test',test)]])
display(splits)
assert train.timestamp.max() < val.timestamp.min()
assert val.timestamp.max() < test.timestamp.min()
display(pd.DataFrame([{'sensor_set':k,'readings':len(v),'columns':', '.join(v)} for k,v in FEATURE_SETS.items()]))""")
    md("""## 5. Train and compare models

We compare Logistic Regression (with training-only standardization), Decision Tree and Random Forest. Hyperparameters and random seed 42 are fixed in `train.py`. Every algorithm is tested with four sensor sets. Configuration selection uses validation F1, then occupied recall, fewer readings and algorithm name. The threshold remains 0.5.

After selecting the configuration, it is refit using train + validation. Test results are opened only after this choice. All test comparisons are descriptive and must not be used for further tuning. A majority baseline exposes the limitations of accuracy.

For positive class **occupied**: precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R). A false-empty prediction is FN and is especially relevant to energy-management decisions.""")
    code("""from train import run_training
summary = run_training()
validation = pd.read_csv(ROOT / 'reports/validation_comparison.csv')
display(validation[['sensor_set','algorithm','readings','accuracy','precision','recall','f1','false_empty']].round(4))
print('Frozen choice:', summary['selected_algorithm'], '/', summary['selected_sensor_set'])""")
    md("""## 6. Held-out evaluation

Inspect false-empty counts and occupied recall alongside accuracy. The same-room test is not a measure of performance in new classrooms. Classifier scores are not assumed to be calibrated probabilities.""")
    code("""display(pd.DataFrame({'Selected model':summary['test_metrics'], 'Majority baseline':summary['baseline_metrics']}).round(4))
display(pd.read_csv(ROOT / 'reports/daily_test_metrics.csv'))
from IPython.display import Image
display(Image(filename=str(ROOT / 'reports/figures/confusion_matrix.png')))
display(pd.read_csv(ROOT / 'reports/test_comparison.csv').round(4))""")
    md("""## 7. Do fewer readings preserve performance?

Compare the validation winner for each sensor set. Feature counts are measurement counts, not verified hardware cost. Validation permutation importance uses a model fitted only on training data. Correlated inputs may share importance.""")
    code("""display(Image(filename=str(ROOT / 'reports/figures/sensor_comparison.png')))
importance = pd.read_csv(ROOT / 'reports/sensor_importance.csv')
display(importance.round(4))""")
    md("""## 8. Inference on a CSV

The saved pipeline preserves scaling and column order. Only source-compatible numeric sensor CSVs are accepted. The occupancy label is never an input. The app warns when input values fall outside the development-data range; being inside that range still does not guarantee validity in another room.""")
    code("""import joblib
from src.core import predict_frame
bundle = joblib.load(ROOT / 'artifacts/occupancy_model.joblib')
sample = pd.read_csv(ROOT / 'data/sample_upload.csv')
display(predict_frame(bundle, sample).head(10))""")
    md("""## 9. Hypothetical energy scenario

Assume 300 W of equipment would otherwise be on during observed intervals, a five-minute vacancy review delay and an illustrative tariff of ₹8/kWh. These are assumptions, not source measurements. Integrate adjacent intervals only; gaps above 90 seconds reset the timer and contribute no energy. The final reading has no assumed duration. False-empty alert minutes are reported separately, because those intervals are not confirmed safe opportunities.

Potential kWh = eligible hours × assumed watts / 1000. Potential cost = kWh × assumed tariff.""")
    code("""predictions = pd.read_csv(ROOT / 'reports/test_predictions.csv', parse_dates=['timestamp'])
scenario = simulate_energy(predictions.timestamp, predictions.predicted_occupied,
                           watts=300, tariff=8, delay_minutes=5, truth=predictions.occupied)
display(pd.Series({k:v for k,v in scenario.items() if k != 'intervals'}, name='Scenario result'))""")
    md("""## 10. Conclusions and limitations

The executed results support a comparison of declared models and sensor sets within this dataset. The validation-selected model is reported independently on later records. A high accuracy is insufficient when most records are empty; occupied recall, F1 and false-empty counts must also be considered.

This is one controlled room, at most three people and a short, discontinuous recording. Sensor calibration and room conditions can change predictions. The simulation estimates conditional potential, not measured savings. No campus deployment, automatic control or actual hardware savings have been demonstrated.

See `reports/PROJECT_REPORT.md`, `reports/RESULTS.md` and `docs/VIVA_GUIDE.md` for the full project explanation. Dataset credit: Singh & Chaudhari, UCI Room Occupancy Estimation, https://doi.org/10.24432/C5P605, CC BY 4.0.""")
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name":"Python (RoomSense)","language":"python","name":"python3"},
                   "language_info":{"name":"python","version":sys.version.split()[0]}}
    path = ROOT / "RoomSense_Analysis.ipynb"
    nbf.write(nb, path)
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(nb, km=km, timeout=240, resources={"metadata":{"path":str(ROOT)}})
    client.execute()
    nbf.write(nb, path)
    print(f"Executed {sum(c.cell_type == 'code' for c in nb.cells)} code cells: {path}")


if __name__ == "__main__":
    build()
