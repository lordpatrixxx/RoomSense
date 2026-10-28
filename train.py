"""Run once to reproduce every saved model, metric, chart and report table.

Usage: python train.py (using the project environment)
"""
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.core import (ROOT, FEATURES, FEATURE_SETS, download_data, load_data,
                      split_data, dataset_hash)

SEED = 42


def metrics(y, probability):
    pred = np.asarray(probability) >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "true_empty": int(tn), "false_occupied": int(fp),
            "false_empty": int(fn), "true_occupied": int(tp)}


def make_models():
    return {
        "Logistic Regression": Pipeline([("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))]),
        "Decision Tree": Pipeline([("model", DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=12, class_weight="balanced", random_state=SEED))]),
        "Random Forest": Pipeline([("model", RandomForestClassifier(
            n_estimators=160, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=SEED, n_jobs=2))]),
    }


def run_training():
    download_data()
    df = load_data()
    train, val, test = split_data(df)
    development = pd.concat([train, val])
    artifacts = ROOT / "artifacts"
    reports = ROOT / "reports"
    figures = reports / "figures"
    for folder in (artifacts, figures):
        folder.mkdir(parents=True, exist_ok=True)
    candidates = []
    # Hyperparameters, sensor sets, split and threshold are declared before test evaluation.
    for feature_name, columns in FEATURE_SETS.items():
        for model_name, model in make_models().items():
            model.fit(train[columns], train.occupied)
            p = model.predict_proba(val[columns])[:, 1]
            candidates.append({"sensor_set": feature_name, "algorithm": model_name,
                               "readings": len(columns), **metrics(val.occupied, p)})
    validation = pd.DataFrame(candidates).sort_values(
        ["f1", "recall", "readings", "algorithm"], ascending=[False, False, True, True],
        kind="stable").reset_index(drop=True)
    selected = validation.iloc[0]
    test_rows = []
    selected_model = None
    # Freeze the winner before opening test results. Refit on train+validation.
    for feature_name, columns in FEATURE_SETS.items():
        for model_name, model in make_models().items():
            model.fit(development[columns], development.occupied)
            p = model.predict_proba(test[columns])[:, 1]
            test_rows.append({"sensor_set": feature_name, "algorithm": model_name,
                              "readings": len(columns), **metrics(test.occupied, p)})
            if feature_name == selected.sensor_set and model_name == selected.algorithm:
                selected_model = model
                selected_columns = columns
                selected_p = p
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(development[FEATURES], development.occupied)
    baseline_metrics = metrics(test.occupied, baseline.predict_proba(test[FEATURES])[:, 1])
    test_results = pd.DataFrame(test_rows)
    selected_metrics = metrics(test.occupied, selected_p)
    predictions = test[["timestamp", "Room_Occupancy_Count", "occupied"]].copy()
    predictions["occupied_score"] = selected_p
    predictions["predicted_occupied"] = (selected_p >= 0.5).astype(int)
    daily = []
    for date, part in predictions.groupby(predictions.timestamp.dt.date):
        daily.append({"date": str(date), "rows": len(part),
                      "occupied_rows": int(part.occupied.sum()),
                      "accuracy": float(accuracy_score(part.occupied, part.predicted_occupied)),
                      "false_empty": int(((part.occupied == 1) & (part.predicted_occupied == 0)).sum()),
                      "false_occupied": int(((part.occupied == 0) & (part.predicted_occupied == 1)).sum())})
    ranges = {c: {"min": float(development[c].min()), "max": float(development[c].max()),
                  "median": float(development[c].median())} for c in FEATURES}
    summary = {
        "title": "Is Anyone There? Predicting Room Occupancy for Smarter Energy Use",
        "dataset_sha256": dataset_hash(), "rows": len(df), "seed": SEED,
        "target": "occupied = 1 when Room_Occupancy_Count > 0; otherwise 0",
        "threshold": 0.5, "python": platform.python_version(), "sklearn": sklearn.__version__,
        "selected_algorithm": selected.algorithm, "selected_sensor_set": selected.sensor_set,
        "features": selected_columns, "selection": "Highest validation F1, then occupied recall, then fewer readings, then algorithm name. Test never selects the model.",
        "validation_metrics": {k: float(selected[k]) for k in ["f1", "precision", "recall", "accuracy"]},
        "test_metrics": selected_metrics, "baseline_metrics": baseline_metrics,
        "splits": {name: {"rows": len(part), "occupied_rows": int(part.occupied.sum()),
                          "start": str(part.timestamp.min()), "end": str(part.timestamp.max())}
                   for name, part in [("train", train), ("validation", val), ("test", test)]},
        "ranges": ranges,
    }
    bundle = {"pipeline": selected_model, "features": selected_columns, "ranges": ranges,
              "algorithm": selected.algorithm, "sensor_set": selected.sensor_set,
              "dataset_sha256": dataset_hash(), "sklearn": sklearn.__version__}
    joblib.dump(bundle, artifacts / "occupancy_model.joblib")
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    validation.to_csv(reports / "validation_comparison.csv", index=False)
    test_results.to_csv(reports / "test_comparison.csv", index=False)
    predictions.to_csv(reports / "test_predictions.csv", index=False)
    pd.DataFrame(daily).to_csv(reports / "daily_test_metrics.csv", index=False)
    # Importance uses validation and a model fit only on the training partition.
    probe = make_models()[selected.algorithm]
    probe.fit(train[selected_columns], train.occupied)
    importance = permutation_importance(probe, val[selected_columns], val.occupied,
                                       scoring="f1", n_repeats=5, random_state=SEED, n_jobs=2)
    importance_df = pd.DataFrame({"sensor": selected_columns, "f1_decrease": importance.importances_mean,
                                  "std": importance.importances_std}).sort_values("f1_decrease", ascending=False)
    importance_df.to_csv(reports / "sensor_importance.csv", index=False)
    df[FEATURES].iloc[::100].head(50).to_csv(ROOT / "data" / "sample_upload.csv", index=False)
    for label, target in [("empty", 0), ("occupied", 1)]:
        df.loc[df.occupied == target, FEATURES].iloc[[0]].to_csv(ROOT / "data" / f"example_{label}.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    fig, ax = plt.subplots(figsize=(6, 4))
    df.occupied.value_counts().sort_index().plot.bar(ax=ax, color=["#c7d2dd", "#007f73"])
    ax.set(xticklabels=["Empty", "Occupied"], ylabel="Observations", title="Dataset target balance")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(figures / "class_balance.png"); plt.close(fig)
    cm = confusion_matrix(test.occupied, selected_p >= .5, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", color="white" if value > cm.max()/2 else "#142b3d", fontsize=18)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Empty", "Occupied"],
           yticklabels=["Empty", "Occupied"], xlabel="Predicted", ylabel="Actual", title="Later-period test confusion matrix")
    fig.tight_layout(); fig.savefig(figures / "confusion_matrix.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    comparison = validation.groupby("sensor_set", sort=False).first().reset_index()
    ax.barh(comparison.sensor_set, comparison.f1, color="#007f73")
    ax.set(xlim=(0, 1.05), xlabel="Validation F1", title="Best validation model for each sensor set")
    fig.tight_layout(); fig.savefig(figures / "sensor_comparison.png"); plt.close(fig)
    # Plain Markdown keeps report editable and inspectable without a document renderer.
    rows = "\n".join(f"| {r.sensor_set} | {r.algorithm} | {r.f1:.4f} | {r.recall:.4f} | {int(r.false_empty)} |" for r in test_results.itertuples())
    result_text = f"""# Reproduced experimental results

Selected model: **{selected.algorithm} / {selected.sensor_set}**. Selection used validation F1 only, with documented tie breakers; test scores did not select the winner. The final model was refit on the first two observed days.

The selected model detects **{selected_metrics['recall']:.1%} of occupied test readings**, with **{selected_metrics['false_empty']} false-empty predictions**. Its {selected_metrics['accuracy']:.1%} accuracy must be interpreted alongside that occupied recall. This is a research prototype and empty predictions are not sufficient for equipment-control decisions.

| Final test metric | Value |
|---|---:|
| Accuracy | {selected_metrics['accuracy']:.4f} |
| Balanced accuracy | {selected_metrics['balanced_accuracy']:.4f} |
| Occupied precision | {selected_metrics['precision']:.4f} |
| Occupied recall | {selected_metrics['recall']:.4f} |
| Occupied F1 | {selected_metrics['f1']:.4f} |
| Average precision | {selected_metrics['average_precision']:.4f} |
| False-empty predictions | {selected_metrics['false_empty']} |
| Majority baseline accuracy | {baseline_metrics['accuracy']:.4f} |
| Majority baseline occupied F1 | {baseline_metrics['f1']:.4f} |

## Diagnostic test comparisons

These comparisons describe the fixed experiments. They must not be used to repeatedly tune the final model on the test period.

| Sensor set | Algorithm | F1 | Occupied recall | False empty |
|---|---|---:|---:|---:|
{rows}

![Confusion matrix](figures/confusion_matrix.png)

![Validation sensor experiment](figures/sensor_comparison.png)

The later test period is dominated by empty readings. Accuracy alone is misleading; inspect occupied recall and the per-day table in `daily_test_metrics.csv`. A single room and a short experiment cannot establish performance in other buildings. The data contains long recording gaps; these are not evidence of vacancy.

Dataset SHA256: `{dataset_hash()}`. Random seed: {SEED}. scikit-learn: {sklearn.__version__}.
"""
    (reports / "RESULTS.md").write_text(result_text, encoding="utf-8")
    print(json.dumps({"selected": [selected.algorithm, selected.sensor_set], "test": selected_metrics,
                      "splits": summary["splits"]}, indent=2))
    return summary


if __name__ == "__main__":
    run_training()
