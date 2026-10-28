# Presentation and viva guide

## Suggested seven-minute demonstration

1. Introduce the problem: predicting empty versus occupied rooms and comparing sensor sets.
2. Show the source dataset, target balance and the seven actual observation dates.
3. Explain the chronological split and why random row splitting would mix nearby readings.
4. Show the validation-selected model and the final test confusion matrix. Explain a false-empty prediction.
5. Compare the full sensor set with smaller sets. Avoid claiming hardware cost savings from column counts.
6. Load an example in Predict occupancy, change a reading and run prediction. Upload `data/sample_upload.csv`, then download results.
7. Show the energy simulation, change equipment power and vacancy delay, and explain why the numbers are hypothetical.

## Questions to prepare for

**Why binary classification?** The main question is whether anyone is present. The source count is converted into 0 for empty and 1 for any count above zero. We do not estimate classroom-sized occupancy.

**Why is this data science?** It includes dataset validation, exploratory analysis, a measurable research question, statistical summaries, feature experiments, classification and independent evaluation.

**Why compare against a baseline?** A dataset dominated by empty readings can give a high accuracy to a model that always predicts empty. Occupied recall and F1 reveal that weakness.

**Why exclude time and CO2 slope?** Time is used to define evaluation periods, not as an occupancy shortcut. The source slope's window computation is not verified, so the baseline study excludes it.

**Why scale Logistic Regression?** Sensor units differ greatly. Standardization makes optimization more stable. Training statistics are learned inside a pipeline and applied unchanged to later data.

**What is data leakage?** Information that would be unavailable during genuine prediction enters training or evaluation. Using occupancy count as an input, scaling before splitting, or tuning on test results would create misleading results.

**What is a false-empty error?** Actual occupancy is 1 but prediction is 0. This could lead to an inappropriate energy-management suggestion, so the app is a manual-review prototype.

**What is the unique experiment?** Compare four predetermined sensor sets to investigate whether fewer measurements can preserve useful validation performance. Do not claim the general idea of occupancy detection is globally new.

**Are the scores certainty?** No. Classifier output is not automatically a calibrated probability, particularly with weighted classes and a different test distribution.

**Does the app need hardware or internet?** The supplied prototype replays recorded data or accepts manually entered readings. No hardware is required. After setup and download, local inference is offline.

**Did you measure electricity savings?** No. Power and tariff are assumptions. The dataset contains no energy meter readings or equipment state. The simulation explicitly excludes recording gaps and reports false-empty review intervals.

**Can it work at PCCoE?** That has not been validated. Room-specific calibration and new labelled data would be required.

**Why not use the best test score?** Test data is for the final evaluation. The deployed configuration is selected using validation F1, and then frozen before test evaluation.

**Why is a statistical p-value not enough?** Nearby readings are correlated, violating the independent-observation assumption of common tests. Descriptive association can be shown, but naive row-wise significance would be overstated.

## Suggested team responsibilities

Pratik: dataset audit, preprocessing and exploratory analysis. Swastik: algorithms, validation and evaluation. Nirjar: GUI, demonstration and report integration. These are suggested responsibilities; every member should understand the full pipeline and be able to explain the limitations.
