# Is Anyone There? Predicting Room Occupancy for Smarter Energy Use

## Project information

Pimpri Chinchwad College of Engineering, Pune  
Department of Information Technology  
Second Year B.Tech, Semester III, Division B  
Course: VSEC Data Science Laboratory, BIT23VS01  
Faculty: JYOTI MANOORKAR  
Group: B3-4

| Name | PRN |
|---|---|
| Pratik Deepak Phapale | 125B1F210 |
| Swastik Sandipkumar Phapale | 125B1F212 |
| Nirjar Rupesh Patil | 125B1F206 |

## Abstract

This project studies whether environmental sensor readings can distinguish an empty room from an occupied room, and whether fewer sensor readings can retain useful classification performance. It uses the public UCI Room Occupancy Estimation dataset and compares Logistic Regression, Decision Tree and Random Forest classifiers across four sensor configurations. Separate chronological training, validation and test periods reduce optimistic evaluation caused by neighbouring time samples. A Streamlit interface supports data exploration, manual and CSV-based prediction, model evaluation and a hypothetical energy-saving calculation. Results are generated from executed experiments in the companion Results document; energy calculations are scenarios, not measured savings. The work is limited to a controlled single-room dataset and does not claim readiness for automatic campus equipment control.

## Problem statement and objectives

Room status can help people decide when to inspect potentially unused equipment. Motion alone may miss stationary occupants, while other environmental readings may add information. The project asks whether temperature, light, sound amplitude, carbon dioxide and motion can classify occupancy, and which combinations provide a useful balance between the number of readings and predictive quality.

Objectives are to prepare and understand the dataset, compare classification methods, evaluate occupied-room errors explicitly, compare smaller sensor sets, provide an understandable GUI and demonstrate a transparent energy scenario. The aim is to support a manual decision rather than operate switches.

## Dataset and preprocessing

The raw CSV contains 10,129 observations with Date, Time, sixteen sensor-related variables and an occupancy count from 0 to 3. The binary target is 1 when the count is greater than zero and 0 otherwise. Fifteen raw sensor values are eligible model inputs. Date and Time are retained for ordering and graphs; the provided CO2 slope is excluded because its window construction has not been verified. Numeric validity, finite values, PIR values and timestamp uniqueness are checked before modelling. The source has no missing sensor values. Uploaded invalid data is rejected with a useful message.

The source summary refers to four collection days, but the CSV covers seven calendar dates from 22 December 2017 to 11 January 2018. Analyses follow the actual timestamps and preserve recording gaps. The data is from one controlled room with at most three occupants; it does not represent PCCoE classrooms.

## Architecture and algorithm

1. Load the source CSV and validate its columns, readings and timestamps.
2. Derive the binary target and sort records chronologically.
3. Assign the first observed date to training, the second date to validation and later dates to test.
4. Fit three algorithms for each of four declared sensor sets on training data only.
5. Select the configuration with highest validation F1. Resolve ties using occupied recall, fewer readings and then algorithm name. Keep a fixed probability threshold of 0.5.
6. Freeze the chosen configuration and refit it on training plus validation data.
7. Evaluate the frozen choice on the test period, alongside a majority baseline and diagnostic fixed comparisons.
8. Save the pipeline, feature order, reference ranges, checksum, results and figures.
9. Load the saved local model in the GUI for inference. Do not train on uploaded CSV files.

The software flow is `CSV → validation → chronological splits → pipeline training → evaluation → saved model → GUI`. Prediction and energy calculations share functions in `src/core.py` so the notebook, GUI and tests use the same definitions.

## Mathematical background

For sensor vector x, Logistic Regression estimates p = 1 / (1 + exp(-(wᵀx + b))). Inputs are standardized using training means and standard deviations. The binary prediction is occupied when p ≥ 0.5. Decision Trees recursively divide feature space; Random Forest averages estimates from multiple trees. Tree depth and minimum leaf sizes constrain complexity; class weighting accounts for imbalance. These outputs are model scores and are not assumed calibrated probabilities.

Occupied precision = TP / (TP + FP). Occupied recall = TP / (TP + FN). F1 = 2 × precision × recall / (precision + recall). Accuracy = (TP + TN) / N. Balanced accuracy averages sensitivity and specificity. A false-empty prediction is FN: a room with occupants is classified as empty. This error receives explicit attention because an occupancy-based equipment decision could inconvenience occupants. ROC AUC and average precision summarize score ranking. Undefined precision/F1 for a majority baseline is reported as zero.

## Sensor experiment

| Configuration | Readings | Research purpose |
|---|---:|---|
| All sensors | 15 | Full raw environmental and motion information |
| Ambient only | 9 | Temperature, light and CO2 without sound or motion |
| Three readings | 3 | S1 temperature, S1 light and S5 CO2 |
| Motion only | 2 | S6 and S7 PIR reference |

Counts refer to feature readings, not a verified number of physical devices or a cost quotation. Permutation importance measures validation F1 decrease when a feature is shuffled, using a model fit only on training data. Correlated readings can share importance, so low importance does not prove a sensor is unnecessary. Comparing subsets is predictive evidence, not a causal experiment.

## Experimental protocol and results

See [Reproduced experimental results](RESULTS.md) for actual metrics, the selected configuration, the confusion matrix and sensor comparison. CSV tables preserve full precision. The final test period is not used to choose a model; all shown test comparisons are descriptive. `daily_test_metrics.csv` separates occupied days from days containing only empty readings. Repeatedly optimizing after viewing these test results would invalidate this held-out estimate.

The notebook `RoomSense_Analysis.ipynb` executes data auditing, exploratory plots, a contingency analysis, reproduction of model training, baseline comparison, feature experiments and energy simulation. `artifacts/summary.json` records software versions and exact date boundaries.

## Graphical interface

- Overview presents the research question, dataset coverage and final evaluation.
- Predict occupancy accepts individual readings or a CSV, reports the model score and warns about readings outside training ranges. Batch outputs are downloadable.
- Explore data provides date filters, sensor distributions, correlation and time plots with recording gaps respected.
- Model lab shows validation and test results, split boundaries, confusion counts, importance and class imbalance.
- Energy sandbox replays held-out predictions and estimates hypothetical equipment-on energy during sustained empty predictions.
- Project guide contains group details, definitions, source credit and report/notebook downloads.

## Energy simulation assumptions

Assume the specified equipment would otherwise remain on throughout the valid observed intervals. Potential energy in kWh = equipment watts × eligible empty-prediction hours / 1000. Potential cost = kWh × the user-entered rupees per kWh. Neither the power nor tariff is inferred from the data, and the default tariff is an illustrative assumption.

An alert is eligible only after a continuous predicted-empty waiting period. Adjacent intervals longer than 90 seconds are excluded and reset the timer; the last observation has no assumed duration. The simulator uses a sample-and-hold approximation between adjacent readings. It also reports eligible minutes during which ground truth indicates occupancy. These are false-empty review alerts, not safe energy-saving opportunities. No actual consumption, savings, occupancy beyond the observation interval or device status is measured.

## Testing and reproducibility

Automated tests cover missing/non-numeric/infinite CSV values, invalid PIR readings, chronological separation, model feature validation, temporal gap handling, delay integration, false-empty intervals and GUI views. The notebook is executed rather than supplied as unrun code. All pseudo-random operations use seed 42. Recreate outputs with `train.py`, then `scripts/build_notebook.py`; run tests with `python -m pytest -q`. Exact installed package versions are saved in `requirements-lock.txt`.

## Limitations and future work

The experiment covers one room, a short time span and very few occupants. Sensor calibration, lighting, ventilation and behaviour may change performance elsewhere. Temporal separation reduces local leakage but does not replace external-room validation. Class imbalance means high accuracy can coexist with poor occupied recall. Manual input outside training conditions should be treated as exploratory. The project does not identify people, estimate large crowds, verify real electricity savings or automatically switch equipment.

Future work could collect consented local environmental measurements with reliable ground truth, test multiple rooms and seasons, calibrate probabilities, evaluate different alert delays and compare estimated energy with real meter readings. Any operational system would need conservative fail-safe controls and independent validation.

## References

1. Singh, A. and Chaudhari, S. (2018). Room Occupancy Estimation. UCI Machine Learning Repository. https://doi.org/10.24432/C5P605. Dataset licensed CC BY 4.0.
2. Singh et al. (2018). Machine Learning-Based Occupancy Estimation Using Multivariate Sensor Nodes. IEEE Globecom Workshops. Linked from the UCI dataset page: https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation.
3. Scikit-learn documentation: https://scikit-learn.org/stable/.
4. Streamlit documentation: https://docs.streamlit.io/.
