# Data card

Dataset: Room Occupancy Estimation, UCI ID 864. Creators: Adarsh Pal Singh and Sachin Chaudhari. DOI: https://doi.org/10.24432/C5P605. CC BY 4.0.

Source download: https://archive.ics.uci.edu/static/public/864/room+occupancy+estimation.zip

The original CSV is preserved in this directory. The raw file has 10,129 observations and 19 columns: Date, Time, sixteen sensor-related columns, and Room_Occupancy_Count. The source repository counts Date and Time among its eighteen input features. This project uses fifteen raw sensor readings; it excludes Date, Time and S5_CO2_Slope from the model inputs.

| Field family | Meaning | Unit |
|---|---|---|
| S1_Temp to S4_Temp | Four temperature readings | Celsius |
| S1_Light to S4_Light | Four illuminance readings | lux |
| S1_Sound to S4_Sound | Four sound sensor amplifier outputs | volts, not decibels or audio recordings |
| S5_CO2 | Carbon dioxide concentration | ppm |
| S5_CO2_Slope | Supplied sliding-window slope; excluded | source-defined |
| S6_PIR, S7_PIR | Passive infrared motion detection | binary 0 or 1 |
| Room_Occupancy_Count | Manually recorded ground truth | 0, 1, 2 or 3 people |

The new target is `occupied = (Room_Occupancy_Count > 0)`. Sample inputs are extracted from the original CSV. No synthetic records are added to the training data. Numeric types are checked, timestamps parsed and records sorted. Invalid readings cause an explicit error. No imputation is required for this complete source; uploaded missing values are rejected rather than silently filled.

Training uses the earliest recorded date, validation uses the second date, and test uses all later dates. The final selected model is refit on training plus validation data after the configuration is frozen. Full split boundaries and a SHA256 checksum are recorded in `artifacts/summary.json`.

The source describes a 6 m × 4.6 m room, no HVAC during collection, and readings roughly every 30 seconds. The CSV actually includes seven calendar dates and long gaps. It is a small, controlled, single-room dataset, not live campus data. It cannot establish performance at PCCoE, in larger crowds, under new sensor calibration, or in rooms with different lighting and ventilation. The later test portion is strongly imbalanced toward empty readings.

Source documentation: https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation
