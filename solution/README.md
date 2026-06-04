# Flipkart Traffic Demand Prediction

This repository contains our solution for the Flipkart Grid Traffic Demand Prediction challenge. The objective of this project is to accurately forecast the normalized traffic demand (0.0 to 1.0) for a given geographical location (`geohash`) at a specific 15-minute time slot.

## 🚀 How Our Model Works

Our approach models the problem as a **Spatial-Temporal Regression** task rather than a traditional time-series forecasting problem. By shifting away from multi-day rolling averages—which struggle due to structural missing data in the challenge—we focus on deeply analyzing specific micro and macro-level spatial trends.

### 1. Granular Time Parsing
Traffic is highly cyclical and predictable based on the time of day. We engineer features to capture these patterns perfectly:
- **Time Boxing:** We break timestamps down into `hour`, `minute`, and specific 15-minute `slot` increments.
- **Cyclical Encoding:** Using sine and cosine transformations, we allow the model to understand that 11:59 PM is mathematically adjacent to 12:00 AM.
- **Human Behavior Flags:** We introduce an `is_rush` feature to explicitly label peak morning commute (7-10 AM) and evening return (4-7 PM) hours.

### 2. Spatial "Target Encodings"
To help the model understand exactly "how busy" a location typically is, we compute historical means. Crucially, these are computed strictly on our training fold to absolutely prevent target leakage.
- **Micro-level (`ghh_encoded`):** The average traffic demand for an exact, specific `geohash` during a specific hour.
- **Macro-level (`gh5h` & `gh4h`):** Sometimes a location is too new or sparse to have a reliable mean. We look at the parent 5-character and 4-character geohash zones. If a street is unknown, but the surrounding district is swamped, the model infers high traffic.

### 3. "Morning Momentum" & Trend Ratios
Traffic isn't just about averages; it's about momentum. Is today a holiday? Did a localized event occur?
To capture this, we track the momentum of the current morning and compare it to historical mornings:
- **`morning_mean`:** How busy a geohash was *this morning* (slots 0-8).
- **`lag1`:** The absolute most recently observed demand for the location right before our forecasting horizon begins.
- **`morning_shift_ratio`:** We divide today's morning mean by yesterday's morning mean. If this ratio is > 1.0, the model learns that this specific area is organically "trending hotter" today and scales its afternoon predictions accordingly.

### 4. Interactive Weather Dynamics
We noticed that weather doesn't affect traffic uniformly. Rain at 3 AM does very little, but rain during a 5 PM rush hour causes significant spikes. We encode these relationships using interaction terms like `temp_x_rush`.

---

## 🛠️ Machine Learning Pipeline

Our entire approach is housed inside a single, deterministic pipeline: `src/pipeline.py`.

1. **Rigorous Cross-Validation:** The pipeline enforces a strict chronological split on Day 48 data. It trains on slots 0-47 and validates on slots 48-95. This perfectly simulates the test environment and prevents the model from looking into the future.
2. **Algorithm:** We utilize **LightGBM** (`LGBMRegressor`). It was chosen for its blazing-fast tabular data processing, native support for categorical variables like `geohash`, and its ability to handle complex non-linear feature interactions seamlessly.
3. **Hyperparameter Tuning:** The model parameters were optimized using an Optuna study. Rather than maximizing raw depth, we constrained the model (e.g., `max_depth: 7`, `num_leaves: 54`) with conservative regularization parameters (`reg_alpha`, `reg_lambda`) to ensure maximum generalizability to unseen data.

## 📂 Repository Structure

- `dataset/`: Contains `train.csv` and `test.csv` (Note: Ensure data is populated before running).
- `src/pipeline.py`: The main orchestrator script. Runs data loading, leak-proof feature engineering, LightGBM training, and generates predictions.
- `requirements.txt`: Python package dependencies necessary to run the pipeline.
- `approach.txt`: A detailed, formal breakdown of our feature engineering for the competition reviewers.
