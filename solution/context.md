# Flipkart Traffic Demand Prediction - Project Context

**Goal:** Predict the normalized traffic demand (0.0 to 1.0) for a given `geohash` at a specific 15-minute time slot. The evaluation metric is **R² (R-squared)**.

**Current Best Leaderboard Score:** 90.72 R² (Strict CV: ~0.498 R²)

## 1. The Great Reset & Data Reality
We discovered a critical structural flaw in our previous assumptions: **Days 1-47 do not exist in the training data.** 
The `train.csv` ONLY contains:
*   **Day 48:** All 96 slots (0-95).
*   **Day 49:** Only morning slots (0-35).
Because we must predict Day 49 afternoon/evening (slots 36+), any "multi-day rolling average" strategy structurally fails. We completely restarted our approach to build a leak-free, mathematically honest pipeline.

## 2. Codebase Architecture
The solution has been sanitized and stripped down to exactly what works:
*   **`src/pipeline_90.72.py`:** The single, definitive script that generates our best submission. It handles strict CV splitting, feature engineering, LightGBM training, and generates `submission_90.72.csv`.
*   **`dataset/`:** Contains `train.csv` and `test.csv`.

## 3. Strict Honest CV Strategy
To prevent leakage and accurately simulate the LB, we use a rigid time-based split for our Cross Validation:
*   **CV Train Fold:** Day 48, slots 0-47.
*   **CV Val Fold:** Day 48, slots 48-95.
All target-encoded features (means, standard deviations, shift ratios) are computed **strictly** on the Train Fold and mapped to the Val Fold. This ensures the model learns real generalization, not target leakage.

## 4. Proven Feature Engineering
Our 90.72 score was achieved through rigorous, isolated A/B testing of features. If a feature dropped the LB score, it was discarded (e.g., weather-hour interactions).

**Time & Environment (Base)**
*   `hour`, `minute`, `slot`, `hour_sin`, `hour_cos` (Cyclical time).
*   `is_rush` (Binary rush hour flag: 7-10 AM, 4-7 PM).
*   `Temperature` (Imputed per weather condition).
*   `temp_x_rush`: Temperature during rush hours (Proven LB boost).
*   `temp_bin_x_hour`: Binned temperature multiplied by the hour (Proven LB boost).

**Spatial Means (The Engine)**
*   `ghh_encoded`: The historical mean demand for the exact geohash at that specific hour.
*   `gh5h_encoded`: Mean demand for the parent 5-char geohash at that specific hour.
*   `gh4h_encoded`: Mean demand for the parent 4-char geohash at that specific hour.

**Morning Momentum (Capturing Today's Trend)**
*   `morning_mean`: The mean demand for the geohash during today's morning (slots 0-8).
*   `lag1`: The absolute last observed demand (slot 8) for the exact geohash.
*   `morning_shift_ratio`: `morning_mean` (today) / `d48_morning_mean` (yesterday). Tells the model if this specific geohash is "hotter" or "colder" today than yesterday.

## 5. Model & Optimization
*   **Algorithm:** LightGBM Regressor (`lgb.LGBMRegressor`).
*   **Optuna Tuning:** We ran a 50-trial Optuna study on the exact feature set, finding parameters that prioritize generalization (`num_leaves`: 54, `max_depth`: 7, conservative regularization).

## 6. Next Experimental Frontiers
We are continuing to test isolated features against the 90.72 baseline:
1.  **Uncertainty Quantification:** `ghh_std` (the standard deviation of the geohash-hour). Tested and yielded massive CV gains (+0.018), but dropped LB to 90.66. (Discarded, but concept remains interesting).
2.  **Parent-Level Momentum:** `gh5_shift_ratio` (Day 49 morning vs Day 48 morning at the 5-char geohash level). Good fallback for cold-start geohashes.
3.  **Spatial Ranking:** `gh5_rank` (Ranking a child geohash's busyness relative to its 5-char parent siblings). Captures stable micro-location effects.
