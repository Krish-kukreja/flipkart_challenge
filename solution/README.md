# Flipkart Traffic Demand Prediction - Our Approach

This repository contains our solution for the Flipkart Grid Traffic Demand Prediction challenge. We built a model to forecast normalized traffic demand (0.0 to 1.0) for given locations (`geohash`) during specific 15-minute time slots.

## 1. Overall Approach

We built our final solution on a single LightGBM Regressor trained using Out-Of-Fold spatial and temporal mean encodings. When we looked at the data, we found that the training set only had data for Day 48 (all slots) and the morning slots of Day 49. The test set wanted us to predict what would happen in Day 49's afternoon and evening slots.

Since we only had a day and a half of history to work with, the usual time series methods like rolling averages or ARIMA were not going to work, especially since we needed to predict in bulk. So we treated it more like a location and time based prediction problem instead. We built a Cross-Validation setup that trains on the first half of Day 48 (slots 0-47) and tests on the second half (slots 48-95), which closely mirrors the actual test situation and makes sure we are not accidentally using future data to predict the past.

## 2. Feature Engineering

We tried to capture demand patterns at two levels: the exact location (geohash) and the broader area around it (parent geohash).

### A. Time and Weather
We took the hour, minute, and slot out of the timestamp and also created two features, `hour_sin` and `hour_cos`, to help the model understand that time wraps around (so 11 PM and midnight are close to each other). We added a simple yes/no flag called `is_rush` to mark the busier parts of the day, specifically 7-10 AM and 4-7 PM. Where temperature data was missing, we filled it in using the typical temperature for that weather condition. We also created two extra features, `temp_x_rush` and `temp_bin_x_hour`, to capture how temperature and time of day work together.

### B. Location Based Averages
To make sure we were not leaking information from the future into our training, all location averages were calculated only from past data and then applied to the validation and test sets.
- `ghh_encoded`: the average past demand for that exact location at that hour.
- `gh5h_encoded` and `gh4h_encoded`: the same thing but for the wider area around the location, using 5 and 4 character parent geohashes.

### C. Morning Trend Features
Since we had Day 49 morning data available, we used it to get a sense of how the day was shaping up compared to Day 48.
- `morning_mean`: the average demand for a location during the morning slots (0-8).
- `lag1`: the last demand value we saw for that location, at slot 8.
- `morning_shift_ratio`: how the morning of Day 49 compared to the morning of Day 48 for the same location. This basically tells the model if a spot is busier or quieter than it was the day before.

## 3. Tools Used

- **Python 3**
- **Pandas and NumPy:** for cleaning and working with the data.
- **LightGBM:** the model we used for predictions. We picked it because it is fast, handles different types of data well, and does not overfit easily.
- **Optuna:** we used this to find the best settings for the model over 50 attempts. We kept the settings on the conservative side (num_leaves: 54, max_depth: 7) to make sure the model generalizes well.
- **Scikit-Learn:** used to measure how well the model performed using `r2_score`.

## 4. Files

- `src/pipeline.ipynb`: the Jupyter Notebook that does everything from loading the data, building features, training the model, and writing out the final predictions in an easy-to-read format.
- `src/pipeline.py`: the exact same pipeline but as a raw Python script.
- `requirements.txt`: the list of Python packages needed to run the code.
- `approach.txt`: a text summary of our approach.
