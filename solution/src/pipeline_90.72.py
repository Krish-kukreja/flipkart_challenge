import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import r2_score
import lightgbm as lgb
import optuna
import os

warnings.filterwarnings('ignore')

dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')

# ==========================================
# 1. LOAD DATA (identical to your 90.64 pipeline)
# ==========================================
train = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))
test = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))
test_indices = test['Index'].values

# ==========================================
# 2. BASE FEATURES (identical to 90.64)
# ==========================================
for df in [train, test]:
    df['RoadType'] = df['RoadType'].fillna('Unknown')
    df['Weather'] = df['Weather'].fillna('Unknown')
    df['NumberofLanes'] = df['NumberofLanes'].fillna(0).astype(int)
    df['LargeVehicles'] = df['LargeVehicles'].map({'Allowed': 1, 'Not Allowed': 0}).fillna(0).astype(int)
    df['Landmarks'] = df['Landmarks'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    
    time_split = df['timestamp'].str.split(':', expand=True).astype(int)
    df['hour'] = time_split[0]
    df['minute'] = time_split[1]
    df['slot'] = (df['hour'] * 60 + df['minute']) // 15
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['is_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 10) | (df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)

# Temperature imputation (identical)
weather_medians = train.groupby('Weather')['Temperature'].median()
train['Temperature'] = train.groupby('Weather')['Temperature'].transform(lambda x: x.fillna(x.median()))
train['Temperature'] = train['Temperature'].fillna(train['Temperature'].median())
test['Temperature'] = test['Weather'].map(weather_medians).fillna(train['Temperature'].median())

# Weather severity (identical)
weather_map = {'Sunny': 0, 'Foggy': 1, 'Rainy': 2, 'Snowy': 3, 'Unknown': 0}
train['weather_sev'] = train['Weather'].map(weather_map)
test['weather_sev'] = test['Weather'].map(weather_map)

# ==========================================
# 3. THE ONLY NEW FEATURES: GEOHASH HIERARCHY
# ==========================================
# 5-char and 4-char parent geohashes
for df in [train, test]:
    df['gh5'] = df['geohash'].str[:5]
    df['gh4'] = df['geohash'].str[:4]

# ==========================================
# 4. HONEST CV (identical split: Day 48 slots 0-47 → 48-95)
# ==========================================
day48 = train[train['day'] == 48].copy()
train_fold = day48[day48['slot'] <= 47].copy()
val_fold = day48[day48['slot'] > 47].copy()

# --- Morning stats (identical) ---
d48_morning = train_fold[train_fold['slot'] <= 8].copy()
morning_stats = d48_morning.groupby('geohash')['demand'].agg(
    morning_mean='mean'
).reset_index()
last_obs = d48_morning.sort_values('slot').groupby('geohash').last().reset_index()[['geohash', 'demand']].rename(columns={'demand': 'lag1'})
morning_stats = morning_stats.merge(last_obs, on='geohash', how='left')

train_fold = train_fold.merge(morning_stats, on='geohash', how='left')
val_fold = val_fold.merge(morning_stats, on='geohash', how='left')

global_morning_mean = d48_morning['demand'].mean()
for df in [train_fold, val_fold]:
    df['morning_mean'] = df['morning_mean'].fillna(global_morning_mean)
    df['lag1'] = df['lag1'].fillna(df['morning_mean'])

# --- ghh_encoded (identical) ---
ghh_mean_cv = train_fold.groupby(['geohash', 'hour'])['demand'].mean()
global_mean_cv = train_fold['demand'].mean()

train_fold['ghh_encoded'] = train_fold.set_index(['geohash', 'hour']).index.map(ghh_mean_cv).fillna(global_mean_cv)
val_fold['ghh_encoded'] = val_fold.set_index(['geohash', 'hour']).index.map(ghh_mean_cv).fillna(global_mean_cv)

# --- city hour momentum (identical) ---
city_hour_mean_cv = train_fold.groupby('hour')['demand'].mean()
train_fold['city_hour_momentum'] = train_fold['hour'].map(city_hour_mean_cv).fillna(global_mean_cv)
val_fold['city_hour_momentum'] = val_fold['hour'].map(city_hour_mean_cv).fillna(global_mean_cv)

# --- morning_shift_ratio (identical) ---
d48_morning_global = train_fold[train_fold['slot'] <= 8].groupby('geohash')['demand'].mean().reset_index()
d48_morning_global = d48_morning_global.rename(columns={'demand': 'd48_morning_mean'})

train_fold = train_fold.merge(d48_morning_global, on='geohash', how='left')
val_fold = val_fold.merge(d48_morning_global, on='geohash', how='left')

train_fold['morning_shift_ratio'] = (train_fold['morning_mean'] / (train_fold['d48_morning_mean'] + 1e-9)).fillna(1.0)
val_fold['morning_shift_ratio'] = (val_fold['morning_mean'] / (val_fold['d48_morning_mean'] + 1e-9)).fillna(1.0)

# --- Proven interactions (identical) ---
train_fold['temp_x_rush'] = train_fold['Temperature'] * train_fold['is_rush']
val_fold['temp_x_rush'] = val_fold['Temperature'] * val_fold['is_rush']

# Temperature bin (identical)
temp_bins = pd.qcut(train_fold['Temperature'], q=5, labels=False, duplicates='drop').unique()
train_fold['temp_bin'] = pd.qcut(train_fold['Temperature'], q=5, labels=False, duplicates='drop')
val_fold['temp_bin'] = pd.cut(val_fold['Temperature'], bins=pd.qcut(train_fold['Temperature'], q=5, retbins=True)[1], labels=False, include_lowest=True).fillna(0).astype(int)

train_fold['temp_bin_x_hour'] = train_fold['temp_bin'] * train_fold['hour']
val_fold['temp_bin_x_hour'] = val_fold['temp_bin'] * val_fold['hour']

# ==========================================
# 5. THE ONLY NEW CODE: GEOHASH HIERARCHY ENCODING
# ==========================================
# Compute from train_fold ONLY (leakage-free for CV)
gh5h_mean_cv = train_fold.groupby(['gh5', 'hour'])['demand'].mean()
gh4h_mean_cv = train_fold.groupby(['gh4', 'hour'])['demand'].mean()

train_fold['gh5h_encoded'] = train_fold.set_index(['gh5', 'hour']).index.map(gh5h_mean_cv).fillna(global_mean_cv)
val_fold['gh5h_encoded'] = val_fold.set_index(['gh5', 'hour']).index.map(gh5h_mean_cv).fillna(global_mean_cv)
train_fold['gh4h_encoded'] = train_fold.set_index(['gh4', 'hour']).index.map(gh4h_mean_cv).fillna(global_mean_cv)
val_fold['gh4h_encoded'] = val_fold.set_index(['gh4', 'hour']).index.map(gh4h_mean_cv).fillna(global_mean_cv)

# ==========================================
# 6. FEATURE LIST (ONLY ADD gh5h, gh4h)
# ==========================================
features = [
    'hour', 'minute', 'slot', 'hour_sin', 'hour_cos', 'is_rush',
    'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Temperature',
    'ghh_encoded', 'morning_mean', 'lag1', 'morning_shift_ratio',
    'temp_x_rush', 'temp_bin_x_hour',
    'gh5h_encoded', 'gh4h_encoded'  # <-- THE ONLY NEW FEATURES
]
cat_features = ['geohash', 'RoadType', 'Weather']

X_tr = train_fold[features + cat_features]
y_tr = train_fold['demand']
X_val = val_fold[features + cat_features]
y_val = val_fold['demand']

for col in cat_features:
    X_tr[col] = X_tr[col].astype('category')
    X_val[col] = X_val[col].astype('category')

# ==========================================
# 7. MODEL: Optuna-tuned params (0.45861)
# ==========================================
lgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.04,
    'num_leaves': 54,
    'max_depth': 7,
    'min_child_samples': 12,
    'reg_lambda': 0.15609878207944705,
    'reg_alpha': 0.02009412265635894,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

model = lgb.LGBMRegressor(**lgb_params)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    categorical_feature=cat_features,
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

val_preds = np.clip(model.predict(X_val), 0, 1)
cv_r2 = r2_score(y_val, val_preds)
print(f"CV R2: {cv_r2:.5f}")

# ==========================================
# 8. FINAL TRAINING & SUBMISSION
# ==========================================
final_train = train[(train['day'] == 48) | ((train['day'] == 49) & (train['slot'] <= 8))].copy()

# Morning stats per day (identical)
d48_morn_train = train[(train['day'] == 48) & (train['slot'] <= 8)]
d48_morning_stats = d48_morn_train.groupby('geohash')['demand'].agg(
    morning_mean='mean'
).reset_index()
d48_last_obs = d48_morn_train.sort_values('slot').groupby('geohash').last().reset_index()[['geohash', 'demand']].rename(columns={'demand': 'lag1'})
d48_morning_stats = d48_morning_stats.merge(d48_last_obs, on='geohash', how='left')
d48_morning_stats['day'] = 48

d49_morn_train = train[(train['day'] == 49) & (train['slot'] <= 8)]
d49_morning_stats = d49_morn_train.groupby('geohash')['demand'].agg(
    morning_mean='mean'
).reset_index()
d49_last_obs = d49_morn_train.sort_values('slot').groupby('geohash').last().reset_index()[['geohash', 'demand']].rename(columns={'demand': 'lag1'})
d49_morning_stats = d49_morning_stats.merge(d49_last_obs, on='geohash', how='left')
d49_morning_stats['day'] = 49

all_morning_stats = pd.concat([d48_morning_stats, d49_morning_stats])
final_train = final_train.merge(all_morning_stats, on=['geohash', 'day'], how='left')

test = test.merge(d49_morning_stats[['geohash', 'morning_mean', 'lag1']], on='geohash', how='left')

global_morning_mean_final = train[train['slot'] <= 8]['demand'].mean()
for df in [final_train, test]:
    df['morning_mean'] = df['morning_mean'].fillna(global_morning_mean_final)
    df['lag1'] = df['lag1'].fillna(df['morning_mean'])

# ghh_encoded from all Day 48 (identical)
d48_hour_mean = train[train['day'] == 48].groupby(['geohash', 'hour'])['demand'].mean()
d48_global_mean = train[train['day'] == 48]['demand'].mean()

final_train['ghh_encoded'] = final_train.set_index(['geohash', 'hour']).index.map(d48_hour_mean).fillna(d48_global_mean)
test['ghh_encoded'] = test.set_index(['geohash', 'hour']).index.map(d48_hour_mean).fillna(d48_global_mean)

# City hour momentum (identical)
final_city_hour_mean = final_train.groupby('hour')['demand'].mean()
final_global_mean = final_train['demand'].mean()
final_train['city_hour_momentum'] = final_train['hour'].map(final_city_hour_mean).fillna(final_global_mean)
test['city_hour_momentum'] = test['hour'].map(final_city_hour_mean).fillna(final_global_mean)

# shift_ratio (identical)
d48_morning_global_final = train[(train['day'] == 48) & (train['slot'] <= 8)].groupby('geohash')['demand'].mean().reset_index()
d48_morning_global_final = d48_morning_global_final.rename(columns={'demand': 'd48_morning_mean'})

final_train = final_train.merge(d48_morning_global_final, on='geohash', how='left')
final_train['morning_shift_ratio'] = np.where(final_train['day'] == 49, final_train['morning_mean'] / (final_train['d48_morning_mean'] + 1e-9), 1.0)
final_train['morning_shift_ratio'] = final_train['morning_shift_ratio'].fillna(1.0)

test = test.merge(d48_morning_global_final, on='geohash', how='left')
test['morning_shift_ratio'] = (test['morning_mean'] / (test['d48_morning_mean'] + 1e-9)).fillna(1.0)

# Proven interactions (identical)
final_train['temp_x_rush'] = final_train['Temperature'] * final_train['is_rush']
test['temp_x_rush'] = test['Temperature'] * test['is_rush']

# Temperature bin (identical)
final_train['temp_bin'] = pd.qcut(final_train['Temperature'], q=5, labels=False, duplicates='drop')
test['temp_bin'] = pd.cut(test['Temperature'], bins=pd.qcut(final_train['Temperature'], q=5, retbins=True)[1], labels=False, include_lowest=True).fillna(0).astype(int)

final_train['temp_bin_x_hour'] = final_train['temp_bin'] * final_train['hour']
test['temp_bin_x_hour'] = test['temp_bin'] * test['hour']

# ==========================================
# 9. FINAL HIERARCHY ENCODING (from all Day 48)
# ==========================================
# Compute from all Day 48 data (final training)
gh5h_mean_final = train[train['day'] == 48].groupby(['gh5', 'hour'])['demand'].mean()
gh4h_mean_final = train[train['day'] == 48].groupby(['gh4', 'hour'])['demand'].mean()

final_train['gh5h_encoded'] = final_train.set_index(['gh5', 'hour']).index.map(gh5h_mean_final).fillna(d48_global_mean)
test['gh5h_encoded'] = test.set_index(['gh5', 'hour']).index.map(gh5h_mean_final).fillna(d48_global_mean)
final_train['gh4h_encoded'] = final_train.set_index(['gh4', 'hour']).index.map(gh4h_mean_final).fillna(d48_global_mean)
test['gh4h_encoded'] = test.set_index(['gh4', 'hour']).index.map(gh4h_mean_final).fillna(d48_global_mean)

# ==========================================
# 10. PREDICT & SUBMIT
# ==========================================
X_final = final_train[features + cat_features].copy()
y_final = final_train['demand']
X_test = test[features + cat_features].copy()

for col in cat_features:
    X_final[col] = X_final[col].astype('category')
    X_test[col] = X_test[col].astype('category')

model.fit(X_final, y_final, categorical_feature=cat_features)

final_preds = np.clip(model.predict(X_test), 0, 1)

submission = pd.DataFrame({'Index': test_indices, 'demand': final_preds})
submission.to_csv('submission_90.72.csv', index=False)

print(f"CV R2: {cv_r2:.5f}")
print(f"Submission mean: {submission['demand'].mean():.5f}")
print("Saved: submission_90.72.csv")
