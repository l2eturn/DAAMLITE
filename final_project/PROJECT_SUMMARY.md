# NYC Taxi Fare Prediction — Project Summary

## Overview

This project predicts NYC yellow-taxi fare amounts from trip coordinates, timestamps, and passenger count. It follows a structured 5-phase ML pipeline using Python (scikit-learn, XGBoost).

---

## Dataset

| File | Rows | Columns |
|---|---|---|
| `raw-data/train_taxi.csv` | ~55,000+ | key, fare_amount, pickup_datetime, pickup/dropoff lon/lat, passenger_count |
| `raw-data/test_taxi.csv` | ~9,900+ | key, pickup_datetime, pickup/dropoff lon/lat, passenger_count |

**Target variable**: `fare_amount` (USD)

---

## Pipeline Phases

### Phase 1 — Data Exploration, Cleaning & Feature Engineering (`eda.py`)

**Data Loading**
- Reads both `train_taxi.csv` and `test_taxi.csv` with datetime parsing.

**EDA (Exploratory Data Analysis)**
- 6-panel plot saved to `eda_plots.png`: fare distribution, fare by passenger count, pickup locations (scatter), median fare by hour, by day of week, and by year.
- Correlation heatmap saved to `correlation.png`.
- Interactive Folium map (`taxi_map.html`) with 4 toggleable layers: pickup heatmap, fare dots, trip O→D lines, airport markers.

**Data Cleaning (train only)**
- Drop rows with NaN values.
- Filter `fare_amount` to range \[$2.50, $500\] (NYC minimum base fare to reasonable upper bound).
- Filter coordinates to NYC bounding box: lon ∈ (−74.30, −72.90), lat ∈ (40.50, 41.80).
- Filter `passenger_count` to \[1, 6\] per NYC regulations.

**Feature Engineering (applied to both train & test)**

| Feature | Description |
|---|---|
| `distance_km` | Haversine distance between pickup and dropoff |
| `delta_lat` / `delta_lon` | Coordinate deltas (direction of travel) |
| `abs_delta_lat` / `abs_delta_lon` | Absolute deltas |
| `hour`, `day_of_week`, `month`, `year` | Datetime decomposition |
| `is_weekend` | 1 if Saturday/Sunday |
| `is_rush_hour` | 1 if 7–9 AM or 4–7 PM |
| `is_night` | 1 if 10 PM–5 AM (NYC night surcharge hours) |
| `is_jfk`, `is_lga`, `is_ewr` | 1 if pickup or dropoff within 2 km of airport |

**Output**: `X_train.npy`, `X_test.npy`, `y_train.npy`

---

### Phase 2 — Model Training & Comparison (`model_comparison.py`)

Trains 8 regression algorithms and compares them with 5-Fold Cross-Validation (RMSE).

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Polynomial Regression (d=2) | Captures non-linear interactions |
| Ridge (α=1) | L2 regularization |
| LASSO (α=0.01) | L1 regularization, feature selection |
| KNN (k=10, distance-weighted) | Instance-based |
| SVM Linear | LinearSVR — fast linear kernel |
| Decision Tree | max_depth=10 |
| Random Forest | 100 estimators, max_depth=12 |
| XGBoost | 300 estimators, lr=0.05, max_depth=6 |

**Outputs**:
- `model_comparison.png` — CV RMSE bar chart, train vs CV RMSE, fold distribution boxplot, training time chart
- `feature_importance.png` — Top feature importances from Random Forest and XGBoost
- `pred_vs_actual.png` — Predicted vs Actual scatter for 4 models
- `best_predictions.npy` — Predictions from the best CV model
- `best_model.txt` — Name of the best model (used by Phase 4)

---

### Phase 3 — Model Comparison (`train_compare.py`)

Identical to Phase 2 (`model_comparison.py`). Appears to be a duplicate script kept for reference.

---

### Phase 4 — Hyperparameter Tuning (`tuning.py`)

Uses **RandomizedSearchCV** (10 iterations per model, 5-Fold CV) to tune all non-linear models.

**Search spaces**:
- Ridge: `alpha` log-uniform [0.001, 1000]
- LASSO: `alpha` log-uniform [0.0001, 10]
- KNN: `n_neighbors` [3, 30], `weights`, `p` (Manhattan/Euclidean)
- SVM Linear: `C` log-uniform [0.01, 100], `epsilon` uniform [0.01, 2]
- Decision Tree: `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_features`
- Random Forest: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
- XGBoost: `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`

**Outputs**:
- `phase4_tuning.png` — Before vs After RMSE, improvement %, random search convergence curves
- `tuning_results.csv` — Ranking table with before/after RMSE per model
- `best_tuned_predictions.npy` — Predictions from the best tuned model
- `best_tuned_model.txt` — Name of the best tuned model

**Results** (from `tuning_results.csv`):

| Rank | Model | Before RMSE | After RMSE | Improvement |
|---|---|---|---|---|
| 1 | XGBoost | $3.3178 | $3.3219 | −0.12% |
| 2 | Random Forest | $3.4452 | $3.5109 | −1.91% |
| 3 | KNN | $3.6356 | $3.5575 | +2.15% |
| 4 | Decision Tree | $3.6608 | $3.5976 | +1.73% |
| 5 | Ridge | $4.3511 | $4.3511 | ≈0% |
| 6 | LASSO | $4.3701 | $4.3511 | +0.43% |
| 7 | SVM (Linear) | $4.5869 | $4.5322 | +1.19% |

> XGBoost achieves the best CV RMSE overall (~$3.32).

---

### Phase 5 — Final Prediction & Submission (`submission.py`)

- Reloads the best tuned model with hardcoded optimal parameters from Phase 4.
- Trains on the full training set.
- Predicts on the test set (floor predictions at $2.50).
- Saves `submission.csv` with columns `key`, `fare_amount`.
- Generates `phase5_final.png` with distribution, actual vs predicted, residuals, boxplot by fare range, CDF comparison, and a summary scorecard.

---

## Key Findings

- **Best feature**: `distance_km` (Haversine distance) — highest correlation with fare
- **Best model**: XGBoost (CV RMSE ≈ $3.32)
- **Airport trips** are a meaningful signal due to NYC flat-rate and surcharge rules
- **Night/rush-hour flags** capture fare surcharge patterns
- Polynomial Regression was excluded from tuning (slow) but tested in Phase 2

---

## File Structure

```
final_project/
├── raw-data/
│   ├── train_taxi.csv
│   └── test_taxi.csv
├── nyc-taxi-fare-predic/
│   ├── eda.py                  # Phase 1
│   ├── model_comparison.py     # Phase 2
│   ├── train_compare.py        # Phase 3 (duplicate of Phase 2)
│   ├── tuning.py               # Phase 4
│   ├── submission.py           # Phase 5
│   ├── submission.csv          # Final output
│   ├── tuning_results.csv      # Tuning comparison
│   ├── best_model.txt          # Best Phase 2 model name
│   ├── best_tuned_model.txt    # Best Phase 4 model name
│   ├── X_train.npy / X_test.npy / y_train.npy
│   ├── best_predictions.npy / best_tuned_predictions.npy
│   ├── eda_plots.png / correlation.png / taxi_map.html
│   ├── model_comparison.png / feature_importance.png / pred_vs_actual.png
│   ├── phase4_tuning.png / phase5_final.png
└── subject/
    └── AY2025_TermProject.pdf
```

---

## How to Run

```bash
cd nyc-taxi-fare-predic/

# Phase 1 — EDA + feature engineering
python eda.py

# Phase 2 — Model comparison
python model_comparison.py

# Phase 4 — Hyperparameter tuning
python tuning.py

# Phase 5 — Final submission
python submission.py
```

> Phase 3 (`train_compare.py`) is optional — identical to Phase 2.
