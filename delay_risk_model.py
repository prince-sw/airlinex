import os
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)


# =====================================================
# CONFIG
# =====================================================

INPUT_FILE = "flight_pair_training_table.csv"

MODEL_DIR = "delay_risk_models"
PREDICTION_OUTPUT_FILE = "delay_risk_predictions.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COLS = [
    "target_delay_gt_15",
    "target_delay_gt_45",
    "target_delay_gt_90",
]

FEATURE_COLS = [
    # Categorical/context
    # Note: route_i and route_j are deliberately excluded because they are high-cardinality
    # string columns that heavily overlap with (and are redundant to) origin_i, dest_i, origin_j, and dest_j.
    "carrier",
    "origin_i",
    "dest_i",
    "origin_j",
    "dest_j",

    # Date/time context
    "month",
    "day_of_month",
    "day_of_week",
    "next_dep_hour",
    "next_arr_hour",

    # Upstream delay signals
    "upstream_arr_delay_min",
    "upstream_dep_delay_min",
    "carrier_delay_i",
    "weather_delay_i",
    "nas_delay_i",
    "security_delay_i",
    "late_aircraft_delay_i",

    # Turnaround features
    # turnaround_slack_min, is_tight_turnaround, is_negative_available_turnaround,
    # is_actually_negative_turnaround are all linear/threshold derivations of the two
    # continuous columns below; CatBoost learns those splits directly from the parent.
    # distance_group_j is a binned version of distance_j (already present).
    # security_delay_i has near-zero importance (~0.007) across all models.
    # estimated_min_turnaround_min is a zero-variance constant.
    "scheduled_turnaround_min",
    "actual_available_turnaround_min",
    "turnaround_pressure_min",

    # Flight size/distance context
    "distance_i",
    "distance_j",
    "crs_elapsed_time_i",
    "crs_elapsed_time_j",
]

CATEGORICAL_COLS = [
    "carrier",
    "origin_i",
    "dest_i",
    "origin_j",
    "dest_j",
]


# =====================================================
# LOAD DATA
# =====================================================

print("Loading engineered dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# =====================================================
# SAFETY CHECKS
# =====================================================

missing_features = [c for c in FEATURE_COLS if c not in df.columns]
missing_targets = [c for c in TARGET_COLS if c not in df.columns]

if missing_features:
    raise ValueError(f"Missing feature columns: {missing_features}")

if missing_targets:
    raise ValueError(f"Missing target columns: {missing_targets}")

if "FL_DATE_i" not in df.columns:
    raise ValueError("Missing FL_DATE_i. Needed for time-based train/test split.")


# =====================================================
# CLEAN FEATURES
# =====================================================

print("\nCleaning features...")

df["FL_DATE_i"] = pd.to_datetime(df["FL_DATE_i"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
df = df[df["FL_DATE_i"].notna()].copy()

# Fill categorical missing values
for col in CATEGORICAL_COLS:
    df[col] = df[col].astype(str).fillna("UNKNOWN")

# Convert numeric columns safely
numeric_cols = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Make sure target columns are integer 0/1
for col in TARGET_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


# =====================================================
# TIME-BASED TRAIN/TEST SPLIT
# =====================================================

print("\nCreating time-based split...")

df = df.sort_values("FL_DATE_i").copy()

unique_dates = sorted(df["FL_DATE_i"].dt.date.unique())
split_index = int(len(unique_dates) * 0.8)

train_dates = unique_dates[:split_index]
test_dates = unique_dates[split_index:]

train_df = df[df["FL_DATE_i"].dt.date.isin(train_dates)].copy()
test_df = df[df["FL_DATE_i"].dt.date.isin(test_dates)].copy()

# Median imputation computed on train only to prevent data leakage
for col in numeric_cols:
    median_value = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_value)
    test_df[col] = test_df[col].fillna(median_value)

print(f"Train dates: {train_dates[0]} to {train_dates[-1]}")
print(f"Test dates:  {test_dates[0]} to {test_dates[-1]}")
print(f"Train rows: {len(train_df):,}")
print(f"Test rows:  {len(test_df):,}")

X_train = train_df[FEATURE_COLS]
X_test = test_df[FEATURE_COLS]

cat_feature_indices = [FEATURE_COLS.index(c) for c in CATEGORICAL_COLS]


# =====================================================
# BASELINE RULE
# =====================================================

print("\nBaseline rule performance")
print("=========================")

# Simple rule:
# If actual available turnaround is less than minimum turnaround (35 min), predict delay risk.
baseline_pred = (
    test_df["actual_available_turnaround_min"] < 35
).astype(int)

for target in TARGET_COLS:
    y_test = test_df[target]

    print(f"\nTarget: {target}")
    print(f"Positive rate: {y_test.mean():.4f}")

    print("Confusion matrix:")
    print(confusion_matrix(y_test, baseline_pred))

    print(classification_report(y_test, baseline_pred, zero_division=0))


# =====================================================
# TRAIN CATBOOST MODELS
# =====================================================

all_predictions = test_df[
    [
        "FL_DATE_i",
        "TAIL_NUM",
        "carrier",
        "route_i",
        "route_j",
        "upstream_arr_delay_min",
        "scheduled_turnaround_min",
        "actual_available_turnaround_min",
        "turnaround_pressure_min",
        # NOTE: downstream_dep_delay_min is the ground truth actual delay for the next flight.
        # This is included in predictions only for validation and debugging purposes.
        # Do NOT use it as a predictor/feature in subsequent downstream steps.
        "downstream_dep_delay_min",
    ]
].copy()

metrics_summary = []

print("\nTraining CatBoost models")
print("========================")

for target in TARGET_COLS:
    print(f"\nTraining model for: {target}")

    y_train = train_df[target]
    y_test = test_df[target]

    print(f"Train positive rate: {y_train.mean():.4f}")
    print(f"Test positive rate:  {y_test.mean():.4f}")

    # Handles imbalance automatically using class weights
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
        auto_class_weights="Balanced",
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )

    pred_proba = model.predict_proba(X_test)[:, 1]
    pred_label_50 = (pred_proba >= 0.50).astype(int)

    roc_auc = roc_auc_score(y_test, pred_proba)
    pr_auc = average_precision_score(y_test, pred_proba)
    brier = brier_score_loss(y_test, pred_proba)

    print(f"\nMetrics for {target}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"Brier Score: {brier:.4f}")

    print("\nConfusion matrix at threshold 0.50:")
    print(confusion_matrix(y_test, pred_label_50))

    print("\nClassification report at threshold 0.50:")
    print(classification_report(y_test, pred_label_50, zero_division=0))

    # Save model
    model_path = os.path.join(MODEL_DIR, f"catboost_{target}.cbm")
    model.save_model(model_path)
    print(f"Saved model to: {model_path}")

    # Save prediction probability
    prob_col = target.replace("target_delay_gt_", "prob_delay_gt_")
    all_predictions[prob_col] = pred_proba
    all_predictions[target] = y_test.values

    # Save metrics
    metrics_summary.append(
        {
            "target": target,
            "train_positive_rate": y_train.mean(),
            "test_positive_rate": y_test.mean(),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "model_path": model_path,
        }
    )

    # Feature importance
    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)

    importance_path = os.path.join(MODEL_DIR, f"feature_importance_{target}.csv")
    importance_df.to_csv(importance_path, index=False)

    print("\nTop 10 features:")
    print(importance_df.head(10))


# =====================================================
# FIX MONOTONIC PROBABILITIES
# =====================================================

print("\nApplying monotonic correction to risk profile...")

prob_cols = [
    "prob_delay_gt_15",
    "prob_delay_gt_45",
    "prob_delay_gt_90",
]

# Enforce:
# P(>15) >= P(>30) >= P(>45) >= P(>60) >= P(>90)
for i in range(1, len(prob_cols)):
    all_predictions[prob_cols[i]] = np.minimum(
        all_predictions[prob_cols[i]],
        all_predictions[prob_cols[i - 1]],
    )


# =====================================================
# SAVE OUTPUTS
# =====================================================

all_predictions.to_csv(PREDICTION_OUTPUT_FILE, index=False)

metrics_df = pd.DataFrame(metrics_summary)
metrics_path = os.path.join(MODEL_DIR, "metrics_summary.csv")
metrics_df.to_csv(metrics_path, index=False)

print("\nDone.")
print(f"Saved predictions to: {PREDICTION_OUTPUT_FILE}")
print(f"Saved metrics to: {metrics_path}")

print("\nMetrics summary:")
print(metrics_df)