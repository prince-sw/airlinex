import pandas as pd
import numpy as np
from pathlib import Path


# =========================
# CONFIG
# =========================

INPUT_FILE = "datasets/april24.csv"
OUTPUT_FILE = "flight_pair_training_table.csv"

MIN_TURNAROUND_MINUTES = 35
MAX_TURNAROUND_HOURS = 8


# =========================
# HELPER FUNCTIONS
# =========================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def require_columns(df: pd.DataFrame, required_cols: list):
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print("\nAvailable columns in your file:")
        for col in df.columns:
            print(repr(col))

        raise ValueError(f"\nMissing required columns: {missing}")


def bts_time_to_minutes(value):
    """
    BTS time format:
    5    -> 00:05
    945  -> 09:45
    1530 -> 15:30
    """
    if pd.isna(value):
        return np.nan

    try:
        value = int(float(value))
    except Exception:
        return np.nan

    hour = value // 100
    minute = value % 100

    if hour == 24:
        hour = 0

    if hour < 0 or hour > 24 or minute < 0 or minute >= 60:
        return np.nan

    return hour * 60 + minute


def make_datetime_from_bts_time(date_series, time_series):
    dates = pd.to_datetime(date_series, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    minutes = time_series.apply(bts_time_to_minutes)

    return dates + pd.to_timedelta(minutes, unit="m")


def fix_midnight_crossing(df: pd.DataFrame) -> pd.DataFrame:
    """
    If arrival time is earlier than departure time, assume arrival is next day.
    """

    sched_cross = df["sched_arrival_dt"] < df["sched_departure_dt"]
    df.loc[sched_cross, "sched_arrival_dt"] += pd.Timedelta(days=1)

    actual_cross = df["actual_arrival_dt"] < df["actual_departure_dt"]
    df.loc[actual_cross, "actual_arrival_dt"] += pd.Timedelta(days=1)

    return df


# =========================
# LOAD DATA
# =========================

print("Loading data...")

datasets_dir = Path("datasets")
csv_files = sorted(list(datasets_dir.glob("*.csv")))

if not csv_files:
    raise FileNotFoundError("Could not find any CSV files in datasets/ directory")

print(f"Found {len(csv_files)} CSV files in datasets/: {[f.name for f in csv_files]}")

dfs = []
for f in csv_files:
    print(f"Loading {f.name}...")
    df_part = pd.read_csv(f, low_memory=False)
    df_part = normalize_columns(df_part)
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)

print(f"Total raw rows loaded: {len(df):,}")
print(f"Raw columns: {len(df.columns):,}")


# =========================
# REQUIRED COLUMNS
# =========================

required_cols = [
    "YEAR",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "TAIL_NUM",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",
    "DEP_DELAY_NEW",
    "DEP_DEL15",
    "CRS_ARR_TIME",
    "ARR_TIME",
    "ARR_DELAY",
    "ARR_DELAY_NEW",
    "ARR_DEL15",
    "CANCELLED",
    "DIVERTED",
    "CRS_ELAPSED_TIME",
    "ACTUAL_ELAPSED_TIME",
    "AIR_TIME",
    "DISTANCE",
    "DISTANCE_GROUP",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
]

require_columns(df, required_cols)


# =========================
# BASIC CLEANING
# =========================

print("Cleaning rows...")

numeric_cols = [
    "YEAR",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",
    "DEP_DELAY_NEW",
    "DEP_DEL15",
    "CRS_ARR_TIME",
    "ARR_TIME",
    "ARR_DELAY",
    "ARR_DELAY_NEW",
    "ARR_DEL15",
    "CANCELLED",
    "DIVERTED",
    "CRS_ELAPSED_TIME",
    "ACTUAL_ELAPSED_TIME",
    "AIR_TIME",
    "DISTANCE",
    "DISTANCE_GROUP",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing delay breakdown values with 0
for col in ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]:
    df[col] = df[col].fillna(0)


# For MVP: remove cancelled/diverted flights.
# These are filtering columns, not model predictors.
df = df[
    (df["CANCELLED"] == 0) &
    (df["DIVERTED"] == 0) &
    (df["TAIL_NUM"].notna()) &
    (df["FL_DATE"].notna()) &
    (df["CRS_DEP_TIME"].notna()) &
    (df["CRS_ARR_TIME"].notna()) &
    (df["DEP_TIME"].notna()) &
    (df["ARR_TIME"].notna()) &
    (df["DEP_DELAY_NEW"].notna()) &
    (df["ARR_DELAY_NEW"].notna())
].copy()

print(f"Rows after cleaning: {len(df):,}")


# =========================
# CREATE DATETIME COLUMNS
# =========================

print("Creating datetime columns...")

df["sched_departure_dt"] = make_datetime_from_bts_time(df["FL_DATE"], df["CRS_DEP_TIME"])
df["actual_departure_dt"] = make_datetime_from_bts_time(df["FL_DATE"], df["DEP_TIME"])
df["sched_arrival_dt"] = make_datetime_from_bts_time(df["FL_DATE"], df["CRS_ARR_TIME"])
df["actual_arrival_dt"] = make_datetime_from_bts_time(df["FL_DATE"], df["ARR_TIME"])

df = fix_midnight_crossing(df)

df = df[
    df["sched_departure_dt"].notna() &
    df["actual_departure_dt"].notna() &
    df["sched_arrival_dt"].notna() &
    df["actual_arrival_dt"].notna()
].copy()

print(f"Rows after datetime cleaning: {len(df):,}")


# =========================
# BUILD AIRCRAFT ROTATION PAIRS
# =========================

print("Building aircraft rotation pairs...")

df = df.sort_values(["TAIL_NUM", "sched_departure_dt"]).copy()

base_cols = [
    "YEAR",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "TAIL_NUM",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",
    "DEP_DELAY_NEW",
    "DEP_DEL15",
    "CRS_ARR_TIME",
    "ARR_TIME",
    "ARR_DELAY",
    "ARR_DELAY_NEW",
    "ARR_DEL15",
    "CRS_ELAPSED_TIME",
    "ACTUAL_ELAPSED_TIME",
    "AIR_TIME",
    "DISTANCE",
    "DISTANCE_GROUP",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "sched_departure_dt",
    "actual_departure_dt",
    "sched_arrival_dt",
    "actual_arrival_dt",
]

df_small = df[base_cols].copy()

# Current flight = i
df_small = df_small.sort_values(["TAIL_NUM", "sched_departure_dt"])
current = df_small.add_suffix("_i")

# Next same-aircraft flight = j (grouped by tail number to avoid bleed)
next_flight = df_small.groupby("TAIL_NUM", group_keys=False).apply(
    lambda g: g.shift(-1)
).add_suffix("_j")

pairs = pd.concat([current, next_flight], axis=1)

# Aircraft ID
pairs["TAIL_NUM"] = pairs["TAIL_NUM_i"]

# Remove rows where there is no next flight
pairs = pairs[pairs["FL_DATE_j"].notna()].copy()


# =========================
# VALID ROTATION FILTER
# =========================

print("Filtering valid aircraft rotations...")

# Valid aircraft rotation means previous destination equals next origin
pairs["same_airport_connection"] = (pairs["DEST_i"] == pairs["ORIGIN_j"]).astype(int)

pairs = pairs[pairs["same_airport_connection"] == 1].copy()

# Scheduled turnaround: next scheduled departure - previous scheduled arrival
pairs["scheduled_turnaround_min"] = (
    pairs["sched_departure_dt_j"] - pairs["sched_arrival_dt_i"]
).dt.total_seconds() / 60

# How much time was actually available before next scheduled departure
pairs["actual_available_turnaround_min"] = (
    pairs["sched_departure_dt_j"] - pairs["actual_arrival_dt_i"]
).dt.total_seconds() / 60

# Remove weird rotations
pairs = pairs[
    (pairs["scheduled_turnaround_min"] >= 0) &
    (pairs["scheduled_turnaround_min"] <= MAX_TURNAROUND_HOURS * 60)
].copy()

print(f"Valid flight-pair rows: {len(pairs):,}")


# =========================
# FEATURE ENGINEERING
# =========================

print("Engineering features...")

# Upstream flight i actual delay info.
# Safe because flight i has already happened.
pairs["upstream_arr_delay_min"] = pairs["ARR_DELAY_NEW_i"]
pairs["upstream_dep_delay_min"] = pairs["DEP_DELAY_NEW_i"]

# Feature engineer upstream delay causes strictly for i (not j) to avoid leakage
pairs["carrier_delay_i"] = pairs["CARRIER_DELAY_i"]
pairs["weather_delay_i"] = pairs["WEATHER_DELAY_i"]
pairs["nas_delay_i"] = pairs["NAS_DELAY_i"]
pairs["security_delay_i"] = pairs["SECURITY_DELAY_i"]
pairs["late_aircraft_delay_i"] = pairs["LATE_AIRCRAFT_DELAY_i"]

# Downstream scheduled/static information.
pairs["carrier"] = pairs["OP_UNIQUE_CARRIER_i"]
pairs["origin_i"] = pairs["ORIGIN_i"]
pairs["dest_i"] = pairs["DEST_i"]
pairs["origin_j"] = pairs["ORIGIN_j"]
pairs["dest_j"] = pairs["DEST_j"]

pairs["route_i"] = pairs["ORIGIN_i"].astype(str) + "-" + pairs["DEST_i"].astype(str)
pairs["route_j"] = pairs["ORIGIN_j"].astype(str) + "-" + pairs["DEST_j"].astype(str)

pairs["month"] = pairs["MONTH_i"]
pairs["day_of_month"] = pairs["DAY_OF_MONTH_i"]
pairs["day_of_week"] = pairs["DAY_OF_WEEK_i"]

pairs["next_dep_hour"] = pairs["sched_departure_dt_j"].dt.hour
pairs["next_arr_hour"] = pairs["sched_arrival_dt_j"].dt.hour

pairs["distance_i"] = pairs["DISTANCE_i"]
pairs["distance_j"] = pairs["DISTANCE_j"]
pairs["distance_group_j"] = pairs["DISTANCE_GROUP_j"]

pairs["crs_elapsed_time_i"] = pairs["CRS_ELAPSED_TIME_i"]
pairs["crs_elapsed_time_j"] = pairs["CRS_ELAPSED_TIME_j"]

# Turnaround pressure features
pairs["estimated_min_turnaround_min"] = MIN_TURNAROUND_MINUTES

pairs["turnaround_slack_min"] = (
    pairs["scheduled_turnaround_min"] - pairs["estimated_min_turnaround_min"]
)

pairs["turnaround_pressure_min"] = (
    pairs["upstream_arr_delay_min"] - pairs["turnaround_slack_min"]
)

pairs["is_tight_turnaround"] = (
    pairs["scheduled_turnaround_min"] < MIN_TURNAROUND_MINUTES
).astype(int)

pairs["is_negative_available_turnaround"] = (
    pairs["actual_available_turnaround_min"] < MIN_TURNAROUND_MINUTES
).astype(int)

pairs["is_actually_negative_turnaround"] = (
    pairs["actual_available_turnaround_min"] < 0
).astype(int)


# =========================
# TARGET CREATION
# =========================

print("Creating target labels...")

# Downstream departure delay.
# This is target/debug only, not predictor.
pairs["downstream_dep_delay_min"] = pairs["DEP_DELAY_NEW_j"]

thresholds = [15, 30, 45, 60, 90]

for t in thresholds:
    pairs[f"target_delay_gt_{t}"] = (pairs["downstream_dep_delay_min"] > t).astype(int)


# =========================
# FINAL TRAINING TABLE
# =========================

feature_cols = [
    # ID/debug
    "TAIL_NUM",
    "FL_DATE_i",
    "carrier",
    "route_i",
    "route_j",

    # Route/context
    "origin_i",
    "dest_i",
    "origin_j",
    "dest_j",
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
    "scheduled_turnaround_min",
    "actual_available_turnaround_min",
    "estimated_min_turnaround_min",
    "turnaround_slack_min",
    "turnaround_pressure_min",
    "is_tight_turnaround",
    "is_negative_available_turnaround",
    "is_actually_negative_turnaround",

    # Scheduled/static flight features
    "distance_i",
    "distance_j",
    "distance_group_j",
    "crs_elapsed_time_i",
    "crs_elapsed_time_j",
]

target_cols = [f"target_delay_gt_{t}" for t in thresholds]

debug_cols = [
    # Keep these for inspection only.
    # Do not use debug cols as ML predictors.
    "downstream_dep_delay_min",
    "DEP_DELAY_NEW_j",
    "ARR_DELAY_NEW_j",
    "CRS_DEP_TIME_j",
    "DEP_TIME_j",
    "CRS_ARR_TIME_j",
    "ARR_TIME_j",
]

final_cols = feature_cols + target_cols + debug_cols

training_df = pairs[final_cols].copy()


# =========================
# SUMMARY
# =========================

print("\nFinal training table summary")
print("============================")
print(f"Rows: {len(training_df):,}")
print(f"Columns: {len(training_df.columns):,}")

print("\nTarget positive rates:")
for t in thresholds:
    col = f"target_delay_gt_{t}"
    print(f"{col}: {training_df[col].mean():.3f}")

print("\nPreview:")
print(training_df.head())


# =========================
# SAVE
# =========================

training_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved final training table to: {OUTPUT_FILE}")