from pathlib import Path
import sys

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from processing.dataloader import load_combined_data

EAQI_COL = "european_aqi (EAQI)"
TARGET_COL = "pm2_5 (μg/m³)"            # current-hour PM2.5 (kept for compatibility)
REGRESSION_TARGET = "pm2_5_next_hour"   # next-hour PM2.5 (regression target)
CATEGORY_TARGET = "eaqi_category_next_hour"  # next-hour EAQI category (classification target)
TIME_COL = "time"

# Official EAQI index bands (your original encoding).
EAQI_BINS = [0, 20, 40, 60, 80, 100, float("inf")]
EAQI_CATEGORIES = ["Good", "Fair", "Moderate", "Poor", "Very Poor", "Extremely Poor"]
HIGH_RISK_CATEGORIES = ["Poor", "Very Poor", "Extremely Poor"]
# Official EAQI PM2.5 sub-index bands (µg/m³) for the regression->category path.
PM25_EAQI_BINS = [0, 10, 20, 25, 50, 75, float("inf")]

FEATURED_PATH = PROJECT_ROOT / "data" / "featured_dataset.csv"

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15


def get_numeric_columns(df):
    return [
        col for col in df.select_dtypes(include="number").columns
        if col != EAQI_COL
    ]


def show_comparison(before, after, columns, title):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    print("\nBefore preprocessing:")
    print(before[columns].head())
    print("\nAfter preprocessing:")
    print(after[columns].head())
    print("\nBefore: descriptive statistics")
    print(before[columns].describe().round(3))
    print("\nAfter: descriptive statistics")
    print(after[columns].describe().round(3))


def feature_scaling(df):
    """DEMO ONLY: fits the scaler on all data. Models must scale inside a
    Pipeline fitted on the training split to avoid leakage."""
    before = df.copy()
    after = df.copy()
    columns = get_numeric_columns(df)
    scaler = StandardScaler()
    after[columns] = scaler.fit_transform(after[columns])
    show_comparison(before, after, columns, "FEATURE SCALING (STANDARD SCALER)")
    return after


def ordinal_encoding(df):
    before = df.copy()
    after = df.copy()
    bins = EAQI_BINS
    labels = EAQI_CATEGORIES
    after["EAQI Category"] = pd.cut(after[EAQI_COL], bins=bins, labels=labels, right=False)
    category_mapping = {name: code for code, name in enumerate(EAQI_CATEGORIES)}
    after["EAQI Ordinal"] = after["EAQI Category"].map(category_mapping).astype("Int64")
    print(f"\n{'=' * 60}")
    print("ORDINAL ENCODING (EAQI)")
    print("=" * 60)
    print("\nBefore preprocessing:")
    print(before[[EAQI_COL]].head())
    print("\nAfter preprocessing:")
    print(after[[EAQI_COL, "EAQI Category", "EAQI Ordinal"]].head())
    print("\nEAQI category counts:")
    print(after["EAQI Category"].value_counts(sort=False))
    print("\nOrdinal value counts:")
    print(after["EAQI Ordinal"].value_counts().sort_index())
    return after


# ---------------------------------------------------------------------------
# Helpers used by the model scripts and eval.py
# ---------------------------------------------------------------------------

def eaqi_category(values):
    """Official EAQI index bands (0-20 Good ... 100+ Extremely Poor)."""
    return pd.cut(values, bins=EAQI_BINS, labels=EAQI_CATEGORIES, right=False)


def pm25_eaqi_category(values):
    """Official EAQI PM2.5 sub-index bands (µg/m³)."""
    return pd.cut(values, bins=PM25_EAQI_BINS, labels=EAQI_CATEGORIES, right=False)


def load_featured_data(add_targets=True):
    df = pd.read_csv(FEATURED_PATH, parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    if add_targets:
        df[CATEGORY_TARGET] = eaqi_category(df[EAQI_COL].shift(-1))
        df = df.dropna(subset=[CATEGORY_TARGET]).reset_index(drop=True)
    return df


def feature_columns(df):
    excluded = {TIME_COL, REGRESSION_TARGET, "EAQI Ordinal", "EAQI Category"}
    return [c for c in df.select_dtypes(include="number").columns
            if c not in excluded]


def chronological_split(df, train_fraction=TRAIN_FRACTION,
                        validation_fraction=VALIDATION_FRACTION):
    """Strict chronological 70/15/15 split (identical ratios to train.py)."""
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    return (df.iloc[:train_end].copy(),
            df.iloc[train_end:validation_end].copy(),
            df.iloc[validation_end:].copy())


def split_features_target(df, target_col=REGRESSION_TARGET):
    return df[feature_columns(df)], df[target_col]


def class_weights(y):
    """Inverse-frequency sample weights for imbalanced EAQI classes."""
    y = pd.Series(y).astype(str)
    counts = y.value_counts()
    weights = len(y) / (len(counts) * counts)
    return y.map(weights).to_numpy()


def main():
    df = load_combined_data()
    print("Original dataset shape:", df.shape)
    scaled_df = feature_scaling(df)
    encoded_df = ordinal_encoding(df)
    print("\nOriginal dataset remains unchanged:")
    print(df.head())


if __name__ == "__main__":
    main()