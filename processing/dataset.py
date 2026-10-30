from pathlib import Path
import sys

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from processing.dataloader import load_combined_data


EAQI_COL = "european_aqi (EAQI)"
TARGET_COL = "pm2_5 (μg/m³)"

# Numerical columns used for scaling.
# Time and EAQI are excluded because EAQI is encoded separately.
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


# Feature Scaling: Standardization
def feature_scaling(df):
    before = df.copy()
    after = df.copy()

    columns = get_numeric_columns(df)

    scaler = StandardScaler()
    after[columns] = scaler.fit_transform(after[columns])

    show_comparison(
        before,
        after,
        columns,
        "FEATURE SCALING (STANDARD SCALER)"
    )

    return after


# Ordinal Encoding: EAQI Categories
def ordinal_encoding(df):
    before = df.copy()
    after = df.copy()

    bins = [0, 20, 40, 60, 80, 100, float("inf")]
    labels = [
        "Good",
        "Fair",
        "Moderate",
        "Poor",
        "Very Poor",
        "Extremely Poor"
    ]

    # Convert EAQI numerical values into ordered categories.
    after["EAQI Category"] = pd.cut(
        after[EAQI_COL],
        bins=bins,
        labels=labels,
        right=False
    )

    # Convert ordered categories into integers from 0 to 5.
    category_mapping = {
        "Good": 0,
        "Fair": 1,
        "Moderate": 2,
        "Poor": 3,
        "Very Poor": 4,
        "Extremely Poor": 5
    }

    after["EAQI Ordinal"] = (
        after["EAQI Category"]
        .map(category_mapping)
        .astype("Int64")
    )

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

    print("\nBefore: EAQI descriptive statistics")
    print(before[[EAQI_COL]].describe().round(3))

    print("\nAfter: ordinal descriptive statistics")
    print(after[["EAQI Ordinal"]].describe().round(3))

    return after


def main():
    df = load_combined_data()

    print("Original dataset shape:", df.shape)
    print("\nOriginal dataset preview:")
    print(df.head())

    # Run each method independently on the original dataset.
    scaled_df = feature_scaling(df)
    encoded_df = ordinal_encoding(df)

    print("\n" + "=" * 60)
    print("FINAL DATASET COMPARISON")
    print("=" * 60)

    print("Original dataset:", df.shape)
    print("Feature-scaled dataset:", scaled_df.shape)
    print("Ordinal-encoded dataset:", encoded_df.shape)

    print("\nOriginal dataset remains unchanged:")
    print(df.head())


if __name__ == "__main__":
    main()