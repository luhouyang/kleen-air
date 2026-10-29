import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys
from pathlib import Path

# Add the project root to Python's module search path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from processing.dataloader import load_combined_data


sns.set_theme(style="whitegrid")

# Column names
TIME_COL = "time"
PM25_COL = "pm2_5 (μg/m³)"
EAQI_COL = "european_aqi (EAQI)"

POLLUTANT_COLS = [
    "pm10 (μg/m³)",
    PM25_COL,
    "carbon_monoxide (μg/m³)",
    "nitrogen_dioxide (μg/m³)",
    "sulphur_dioxide (μg/m³)",
    "ozone (μg/m³)"
]

WEATHER_COLS = [
    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "rain (mm)",
    "wind_speed_100m (km/h)",
    "wind_direction_100m (°)",
    "precipitation (mm)"
]


def summary_statistics(df):
    """Section 2.2.1: Summary statistics and data checks."""

    # Convert timestamps before performing time calculations.
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="raise")

    print("\n========== DATASET OVERVIEW ==========")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Start date:", df[TIME_COL].min())
    print("End date:", df[TIME_COL].max())
    print("Column names:", df.columns.tolist())

    print("\n========== MISSING VALUES ==========")
    missing = pd.DataFrame({
        "Missing Count": df.isna().sum(),
        "Missing Percentage (%)": (
            df.isna().mean() * 100
        ).round(2)
    })
    print(missing)

    print("\n========== DUPLICATE CHECKS ==========")
    print("Duplicate rows:", df.duplicated().sum())
    print("Duplicate timestamps:", df[TIME_COL].duplicated().sum())

    print("\n========== TIMESTAMP CHECKS ==========")
    time_diff = df[TIME_COL].sort_values().diff()

    print("Time column type:", df[TIME_COL].dtype)
    print("First timestamp:", df[TIME_COL].min())
    print("Last timestamp:", df[TIME_COL].max())
    print("Most common time interval:")
    print(time_diff.value_counts().head())

    print(
        "Hourly intervals:",
        (time_diff == pd.Timedelta(hours=1)).sum()
    )
    print(
        "Unexpected intervals:",
        (
            time_diff.notna()
            & (time_diff != pd.Timedelta(hours=1))
        ).sum()
    )

    numeric_df = df.select_dtypes(include="number")

    print("\n========== DESCRIPTIVE STATISTICS ==========")
    print(numeric_df.describe().T.round(2))

    # EAQI-specific summary
    if EAQI_COL in df.columns:
        print("\n========== EUROPEAN AQI SUMMARY ==========")
        print("Minimum EAQI:", df[EAQI_COL].min())
        print("Maximum EAQI:", df[EAQI_COL].max())
        print("Mean EAQI:", round(df[EAQI_COL].mean(), 2))
        print("Median EAQI:", df[EAQI_COL].median())

def inspect_outliers_and_validity(df):
    print("\n========== OUTLIER INSPECTION (IQR) ==========")

    numerical_cols = df.select_dtypes(include="number").columns
    outlier_results = []

    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df[
            (df[col] < lower_bound) |
            (df[col] > upper_bound)
        ]

        outlier_results.append({
            "Variable": col,
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2),
            "Potential Outliers": len(outliers),
            "Percentage (%)": round(
                len(outliers) / len(df) * 100, 2
            )
        })

    print(pd.DataFrame(outlier_results).to_string(index=False))

    print("\n========== PHYSICAL VALIDITY CHECKS ==========")

    checks = {
        "Negative PM10": df["pm10 (μg/m³)"] < 0,
        "Negative PM2.5": df["pm2_5 (μg/m³)"] < 0,
        "Negative carbon monoxide":
            df["carbon_monoxide (μg/m³)"] < 0,
        "Negative nitrogen dioxide":
            df["nitrogen_dioxide (μg/m³)"] < 0,
        "Negative sulphur dioxide":
            df["sulphur_dioxide (μg/m³)"] < 0,
        "Negative ozone": df["ozone (μg/m³)"] < 0,
        "Humidity outside 0-100%":
            ~df["relative_humidity_2m (%)"].between(0, 100),
        "Negative temperature-independent rainfall":
            df["rain (mm)"] < 0,
        "Negative precipitation":
            df["precipitation (mm)"] < 0,
        "Negative wind speed":
            df["wind_speed_100m (km/h)"] < 0,
        "Wind direction outside 0-360 degrees":
            ~df["wind_direction_100m (°)"].between(0, 360),
    }

    for description, invalid_mask in checks.items():
        print(f"{description}: {invalid_mask.sum()}")

    print("\nNote: IQR flags unusual values, not necessarily errors.")
    print("No observations have been removed.")


def distribution_analysis(df):
    """Section 2.2.2: Variable distributions and class balance."""

    # PM2.5 histogram
    plt.figure(figsize=(9, 5))
    sns.histplot(df[PM25_COL].dropna(), bins=50, kde=True)
    plt.title("Distribution of PM2.5 Concentrations")
    plt.xlabel("PM2.5 (μg/m³)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Boxplots for pollutant concentrations
    available_pollutants = [
        col for col in POLLUTANT_COLS if col in df.columns
    ]

    df[available_pollutants].plot(
        kind="box",
        subplots=True,
        layout=(2, 3),
        figsize=(14, 8),
        sharey=False
    )
    plt.suptitle("Distributions of Air Pollutant Concentrations")
    plt.tight_layout()
    plt.show()

    # EAQI distribution
    if EAQI_COL in df.columns:
        plt.figure(figsize=(9, 5))
        sns.histplot(
            df[EAQI_COL].dropna(),
            bins=40,
            kde=True
        )
        plt.title("Distribution of European Air Quality Index")
        plt.xlabel("European AQI (EAQI)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        # Categorize European Air Quality Index (EAQI)
    eaqi_bins = [
        0, 20, 40, 60, 80, 100, float("inf")
    ]

    eaqi_labels = [
        "Good",
        "Fair",
        "Moderate",
        "Poor",
        "Very Poor",
        "Extremely Poor"
    ]

    eaqi_categories = pd.cut(
        df[EAQI_COL],
        bins=eaqi_bins,
        labels=eaqi_labels,
        right=False
    )

    # Display category counts
    eaqi_counts = eaqi_categories.value_counts(sort=False)

    print("\n========== EAQI CATEGORY COUNTS ==========")
    print(eaqi_counts)

    # Display category percentages
    eaqi_percentages = (
        eaqi_categories.value_counts(
            normalize=True,
            sort=False
        ) * 100
    ).round(2)

    print("\n========== EAQI CATEGORY PERCENTAGES (%) ==========")
    print(eaqi_percentages)

    # Plot category distribution
    plt.figure(figsize=(10, 5))
    ax = eaqi_counts.plot(kind="bar")

    ax.set_title("European AQI Category Distribution")
    ax.set_xlabel("EAQI Category")
    ax.set_ylabel("Number of Observations")
    ax.tick_params(axis="x", rotation=30)

    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)

    plt.tight_layout()
    plt.show()

    # Histograms for all numerical variables
    numeric_df = df.select_dtypes(include="number")

    numeric_df.hist(
        bins=40,
        figsize=(16, 12)
    )
    plt.suptitle("Distributions of Numerical Variables")
    plt.tight_layout()
    plt.show()

    # Retain provisional PM2.5 categories for comparison.
    # Verify the chosen methodology and thresholds before reporting.
    pm25_bins = [
        float("-inf"), 5, 15, 50, 90, 140, float("inf")
    ]

    pm25_labels = [
        "Good", "Fair", "Moderate",
        "Poor", "Very Poor", "Extremely Poor"
    ]

    pm25_categories = pd.cut(
        df[PM25_COL],
        bins=pm25_bins,
        labels=pm25_labels,
        right=False
    )

    pm25_counts = pm25_categories.value_counts(sort=False)

    print("\n========== PROVISIONAL PM2.5 CATEGORY COUNTS ==========")
    print(pm25_counts)

    print("\n========== PROVISIONAL PM2.5 CATEGORY PERCENTAGES (%) ==========")
    print(
        (
            pm25_categories.value_counts(
                normalize=True,
                sort=False
            ) * 100
        ).round(2)
    )

    plt.figure(figsize=(9, 5))
    ax = pm25_counts.plot(kind="bar")

    ax.set_title("Provisional PM2.5 Category Distribution")
    ax.set_xlabel("PM2.5 Category")
    ax.set_ylabel("Number of Observations")
    ax.tick_params(axis="x", rotation=30)

    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)

    plt.tight_layout()
    plt.show()


def correlation_analysis(df):
    """Section 2.2.3: Pearson correlation analysis."""

    numeric_df = df.select_dtypes(include="number")
    correlation = numeric_df.corr(method="pearson")

    print("\n========== CORRELATION WITH PM2.5 ==========")
    print(
        correlation[PM25_COL]
        .sort_values(ascending=False)
        .round(3)
    )

    if EAQI_COL in correlation.columns:
        print("\n========== CORRELATION WITH EAQI ==========")
        print(
            correlation[EAQI_COL]
            .sort_values(ascending=False)
            .round(3)
        )

    # Correlation heatmap now includes EAQI.
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7}
    )

    plt.title("Correlation Heatmap: Air Quality and Weather")
    plt.tight_layout()
    plt.show()


def main():
    df = load_combined_data()

    # Check required columns before analysis.
    required_cols = [TIME_COL, PM25_COL, EAQI_COL]
    missing_cols = [
        col for col in required_cols if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            "Check the CSV headers and dataloader."
        )

    summary_statistics(df)
    inspect_outliers_and_validity(df)
    distribution_analysis(df)
    correlation_analysis(df)


if __name__ == "__main__":
    main()