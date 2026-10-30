from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AIR_QUALITY_PATH = DATA_DIR / "air-quality.csv"
WEATHER_PATH = DATA_DIR / "weather.csv"

TIME_COL = "time"
PM25_COL = "pm2_5 (μg/m³)"

# Physically valid ranges (proposal Section 3.3).
VALID_RANGES = {
    "pm10 (μg/m³)": (0, 2000),
    PM25_COL: (0, 1000),
    "carbon_monoxide (μg/m³)": (0, 100000),
    "nitrogen_dioxide (μg/m³)": (0, 5000),
    "sulphur_dioxide (μg/m³)": (0, 5000),
    "ozone (μg/m³)": (0, 5000),
    "european_aqi (EAQI)": (0, 400),
    "temperature_2m (°C)": (-60, 60),
    "relative_humidity_2m (%)": (0, 100),
    "rain (mm)": (0, 500),
    "wind_speed_100m (km/h)": (0, 400),
    "wind_direction_100m (°)": (0, 360),
    "precipitation (mm)": (0, 500),
}

# Gaps of up to this many consecutive hours are interpolated; longer outages are dropped.
MAX_GAP_HOURS = 6


def load_air_quality():
    """Load air quality data, skipping metadata."""
    return pd.read_csv(AIR_QUALITY_PATH, skiprows=3)


def load_weather():
    """Load weather data, skipping metadata."""
    return pd.read_csv(WEATHER_PATH, skiprows=3)


def load_combined_data():
    """Merge both streams on time; duplicates removed (proposal 3.3).

    Note: validate="one_to_one" was replaced by explicit deduplication so that
    duplicate timestamps are removed instead of raising a MergeError.
    """
    air_quality = load_air_quality()
    weather = load_weather()
    air_quality[TIME_COL] = pd.to_datetime(air_quality[TIME_COL])
    weather[TIME_COL] = pd.to_datetime(weather[TIME_COL])
    air_quality = air_quality.drop_duplicates(subset=TIME_COL, keep="first")
    weather = weather.drop_duplicates(subset=TIME_COL, keep="first")
    df = pd.merge(air_quality, weather, on=TIME_COL, how="inner")
    return df.sort_values(TIME_COL).reset_index(drop=True)


def remove_implausible(df):
    """Drop physically implausible records; returns (clean_df, n_dropped)."""
    bad = pd.Series(False, index=df.index)
    for col, (low, high) in VALID_RANGES.items():
        if col in df.columns:
            bad |= df[col].notna() & ((df[col] < low) | (df[col] > high))
    return df.loc[~bad].copy(), int(bad.sum())


def fill_short_gaps(df):
    """Interpolate short gaps on a complete hourly grid; drop long outages."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df = df.set_index(TIME_COL)
    hourly_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(hourly_index)
    df.index.name = TIME_COL
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="linear", limit=MAX_GAP_HOURS
    )
    df = df.dropna(subset=numeric_cols)
    return df.reset_index()


def load_clean_data():
    """Cleaned hourly dataset for feature engineering (workflow step 2)."""
    df = load_combined_data()
    df, _ = remove_implausible(df)
    df = fill_short_gaps(df)
    return df.reset_index(drop=True)


def inspect_dataset(df, name):
    """Display basic information about a dataset."""
    print(f"\n{'=' * 50}")
    print(f"{name} DATASET")
    print(f"{'=' * 50}")
    print("\nShape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData types and non-null counts:")
    df.info()
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())


if __name__ == "__main__":
    inspect_dataset(load_combined_data(), "COMBINED (MERGED)")
    inspect_dataset(load_clean_data(), "CLEANED")