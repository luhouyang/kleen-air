from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

AIR_QUALITY_PATH = DATA_DIR / "air-quality.csv"
WEATHER_PATH = DATA_DIR / "weather.csv"


def load_air_quality():
    """Load air quality data, skipping metadata."""
    return pd.read_csv(AIR_QUALITY_PATH, skiprows=3)


def load_weather():
    """Load weather data, skipping metadata."""
    return pd.read_csv(WEATHER_PATH, skiprows=3)


def load_combined_data():
    air_quality = load_air_quality()
    weather = load_weather()

    # Convert time columns to datetime
    air_quality["time"] = pd.to_datetime(air_quality["time"])
    weather["time"] = pd.to_datetime(weather["time"])

    # Merge both datasets
    df = pd.merge(
        air_quality,
        weather,
        on="time",
        how="inner",
        validate="one_to_one"
    )

    return df.sort_values("time").reset_index(drop=True)


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
    air_quality = load_air_quality()
    weather = load_weather()
    PM25_prediction_df = load_combined_data()
    inspect_dataset(PM25_prediction_df, "PM2.5 PREDICTION DATASET")