from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from processing.dataloader import load_combined_data


PM25_COL = "pm2_5 (μg/m³)"
TIME_COL = "time"


def create_features(df):
    df_features = df.copy()

    df_features[TIME_COL] = pd.to_datetime(df_features[TIME_COL])
    df_features = df_features.sort_values(TIME_COL).reset_index(drop=True)

    # Create historical PM2.5 features
    df_features["pm2_5_lag_1"] = df_features[PM25_COL].shift(1)
    df_features["pm2_5_lag_3"] = df_features[PM25_COL].shift(3)
    df_features["pm2_5_lag_24"] = df_features[PM25_COL].shift(24)

    # Calculate the average PM2.5 from the previous 24 hours
    df_features["pm2_5_rolling_mean_24"] = (
        df_features[PM25_COL]
        .shift(1)
        .rolling(window=24)
        .mean()
    )

    # Create time-based features
    df_features["hour"] = df_features[TIME_COL].dt.hour
    df_features["month"] = df_features[TIME_COL].dt.month

    # Set the target as the next hour's PM2.5
    df_features["pm2_5_next_hour"] = df_features[PM25_COL].shift(-1)

    # Remove rows with missing lag or target values
    df_features = df_features.dropna().reset_index(drop=True)

    return df_features


def show_comparison(original_df, featured_df):
    new_features = [
        "pm2_5_lag_1",
        "pm2_5_lag_3",
        "pm2_5_lag_24",
        "pm2_5_rolling_mean_24",
        "hour",
        "month",
        "pm2_5_next_hour",
    ]

    print("\n===== FEATURE ENGINEERING COMPARISON =====")

    print("\nOriginal dataset shape:")
    print(original_df.shape)

    print("\nDataset after feature engineering:")
    print(featured_df.shape)

    print("\nNew features:")
    print(featured_df[new_features].head())

    print("\nOriginal PM2.5 values:")
    print(original_df[PM25_COL].head())

    print("\nNext-hour PM2.5 target:")
    print(featured_df[["time", "pm2_5_next_hour"]].head())


def main():
    df = load_combined_data()

    df_features = create_features(df)

    show_comparison(df, df_features)

    output_path = Path("data/featured_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)

    print(f"\nFeatured dataset saved to: {output_path}")


if __name__ == "__main__":
    main()