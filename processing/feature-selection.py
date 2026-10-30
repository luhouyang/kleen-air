from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "featured_dataset.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "feature_importance.csv"

TARGET_COL = "pm2_5_next_hour"
TIME_COL = "time"


def correlation_analysis(df):
    correlations = (
        df.select_dtypes(include="number")
        .corr()[TARGET_COL]
        .drop(TARGET_COL)
        .sort_values(key=abs, ascending=False)
    )

    print("\n===== CORRELATION WITH NEXT-HOUR PM2.5 =====")
    print(correlations)

    return correlations


def feature_importance_scoring(df):
    data = df.copy()

    X = data.drop(columns=[TARGET_COL, TIME_COL], errors="ignore")
    y = data[TARGET_COL]

    split_index = int(len(data) * 0.8)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns,
        name="Importance"
    ).sort_values(ascending=False)

    print("\n===== RANDOM FOREST FEATURE IMPORTANCE =====")
    print(importance)

    # importance.to_csv(OUTPUT_PATH, header=True)
    # print(f"\nFeature importance saved to: {OUTPUT_PATH}")

    return importance


def plot_correlation(correlations):
    plt.figure(figsize=(10, 6))
    correlations.sort_values().plot(kind="barh")
    plt.title("Correlation with Next-Hour PM2.5")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    numerical_df = df.select_dtypes(include="number")
    correlation_matrix = numerical_df.corr()

    # Mask the upper triangle
    mask = np.triu(
        np.ones_like(correlation_matrix, dtype=bool)
    )

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(16, 12))

    sns.heatmap(
        correlation_matrix,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.7},
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap of Air Quality and Weather Features",
        fontsize=13,
        pad=25
    )

    ax.tick_params(axis="x", labelsize=8, rotation=45, pad=8)
    ax.tick_params(axis="y", labelsize=8, rotation=0, pad=8)

    # Adjust spacing to prevent labels from being cut off
    fig.subplots_adjust(
        top=0.90,
        bottom=0.25,
        left=0.25,
        right=0.92
    )

    plt.show()
    plt.close(fig)


def plot_feature_importance(importance):
    plt.figure(figsize=(10, 6))
    importance.sort_values().plot(kind="barh")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])

    print("Dataset shape:", df.shape)

    correlations = correlation_analysis(df)
    importance = feature_importance_scoring(df)

    plot_correlation(correlations)
    plot_correlation_heatmap(df)
    plot_feature_importance(importance)


if __name__ == "__main__":
    main()