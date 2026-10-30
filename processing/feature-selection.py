import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from contextlib import redirect_stdout

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "featured_dataset.csv"

# --- Output Directories ---
OUTPUT_DIR = ROOT_DIR / "output" / "feature_selection"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "pm2_5_next_hour"
TIME_COL = "time"

def save_text(filename, content):
    path = OUTPUT_DIR / filename
    path.write_text(content, encoding="utf-8")
    print(f"[Saved] {path}")

def save_fig(fig, filename):
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")

def correlation_analysis(df):
    correlations = (
        df.select_dtypes(include="number")
        .corr()[TARGET_COL]
        .drop(TARGET_COL)
        .sort_values(key=abs, ascending=False)
    )
    f = io.StringIO()
    with redirect_stdout(f):
        print("\n===== CORRELATION WITH NEXT-HOUR PM2.5 =====")
        print(correlations)
    save_text("correlation_analysis.txt", f.getvalue())
    return correlations

def feature_importance_scoring(df):
    data = df.copy()
    X = data.drop(columns=[TARGET_COL, TIME_COL], errors="ignore")
    y = data[TARGET_COL]
    
    split_index = int(len(data) * 0.8)
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    importance = pd.Series(
        model.feature_importances_, index=X.columns, name="Importance"
    ).sort_values(ascending=False)
    
    f = io.StringIO()
    with redirect_stdout(f):
        print("\n===== RANDOM FOREST FEATURE IMPORTANCE =====")
        print(importance)
    save_text("feature_importance.txt", f.getvalue())
    
    # Save importance to CSV as well
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", header=True)
    print(f"[Saved] {OUTPUT_DIR / 'feature_importance.csv'}")
    
    return importance

def plot_correlation(correlations):
    fig, ax = plt.subplots(figsize=(10, 6))
    correlations.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Correlation with Next-Hour PM2.5")
    ax.set_xlabel("Correlation Coefficient")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    save_fig(fig, "correlation_bar.png")

def plot_correlation_heatmap(df):
    numerical_df = df.select_dtypes(include="number")
    correlation_matrix = numerical_df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        correlation_matrix, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.3,
        cbar_kws={"shrink": 0.7}, ax=ax
    )
    ax.set_title("Correlation Heatmap of Air Quality and Weather Features", fontsize=13, pad=25)
    ax.tick_params(axis="x", labelsize=8, rotation=45, pad=8)
    ax.tick_params(axis="y", labelsize=8, rotation=0, pad=8)
    fig.subplots_adjust(top=0.90, bottom=0.25, left=0.25, right=0.92)
    save_fig(fig, "correlation_heatmap.png")

def plot_feature_importance(importance):
    fig, ax = plt.subplots(figsize=(10, 6))
    importance.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    save_fig(fig, "feature_importance.png")

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    print("Dataset shape:", df.shape)
    
    print("Running Feature Selection and saving outputs to 'output/feature_selection/'...")
    correlations = correlation_analysis(df)
    importance = feature_importance_scoring(df)
    
    plot_correlation(correlations)
    plot_correlation_heatmap(df)
    plot_feature_importance(importance)
    print("Feature selection complete. All text and figures saved.")

if __name__ == "__main__":
    main()