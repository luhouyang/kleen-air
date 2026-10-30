from pathlib import Path
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Load the featured dataset
df = pd.read_csv("data/featured_dataset.csv")

# Sort data by time to preserve chronological order
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

# Remove rows with missing values
df = df.dropna().reset_index(drop=True)

# Separate input features and target
X = df.drop(columns=["time", "pm2_5_next_hour"])
y = df["pm2_5_next_hour"]

# Split dataset chronologically: 70% train, 15% validation, 15% test
total_rows = len(df)
train_end = int(total_rows * 0.70)
validation_end = int(total_rows * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:validation_end]
y_val = y.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]

# Time-series cross-validation on training data
tscv = TimeSeriesSplit(n_splits=5)

# Display dataset split information
print("===== DATASET SPLITTING =====")
print(f"Total rows: {total_rows}")

print(f"\nTraining set: {len(X_train)} rows")
print(f"Time range: {df['time'].iloc[0]} to "
      f"{df['time'].iloc[train_end - 1]}")

print(f"\nValidation set: {len(X_val)} rows")
print(f"Time range: {df['time'].iloc[train_end]} to "
      f"{df['time'].iloc[validation_end - 1]}")

print(f"\nTesting set: {len(X_test)} rows")
print(f"Time range: {df['time'].iloc[validation_end]} to "
      f"{df['time'].iloc[-1]}")

print("\n===== TIME SERIES CROSS-VALIDATION =====")

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
    print(
        f"Fold {fold}: "
        f"{len(train_idx)} training rows, "
        f"{len(val_idx)} validation rows"
    )