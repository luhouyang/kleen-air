"""Baseline model module: Linear Regression (regression) + Logistic Regression
(classification). Trained via train.py; exposes MODEL_NAME and train(split)."""

from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from eval import build_learning_curve_history

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_NAME = "linear_logistic_baseline"


def build_regressor():
    return Pipeline([("scaler", StandardScaler()),
                     ("model", LinearRegression())])


def build_classifier():
    return Pipeline([("scaler", StandardScaler()),
                     ("model", LogisticRegression(max_iter=5000,
                                                  class_weight="balanced"))])


def train(split):
    """Fit both tasks and return models plus a training trace (5-fold TSCV)."""
    X_train, y_train = split["X_train"], split["y_train"]
    X_val, y_val = split["X_val"], split["y_val"]
    y_train_cat, y_val_cat = split["y_train_cat"], split["y_val_cat"]

    regressor, classifier = build_regressor(), build_classifier()
    tscv = TimeSeriesSplit(n_splits=5)
    trace = ["Time-series cross-validation on the training block (5 folds):"]

    reg_cv = cross_validate(regressor, X_train, y_train, cv=tscv,
                            scoring=("neg_root_mean_squared_error",
                                     "neg_mean_absolute_error", "r2"))
    cls_cv = cross_validate(classifier, X_train, y_train_cat, cv=tscv,
                            scoring=("f1_macro", "accuracy"))

    rmse_f = -reg_cv["test_neg_root_mean_squared_error"]
    mae_f = -reg_cv["test_neg_mean_absolute_error"]
    r2_f = reg_cv["test_r2"]
    f1_f = cls_cv["test_f1_macro"]
    acc_f = cls_cv["test_accuracy"]
    for i in range(len(rmse_f)):
        trace.append(f"  Fold {i + 1}: RMSE={rmse_f[i]:.4f} MAE={mae_f[i]:.4f} "
                     f"R2={r2_f[i]:.4f} | F1_macro={f1_f[i]:.4f} Acc={acc_f[i]:.4f}")
    trace.append(f"  CV mean: RMSE={rmse_f.mean():.4f} +/- {rmse_f.std():.4f} | "
                 f"F1_macro={f1_f.mean():.4f} +/- {f1_f.std():.4f}")

    regressor.fit(X_train, y_train)
    classifier.fit(X_train, y_train_cat)

    val_rmse = float(np.sqrt(mean_squared_error(y_val, regressor.predict(X_val))))
    val_f1 = f1_score(y_val_cat, classifier.predict(X_val), average="macro",
                      zero_division=0)
    trace.append(f"Validation block: RMSE={val_rmse:.4f} | F1_macro={val_f1:.4f}")

    history = {"panels": [
        build_learning_curve_history(
            X_train, y_train, build_regressor(),
            "Linear Regression learning curve",
            "neg_root_mean_squared_error", "RMSE", transform=lambda s: -s),
        build_learning_curve_history(
            X_train, y_train_cat, build_classifier(),
            "Logistic Regression learning curve", "f1_macro", "Macro F1"),
    ]}
    return {"regressor": regressor, "classifier": classifier, "trace": trace,
            "history": history}


if __name__ == "__main__":
    print(f"'{MODEL_NAME}' is trained by the orchestrator. Run: python train.py")