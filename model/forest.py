"""Random Forest module: regressor & classifier tuned with TimeSeriesSplit
grid search. Trained via train.py; exposes MODEL_NAME and train(split)."""

from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from eval import build_learning_curve_history

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_NAME = "random_forest"

REGRESSOR_GRID = {"n_estimators": [200], "max_depth": [12, None],
                  "min_samples_split": [5]}
CLASSIFIER_GRID = {"n_estimators": [200], "max_depth": [16, None],
                   "min_samples_split": [5], "class_weight": ["balanced"]}


def train(split):
    X_train, y_train = split["X_train"], split["y_train"]
    X_val, y_val = split["X_val"], split["y_val"]
    y_train_cat, y_val_cat = split["y_train_cat"], split["y_val_cat"]

    trace = ["Time-series CV grid search (TimeSeriesSplit, 3 folds):"]

    reg_search = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                              REGRESSOR_GRID, cv=TimeSeriesSplit(n_splits=3),
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
    reg_search.fit(X_train, y_train)
    trace.append("  Regressor candidates (neg-RMSE scoring):")
    for params, mean_s, std_s in zip(reg_search.cv_results_["params"],
                                     reg_search.cv_results_["mean_test_score"],
                                     reg_search.cv_results_["std_test_score"]):
        trace.append(f"    score={mean_s:.4f} +/- {std_s:.4f}  params={params}")
    trace.append(f"  Best regressor params: {reg_search.best_params_}")

    cls_search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                              CLASSIFIER_GRID, cv=TimeSeriesSplit(n_splits=3),
                              scoring="f1_macro", n_jobs=-1)
    cls_search.fit(X_train, y_train_cat)
    trace.append("  Classifier candidates (macro-F1 scoring):")
    for params, mean_s, std_s in zip(cls_search.cv_results_["params"],
                                     cls_search.cv_results_["mean_test_score"],
                                     cls_search.cv_results_["std_test_score"]):
        trace.append(f"    score={mean_s:.4f} +/- {std_s:.4f}  params={params}")
    trace.append(f"  Best classifier params: {cls_search.best_params_}")

    regressor, classifier = reg_search.best_estimator_, cls_search.best_estimator_
    val_rmse = float(np.sqrt(mean_squared_error(y_val, regressor.predict(X_val))))
    val_f1 = f1_score(y_val_cat, classifier.predict(X_val), average="macro",
                      zero_division=0)
    trace.append(f"Validation block: RMSE={val_rmse:.4f} | F1_macro={val_f1:.4f}")

    from sklearn.ensemble import RandomForestClassifier as _RFC, RandomForestRegressor as _RFR
    history = {"panels": [
        build_learning_curve_history(
            X_train, y_train, _RFR(n_estimators=100, n_jobs=-1, random_state=42),
            "Random Forest regressor learning curve",
            "neg_root_mean_squared_error", "RMSE", transform=lambda s: -s),
        build_learning_curve_history(
            X_train, y_train_cat, _RFC(n_estimators=100, n_jobs=-1, random_state=42),
            "Random Forest classifier learning curve", "f1_macro", "Macro F1"),
    ]}
    return {"regressor": regressor, "classifier": classifier, "trace": trace,
            "history": history}


if __name__ == "__main__":
    print(f"'{MODEL_NAME}' is trained by the orchestrator. Run: python train.py")