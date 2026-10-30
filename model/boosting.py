"""XGBoost multi-task model for the Kleen-Air pipeline.

Final candidate model of the project proposal: gradient-boosted trees for
(a) next-hour PM2.5 regression and (b) multi-class European Air Quality
Index (EAQI) category classification, with

* chronological early stopping on the validation split (no future leakage),
* inverse-frequency sample weights so rare high-risk air hours
  ("Poor" ... "Extremely Poor") are penalised heavily when misclassified,
* integer coding of EAQI category labels (XGBoost requires 0..k-1 codes),
  transparently mapped back to category names for every downstream consumer,
* a training-curve history (validation loss per boosting iteration) that
  eval.py turns into ``figures/training_curves_xgboost.png``.

If the ``xgboost`` package is not installed the module falls back to
scikit-learn's histogram gradient boosting estimators so the pipeline keeps
running (the training-curve history is then empty).

Interface used by train.py
--------------------------
MODEL_NAME : str
train(split_pack) -> dict with keys
    "regressor"   : fitted regressor  (predict -> PM2.5 in µg/m³)
    "classifier"  : fitted classifier (predict -> EAQI category names)
    "trace"       : list[str] training trace lines
    "history"     : dict with key "panels" (list of plot panels)

Expected split_pack keys
------------------------
X_train / X_val / X_test      : pandas.DataFrame of numeric features
y_train / y_val / y_test      : pandas.Series next-hour PM2.5 (µg/m³)
y_train_cat / y_val_cat / y_test_cat : pandas.Series EAQI category names
"""

import inspect
import time as _time

import numpy as np
import pandas as pd

from processing.dataset import EAQI_CATEGORIES

try:  # preferred: native XGBoost (sample weights + early stopping + evals_result)
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:  # graceful fallback keeps the pipeline runnable
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    XGBOOST_AVAILABLE = False

MODEL_NAME = "xgboost"

# hyper-parameters
RANDOM_STATE = 42
N_ESTIMATORS = 1000          # upper bound; early stopping picks the final round
EARLY_STOPPING_ROUNDS = 50
LEARNING_RATE = 0.05
MAX_DEPTH = 6
SUBSAMPLE = 0.9
COLSAMPLE_BYTREE = 0.9
N_JOBS = -1

# XGBoost moved ``early_stopping_rounds`` from fit() to the constructor in
# v1.6; detect where the installed version expects it so both work.
if XGBOOST_AVAILABLE:
    _FIT_ACCEPTS_ESR = (
        "early_stopping_rounds"
        in inspect.signature(XGBRegressor.fit).parameters
    )
else:
    _FIT_ACCEPTS_ESR = False


# label coding
def _build_label_maps(y_train_cat):
    """Contiguous integer codes 0..k-1 in official EAQI severity order.

    Built only from categories present in the training split so XGBoost's
    ``Expected: [0 1 2 3 4]`` check always passes (e.g. when "Extremely Poor"
    never occurs in training).
    """
    present = [c for c in EAQI_CATEGORIES if c in set(y_train_cat)]
    cat_to_code = {name: code for code, name in enumerate(present)}
    code_to_cat = {code: name for name, code in cat_to_code.items()}
    return cat_to_code, code_to_cat


def _class_sample_weights(y_train_cat):
    """Inverse-frequency weights: rare high-risk hours count more per sample."""
    counts = y_train_cat.value_counts()
    n = len(y_train_cat)
    weight_per_class = {c: n / (len(counts) * counts[c]) for c in counts.index}
    weights = y_train_cat.map(weight_per_class).to_numpy(dtype=float)
    return weights, weight_per_class


# estimator builders
def _xgb_params():
    params = dict(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    if not _FIT_ACCEPTS_ESR:  # modern XGBoost: constructor parameter
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return params


def _build_regressor():
    if XGBOOST_AVAILABLE:
        return XGBRegressor(**_xgb_params())
    return HistGradientBoostingRegressor(
        max_iter=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_STATE,
    )


def _build_classifier():
    if XGBOOST_AVAILABLE:
        return XGBClassifier(**_xgb_params())
    return HistGradientBoostingClassifier(
        max_iter=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_STATE,
    )


# prediction wrapper
class _CodedCategoryClassifier:
    """Wraps an integer-coded classifier so ``predict`` returns EAQI names."""

    def __init__(self, estimator, code_to_cat):
        self.estimator = estimator
        self.code_to_cat = code_to_cat

    @property
    def classes_(self):
        return np.array([self.code_to_cat[c] for c in sorted(self.code_to_cat)])

    def predict(self, X):
        codes = self.estimator.predict(X)
        return pd.Series(codes, index=X.index).map(self.code_to_cat)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def __getattr__(self, name):
        # Guard against infinite recursion: while unpickling (joblib.load) the
        # instance dict is still empty, so a normal `self.estimator` lookup
        # would re-enter __getattr__ forever. Private/dunder probes (e.g.
        # pickle's "__setstate__" check) must raise AttributeError immediately.
        if name.startswith("_"):
            raise AttributeError(name)
        estimator = self.__dict__.get("estimator")
        if estimator is None:
            raise AttributeError(
                f"{type(self).__name__} object has no attribute {name!r}"
            )
        return getattr(estimator, name)


# training history
def _build_history(regressor, classifier):
    """Validation loss per boosting iteration (empty for the sklearn fallback)."""
    panels = []
    if XGBOOST_AVAILABLE:
        for model, task in ((regressor, "regressor"), (classifier, "classifier")):
            results = model.evals_result() if hasattr(model, "evals_result") else {}
            for split_name, metric_dict in results.items():
                for metric_name, values in metric_dict.items():
                    panels.append({
                        "title": f"XGBoost {task}: {metric_name}",
                        "x_label": "Boosting iteration",
                        "y_label": metric_name,
                        "lines": [{
                            "label": split_name,
                            "x": list(range(1, len(values) + 1)),
                            "y": [float(v) for v in values],
                        }],
                    })
    return {"panels": panels}


# training entry point
def train(split_pack):
    started = _time.perf_counter()

    X_train = split_pack["X_train"]
    y_train = split_pack["y_train"]
    y_train_cat = split_pack["y_train_cat"]
    X_val = split_pack["X_val"]
    y_val = split_pack["y_val"]
    y_val_cat = split_pack["y_val_cat"]

    trace = [
        f"Model: {MODEL_NAME} (native xgboost available: {XGBOOST_AVAILABLE})",
        f"Training rows: {len(X_train)} | validation rows: {len(X_val)}",
        f"Hyper-parameters: n_estimators<={N_ESTIMATORS}, learning_rate={LEARNING_RATE}, "
        f"max_depth={MAX_DEPTH}, subsample={SUBSAMPLE}, "
        f"colsample_bytree={COLSAMPLE_BYTREE}, early_stopping_rounds={EARLY_STOPPING_ROUNDS}",
    ]

    # ---- regressor: next-hour PM2.5 -------------------------------------------
    regressor = _build_regressor()
    if XGBOOST_AVAILABLE:
        fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
        if _FIT_ACCEPTS_ESR:  # legacy XGBoost: fit-time parameter
            fit_kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        regressor.fit(X_train, y_train, **fit_kwargs)
        if hasattr(regressor, "best_iteration"):
            trace.append(f"Regressor best iteration: {regressor.best_iteration}")
    else:
        regressor.fit(X_train, y_train)
        trace.append("Regressor: HistGradientBoosting fallback (internal early stopping)")

    # ---- classifier: EAQI category (integer codes + sample weights) -----------
    cat_to_code, code_to_cat = _build_label_maps(y_train_cat)
    y_train_codes = y_train_cat.map(cat_to_code).to_numpy(dtype=int)
    y_val_codes = y_val_cat.map(cat_to_code)
    weights, weight_per_class = _class_sample_weights(y_train_cat)
    trace.append("Inverse-frequency class weights: "
                 + ", ".join(f"{c}={w:.3f}" for c, w in sorted(weight_per_class.items())))

    classifier = _build_classifier()
    if XGBOOST_AVAILABLE:
        fit_kwargs = {"sample_weight": weights, "verbose": False}
        if y_val_codes.isna().any():
            trace.append("Warning: validation set contains categories unseen in "
                         "training; disabling XGBoost early-stopping eval set.")
        else:
            fit_kwargs["eval_set"] = [(X_val, y_val_codes.to_numpy(dtype=int))]
            if _FIT_ACCEPTS_ESR:
                fit_kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        classifier.fit(X_train, y_train_codes, **fit_kwargs)
        if hasattr(classifier, "best_iteration"):
            trace.append(f"Classifier best iteration: {classifier.best_iteration}")
    else:
        classifier.fit(X_train, y_train_codes, sample_weight=weights)
        trace.append("Classifier: HistGradientBoosting fallback (internal early stopping)")

    elapsed = _time.perf_counter() - started
    trace.append(f"Training wall-time (regressor + classifier): {elapsed:.1f} s")

    return {
        "regressor": regressor,
        "classifier": _CodedCategoryClassifier(classifier, code_to_cat),
        "code_to_cat": code_to_cat,
        "trace": trace,
        "history": _build_history(regressor, classifier),
    }


# standalone smoke test
if __name__ == "__main__":
    from processing.dataset import (CATEGORY_TARGET, REGRESSION_TARGET,
                                    chronological_split, load_featured_data)

    df = load_featured_data()
    train_df, val_df, test_df = chronological_split(df)
    feature_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in (REGRESSION_TARGET, CATEGORY_TARGET)]

    pack = {
        "X_train": train_df[feature_cols], "y_train": train_df[REGRESSION_TARGET],
        "y_train_cat": train_df[CATEGORY_TARGET],
        "X_val": val_df[feature_cols], "y_val": val_df[REGRESSION_TARGET],
        "y_val_cat": val_df[CATEGORY_TARGET],
        "X_test": test_df[feature_cols], "y_test": test_df[REGRESSION_TARGET],
        "y_test_cat": test_df[CATEGORY_TARGET],
    }
    result = train(pack)

    from eval import classification_report, regression_report
    y_pred_reg = result["regressor"].predict(pack["X_test"])
    y_pred_cat = result["classifier"].predict(pack["X_test"])
    print("\n".join(result["trace"]))
    print("Test regression:", regression_report(pack["y_test"], y_pred_reg))
    print("Test classification:", classification_report(pack["y_test_cat"], y_pred_cat))