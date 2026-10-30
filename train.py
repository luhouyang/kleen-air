"""Orchestrator: chronological split -> train all three model families ->
save training traces, models, test predictions and metrics -> run the full
evaluation/comparison/statistical-analysis suite in eval.py."""

from pathlib import Path
import sys
import time

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval import (classification_report,
                  regression_report,
                  run_full_evaluation,
                  save_metrics,
                  save_model,
                  save_predictions,
                  save_trace,
                  save_training_history)
from model import boosting, forest, regression
from processing.dataset import (CATEGORY_TARGET,
                                REGRESSION_TARGET,
                                TIME_COL,
                                chronological_split,
                                load_featured_data,
                                split_features_target)

MODEL_MODULES = [regression, forest, boosting]


def main(show_plots=False):
    df = load_featured_data()
    train_df, val_df, test_df = chronological_split(df)

    X_train, y_train = split_features_target(train_df, REGRESSION_TARGET)
    X_val, y_val = split_features_target(val_df, REGRESSION_TARGET)
    X_test, y_test = split_features_target(test_df, REGRESSION_TARGET)
    y_train_cat = train_df[CATEGORY_TARGET].astype(str)
    y_val_cat = val_df[CATEGORY_TARGET].astype(str)
    y_test_cat = test_df[CATEGORY_TARGET].astype(str)

    total_rows = len(df)
    print("===== DATASET SPLITTING =====")
    print(f"Total rows: {total_rows}")
    print(f"\nTraining set: {len(X_train)} rows | "
          f"{train_df[TIME_COL].iloc[0]} to {train_df[TIME_COL].iloc[-1]}")
    print(f"Validation set: {len(X_val)} rows | "
          f"{val_df[TIME_COL].iloc[0]} to {val_df[TIME_COL].iloc[-1]}")
    print(f"Testing set: {len(X_test)} rows | "
          f"{test_df[TIME_COL].iloc[0]} to {test_df[TIME_COL].iloc[-1]}")

    print("\n===== TIME SERIES CROSS-VALIDATION (overview) =====")
    for fold, (tr_idx, va_idx) in enumerate(TimeSeriesSplit(n_splits=5).split(X_train), 1):
        print(
            f"Fold {fold}: {len(tr_idx)} training rows, {len(va_idx)} validation rows"
        )

    split_pack = dict(X_train=X_train,
                      y_train=y_train,
                      X_val=X_val,
                      y_val=y_val,
                      y_train_cat=y_train_cat,
                      y_val_cat=y_val_cat)

    for module in MODEL_MODULES:
        print(f"\n===== TRAINING {module.MODEL_NAME} =====")
        started = time.perf_counter()
        result = module.train(split_pack)
        elapsed = time.perf_counter() - started

        trace = list(
            result["trace"]) + [f"Total training wall-time: {elapsed:.1f} s"]
        save_trace(module.MODEL_NAME, trace)
        if result.get("history"):
            save_training_history(module.MODEL_NAME, result["history"])

        regressor, classifier = result["regressor"], result["classifier"]
        y_pred_reg = regressor.predict(X_test)
        y_pred_cat = classifier.predict(X_test)

        reg_metrics = regression_report(y_test, y_pred_reg)
        cls_metrics = classification_report(y_test_cat, y_pred_cat)
        print("Test regression:",
              {
                  k: round(v,
                           4)
                  for k, v in reg_metrics.items()
              })
        print("Test classification:",
              {
                  k: round(v,
                           4)
                  for k, v in cls_metrics.items()
              })

        save_metrics(module.MODEL_NAME,
                     {
                         "regression": reg_metrics,
                         "classification": cls_metrics
                     })
        save_predictions(
            module.MODEL_NAME,
            pd.DataFrame({
                TIME_COL:
                test_df[TIME_COL].to_numpy(),
                "y_true_reg":
                y_test.to_numpy(),
                "y_pred_reg":
                y_pred_reg,
                "y_true_cat":
                y_test_cat.to_numpy(),
                "y_pred_cat":
                pd.Series(y_pred_cat).astype(str).to_numpy(),
            }))
        save_model(module.MODEL_NAME,
                   {
                       "regressor": regressor,
                       "classifier": classifier
                   })

    print("\n===== EVALUATION, COMPARISON & STATISTICAL ANALYSIS =====")
    run_full_evaluation(show_plots=show_plots)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train and evaluate all models.")
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="display matplotlib windows in addition to saving")
    args = parser.parse_args()
    main(show_plots=args.show_plots)
