# Kleen-Air — Explainable Next-Hour PM2.5 Forecasting & EAQI Classification

TEB3123 Machine Learning · Group 2 · UTP Grand Challenge: Sustainable Living · SDG 11 (Target 11.6)

Forecasts PM2.5 one hour ahead from co-pollutants (PM10, CO, NO₂, SO₂, O₃) and weather
(Open-Meteo, Europe, 2016–2026), classifies each forecast into the six EAQI categories,
and explains the drivers with SHAP.

## Structure

```
analysis/EDA.py                    # data understanding: gaps, outliers, imbalance, distributions
processing/dataloader.py           # merge streams (skiprows=3), dedup, plausibility filter, gap interpolation
processing/dataset.py              # EAQI bands/encoding, targets, chronological splits, class weights
processing/feature-engineering.py  # lags, 24 h rolling mean, hour/month -> data/featured_dataset.csv
processing/feature-selection.py    # correlation + Random Forest importance
train.py                           # ORCHESTRATOR: split -> train all models -> traces/models/predictions -> evaluation
model/regression.py                # baseline: Linear + Logistic Regression (module)
model/forest.py                    # Random Forest reg/cls, TimeSeriesSplit grid search (module)
model/boosting.py                  # XGBoost reg/cls, early stopping + inverse-frequency weights (module)
eval.py                            # comparison tables, plots, bootstrap CIs, paired/McNemar tests, SHAP
```

## Usage

```bash
python analysis/EDA.py
python processing/feature-engineering.py
python processing/feature-selection.py
python train.py                 # trains all 3 model families + full evaluation (--show-plots optional)
python eval.py                  # optional: re-run evaluation suite on saved artifacts
```

## Outputs (`results/`, `models/`)

- `metrics_<model>.json`, `predictions/predictions_<model>.csv`
- `traces/<model>_training_trace.txt` — CV folds / grid search / early-stopping history
- `tables/model_comparison.txt|.csv` — RMSE, MAE, R², Acc, Precision/Recall/F1 (macro, weighted), high-risk recall, naive persistence baseline
- `tables/derived_eaqi_comparison.txt`, `tables/statistical_analysis.txt` — bootstrap CIs, paired t/Wilcoxon, McNemar, residual diagnostics, SHAP ranking
- `figures/` — comparison bars, test-fit & residual plots, confusion matrices, SHAP plots
- `models/<model>.joblib` — regressor + classifier per family

## Requirements

pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, joblib · optional: xgboost, shap
