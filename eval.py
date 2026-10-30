"""Evaluation engine: metric helpers used by train.py plus run_full_evaluation(),
which produces comparison tables (.txt/.csv), training-curve and performance
figures, bootstrap CIs, paired significance tests, McNemar tests, residual
diagnostics, per-model confusion matrices and SHAP source attribution.
All text outputs are written with explicit UTF-8 encoding (Windows cp1252 safe)."""

import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.metrics import (accuracy_score, classification_report as sk_classification_report,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)
from sklearn.model_selection import learning_curve

from processing.dataset import (CATEGORY_TARGET, EAQI_CATEGORIES,
                                HIGH_RISK_CATEGORIES, REGRESSION_TARGET,
                                TIME_COL, chronological_split,
                                load_featured_data, pm25_eaqi_category)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "output"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
TRACES_DIR = RESULTS_DIR / "traces"

POLLUTANT_SOURCES = {
    "carbon_monoxide (μg/m³)": "incomplete combustion / road traffic",
    "nitrogen_dioxide (μg/m³)": "road traffic & fuel combustion",
    "sulphur_dioxide (μg/m³)": "industry & power generation",
    "ozone (μg/m³)": "photochemical smog (secondary formation)",
    "pm10 (μg/m³)": "coarse particles (dust, resuspension, combustion)",
    "pm2_5 (μg/m³)": "current fine-particle level (persistence)",
    "european_aqi (EAQI)": "overall current air quality (aggregate index)",
    "pm2_5_lag_1": "recent pollution history",
    "pm2_5_lag_3": "recent pollution history",
    "pm2_5_lag_24": "day-ahead pollution level",
    "pm2_5_rolling_mean_24": "24 h background pollution level",
    "hour": "diurnal activity pattern (traffic, heating)",
    "month": "seasonal pattern (heating season, meteorology)",
    "temperature_2m (°C)": "meteorology (reaction rates, inversions)",
    "relative_humidity_2m (%)": "meteorology (particle formation/growth)",
    "wind_speed_100m (km/h)": "meteorology (dispersion)",
    "wind_direction_100m (°)": "meteorology (transport direction)",
    "rain (mm)": "meteorology (wet deposition)",
    "precipitation (mm)": "meteorology (wet deposition)",
}


# --------------------------------------------------------------------------- storage
def _ensure_dirs():
    for directory in (RESULTS_DIR, PREDICTIONS_DIR, MODELS_DIR,
                      FIGURES_DIR, TABLES_DIR, TRACES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_trace(model_name, trace_lines):
    _ensure_dirs()
    path = TRACES_DIR / f"{model_name}_training_trace.txt"
    header = (f"Training trace - {model_name}\n"
              f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n" + "-" * 60)
    path.write_text(header + "\n" + "\n".join(trace_lines) + "\n", encoding="utf-8")
    return path


def save_training_history(model_name, history):
    _ensure_dirs()
    path = TRACES_DIR / f"{model_name}_history.json"
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_table(table_name, text):
    _ensure_dirs()
    path = TABLES_DIR / f"{table_name}.txt"
    path.write_text(text + "\n", encoding="utf-8")
    return path


def save_metrics(model_name, metrics):
    _ensure_dirs()
    path = RESULTS_DIR / f"metrics_{model_name}.json"
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_predictions(model_name, predictions_df):
    _ensure_dirs()
    path = PREDICTIONS_DIR / f"predictions_{model_name}.csv"
    predictions_df.to_csv(path, index=False, encoding="utf-8")
    return path


def save_model(model_name, payload):
    _ensure_dirs()
    path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(payload, path)
    return path


def load_metrics():
    return {p.stem.replace("metrics_", ""): json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(RESULTS_DIR.glob("metrics_*.json"))}


def save_figure(fig, name, show=False):
    _ensure_dirs()
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- metrics
def regression_report(y_true, y_pred):
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred))}


def classification_report(y_true, y_pred):
    y_true = pd.Series(y_true).astype(str).to_numpy()
    y_pred = pd.Series(y_pred).astype(str).to_numpy()
    hr_true = np.isin(y_true, HIGH_RISK_CATEGORIES)
    hr_pred = np.isin(y_pred, HIGH_RISK_CATEGORIES)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall_high_risk": float(recall_score(hr_true, hr_pred, zero_division=0)),
    }


def naive_persistence_report(df_test):
    return regression_report(df_test[REGRESSION_TARGET], df_test["pm2_5 (μg/m³)"])


def build_comparison_table(metrics, naive_report):
    rows = {name: {**m.get("regression", {}), **m.get("classification", {})}
            for name, m in metrics.items()}
    rows["naive_persistence"] = naive_report
    columns = ["RMSE", "MAE", "R2", "Accuracy", "Precision_macro", "Recall_macro",
               "F1_macro", "F1_weighted", "Recall_high_risk"]
    return pd.DataFrame(rows).T.reindex(columns=columns)


# ------------------------------------------------------------------- statistical analysis
def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx = np.arange(len(y_true))
    values = [metric_fn(y_true[s], y_pred[s])
              for s in (rng.choice(idx, size=len(idx), replace=True)
                        for _ in range(n_bootstrap))]
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def compare_regression_predictions(y_true, pred_a, pred_b, name_a, name_b):
    y = np.asarray(y_true, dtype=float)
    ea = (y - np.asarray(pred_a, dtype=float)) ** 2
    eb = (y - np.asarray(pred_b, dtype=float)) ** 2
    lines = [f"{name_a} vs {name_b}:"]
    t_stat, p_val = scipy_stats.ttest_rel(ea, eb)
    lines.append(f"  paired t-test on squared errors: t={t_stat:.3f}, p={p_val:.3e}")
    try:
        w_stat, w_p = scipy_stats.wilcoxon(np.abs(y - np.asarray(pred_a)),
                                           np.abs(y - np.asarray(pred_b)),
                                           zero_method="zsplit")
        lines.append(f"  Wilcoxon signed-rank on |errors|: W={w_stat:.1f}, p={w_p:.3e}")
    except ValueError:
        lines.append("  Wilcoxon signed-rank: not computable (identical errors)")
    return lines


def mcnemar_test(y_true, pred_a, pred_b, name_a, name_b):
    y = pd.Series(y_true).astype(str).to_numpy()
    a = pd.Series(pred_a).astype(str).to_numpy()
    b = pd.Series(pred_b).astype(str).to_numpy()
    disc = int(np.sum((a == y) & (b != y)))
    conc = int(np.sum((a != y) & (b == y)))
    if disc + conc == 0:
        return [f"{name_a} vs {name_b}: McNemar not computable (identical predictions)"]
    chi2 = (abs(disc - conc) - 1) ** 2 / (disc + conc)
    return [f"{name_a} vs {name_b}: McNemar (continuity-corrected) "
            f"b={disc}, c={conc}, chi2={chi2:.3f}, p={float(scipy_stats.chi2.sf(chi2, df=1)):.3e}"]


def residual_diagnostics(y_true, y_pred):
    resid = pd.Series(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    sample = resid.sample(n=min(5000, len(resid)), random_state=42)
    shapiro_stat, shapiro_p = scipy_stats.shapiro(sample)
    return {"mean": resid.mean(), "std": resid.std(), "skew": resid.skew(),
            "lag1_autocorr": resid.autocorr(1),
            "shapiro_W": shapiro_stat, "shapiro_p": shapiro_p}


# ------------------------------------------------------------------- training histories
def build_learning_curve_history(X, y, estimator, title, scoring, y_label,
                                 transform=None, max_rows=30000, n_sizes=4, cv=3):
    """Chronological learning curve (train vs validation score vs training rows)."""
    step = max(1, len(X) // max_rows)
    Xs = X.iloc[::step]
    ys = y.iloc[::step]
    sizes, train_scores, test_scores = learning_curve(
        estimator, Xs, ys, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, n_sizes), scoring=scoring)
    fn = transform if transform is not None else (lambda a: a)
    x = [int(s) for s in sizes]
    return {"title": title, "x_label": "Training rows", "y_label": y_label,
            "lines": [
                {"label": "train", "x": x, "y": [float(v) for v in fn(train_scores.mean(axis=1))]},
                {"label": "validation", "x": x, "y": [float(v) for v in fn(test_scores.mean(axis=1))]},
            ]}


# --------------------------------------------------------------------------- plots
def plot_training_curves(model_name, show=False):
    path = TRACES_DIR / f"{model_name}_history.json"
    if not path.exists():
        return None
    panels = [p for p in json.loads(path.read_text(encoding="utf-8")).get("panels", [])
              if p.get("lines")]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        for line in panel["lines"]:
            ax.plot(line["x"], line["y"], label=line["label"], lw=1.4)
        ax.set_title(panel["title"], fontsize=10)
        ax.set_xlabel(panel["x_label"])
        ax.set_ylabel(panel["y_label"])
        ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, f"training_curves_{model_name}.png", show=show)


def plot_performance_comparison(table, show=False):
    panels = [("RMSE", "lower better"), ("MAE", "lower better"), ("R2", "higher better"),
              ("Accuracy", "higher better"), ("F1_macro", "higher better"),
              ("Recall_high_risk", "higher better")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (col, note) in zip(axes.ravel(), panels):
        series = table[col].dropna().sort_values()
        series.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"{col} ({note})")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Model Performance Comparison (test set)", fontsize=13)
    fig.tight_layout()
    return save_figure(fig, "performance_comparison.png", show=show)


def plot_predictions_overlay(preds, show=False, tail=500):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    first = next(iter(preds.values())).tail(tail)
    ax.plot(first[TIME_COL], first["y_true_reg"], label="Actual", lw=1.4, color="black")
    for name, d in preds.items():
        sample = d.tail(tail)
        ax.plot(sample[TIME_COL], sample["y_pred_reg"], lw=1.0, ls="--", label=name)
    ax.set_title(f"Next-hour PM2.5 forecasts (last {tail} test hours)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, "predictions_overlay.png", show=show)


def plot_per_class_recall(preds, show=False):
    rows = {}
    for name, d in preds.items():
        rows[name] = recall_score(d["y_true_cat"], d["y_pred_cat"],
                                  labels=EAQI_CATEGORIES, average=None, zero_division=0)
    frame = pd.DataFrame(rows, index=EAQI_CATEGORIES)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    frame.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Per-class Recall by Model (test set)")
    ax.set_ylabel("Recall")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, "per_class_recall.png", show=show)


def plot_confusion_matrix(y_true, y_pred, title, file_name, show=False):
    cm = confusion_matrix(y_true, y_pred, labels=EAQI_CATEGORIES)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True,
                xticklabels=EAQI_CATEGORIES, yticklabels=EAQI_CATEGORIES, ax=ax)
    ax.set_xlabel("Predicted EAQI category")
    ax.set_ylabel("Actual EAQI category")
    ax.set_title(title)
    fig.tight_layout()
    return save_figure(fig, file_name, show=show)


def plot_diagnostics(preds_df, model_name, show=False):
    y = preds_df["y_true_reg"].to_numpy()
    p = preds_df["y_pred_reg"].to_numpy()
    resid = y - p
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].scatter(p, y, s=3, alpha=0.3)
    lims = [min(y.min(), p.min()), max(y.max(), p.max())]
    axes[0].plot(lims, lims, "r--", lw=1)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_title(f"{model_name}: actual vs predicted")
    axes[1].hist(resid, bins=60)
    axes[1].set_title("Residual distribution")
    axes[2].scatter(p, resid, s=3, alpha=0.3)
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Residual")
    axes[2].set_title("Residuals vs predicted")
    fig.tight_layout()
    return save_figure(fig, f"diagnostics_{model_name}.png", show=show)


def shap_driver_analysis(metrics, test, show=False):
    try:
        import shap
    except ImportError:
        print("shap is not installed - skipping SHAP interpretation.")
        return []
    candidates = [n for n in metrics if n in ("random_forest", "xgboost")]
    if not candidates:
        return []
    best = min(candidates, key=lambda n: metrics[n]["regression"]["RMSE"])
    regressor = joblib.load(MODELS_DIR / f"{best}.joblib")["regressor"]

    X_test = test[[c for c in test.select_dtypes(include="number").columns
                   if c != REGRESSION_TARGET]]
    sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    shap_values = shap.TreeExplainer(regressor).shap_values(sample)

    mean_abs = (pd.Series(np.abs(shap_values).mean(axis=0), index=sample.columns)
                .sort_values(ascending=False))
    lines = [f"SHAP global importance (best tree regressor: {best}):"]
    for feature, value in mean_abs.items():
        lines.append(f"  {feature:28s} mean|SHAP| = {value:8.4f}  -> "
                     f"{POLLUTANT_SOURCES.get(feature, 'engineered feature')}")

    fig, ax = plt.subplots(figsize=(9, 6))
    mean_abs.sort_values().plot(kind="barh", ax=ax, color="teal")
    ax.set_title(f"Mean |SHAP| - {best}")
    fig.tight_layout()
    save_figure(fig, f"shap_importance_{best}.png", show=show)
    shap.summary_plot(shap_values, sample, show=False)
    save_figure(plt.gcf(), f"shap_summary_{best}.png", show=show)
    return lines


# --------------------------------------------------------------------------- main suite
def run_full_evaluation(show_plots=False):
    _ensure_dirs()
    df = load_featured_data()
    _, _, test = chronological_split(df)
    metrics = load_metrics()
    if not metrics:
        print("No metrics found - run train.py first.")
        return

    preds = {p.stem.replace("predictions_", ""):
             pd.read_csv(p, parse_dates=[TIME_COL], encoding="utf-8")
             for p in sorted(PREDICTIONS_DIR.glob("predictions_*.csv"))}

    naive = naive_persistence_report(test)
    table = build_comparison_table(metrics, naive)
    table_txt = ("MODEL COMPARISON ON THE TEST SET\n" + "=" * 60 + "\n"
                 + table.round(4).to_string())
    print("\n" + table_txt)
    save_table("model_comparison", table_txt)
    table.round(6).to_csv(TABLES_DIR / "model_comparison.csv", encoding="utf-8")

    # ---- derived early-warning path -------------------------------------------
    # (a) Conversion fidelity: does "regressed PM2.5 -> EAQI PM2.5 band"
    #     reproduce the band of the OBSERVED next-hour PM2.5?
    derived = {name: classification_report(
        pm25_eaqi_category(d["y_true_reg"]),
        pm25_eaqi_category(d["y_pred_reg"]),
    ) for name, d in preds.items()}
    derived_txt = ("EARLY-WARNING PATH: PM2.5 FORECAST -> EAQI PM2.5 BAND\n"
                   "(scored against the EAQI PM2.5 band of the OBSERVED "
                   "next-hour PM2.5)\n" + "=" * 60 + "\n"
                   + pd.DataFrame(derived).T.round(4).to_string())
    print("\n" + derived_txt)
    save_table("derived_eaqi_comparison", derived_txt)

    # (b) Label-space diagnostic: the official EAQI category is the WORST of
    #     five pollutants, so it usually exceeds the PM2.5-only band. This
    #     crosswalk explains any gap between (a) and the official-category
    #     scores in model_comparison.txt.
    crosswalk = pd.crosstab(
        pm25_eaqi_category(test[REGRESSION_TARGET]).astype(str),
        test[CATEGORY_TARGET].astype(str),
        normalize="index",
    ).round(3)
    crosswalk_txt = ("LABEL-SPACE CROSSWALK (rows: PM2.5 band of observed "
                     "next-hour PM2.5, cols: official next-hour EAQI category,"
                     " row-normalised)\n" + "=" * 60 + "\n"
                     + crosswalk.to_string())
    print("\n" + crosswalk_txt)
    save_table("label_space_crosswalk", crosswalk_txt)

    # ---- statistical analysis -------------------------------------------------
    stat = ["STATISTICAL ANALYSIS (test set)", "=" * 60]
    stat.append("\n[1] Bootstrap 95% confidence intervals (1000 resamples):")
    for name, d in preds.items():
        lo, hi = bootstrap_ci(d["y_true_reg"], d["y_pred_reg"],
                              lambda t, p: np.sqrt(mean_squared_error(t, p)))
        stat.append(f"  {name}: RMSE 95% CI [{lo:.4f}, {hi:.4f}]")
        lo, hi = bootstrap_ci(d["y_true_cat"], d["y_pred_cat"],
                              lambda t, p: f1_score(t, p, average="macro", zero_division=0))
        stat.append(f"  {name}: F1_macro 95% CI [{lo:.4f}, {hi:.4f}]")

    best_reg = min(metrics, key=lambda n: metrics[n]["regression"]["RMSE"])
    best_cls = max(metrics, key=lambda n:
                   metrics[n].get("classification", {}).get("F1_macro", -1))
    ref, ref_c = preds[best_reg], preds[best_cls]

    stat.append(f"\n[2] Paired error tests vs reference regressor '{best_reg}':")
    for name, d in preds.items():
        if name != best_reg:
            stat += ["  " + l for l in compare_regression_predictions(
                d["y_true_reg"], ref["y_pred_reg"], d["y_pred_reg"], best_reg, name)]
    stat += ["  " + l for l in compare_regression_predictions(
        test[REGRESSION_TARGET], ref["y_pred_reg"],
        test["pm2_5 (μg/m³)"], best_reg, "naive_persistence")]

    stat.append(f"\n[3] McNemar tests vs reference classifier '{best_cls}':")
    for name, d in preds.items():
        if name != best_cls:
            stat += ["  " + l for l in mcnemar_test(
                d["y_true_cat"], ref_c["y_pred_cat"], d["y_pred_cat"], best_cls, name)]

    stat.append(f"\n[4] Residual diagnostics for '{best_reg}':")
    stat += [f"  {k}: {v:.4f}" for k, v in
             residual_diagnostics(ref["y_true_reg"], ref["y_pred_reg"]).items()]

    stat.append("\n[5] SHAP driver ranking with plausible emission sources:")
    shap_lines = shap_driver_analysis(metrics, test, show=show_plots)
    stat += ["  " + l for l in shap_lines]

    stat_txt = "\n".join(stat)
    print("\n" + stat_txt)
    save_table("statistical_analysis", stat_txt)

    # ---- figures ---------------------------------------------------------------
    for name in preds:
        plot_training_curves(name, show=show_plots)
        plot_diagnostics(preds[name], name, show=show_plots)
        plot_confusion_matrix(preds[name]["y_true_cat"], preds[name]["y_pred_cat"],
                              f"Confusion Matrix - {name}",
                              f"confusion_{name}.png", show=show_plots)
    plot_performance_comparison(table, show=show_plots)
    plot_predictions_overlay(preds, show=show_plots)
    plot_per_class_recall(preds, show=show_plots)

    print("\nTables saved to:", TABLES_DIR)
    print("Figures saved to:", FIGURES_DIR)
    print("Traces & histories saved to:", TRACES_DIR)


def main():
    run_full_evaluation(show_plots=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluation & comparison suite.")
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()
    run_full_evaluation(show_plots=args.show_plots)