"""Kleen-Air Streamlit dashboard.

Live demo of the trained pipeline: next-hour PM2.5 forecast, EAQI category,
high-risk alert and SHAP explanation for any test-set hour, plus the saved
model results and an alert log.

Run:
    streamlit run app.py
If output/models/*.joblib is missing (e.g. on Streamlit Cloud), XGBoost is
trained once on startup with the same settings as model/boosting.py.
"""
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from processing.dataset import (CATEGORY_TARGET, HIGH_RISK_CATEGORIES,  # noqa: E402
                                REGRESSION_TARGET, TIME_COL,
                                chronological_split, feature_columns,
                                load_featured_data)
import model.boosting  # noqa: E402,F401  (needed so joblib can unpickle the classifier wrapper)

OUTPUT = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT / "models"
FIGURES_DIR = OUTPUT / "figures"
PM25 = "pm2_5 (μg/m³)"

# Official EAQI colours
EAQI_COLOURS = {"Good": "#50F0E6", "Fair": "#50CCAA", "Moderate": "#F0E641",
                "Poor": "#FF5050", "Very Poor": "#960032", "Extremely Poor": "#7D2181"}
MODEL_LABELS = {"xgboost": "XGBoost", "random_forest": "Random Forest",
                "linear_logistic_baseline": "Linear / Logistic"}

st.set_page_config(page_title="Kleen-Air PM2.5 Forecast", page_icon="🌫️", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_data
def load_test_set():
    df = load_featured_data()
    _, _, test = chronological_split(df)
    return test.reset_index(drop=True)


@st.cache_resource(show_spinner="Training XGBoost (first start only, about 30 seconds)...")
def train_xgboost():
    """Used on Streamlit Cloud, where the .joblib files are not in the repo."""
    from processing.dataset import CATEGORY_TARGET as CAT, split_features_target
    df = load_featured_data()
    train_df, val_df, _ = chronological_split(df)
    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)
    result = model.boosting.train(dict(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        y_train_cat=train_df[CAT].astype(str), y_val_cat=val_df[CAT].astype(str)))
    return {"regressor": result["regressor"], "classifier": result["classifier"]}


@st.cache_resource
def load_model(name):
    path = MODELS_DIR / f"{name}.joblib"
    if path.exists():
        return joblib.load(path)
    return train_xgboost()


@st.cache_data
def load_predictions(name):
    path = OUTPUT / "predictions" / f"predictions_{name}.csv"
    return pd.read_csv(path, parse_dates=[TIME_COL]) if path.exists() else None


@st.cache_resource
def shap_explainer(name):
    import shap
    return shap.TreeExplainer(load_model(name)["regressor"])


def predict_categories(classifier, X):
    """Category names as a pandas Series, whatever the model returns."""
    return pd.Series(np.asarray(classifier.predict(X)), index=X.index).astype(str)


def badge(category):
    colour = EAQI_COLOURS.get(category, "#999999")
    text = "#FFFFFF" if category in ("Poor", "Very Poor", "Extremely Poor") else "#14213D"
    return (f"<span style='background:{colour};color:{text};padding:6px 16px;"
            f"border-radius:999px;font-weight:600;font-size:1.1rem'>{category}</span>")


# Use saved models when they exist; otherwise train XGBoost once on startup.
available = [m for m in MODEL_LABELS if (MODELS_DIR / f"{m}.joblib").exists()] or ["xgboost"]

test = load_test_set()
features = feature_columns(test)

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Kleen-Air")
st.sidebar.caption("TEB3123 Machine Learning · Group 2")
model_name = st.sidebar.selectbox("Model", available,
                                  format_func=lambda m: MODEL_LABELS[m])
models = load_model(model_name)

mode = st.sidebar.radio("Input", ["Pick an hour from the test set", "Enter values manually"])
high_risk_rows = test.index[test[CATEGORY_TARGET].astype(str).isin(HIGH_RISK_CATEGORIES)]
# Start on a high-risk hour the model catches, so the demo opens on an alert.
hits = high_risk_rows[predict_categories(models["classifier"], test.loc[high_risk_rows, features])
                      .isin(HIGH_RISK_CATEGORIES).to_numpy()]
if "row" not in st.session_state:
    st.session_state["row"] = int(hits[0]) if len(hits) else int(high_risk_rows[0])
if st.sidebar.button("Jump to a random high-risk hour"):
    st.session_state["row"] = int(np.random.choice(high_risk_rows))
row_idx = st.sidebar.slider("Test-set hour", 0, len(test) - 1, key="row")
row = test.loc[[row_idx]]
st.sidebar.write(f"**{row[TIME_COL].iloc[0]:%d %b %Y, %H:%M}**")

X = row[features].copy()
if mode == "Enter values manually":
    with st.sidebar.expander("Edit inputs", expanded=True):
        for col in features:
            X[col] = st.number_input(col, value=float(X[col].iloc[0]), format="%.2f")

# ---------------------------------------------------------------- header
st.title("Next-hour PM2.5 forecast and air quality alert")
st.caption("Open-Meteo air quality and weather data, one location in Europe, 2016–2026. "
           "Test period: Apr 2025 – Oct 2026 (13,151 hours).")

tab_live, tab_results, tab_alerts = st.tabs(["Live forecast", "Model results", "Alert log"])

# ---------------------------------------------------------------- live forecast
with tab_live:
    pred_pm = float(models["regressor"].predict(X)[0])
    pred_cat = predict_categories(models["classifier"], X).iloc[0]
    is_high = pred_cat in HIGH_RISK_CATEGORIES

    if is_high:
        st.error(f"⚠️ HIGH-RISK ALERT: next hour predicted **{pred_cat}**. "
                 "Notify schools and clinics; consider traffic measures.")
    else:
        st.success(f"No alert: next hour predicted {pred_cat}.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current PM2.5", f"{X[PM25].iloc[0]:.1f} µg/m³")
    c2.metric("Predicted next hour", f"{pred_pm:.1f} µg/m³",
              f"{pred_pm - X[PM25].iloc[0]:+.1f}")
    if mode == "Pick an hour from the test set":
        c3.metric("Actual next hour", f"{row[REGRESSION_TARGET].iloc[0]:.1f} µg/m³")
        c4.markdown("**Predicted / actual EAQI category**<br>" + badge(pred_cat) + " &nbsp; "
                    + badge(str(row[CATEGORY_TARGET].iloc[0])), unsafe_allow_html=True)
    else:
        c4.markdown("**Predicted category**<br>" + badge(pred_cat), unsafe_allow_html=True)

    st.caption("The category is the official European Air Quality Index, which uses all five "
               "pollutants, so it can be Poor even when PM2.5 alone looks moderate.")

    left, right = st.columns(2)
    with left:
        st.subheader("Why this prediction? (SHAP)")
        if model_name in ("xgboost", "random_forest"):
            values = shap_explainer(model_name).shap_values(X)[0]
            contrib = pd.Series(values, index=features)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:8])[::-1]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(top.index, top.values,
                    color=["#B85A16" if v > 0 else "#17736E" for v in top.values])
            ax.axvline(0, color="#999999", lw=1)
            ax.set_xlabel("Push on predicted PM2.5 (µg/m³)")
            fig.tight_layout()
            st.pyplot(fig)
            st.caption("Orange pushes the forecast up, teal pushes it down.")
        else:
            st.info("SHAP explanation is shown for the tree models (XGBoost, Random Forest).")

    with right:
        st.subheader("Category probabilities")
        proba = models["classifier"].predict_proba(X)[0]
        classes = list(models["classifier"].classes_)
        prob = pd.Series(proba, index=classes)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(prob.index, prob.values, color=[EAQI_COLOURS.get(c, "#999") for c in prob.index],
               edgecolor="#14213D")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        fig.tight_layout()
        st.pyplot(fig)

    preds = load_predictions(model_name)
    if preds is not None and mode == "Pick an hour from the test set":
        st.subheader("Actual vs predicted, 48 hours around this hour")
        window = preds.iloc[max(0, row_idx - 24): row_idx + 24].set_index(TIME_COL)
        st.line_chart(window[["y_true_reg", "y_pred_reg"]].rename(
            columns={"y_true_reg": "Actual", "y_pred_reg": "Predicted"}))

# ---------------------------------------------------------------- results
with tab_results:
    table_path = OUTPUT / "tables" / "model_comparison.csv"
    if table_path.exists():
        table = pd.read_csv(table_path, index_col=0).rename(index=lambda i: MODEL_LABELS.get(
            i, "Naive persistence" if i == "naive_persistence" else i))
        st.subheader("Test-set comparison")
        st.dataframe(table.style.format(lambda v: "–" if pd.isna(v) else f"{v:.4f}")
                     .highlight_min(subset=["RMSE", "MAE"], color="#D3E7E3")
                     .highlight_max(subset=["R2", "Accuracy", "F1_macro"], color="#D3E7E3"),
                     width="stretch")
    figs = [("performance_comparison.png", "Model comparison"),
            ("predictions_overlay.png", "Next-hour PM2.5: actual vs predicted"),
            (f"confusion_{model_name}.png", "Confusion matrix"),
            ("shap_importance_xgboost.png", "Global SHAP importance (XGBoost)")]
    cols = st.columns(2)
    for i, (file, caption) in enumerate(figs):
        if (FIGURES_DIR / file).exists():
            cols[i % 2].image(str(FIGURES_DIR / file), caption=caption, width="stretch")

# ---------------------------------------------------------------- alert log
with tab_alerts:
    preds = load_predictions(model_name)
    if preds is None:
        st.info("No saved predictions for this model. Run `python train.py`.")
    else:
        actual_high = preds["y_true_cat"].isin(HIGH_RISK_CATEGORIES)
        pred_high = preds["y_pred_cat"].isin(HIGH_RISK_CATEGORIES)
        a, b, c = st.columns(3)
        a.metric("High-risk hours in test set", int(actual_high.sum()))
        b.metric("Caught by alerts", int((actual_high & pred_high).sum()),
                 f"{(actual_high & pred_high).sum() / max(actual_high.sum(), 1):.1%} recall")
        c.metric("False alerts", int((~actual_high & pred_high).sum()))
        log = preds.loc[pred_high, [TIME_COL, "y_pred_reg", "y_pred_cat", "y_true_cat"]].rename(
            columns={TIME_COL: "Time", "y_pred_reg": "Predicted PM2.5",
                     "y_pred_cat": "Predicted", "y_true_cat": "Actual"})
        log["Correct"] = log["Actual"].isin(HIGH_RISK_CATEGORIES)
        st.dataframe(log, width="stretch", hide_index=True)
        st.download_button("Download alert log (CSV)", log.to_csv(index=False),
                           file_name=f"alert_log_{model_name}.csv")
