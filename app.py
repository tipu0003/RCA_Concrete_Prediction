#!/usr/bin/env python
# streamlit run app.py
"""
Interactive regression predictor with
• feature labels that include units
• model performance metrics shown with predictions
"""

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import cloudpickle


st.set_page_config(page_title="Model Zoo – Interactive Predictor",
                   page_icon="✨", layout="centered")

# ─────────────────────── paths ─────────────────────────────
MODEL_DIR = Path("saved_models")
IMP_DIR   = Path("feature_importance")

OUTPUT_LABEL = "Predicted Compressive Strength"
OUTPUT_UNIT  = "MPa"              # change to your real target-unit


# ───────────────────── load artefacts ──────────────────────
with open(MODEL_DIR / "scaler.pkl", "rb") as f:
    scaler = cloudpickle.load(f)

model_paths = [p for p in MODEL_DIR.glob("*.pkl") if p.name != "scaler.pkl"]
# models = {p.stem: cloudpickle.load(p) for p in model_paths}
models = {}
for p in model_paths:
    with open(p, "rb") as f:
        models[p.stem] = cloudpickle.load(f)
# Load one importance file only to extract column order
feature_names = list(
    pd.read_csv(IMP_DIR / f"{model_paths[0].stem}_imp.csv")["feature"]
)

# test-set metrics saved by model.py
metrics_df = pd.read_excel(MODEL_DIR / "test_metrics.xlsx").set_index("Model")

# ─────────────── add units for nicer labels ───────────────
# Fill the dict with correct units for every column
# ---- UNITS dictionary (fill in real ones) ----
UNITS = {
    "w/c"        : "-",      # ratio
    "Cement"     : "kg/m³",
    "Curing Days": "days",
    "SP (%)"     : "%",
    "NCA (%)"    : "%",
    "RCA (%)"    : "%",
    "FA"         : "kg/m³"
}

def make_label(col):
    unit = UNITS.get(col, "")
    return f"{col} [{unit}]" if unit else col

display_labels = [make_label(c) for c in feature_names]

print(display_labels)

# ─────────────────────── sidebar ───────────────────────────
st.sidebar.title("🔧 Settings")
model_choice = st.sidebar.selectbox(
    "Choose a model (or predict with **all**):",
    ["All Models"] + sorted(models.keys())
)

# ──────────────────────── title  ───────────────────────────
st.title("📈 Interactive Regression Predictor")
st.markdown(
    "Enter feature values, pick a model, then hit **Predict**. "
    "The app scales the input, returns predictions, plots feature importance, "
    "and shows the model’s test-set accuracy."
)

# ──────────────────── input form ───────────────────────────
with st.form("input_form"):
    cols = st.columns(2)
    user_vals = []
    for idx, label in enumerate(display_labels):
        with cols[idx % 2]:
            val = st.number_input(label, format="%.5f")
            user_vals.append(val)
    submitted = st.form_submit_button("Predict")

# ────────── helper: plot permutation importance ────────────
def plot_importance(model_name):
    path = IMP_DIR / f"{model_name}_imp.csv"
    if not path.exists():
        st.info("Importance not available for this model.")
        return
    imp = pd.read_csv(path).head(20)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1])
    ax.set_xlabel("Permutation Importance")
    ax.set_ylabel("")
    ax.set_title(f"Feature Importance — {model_name}")
    st.pyplot(fig)

# ─────────────────── prediction block ──────────────────────
if submitted:
    X_scaled = scaler.transform(np.array(user_vals).reshape(1, -1))

    if model_choice == "All Models":
        # predictions
        preds = {name: mdl.predict(X_scaled)[0] for name, mdl in models.items()}
        preds_df = pd.DataFrame(preds, index=["Prediction"]).T
        preds_df.rename(columns={"Prediction": f"Prediction ({OUTPUT_UNIT})"},
                        inplace=True)

        # attach metrics
        show_cols = ["R2", "RMSE", "MAE", "MAPE"]
        preds_df = preds_df.join(metrics_df[show_cols])

        st.subheader("Predictions and Accuracy (all models)")
        st.dataframe(preds_df.style.format({"Prediction": "{:.4f}"}))
        st.markdown("---")

        picked = st.selectbox("Select a model to inspect feature importance",
                              sorted(models.keys()))
        plot_importance(picked)

    else:
        mdl = models[model_choice]
        pred = mdl.predict(X_scaled)[0]

        st.subheader(f"Prediction using **{model_choice}**")
        st.metric(OUTPUT_LABEL,
                  f"{pred:,.4f} {OUTPUT_UNIT}")

        # display key metrics
        row = metrics_df.loc[model_choice]
        st.markdown(
            f"**Test-set accuracy** – R² `{row['R2']:.3f}`, "
            f"RMSE `{row['RMSE']:.3f}`, MAE `{row['MAE']:.3f}`, "
            f"MAPE `{row['MAPE']:.2%}`"
        )

        st.markdown("---")
        plot_importance(model_choice)

st.caption("© 2025 Dr. Rupesh Kumar Tipu — Streamlit")
