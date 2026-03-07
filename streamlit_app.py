from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
META_DIR = ARTIFACTS_DIR / "meta"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

st.set_page_config(
    page_title="MSIS 522 HW1 - COVID Mortality Analytics",
    page_icon="🩺",
    layout="wide",
)


def positive_class_shap_values(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[-1])

    arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr[:, :, 1]
    return arr


def positive_class_base_value(expected_value: Any) -> float:
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        if len(expected_value) == 2:
            return float(expected_value[1])
        return float(expected_value[-1])
    return float(expected_value)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "Logistic Regression": joblib.load(MODELS_DIR / "logistic_pipeline.joblib"),
        "Decision Tree": joblib.load(MODELS_DIR / "decision_tree.joblib"),
        "Random Forest": joblib.load(MODELS_DIR / "random_forest.joblib"),
        "LightGBM": joblib.load(MODELS_DIR / "lightgbm.joblib"),
    }

    # TensorFlow model is optional at app-load time; everything else remains usable if unavailable.
    try:
        from tensorflow import keras

        models["Neural Network"] = keras.models.load_model(MODELS_DIR / "keras_mlp.keras")
    except Exception as e:
        st.warning(f"Neural Network model not loaded: {e}")

    return models


def check_artifacts() -> bool:
    required = [
        META_DIR / "summary.json",
        META_DIR / "feature_schema.json",
        META_DIR / "model_comparison.csv",
        META_DIR / "best_params.json",
        META_DIR / "best_tree_model.txt",
        MODELS_DIR / "logistic_pipeline.joblib",
        MODELS_DIR / "decision_tree.joblib",
        MODELS_DIR / "random_forest.joblib",
        MODELS_DIR / "lightgbm.joblib",
        PLOTS_DIR / "target_distribution_raw.png",
        PLOTS_DIR / "correlation_heatmap.png",
        PLOTS_DIR / "roc_overlay.png",
        PLOTS_DIR / "shap_summary.png",
        PLOTS_DIR / "shap_bar.png",
    ]
    return all(p.exists() for p in required)


def make_input_dataframe(feature_schema: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    st.subheader("Set Patient Features")
    st.caption(
        "Choose either a quick clinical form (core features) or a full feature form. "
        "For binary fields, use 0 (No) or 1 (Yes)."
    )

    mode = st.radio(
        "Input mode",
        options=["Core clinical features (recommended)", "All features"],
        horizontal=True,
    )

    core_candidates = [
        "AGE",
        "HOSPITALIZED",
        "PNEUMONIA",
        "DIABETES",
        "HYPERTENSION",
        "CARDIOVASCULAR",
        "RENAL_CHRONIC",
        "OBESITY",
        "COVID_POSITIVE",
        "SEX",
    ]
    core_features = [f for f in core_candidates if f in feature_schema]
    if mode.startswith("Core"):
        edited_features = core_features
        mode_key = "core"
    else:
        edited_features = list(feature_schema.keys())
        mode_key = "all"

    values: Dict[str, float] = {}
    columns = st.columns(2)
    for i, feature in enumerate(edited_features):
        meta = feature_schema[feature]
        col = columns[i % 2]
        with col:
            if meta["type"] == "binary":
                default = int(round(meta["mean"]))
                values[feature] = st.selectbox(
                    feature,
                    options=[0, 1],
                    index=default,
                    key=f"{mode_key}_{feature}_binary",
                )
            else:
                min_val = int(np.floor(meta["min"]))
                max_val = int(np.ceil(meta["max"]))
                default = int(round(meta["mean"]))
                values[feature] = st.slider(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=min(max(default, min_val), max_val),
                    key=f"{mode_key}_{feature}_num",
                )

    # For non-edited features, use sample means as assignment allows.
    for feature, meta in feature_schema.items():
        if feature not in values:
            if meta["type"] == "binary":
                values[feature] = int(round(meta["mean"]))
            else:
                values[feature] = float(meta["mean"])

    if mode.startswith("Core"):
        st.caption(
            "Unedited features are automatically filled with sample-average values. "
            "This keeps prediction interactive while still using the full trained model input schema."
        )

    ordered_columns = list(feature_schema.keys())
    return pd.DataFrame([values])[ordered_columns]


def predict_probability(model_name: str, model: Any, input_df: pd.DataFrame) -> float:
    if model_name == "Neural Network":
        proba = float(model.predict(input_df, verbose=0).ravel()[0])
    else:
        proba = float(model.predict_proba(input_df)[:, 1][0])
    return proba


def render_waterfall(model: Any, input_df: pd.DataFrame, title: str) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = positive_class_shap_values(explainer.shap_values(input_df))[0]
    base_val = positive_class_base_value(explainer.expected_value)

    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_val,
        data=input_df.iloc[0].values,
        feature_names=input_df.columns.tolist(),
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt.gcf(), clear_figure=True)


def plot_metric_bars(comparison_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = axes[0].bar(comparison_df["Model"], comparison_df["F1 Score"], color="teal")
    axes[0].set_title("F1 Score by Model")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].tick_params(axis="x", rotation=20)
    for b in bars1:
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.3f}", ha="center", fontsize=8)

    bars2 = axes[1].bar(comparison_df["Model"], comparison_df["AUC-ROC"], color="slateblue")
    axes[1].set_title("AUC-ROC by Model")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].tick_params(axis="x", rotation=20)
    for b in bars2:
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def plot_individual_rocs(metrics_by_model: Dict[str, Any]) -> None:
    colors = {
        "Logistic Regression": "steelblue",
        "Decision Tree": "darkorange",
        "Random Forest": "forestgreen",
        "LightGBM": "crimson",
        "Neural Network": "purple",
    }
    for model_name, model_metrics in metrics_by_model.items():
        with st.expander(f"ROC Curve - {model_name}", expanded=False):
            fig = plt.figure(figsize=(7, 5))
            fpr = model_metrics["roc_curve"]["fpr"]
            tpr = model_metrics["roc_curve"]["tpr"]
            auc = model_metrics["auc_roc"]
            plt.plot(fpr, tpr, color=colors.get(model_name, "black"), lw=2, label=f"AUC = {auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=1.5)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{model_name} ROC")
            plt.legend(loc="lower right")
            plt.grid(True)
            st.pyplot(fig, clear_figure=True)


def show_part2_outputs(metrics_by_model: Dict[str, Any], best_params: Dict[str, Any]) -> None:
    for model_name, model_metrics in metrics_by_model.items():
        with st.expander(f"Detailed Test Metrics - {model_name}", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Accuracy", f"{model_metrics['accuracy']:.4f}")
                st.metric("Precision", f"{model_metrics['precision']:.4f}")
                st.metric("Recall", f"{model_metrics['recall']:.4f}")
            with c2:
                st.metric("F1 Score", f"{model_metrics['f1']:.4f}")
                st.metric("AUC-ROC", f"{model_metrics['auc_roc']:.4f}")
                st.write("Best hyperparameters:")
                st.json(best_params.get(model_name, {}))

            cm = np.array(model_metrics["confusion_matrix"])
            st.write("Confusion matrix (rows: actual [0,1], columns: predicted [0,1]):")
            st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))


st.title("MSIS 522 HW1: COVID-19 Mortality Prediction")

if not check_artifacts():
    st.error(
        "Missing exported artifacts. Run `python3 train_and_export.py --data-path covid.csv` first, "
        "then reload this app."
    )
    st.stop()

summary = load_json(META_DIR / "summary.json")
feature_schema = load_json(META_DIR / "feature_schema.json")
comparison_df = load_csv(META_DIR / "model_comparison.csv")
best_params = load_json(META_DIR / "best_params.json")
best_tree_model_name = (META_DIR / "best_tree_model.txt").read_text().strip()
metrics_by_model = summary["metrics"]
best_f1_row = comparison_df.loc[comparison_df["F1 Score"].idxmax()]
best_auc_row = comparison_df.loc[comparison_df["AUC-ROC"].idxmax()]
best_recall_row = comparison_df.loc[comparison_df["Recall"].idxmax()]

models = load_models()


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Summary",
        "Descriptive Analytics",
        "Model Performance",
        "Explainability & Interactive Prediction",
    ]
)

with tab1:
    st.header("Dataset and Prediction Task")
    st.markdown(
        """
        This project predicts **COVID-19 patient mortality (`DEATH`)** using a tabular healthcare dataset
        containing over one million anonymized records. The target variable is binary (`DEATH`: 0 = lived,
        1 = died), and the features include demographics (`AGE`, `SEX`), severity signals (`HOSPITALIZED`,
        `PNEUMONIA`, `COVID_POSITIVE`), and comorbidities (`DIABETES`, `HYPERTENSION`, `CARDIOVASCULAR`,
        `RENAL_CHRONIC`, `OBESITY`, and others). In plain terms, this is a risk-stratification problem:
        given a patient profile, estimate the probability of mortality.
        """
    )

    st.header("Why This Matters")
    st.markdown(
        """
        Mortality prediction matters because false negatives can be costly: if a high-risk patient is missed,
        escalation can be delayed. At the same time, too many false positives can overuse limited care resources.
        For healthcare operations, this becomes a triage and resource-allocation problem: hospitals need to
        identify who requires aggressive monitoring first, while avoiding alert fatigue from excessive false alarms.
        A useful model should therefore balance **recall** (catching high-risk patients) and **precision**
        (ensuring flagged cases are credible).
        """
    )

    st.header("Approach and Key Findings")
    st.markdown(
        f"""
        The workflow used a complete modeling pipeline: descriptive analytics, a logistic-regression baseline,
        cross-validated tree models, a neural network benchmark, and SHAP explainability for transparency.
        To keep results reproducible and comparable, we trained with `random_state=42` on a balanced working
        sample of **{summary['sampled_rows']:,} records** (from **{summary['raw_rows']:,} raw records**) and
        **{summary['n_features']} predictors**.

        The model results are tightly clustered but actionable. **{best_f1_row['Model']}** leads on F1
        (**{best_f1_row['F1 Score']:.4f}**), while **{best_auc_row['Model']}** has the highest AUC-ROC
        (**{best_auc_row['AUC-ROC']:.4f}**), and **{best_recall_row['Model']}** has the highest recall
        (**{best_recall_row['Recall']:.4f}**). In practical terms, the top models all identify high-risk
        patients well, with differences mostly in false-positive behavior rather than missed-case behavior.
        """
    )

    st.subheader("Executive Recommendation")
    st.markdown(
        f"""
        For operational deployment, **Random Forest** is a strong default choice in this run because it
        provides the best overall balance of recall and precision with top-tier discrimination
        (F1 **{metrics_by_model['Random Forest']['f1']:.4f}**, AUC **{metrics_by_model['Random Forest']['auc_roc']:.4f}**,
        Recall **{metrics_by_model['Random Forest']['recall']:.4f}**). It also offers better auditability than
        a neural network and can be explained using SHAP for case-level justification.

        Implementation should still include safeguards: treat predictions as decision support, not automated
        diagnosis; monitor drift quarterly; and validate calibration on real prevalence before policy use,
        because training used a balanced sample to improve learning stability.
        """
    )

    st.subheader("Model Snapshot (Test Set)")
    st.dataframe(
        comparison_df[["Model", "F1 Score", "AUC-ROC", "Recall", "Precision", "Accuracy"]],
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.header("Descriptive Analytics")

    st.subheader("Target Distribution")
    st.image(str(PLOTS_DIR / "target_distribution_raw.png"), use_container_width=True)
    st.caption(
        "The raw dataset is strongly imbalanced toward survivors, which is common in clinical outcomes data. "
        "To avoid biased learning, model training used a balanced sample so recall and precision are both meaningful."
    )

    st.subheader("Age Distribution")
    st.image(str(PLOTS_DIR / "age_histogram.png"), use_container_width=True)
    st.caption(
        "Age is right-skewed with concentration in middle and older ranges. "
        "This matters because age contributes strongly to mortality risk in downstream models."
    )

    st.subheader("Age vs Mortality")
    st.image(str(PLOTS_DIR / "age_boxplot_by_death.png"), use_container_width=True)
    st.caption(
        "Patients in the death class show higher median age and broader upper-tail spread. "
        "The chart supports age as a clinically relevant risk factor."
    )

    st.subheader("Comorbidity Risk Patterns")
    st.image(str(PLOTS_DIR / "mortality_by_comorbidity.png"), use_container_width=True)
    st.caption(
        "Comorbidities such as renal chronic disease, cardiovascular issues, and diabetes show higher mortality. "
        "These relationships help clinicians identify vulnerable groups for earlier intervention."
    )

    st.subheader("Age Group and Hospitalization")
    st.image(str(PLOTS_DIR / "mortality_age_hospitalized.png"), use_container_width=True)
    st.caption(
        "Mortality rises with age and is materially higher among hospitalized patients across age bands. "
        "Hospitalization is both a severity signal and a strong predictor in the models."
    )

    st.subheader("Correlation Heatmap")
    st.image(str(PLOTS_DIR / "correlation_heatmap.png"), use_container_width=True)
    st.caption(
        "The heatmap highlights strong relationships between the target and hospitalization, pneumonia, and age. "
        "It also shows clinically plausible co-occurrence patterns among chronic conditions."
    )

with tab3:
    st.header("Model Performance")
    st.caption(
        "This tab surfaces all major Part 2 outputs so model quality can be evaluated without opening code."
    )

    st.subheader("Comparison Table")
    st.dataframe(comparison_df, use_container_width=True)

    st.subheader("Model Comparison Charts (Section 2.7)")
    plot_metric_bars(comparison_df)

    st.subheader("ROC Curves")
    st.image(str(PLOTS_DIR / "roc_overlay.png"), use_container_width=True)
    plot_individual_rocs(metrics_by_model)

    st.subheader("Best Hyperparameters")
    st.json(best_params)

    st.subheader("Detailed Metrics and Confusion Matrices")
    show_part2_outputs(metrics_by_model, best_params)

    st.subheader("Additional Part 2 Visual Outputs")
    if (PLOTS_DIR / "best_decision_tree.png").exists():
        st.image(str(PLOTS_DIR / "best_decision_tree.png"), use_container_width=True)
    if (PLOTS_DIR / "mlp_training_history.png").exists():
        st.image(str(PLOTS_DIR / "mlp_training_history.png"), use_container_width=True)

with tab4:
    st.header("Explainability & Interactive Prediction")

    st.subheader("Global SHAP Explainability")
    st.image(str(PLOTS_DIR / "shap_summary.png"), use_container_width=True)
    st.image(str(PLOTS_DIR / "shap_bar.png"), use_container_width=True)
    st.caption(
        "The SHAP summary plot shows both importance and direction of feature effects, while the bar plot "
        "ranks average absolute contribution size. Together they explain which factors matter most and how they "
        "shift mortality risk."
    )

    st.markdown("---")
    st.subheader("Interactive Prediction")

    selected_model_name = st.selectbox("Select model for prediction", list(models.keys()))
    input_df = make_input_dataframe(feature_schema)

    if st.button("Run Prediction", type="primary"):
        selected_model = models[selected_model_name]
        probability = predict_probability(selected_model_name, selected_model, input_df)
        predicted_class = int(probability >= 0.5)

        st.metric("Predicted Probability of Death", f"{probability:.3f}")
        st.metric("Predicted Class", "Died (1)" if predicted_class == 1 else "Lived (0)")

        st.subheader("SHAP Waterfall for This Custom Input")

        if selected_model_name in {"Decision Tree", "Random Forest", "LightGBM"}:
            tree_model = models[selected_model_name]
            render_waterfall(tree_model, input_df, f"Waterfall ({selected_model_name})")
        else:
            st.info(
                "SHAP waterfall is generated with the best tree-based model for fast and stable local explanation."
            )
            tree_model = models[best_tree_model_name]
            render_waterfall(tree_model, input_df, f"Waterfall ({best_tree_model_name})")

    st.caption(
        "Note: The app loads pre-trained models and does not retrain at runtime, as required by the assignment."
    )
