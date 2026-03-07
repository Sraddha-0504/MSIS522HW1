#!/usr/bin/env python3
"""Train HW1 models and export artifacts for the Streamlit app."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import tensorflow as tf
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models and export artifacts for HW1")
    parser.add_argument(
        "--data-path",
        default="covid.csv",
        help="Path to the CSV dataset (default: covid.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Output directory for all exported artifacts (default: artifacts)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Rows per class for balanced sampling (default: 5000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs for MLP (default: 30)",
    )
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": output_dir,
        "models": output_dir / "models",
        "plots": output_dir / "plots",
        "meta": output_dir / "meta",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def load_and_sample(data_path: Path, sample_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Place your dataset there before running training."
        )

    raw_data = pd.read_csv(data_path, usecols=lambda c: c != "Unnamed: 0")

    if "DEATH" not in raw_data.columns:
        raise ValueError("Dataset must contain DEATH column.")

    died = raw_data[raw_data["DEATH"] == 1]
    lived = raw_data[raw_data["DEATH"] == 0]

    n = min(sample_size, len(died), len(lived))
    if n == 0:
        raise ValueError("Unable to sample classes because one class has zero rows.")

    sampled = pd.concat(
        [
            died.sample(n=n, random_state=RANDOM_STATE),
            lived.sample(n=n, random_state=RANDOM_STATE),
        ]
    ).sample(frac=1.0, random_state=RANDOM_STATE)

    return raw_data, sampled.reset_index(drop=True)


def build_feature_schema(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    schema: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        s = df[col]
        unique_values = sorted(s.dropna().unique().tolist())
        is_binary = set(unique_values).issubset({0, 1}) and len(unique_values) <= 2
        schema[col] = {
            "type": "binary" if is_binary else "numeric",
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
        }
    return schema


def eval_classification(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }


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


def save_descriptive_plots(raw_data: pd.DataFrame, sampled: pd.DataFrame, plot_dir: Path) -> None:
    # Target distribution (raw)
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x="DEATH", data=raw_data, palette="Set2")
    plt.bar_label(ax.containers[0])
    plt.title("Target Distribution (Raw Data)")
    plt.xlabel("DEATH (0=Lived, 1=Died)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plot_dir / "target_distribution_raw.png", dpi=200)
    plt.close()

    # Age histogram (sampled)
    plt.figure(figsize=(8, 5))
    sns.histplot(sampled["AGE"], bins=25, kde=True, color="teal")
    plt.title("Age Distribution (Sampled Data)")
    plt.xlabel("AGE")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plot_dir / "age_histogram.png", dpi=200)
    plt.close()

    # Age boxplot vs target
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="DEATH", y="AGE", data=sampled, palette="viridis")
    plt.title("Age by Mortality")
    plt.xlabel("DEATH")
    plt.ylabel("AGE")
    plt.tight_layout()
    plt.savefig(plot_dir / "age_boxplot_by_death.png", dpi=200)
    plt.close()

    # Comorbidity mortality in COVID positive population
    covid_positive = sampled[sampled["COVID_POSITIVE"] == 1].copy()
    comorbidity_cols = [
        "DIABETES",
        "COPD",
        "ASTHMA",
        "IMMUNOSUPPRESSION",
        "HYPERTENSION",
        "CARDIOVASCULAR",
        "RENAL_CHRONIC",
        "OTHER_DISEASE",
        "OBESITY",
        "TOBACCO",
    ]
    mortality_rates = {}
    for col in comorbidity_cols:
        group = covid_positive[covid_positive[col] == 1]
        mortality_rates[col] = float(group["DEATH"].mean() * 100) if len(group) else 0.0

    mortality_df = pd.DataFrame(
        list(mortality_rates.items()), columns=["Comorbidity", "Mortality Rate (%)"]
    ).sort_values("Mortality Rate (%)", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=mortality_df, x="Mortality Rate (%)", y="Comorbidity", palette="Reds_d")
    plt.title("Mortality Rate by Comorbidity (COVID Positive)")
    plt.tight_layout()
    plt.savefig(plot_dir / "mortality_by_comorbidity.png", dpi=200)
    plt.close()

    # Age group x hospitalization
    age_bins = [0, 18, 40, 60, 80, 120]
    age_labels = ["0-17", "18-39", "40-59", "60-79", "80+"]
    covid_positive = covid_positive.copy()
    covid_positive["Age_Group"] = pd.cut(
        covid_positive["AGE"], bins=age_bins, labels=age_labels, right=False
    )

    age_hosp = (
        covid_positive.groupby(["Age_Group", "HOSPITALIZED"])["DEATH"]
        .mean()
        .unstack(fill_value=0)
    )

    ax = age_hosp.plot(kind="bar", figsize=(10, 6), colormap="viridis")
    ax.set_title("Mortality by Age Group and Hospitalization")
    ax.set_ylabel("Mortality Rate")
    ax.set_xlabel("Age Group")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(plot_dir / "mortality_age_hospitalized.png", dpi=200)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(sampled.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(plot_dir / "correlation_heatmap.png", dpi=200)
    plt.close()


def train_models(sampled: pd.DataFrame, model_dir: Path, meta_dir: Path, epochs: int) -> Dict[str, Any]:
    X = sampled.drop(columns=["DEATH"])
    y = sampled["DEATH"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    schema = build_feature_schema(X_train)
    with (meta_dir / "feature_schema.json").open("w") as f:
        json.dump(schema, f, indent=2)
    with (meta_dir / "model_input_columns.json").open("w") as f:
        json.dump(list(X_train.columns), f, indent=2)

    X_train.head(500).to_csv(meta_dir / "x_train_reference.csv", index=False)

    artifacts: Dict[str, Any] = {
        "metrics": {},
        "best_params": {},
    }

    # 1) Logistic Regression baseline
    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    logistic.fit(X_train, y_train)
    y_pred = logistic.predict(X_test)
    y_proba = logistic.predict_proba(X_test)[:, 1]
    artifacts["metrics"]["Logistic Regression"] = eval_classification(y_test, y_pred, y_proba)
    artifacts["best_params"]["Logistic Regression"] = {
        "baseline": True,
        "scaled": True,
    }
    joblib.dump(logistic, model_dir / "logistic_pipeline.joblib")

    # 2) Decision Tree with CV
    dt_grid = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid={
            "max_depth": [3, 5, 7, 10],
            "min_samples_leaf": [5, 10, 20, 50],
        },
        scoring="f1",
        cv=5,
        # Use a single process to avoid semaphore restrictions in some sandboxed environments.
        n_jobs=1,
        verbose=1,
    )
    dt_grid.fit(X_train, y_train)
    best_dt = dt_grid.best_estimator_
    y_pred = best_dt.predict(X_test)
    y_proba = best_dt.predict_proba(X_test)[:, 1]
    artifacts["metrics"]["Decision Tree"] = eval_classification(y_test, y_pred, y_proba)
    artifacts["best_params"]["Decision Tree"] = dt_grid.best_params_
    joblib.dump(best_dt, model_dir / "decision_tree.joblib")

    plt.figure(figsize=(16, 8))
    plot_tree(best_dt, feature_names=X_train.columns, class_names=["Lived", "Died"], filled=True)
    plt.title("Best Decision Tree")
    plt.tight_layout()
    plt.savefig(model_dir.parent / "plots" / "best_decision_tree.png", dpi=200)
    plt.close()

    # 3) Random Forest with CV
    rf_grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid={
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 8],
        },
        scoring="f1",
        cv=5,
        n_jobs=1,
        verbose=1,
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    y_pred = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)[:, 1]
    artifacts["metrics"]["Random Forest"] = eval_classification(y_test, y_pred, y_proba)
    artifacts["best_params"]["Random Forest"] = rf_grid.best_params_
    joblib.dump(best_rf, model_dir / "random_forest.joblib")

    # 4) LightGBM with CV
    lgbm_grid = GridSearchCV(
        estimator=LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
        param_grid={
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1],
        },
        scoring="f1",
        cv=5,
        n_jobs=1,
        verbose=1,
    )
    lgbm_grid.fit(X_train, y_train)
    best_lgbm = lgbm_grid.best_estimator_
    y_pred = best_lgbm.predict(X_test)
    y_proba = best_lgbm.predict_proba(X_test)[:, 1]
    artifacts["metrics"]["LightGBM"] = eval_classification(y_test, y_pred, y_proba)
    artifacts["best_params"]["LightGBM"] = lgbm_grid.best_params_
    joblib.dump(best_lgbm, model_dir / "lightgbm.joblib")

    # 5) Neural Network (Keras MLP)
    nn_model = keras.Sequential(
        [
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    nn_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    history = nn_model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=256,
        verbose=0,
    )

    nn_proba = nn_model.predict(X_test, verbose=0).ravel()
    nn_pred = (nn_proba >= 0.5).astype(int)
    artifacts["metrics"]["Neural Network"] = eval_classification(y_test, nn_pred, nn_proba)
    artifacts["best_params"]["Neural Network"] = {
        "architecture": "128-128-ReLU",
        "optimizer": "Adam",
        "epochs": epochs,
        "batch_size": 256,
    }
    nn_model.save(model_dir / "keras_mlp.keras")

    # Save NN learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("MLP Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("MLP Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(model_dir.parent / "plots" / "mlp_training_history.png", dpi=200)
    plt.close()

    # Save split for reproducibility
    X_test.to_csv(meta_dir / "x_test.csv", index=False)
    y_test.to_csv(meta_dir / "y_test.csv", index=False)

    return {
        "artifacts": artifacts,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "models": {
            "Decision Tree": best_dt,
            "Random Forest": best_rf,
            "LightGBM": best_lgbm,
        },
    }


def save_model_comparison(artifacts: Dict[str, Any], output_dir: Path) -> pd.DataFrame:
    rows = []
    for model_name, m in artifacts["metrics"].items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": m["accuracy"],
                "AUC-ROC": m["auc_roc"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1 Score": m["f1"],
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
    comparison_df.to_csv(output_dir / "meta" / "model_comparison.csv", index=False)

    # F1 comparison bar
    plt.figure(figsize=(9, 5))
    sns.barplot(data=comparison_df, x="Model", y="F1 Score", palette="Set2")
    plt.xticks(rotation=20)
    plt.title("Model Comparison: F1 Score")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "model_comparison_f1.png", dpi=200)
    plt.close()

    return comparison_df


def save_roc_overlay(artifacts: Dict[str, Any], plot_dir: Path) -> None:
    colors = {
        "Logistic Regression": "steelblue",
        "Decision Tree": "darkorange",
        "Random Forest": "forestgreen",
        "LightGBM": "crimson",
        "Neural Network": "purple",
    }

    plt.figure(figsize=(9, 7))
    for model_name, metrics in artifacts["metrics"].items():
        fpr = metrics["roc_curve"]["fpr"]
        tpr = metrics["roc_curve"]["tpr"]
        auc = metrics["auc_roc"]
        plt.plot(fpr, tpr, color=colors.get(model_name, None), lw=2, label=f"{model_name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=1.5)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / "roc_overlay.png", dpi=200)
    plt.close()


def save_shap_outputs(
    artifacts: Dict[str, Any],
    tree_models: Dict[str, Any],
    X_test: pd.DataFrame,
    plot_dir: Path,
    meta_dir: Path,
) -> None:
    # Use best tree-based model by F1
    tree_names = ["Decision Tree", "Random Forest", "LightGBM"]
    best_tree_name = max(tree_names, key=lambda n: artifacts["metrics"][n]["f1"])
    best_tree_model = tree_models[best_tree_name]

    sample = X_test.sample(min(1000, len(X_test)), random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(best_tree_model)
    shap_values = positive_class_shap_values(explainer.shap_values(sample))

    shap.summary_plot(shap_values, sample, show=False)
    plt.title(f"SHAP Summary Plot ({best_tree_name})")
    plt.tight_layout()
    plt.savefig(plot_dir / "shap_summary.png", dpi=200)
    plt.close()

    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.title(f"SHAP Mean |Value| ({best_tree_name})")
    plt.tight_layout()
    plt.savefig(plot_dir / "shap_bar.png", dpi=200)
    plt.close()

    # Waterfall for highest-risk record in test set
    proba = best_tree_model.predict_proba(X_test)[:, 1]
    idx = int(np.argmax(proba))
    one_row = X_test.iloc[[idx]]

    one_shap = positive_class_shap_values(explainer.shap_values(one_row))[0]
    base_val = positive_class_base_value(explainer.expected_value)

    explanation = shap.Explanation(
        values=one_shap,
        base_values=base_val,
        data=one_row.iloc[0].values,
        feature_names=one_row.columns.tolist(),
    )

    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(plot_dir / "shap_waterfall_example.png", dpi=200)
    plt.close()

    with (meta_dir / "best_tree_model.txt").open("w") as f:
        f.write(best_tree_name)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    paths = ensure_dirs(output_dir)

    raw_data, sampled = load_and_sample(Path(args.data_path), args.sample_size)

    save_descriptive_plots(raw_data=raw_data, sampled=sampled, plot_dir=paths["plots"])

    training_output = train_models(
        sampled=sampled,
        model_dir=paths["models"],
        meta_dir=paths["meta"],
        epochs=args.epochs,
    )

    comparison_df = save_model_comparison(training_output["artifacts"], output_dir)
    save_roc_overlay(training_output["artifacts"], paths["plots"])
    save_shap_outputs(
        artifacts=training_output["artifacts"],
        tree_models=training_output["models"],
        X_test=training_output["X_test"],
        plot_dir=paths["plots"],
        meta_dir=paths["meta"],
    )

    summary = {
        "random_state": RANDOM_STATE,
        "raw_rows": int(len(raw_data)),
        "sampled_rows": int(len(sampled)),
        "n_features": int(sampled.drop(columns=["DEATH"]).shape[1]),
        "class_balance_sampled": sampled["DEATH"].value_counts(normalize=True).to_dict(),
        "best_params": training_output["artifacts"]["best_params"],
        "metrics": training_output["artifacts"]["metrics"],
        "best_model_by_f1": comparison_df.iloc[0]["Model"],
    }

    with (paths["meta"] / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (paths["meta"] / "best_params.json").open("w") as f:
        json.dump(training_output["artifacts"]["best_params"], f, indent=2)

    print("Training and export completed.")
    print(f"Artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
