import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Import preprocessing pipeline
from data_preprocessing import load_and_preprocess_data


def get_kmeans_predictions_and_scores(kmeans, X_train, y_train, X_test):
    """
    Deterministically convert KMeans cluster assignments (0, 1) into predicted class labels.
    The cluster index with the higher proportion of positive targets (y=1) in X_train is assigned as class 1.
    Continuous scores for ROC-AUC are computed as negative distance to the positive cluster centroid.
    """
    train_clusters = kmeans.predict(X_train)

    c0_pos_rate = y_train[train_clusters == 0].mean() if (train_clusters == 0).sum() > 0 else 0
    c1_pos_rate = y_train[train_clusters == 1].mean() if (train_clusters == 1).sum() > 0 else 0

    positive_cluster = 1 if c1_pos_rate > c0_pos_rate else 0

    test_clusters = kmeans.predict(X_test)
    y_pred = (test_clusters == positive_cluster).astype(int)

    # Shorter distance to positive cluster centroid -> higher probability score
    distances = kmeans.transform(X_test)
    y_scores = -distances[:, positive_cluster]

    return y_pred, y_scores


def get_model_predictions_and_scores(model, model_key, X_train, y_train, X_test):
    """
    Obtain binary predictions (y_pred) and continuous probability/decision scores (y_score).
    """
    if model_key == "kmeans":
        return get_kmeans_predictions_and_scores(model, X_train, y_train, X_test)

    y_pred = model.predict(X_test)

    y_score = None
    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X_test)[:, 1]
        except Exception:
            pass
    elif hasattr(model, "decision_function"):
        try:
            y_score = model.decision_function(X_test)
        except Exception:
            pass

    return y_pred, y_score


def evaluate_baseline_models(
    models_dir="models",
    metrics_dir="results/metrics",
    figures_dir="results/figures",
    random_state=42
):
    """
    Load saved baseline models, evaluate performance metrics on X_test,
    save baseline_metrics.csv, and save confusion matrix figures.
    """
    # Support running script from workspace root or src/ directory
    if not os.path.exists(models_dir) and os.path.exists(os.path.join("..", models_dir)):
        models_dir = os.path.join("..", models_dir)
        metrics_dir = os.path.join("..", metrics_dir)
        figures_dir = os.path.join("..", figures_dir)

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("Loading preprocessed dataset for baseline evaluation...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(random_state=random_state)

    model_registry = [
        ("Perceptron", "perceptron.joblib", "perceptron"),
        ("Logistic Regression", "logistic_regression.joblib", "logistic_regression"),
        ("KMeans", "kmeans.joblib", "kmeans"),
        ("SVC", "svc.joblib", "svc"),
        ("MLPClassifier", "mlp.joblib", "mlp")
    ]

    metrics_list = []

    for display_name, file_name, model_key in model_registry:
        file_path = os.path.join(models_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: Artifact '{file_name}' not found in {models_dir}. Skipping.")
            continue

        print(f"Evaluating {display_name}...")
        model = joblib.load(file_path)

        y_pred, y_score = get_model_predictions_and_scores(model, model_key, X_train, y_train, X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        auc = np.nan
        if y_score is not None:
            try:
                auc = roc_auc_score(y_test, y_score)
            except Exception as err:
                print(f"  Note: Could not calculate ROC-AUC for {display_name}: {err}")

        metrics_list.append({
            "Model": display_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

        # Generate and save Confusion Matrix figure
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["No Loan (0)", "Loan (1)"],
                    yticklabels=["No Loan (0)", "Loan (1)"])
        plt.title(f"Confusion Matrix - {display_name}", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=10)
        plt.ylabel("True Label", fontsize=10)
        plt.tight_layout()

        fig_file = os.path.join(figures_dir, f"{model_key}_cm.png")
        plt.savefig(fig_file, dpi=300)
        plt.close()
        print(f"  Saved confusion matrix plot: {fig_file}")

    # Build DataFrame, save to CSV, and print terminal table
    metrics_df = pd.DataFrame(metrics_list)
    csv_file = os.path.join(metrics_dir, "baseline_metrics.csv")
    metrics_df.to_csv(csv_file, index=False)
    print(f"\nSaved baseline evaluation table to: {csv_file}")

    print("\n=========================================================================")
    print("                    BASELINE MODEL EVALUATION SUMMARY                    ")
    print("=========================================================================")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=========================================================================\n")

    return metrics_df


def evaluate_improved_model(
    models_dir="models",
    metrics_dir="results/metrics",
    figures_dir="results/figures",
    random_state=42
):
    """
    Load saved models/mlp_smote.joblib model, evaluate on X_test/y_test,
    save results/metrics/improved_metrics.csv, and save results/figures/mlp_smote_cm.png.
    """
    if not os.path.exists(models_dir) and os.path.exists(os.path.join("..", models_dir)):
        models_dir = os.path.join("..", models_dir)
        metrics_dir = os.path.join("..", metrics_dir)
        figures_dir = os.path.join("..", figures_dir)

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    smote_model_path = os.path.join(models_dir, "mlp_smote.joblib")
    if not os.path.exists(smote_model_path):
        raise FileNotFoundError(f"SMOTE model artifact not found at {smote_model_path}")

    print("Loading preprocessed dataset for improved model evaluation...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(random_state=random_state)

    print("Evaluating MLPClassifier (SMOTE)...")
    mlp_smote = joblib.load(smote_model_path)

    y_pred, y_score = get_model_predictions_and_scores(mlp_smote, "mlp_smote", X_train, y_train, X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_score) if y_score is not None else np.nan

    improved_metrics = [{
        "Model": "MLPClassifier (SMOTE)",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    }]

    improved_df = pd.DataFrame(improved_metrics)

    # Save to results/metrics/improved_metrics.csv
    csv_file = os.path.join(metrics_dir, "improved_metrics.csv")
    improved_df.to_csv(csv_file, index=False)
    print(f"Saved improved model evaluation table to: {csv_file}")

    # Generate and save Confusion Matrix figure
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Loan (0)", "Loan (1)"],
                yticklabels=["No Loan (0)", "Loan (1)"])
    plt.title("Confusion Matrix - MLPClassifier (SMOTE)", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)
    plt.tight_layout()

    fig_file = os.path.join(figures_dir, "mlp_smote_cm.png")
    plt.savefig(fig_file, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot: {fig_file}")

    print("\n=========================================================================")
    print("                    IMPROVED MODEL (SMOTE) SUMMARY                      ")
    print("=========================================================================")
    print(improved_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=========================================================================\n")

    return improved_df


if __name__ == "__main__":
    evaluate_baseline_models()
    evaluate_improved_model()
