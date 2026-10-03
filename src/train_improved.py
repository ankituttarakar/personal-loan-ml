import os
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier

# Import existing preprocessing module
from data_preprocessing import load_and_preprocess_data


def train_improved_model(models_dir="models", random_state=42):
    """
    Train an improved MLPClassifier addressing class imbalance using SMOTE.
    SMOTE is applied ONLY to X_train and y_train.
    The resulting model is saved as models/mlp_smote.joblib.
    """
    # Support running script from workspace root or src/ directory
    if not os.path.exists(models_dir):
        alt_dir = os.path.join("..", models_dir)
        if os.path.exists(alt_dir):
            models_dir = alt_dir

    os.makedirs(models_dir, exist_ok=True)

    print("Loading preprocessed dataset...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(random_state=random_state)

    print("\n--- Class Distribution Before SMOTE (Training Set) ---")
    print(f"Class 0 (No Loan) : {(y_train == 0).sum()}")
    print(f"Class 1 (Loan)    : {(y_train == 1).sum()}")
    print(f"Positive Ratio   : {(y_train == 1).mean() * 100:.2f}%")

    # Apply SMOTE ONLY to training data (X_train, y_train)
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("\n--- Class Distribution After SMOTE (Training Set) ---")
    print(f"Class 0 (No Loan) : {(y_train_resampled == 0).sum()}")
    print(f"Class 1 (Loan)    : {(y_train_resampled == 1).sum()}")
    print(f"Positive Ratio   : {(y_train_resampled == 1).mean() * 100:.2f}%")

    print("\nTraining MLPClassifier on SMOTE-balanced training data...")
    mlp_smote = MLPClassifier(random_state=random_state, max_iter=500)
    mlp_smote.fit(X_train_resampled, y_train_resampled)

    model_path = os.path.join(models_dir, "mlp_smote.joblib")
    joblib.dump(mlp_smote, model_path)
    print(f"Saved improved model artifact to: {model_path}")

    return mlp_smote, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train_improved_model()
