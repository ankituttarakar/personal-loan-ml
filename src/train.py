import os
import joblib
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Import preprocessing function from data_preprocessing module
from data_preprocessing import load_and_preprocess_data


def get_baseline_models(random_state=42):
    """
    Instantiate baseline classification & clustering models matching the reference paper:
    1. Perceptron
    2. Logistic Regression
    3. KMeans (n_clusters=2)
    4. Support Vector Classifier (SVC)
    5. Multi-Layer Perceptron (MLPClassifier)
    """
    models = {
        "perceptron": Perceptron(random_state=random_state),
        "logistic_regression": LogisticRegression(random_state=random_state),
        "kmeans": KMeans(n_clusters=2, random_state=random_state, n_init=10),
        "svc": SVC(random_state=random_state, probability=True),
        "mlp": MLPClassifier(random_state=random_state, max_iter=500)
    }
    return models


def train_and_save_baseline_models(models_dir="models", random_state=42):
    """
    Train baseline models on preprocessed training data and save model artifacts using joblib.
    """
    # Support running script from workspace root or src/ directory
    if not os.path.exists(models_dir):
        alt_dir = os.path.join("..", models_dir)
        if os.path.exists(alt_dir):
            models_dir = alt_dir

    os.makedirs(models_dir, exist_ok=True)

    print("Loading preprocessed dataset...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(random_state=random_state)

    # Save fitted MinMaxScaler artifact
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved MinMaxScaler artifact to: {scaler_path}")

    # Instantiate baseline models matching reference paper
    models = get_baseline_models(random_state=random_state)
    trained_models = {}

    print("\n--- Training Baseline Models (Reference Paper Suite) ---")
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        if model_name == "kmeans":
            # Unsupervised clustering with 2 clusters
            model.fit(X_train)
        else:
            model.fit(X_train, y_train)

        model_file = os.path.join(models_dir, f"{model_name}.joblib")
        joblib.dump(model, model_file)
        print(f"  Saved artifact: {model_file}")

        trained_models[model_name] = model

    print("\nBaseline model training complete. All artifacts saved to models/")
    return trained_models, X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    train_and_save_baseline_models()
