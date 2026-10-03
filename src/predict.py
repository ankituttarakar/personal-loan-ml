import os
import joblib
import pandas as pd
import numpy as np


def load_model_and_scaler(model_path="models/mlp.joblib", scaler_path="models/scaler.joblib"):
    """
    Load trained model and fitted scaler artifacts.
    Supports resolution from workspace root or src/ directory.
    """
    if not os.path.exists(model_path):
        alt_model = os.path.join("..", model_path)
        alt_scaler = os.path.join("..", scaler_path)
        if os.path.exists(alt_model) and os.path.exists(alt_scaler):
            model_path = alt_model
            scaler_path = alt_scaler
        else:
            raise FileNotFoundError(f"Model artifact '{model_path}' or Scaler artifact '{scaler_path}' not found.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_personal_loan(customer_data, model_path="models/mlp.joblib", scaler_path="models/scaler.joblib"):
    """
    Predict personal loan acceptance for a new customer.
    
    Parameters:
        customer_data (dict or pd.DataFrame): 12 customer feature attributes.
        model_path (str): Path to saved model artifact.
        scaler_path (str): Path to saved scaler artifact.
        
    Returns:
        dict: Contains predicted class (0 or 1) and loan acceptance probability.
    """
    model, scaler = load_model_and_scaler(model_path=model_path, scaler_path=scaler_path)

    # Format input into a pandas DataFrame
    if isinstance(customer_data, dict):
        df_input = pd.DataFrame([customer_data])
    elif isinstance(customer_data, pd.DataFrame):
        df_input = customer_data.copy()
    else:
        raise ValueError("customer_data must be a dictionary or pandas DataFrame.")

    # List of 12 required features in exact order
    required_features = [
        "Age", "Experience", "Income", "ZIP Code", "Family",
        "CCAvg", "Education", "Mortgage", "Securities Account",
        "CD Account", "Online", "CreditCard"
    ]

    missing_cols = [col for col in required_features if col not in df_input.columns]
    if missing_cols:
        raise ValueError(f"Missing required input features: {missing_cols}")

    # Reorder columns
    df_input = df_input[required_features]

    # Clean negative Experience values if present
    if (df_input["Experience"] < 0).any():
        df_input["Experience"] = df_input["Experience"].abs()

    # Scale input feature vector and preserve DataFrame feature names
    X_scaled_array = scaler.transform(df_input)
    X_scaled = pd.DataFrame(X_scaled_array, columns=required_features)

    # Generate prediction and probability
    pred_class = int(model.predict(X_scaled)[0])
    pred_proba = float(model.predict_proba(X_scaled)[0][1])

    return {
        "prediction": pred_class,
        "probability": pred_proba,
        "status": "Accepted Personal Loan (1)" if pred_class == 1 else "Did Not Accept Personal Loan (0)"
    }


if __name__ == "__main__":
    print("Executing Personal Loan Prediction Module...")

    # Sample test customer feature dictionary
    sample_customer = {
        "Age": 45,
        "Experience": 19,
        "Income": 120,          # $120k annual income
        "ZIP Code": 94720,
        "Family": 3,
        "CCAvg": 3.8,           # $3.8k monthly credit card spend
        "Education": 3,         # Advanced/Professional
        "Mortgage": 0,
        "Securities Account": 1,
        "CD Account": 1,
        "Online": 1,
        "CreditCard": 0
    }

    print("\nSample Customer Profile:")
    for feature, val in sample_customer.items():
        print(f"  {feature:20s}: {val}")

    try:
        result = predict_personal_loan(sample_customer)
        print("\n=======================================================")
        print("                 INFERENCE PREDICTION                   ")
        print("=======================================================")
        print(f"Predicted Class       : {result['prediction']}")
        print(f"Loan Acceptance Prob  : {result['probability'] * 100:.2f}%")
        print(f"Prediction Status     : {result['status']}")
        print("=======================================================\n")
    except FileNotFoundError as err:
        print(f"\nExecution error: {err}")
