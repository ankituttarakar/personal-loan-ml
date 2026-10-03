import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data(file_path="data/Bank_Personal_Loan_Modelling.xlsx", sheet_name="Data"):
    """
    Load dataset from Excel file.
    Supports resolution from both workspace root and src/ directory.
    """
    if not os.path.exists(file_path):
        alt_path = os.path.join("..", file_path)
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset file not found at '{file_path}' or '{alt_path}'.")

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df


def clean_experience(df):
    """
    Investigate and handle negative values in the 'Experience' column (e.g. -3, -2, -1).
    Negative experience values are replaced with their absolute value.
    """
    df = df.copy()
    if 'Experience' in df.columns:
        neg_count = (df['Experience'] < 0).sum()
        if neg_count > 0:
            # Convert negative experience values to positive absolute values
            df['Experience'] = df['Experience'].abs()
    return df


def preprocess_data(df, target_col="Personal Loan", drop_cols=None, test_size=0.2, random_state=42):
    """
    Preprocess Personal Loan Modeling dataset:
    1. Remove identifier column 'ID'
    2. Clean negative values in 'Experience' column
    3. Separate feature matrix X and target vector y ('Personal Loan')
    4. Perform stratified 80/20 train/test split
    5. Fit MinMaxScaler ONLY on training data, then transform X_train and X_test
    
    Returns:
        X_train_scaled (pd.DataFrame): Scaled 80% training feature set
        X_test_scaled (pd.DataFrame): Scaled 20% test feature set
        y_train (pd.Series): Training target labels
        y_test (pd.Series): Test target labels
        scaler (MinMaxScaler): Fitted MinMaxScaler object
    """
    if drop_cols is None:
        drop_cols = ["ID"]

    # Drop identifier columns if present
    cols_to_drop = [col for col in drop_cols if col in df.columns]
    df_cleaned = df.drop(columns=cols_to_drop)

    # Clean negative values in Experience column
    df_cleaned = clean_experience(df_cleaned)

    # Separate target y and feature matrix X
    if target_col not in df_cleaned.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")

    X = df_cleaned.drop(columns=[target_col])
    y = df_cleaned[target_col]
    feature_names = list(X.columns)

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit MinMaxScaler on training data ONLY, then transform both train and test sets
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Return as DataFrames to retain feature column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)

    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler


def load_and_preprocess_data(file_path="data/Bank_Personal_Loan_Modelling.xlsx", sheet_name="Data", test_size=0.2, random_state=42):
    """
    Convenience function to load and preprocess the dataset in a single step.
    """
    df = load_data(file_path=file_path, sheet_name=sheet_name)
    return preprocess_data(df, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    print("Initializing Personal Loan data preprocessing pipeline...")
    try:
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
        print("\n=== Preprocessing Pipeline Execution Successful ===")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape : {X_test.shape}")
        print(f"y_train shape: {y_train.shape} (Positive target count: {y_train.sum()} / {len(y_train)})")
        print(f"y_test shape : {y_test.shape} (Positive target count: {y_test.sum()} / {len(y_test)})")
        print(f"Features scaled: {scaler.n_features_in_}")
        print(f"Scaled range  : [{X_train.min().min():.2f}, {X_train.max().max():.2f}]")
    except FileNotFoundError as err:
        print(f"\nModule compiled cleanly. Data notice: {err}")
