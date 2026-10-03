# Personal Loan Acceptance Prediction - Methodology Document

## 1. Problem Definition
Commercial banks expend substantial resources marketing personal loans to existing customers. However, personal loan conversion rates are typically low (~10%), making un-targeted marketing campaigns costly and inefficient. The primary objective of this project is to build machine learning models to accurately predict whether a bank customer will accept a personal loan offer based on their demographic, financial, and banking attributes.

## 2. Dataset
The project utilizes the Kaggle **Personal Loan Modeling** dataset (`Bank_Personal_Loan_Modelling.xlsx`, sheet `"Data"`).
- **Total Records:** 5,000 customer records
- **Raw Columns (14):**
  - **Identifier:** `ID` (customer ID)
  - **Input Features (12):** `Age`, `Experience`, `Income`, `ZIP Code`, `Family`, `CCAvg`, `Education`, `Mortgage`, `Securities Account`, `CD Account`, `Online`, `CreditCard`
  - **Target Variable:** `Personal Loan` (binary: 0 = Did not accept, 1 = Accepted)
- **Class Balance:** 4,520 negative instances (90.4%) and 480 positive instances (9.6%).

## 3. Exploratory Data Analysis
Exploratory Data Analysis was conducted in `notebooks/exploration.ipynb`:
- Verified dataset shape (5,000 rows, 14 columns) and feature data types.
- Confirmed zero missing values across all columns.
- Identified invalid negative values in the `Experience` column (e.g., `-3`, `-2`, `-1`).
- Documented severe class imbalance in the `Personal Loan` target (~9.6% positive rate).

## 4. Data Preprocessing
Data preprocessing was implemented in `src/data_preprocessing.py`:
- **Identifier Removal:** The `ID` column was removed as it serves only as a row index and carries no predictive value.
- **Experience Cleaning:** Negative values in the `Experience` column were converted to positive absolute values using `.abs()`.
- **Feature Scaling:** `MinMaxScaler` was applied to scale feature values to the range $[0, 1]$. To prevent data leakage, the scaler was fitted **only on the training data** (`X_train`) and subsequently used to transform both `X_train` and `X_test`.

## 5. Train-Test Split
The dataset was split into an **80% training set** (4,000 records) and a **20% test set** (1,000 records) using `train_test_split` with `random_state=42`.
- **Stratified Splitting:** Stratification on `Personal Loan` was enforced to maintain the ~9.6% positive class ratio in both training (384 positive targets out of 4,000) and testing splits (96 positive targets out of 1,000).

## 6. Baseline Models
Five baseline models matching the reference paper were implemented in `src/train.py`:
1. **Perceptron:** Classic linear classifier.
2. **Logistic Regression:** Standard linear probabilistic classifier.
3. **KMeans:** Unsupervised clustering algorithm ($k=2$).
4. **Support Vector Classifier (SVC):** Non-linear margin-based classifier.
5. **Multi-Layer Perceptron (MLPClassifier):** Neural network classifier.

All models and the fitted `MinMaxScaler` were saved to `models/` using `joblib`.

## 7. Baseline Evaluation
Baseline evaluation was implemented in `src/evaluate.py`. For KMeans, cluster assignments (0, 1) were deterministically mapped to target class labels based on the positive target rate in `X_train`, using negative cluster distance as the continuous decision score.

### Baseline Evaluation Results:
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Perceptron** | 0.9210 | 0.5702 | 0.7188 | 0.6359 | 0.9471 |
| **Logistic Regression** | 0.9530 | 0.8551 | 0.6146 | 0.7152 | 0.9626 |
| **KMeans** | 0.4260 | 0.0935 | 0.5729 | 0.1608 | 0.3610 |
| **SVC** | 0.9760 | 0.9737 | 0.7708 | 0.8605 | 0.9902 |
| **MLPClassifier** | 0.9850 | 0.9175 | 0.9271 | 0.9223 | 0.9956 |

## 8. Class Imbalance Problem
The dataset exhibits a ~90:10 class imbalance. Standard classifiers evaluated on imbalanced data tend to bias predictions toward the majority class, leading to lower recall for the minority positive class. In personal loan marketing, missing potential loan acceptors (False Negatives) represents lost revenue opportunity.

## 9. Proposed Improvement using SMOTE
To address class imbalance, Synthetic Minority Over-sampling Technique (SMOTE) was implemented in `src/train_improved.py`:
- **Strict Oversampling Scope:** SMOTE was applied **ONLY to `X_train` and `y_train`**. It was **never** applied to `X_test` or `y_test` to prevent synthetic data leakage into test set evaluation.
- **Resampling Impact:** Balanced the positive class count in training data from 384 (9.6%) to 3,616 (50.0%).
- **Model Training:** An `MLPClassifier` was trained on the SMOTE-balanced training set and saved as `models/mlp_smote.joblib`.

## 10. SMOTE + MLP Evaluation
The SMOTE-trained MLPClassifier was evaluated on the exact same 20% test set (`X_test`, `y_test`) in `src/evaluate.py`.

### Comparison: Baseline vs. SMOTE MLP
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MLPClassifier (Baseline)** | 0.9850 | 0.9175 | 0.9271 | 0.9223 | 0.9956 |
| **MLPClassifier (SMOTE)** | 0.9850 | 0.8857 | **0.9688** | **0.9254** | 0.9950 |

*Observations:* Applying SMOTE increased **Recall from 92.71% to 96.88%** (capturing 93 out of 96 loan acceptors on the test set vs. 89 for baseline MLP) and slightly improved F1-score (0.9254 vs. 0.9223), with a slight reduction in Precision (88.57% vs. 91.75%).

## 11. Evaluation Metrics
The project evaluated models across six core criteria:
- **Accuracy:** Overall prediction correctness.
- **Precision:** Ratio of true loan acceptors among all predicted loan acceptors.
- **Recall:** Proportion of actual loan acceptors correctly identified by the model.
- **F1-Score:** Harmonic mean of Precision and Recall.
- **ROC-AUC:** Area under Receiver Operating Characteristic curve across classification thresholds.
- **Confusion Matrix:** Detailed breakdown of True Positives, False Positives, True Negatives, and False Negatives.

## 12. Limitations
- **Dataset Scope:** Evaluation is limited to the single Kaggle Personal Loan Modeling dataset (5,000 records).
- **Default Hyperparameters:** Baseline and SMOTE models currently use standard default hyperparameters without automated grid search or hyperparameter tuning.
- **Precision-Recall Trade-off:** SMOTE oversampling enhances loan capture (Recall) but introduces slightly more false positives (lower Precision). Model choice depends on bank business priorities—whether the primary goal is capturing every potential loan customer or minimizing unnecessary marketing outreach.
