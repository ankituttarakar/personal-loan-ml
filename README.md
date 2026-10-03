# Personal Loan Acceptance Prediction

## Problem Statement

This project predicts whether a bank customer will accept a personal loan offer using demographic, financial, and banking-related information.

The task is a binary classification problem with an imbalanced target, where only a small proportion of customers accept the loan. Therefore, accuracy alone is not sufficient for evaluating model performance.

## Objective

The project aims to:

- Implement a baseline methodology inspired by the reference paper using our own code.
- Evaluate multiple machine learning algorithms.
- Analyze the effect of class imbalance.
- Apply SMOTE to the training data.
- Train an MLP model using the SMOTE-balanced training data.
- Compare baseline and SMOTE-based models using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
- Analyze the trade-offs introduced by SMOTE.

## Reference Paper

**"Predicting Acceptance of Personal Loans Using Machine Learning Algorithms"**  
AJ Rossman and Conor Casey

### Key Points

- Dataset: Personal Loan Modeling dataset
- Records: 5,000 customers
- Input features: 12
- Target: Personal Loan acceptance
- Class distribution: approximately 90% negative and 10% positive
- Algorithms: Perceptron, Logistic Regression, K-Means, SVC, and MLP
- Evaluation includes accuracy, ROC curves, and AUC
- The paper reports approximately 98% accuracy and 0.99 ROC-AUC for its MLP approach with MinMaxScaler

This project is a **reference-inspired implementation**, not an exact reproduction of the original authors' code.

## Dataset

### Source

**Kaggle - Personal Loan Modeling dataset**

The dataset contains 5,000 customer records with 12 input features and one target variable.

### Dataset Overview

- Records: 5,000
- Raw columns: 14
- Identifier: `ID`
- Input features: 12
- Target: `Personal Loan`
- `0` = Did not accept personal loan
- `1` = Accepted personal loan
- Class 0: 4,520
- Class 1: 480
- Positive class: 9.60%

### Features

- Age
- Experience
- Income
- ZIP Code
- Family
- CCAvg
- Education
- Mortgage
- Securities Account
- CD Account
- Online
- CreditCard

The raw dataset is not stored in this repository. See `data/README.md` for dataset source information.

## Methodology

### 1. Data Preprocessing

Implemented in `src/data_preprocessing.py`.

Steps:

1. Load the Excel dataset from the `Data` sheet.
2. Remove the `ID` identifier column.
3. Convert negative `Experience` values to their absolute values.
4. Separate input features and target.
5. Perform an 80/20 stratified train-test split using `random_state=42`.
6. Fit `MinMaxScaler` only on the training data.
7. Transform both training and test data using the fitted scaler.

Resulting split:

- Training set: 4,000 records
- Test set: 1,000 records
- Training positives: 384
- Test positives: 96

### 2. Baseline Models

Implemented in `src/train.py`.

Models:

1. Perceptron
2. Logistic Regression
3. K-Means
4. Support Vector Classifier (SVC)
5. Multi-Layer Perceptron (MLP)

### 3. Class Imbalance Analysis

Training distribution before SMOTE:

- Class 0: 3,616
- Class 1: 384
- Positive ratio: 9.60%

Because the classes are imbalanced, accuracy alone can be misleading. Precision, recall, F1-score, ROC-AUC, and confusion matrices are also used.

### 4. Proposed Improvement: SMOTE + MLP

Implemented in `src/train_improved.py`.

SMOTE is applied **only to the training data**.

Before SMOTE:

- Class 0: 3,616
- Class 1: 384

After SMOTE:

- Class 0: 3,616
- Class 1: 3,616

An MLPClassifier is trained on the balanced training data.

The original test set remains untouched and is used only for final evaluation.

### 5. Evaluation

Implemented in `src/evaluate.py`.

Metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

## Results

### Baseline Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Perceptron | 92.10% | 57.02% | 71.88% | 63.59% | 94.71% |
| Logistic Regression | 95.30% | 85.51% | 61.46% | 71.52% | 96.26% |
| K-Means | 42.60% | 9.35% | 57.29% | 16.08% | 36.10% |
| SVC | 97.60% | 97.37% | 77.08% | 86.05% | 99.02% |
| MLPClassifier | 98.50% | 91.75% | 92.71% | 92.23% | 99.56% |

The baseline MLP achieved 98.50% accuracy, 92.71% recall, 92.23% F1-score, and 99.56% ROC-AUC on the test set.

### Baseline MLP vs SMOTE + MLP

| Metric | Baseline MLP | SMOTE + MLP |
|---|---:|---:|
| Accuracy | 98.50% | 98.50% |
| Precision | 91.75% | 88.57% |
| Recall | 92.71% | 96.88% |
| F1-Score | 92.23% | 92.54% |
| ROC-AUC | 99.56% | 99.50% |

SMOTE increased recall from 92.71% to 96.88% and F1-score from 92.23% to 92.54%.

Precision decreased from 91.75% to 88.57%, while ROC-AUC decreased slightly from 99.56% to 99.50%.

Therefore, SMOTE improved minority-class detection but introduced a trade-off rather than improving every metric.

### Confusion Matrix Comparison

Baseline MLP:

- TN = 896
- FP = 8
- FN = 7
- TP = 89

SMOTE + MLP:

- TN = 892
- FP = 12
- FN = 3
- TP = 93

SMOTE reduced false negatives from 7 to 3 and increased true positives from 89 to 93, while false positives increased from 8 to 12.

## Sample Inference

Implemented in `src/predict.py`.

The prediction module loads the trained baseline MLP and scaler, applies the same preprocessing, and produces a prediction and probability.

Example output:

`Predicted Class : 1`  
`Loan Acceptance Prob : 99.60%`  
`Prediction Status : Accepted Personal Loan (1)`

This is a sample model inference and not a real banking decision.

## Project Structure

    personal-loan-ml/
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── data/
    │   └── README.md
    ├── docs/
    │   └── methodology.md
    ├── notebooks/
    │   └── exploration.ipynb
    ├── models/
    ├── results/
    │   ├── figures/
    │   └── metrics/
    └── src/
        ├── data_preprocessing.py
        ├── train.py
        ├── train_improved.py
        ├── evaluate.py
        └── predict.py

## Setup

### 1. Clone the Repository

`git clone https://github.com/ankituttarakar/personal-loan-ml.git`

`cd personal-loan-ml`

### 2. Create a Virtual Environment

Windows PowerShell:

`python -m venv .venv`

`.venv\Scripts\activate`

Linux/macOS:

`python3 -m venv .venv`

`source .venv/bin/activate`

### 3. Install Dependencies

`pip install -r requirements.txt`

### 4. Add the Dataset

Download the **Personal Loan Modeling** dataset from Kaggle.

Place the downloaded file:

`Bank_Personal_Loan_Modelling.xlsx`

inside:

`data/`

The Excel sheet used by the project is:

`Data`

See `data/README.md` for the dataset source information.

## Running the Project

Run the following commands from the project root.

### 1. Preprocess the Data

`python src/data_preprocessing.py`

### 2. Train Baseline Models

`python src/train.py`

### 3. Train the SMOTE-Based Model

`python src/train_improved.py`

### 4. Evaluate the Models

`python src/evaluate.py`

### 5. Run Sample Prediction

`python src/predict.py`

## Generated Outputs

Trained model files are generated locally inside `models/`:

- `scaler.joblib`
- `perceptron.joblib`
- `logistic_regression.joblib`
- `kmeans.joblib`
- `svc.joblib`
- `mlp.joblib`
- `mlp_smote.joblib`

Evaluation results are saved in:

- `results/metrics/baseline_metrics.csv`
- `results/metrics/improved_metrics.csv`

Confusion matrices and comparison figures are saved in:

`results/figures/`

Dataset and trained model artifacts are excluded from GitHub using `.gitignore`.

## Limitations

- The evaluation uses one dataset.
- Baseline models use fixed/default hyperparameters.
- SMOTE does not improve every evaluation metric.
- Results may not generalize to other banking datasets.
- K-Means is an unsupervised clustering algorithm and is included because it was part of the reference-paper model suite.
- The project does not represent a real-world banking approval system.
- The SMOTE-trained MLP reached the configured iteration limit during the recorded run, so further hyperparameter tuning could be explored.

## Future Work

Potential extensions include:

- Hyperparameter tuning
- Cross-validation
- Decision-threshold optimization
- Evaluation on additional datasets
- Model explainability techniques
- Testing alternative class-imbalance strategies

## Conclusion

This project implements a complete machine-learning pipeline for personal loan acceptance prediction.

The baseline experiments evaluate five algorithms, with the MLP achieving 98.50% accuracy and 99.56% ROC-AUC on the test set.

SMOTE was then applied only to the training data to address class imbalance. The resulting MLP increased recall from 92.71% to 96.88% and reduced false negatives from 7 to 3, while introducing a precision trade-off.

The results demonstrate why imbalanced classification should be evaluated using multiple metrics rather than accuracy alone.
