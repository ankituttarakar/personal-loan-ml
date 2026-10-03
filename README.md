# Personal Loan Acceptance Prediction Using Machine Learning

## 1. Problem Statement

This project focuses on predicting whether a bank customer will accept a personal loan offer based on demographic, financial, and banking-related information.

The problem is formulated as a **binary classification task**:

- `0` = Customer did not accept the personal loan
- `1` = Customer accepted the personal loan

The dataset is highly imbalanced, with approximately 90% negative samples and 10% positive samples. Therefore, accuracy alone is not sufficient to evaluate model performance.

---

## 2. Objective

The objectives of this project are to:

- Implement a baseline machine learning pipeline inspired by the reference paper.
- Evaluate the reference-paper model suite using our own implementation.
- Analyze the effect of class imbalance on model performance.
- Implement an improved approach using **SMOTE with an MLP classifier**.
- Compare the baseline MLP and SMOTE-based MLP using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
- Demonstrate the final trained model through a sample inference.

---

## 3. Reference Paper

The primary reference for this project is:

> **"Predicting Acceptance of Personal Loans Using Machine Learning Algorithms"**  
> *AJ Rossman and Conor Casey*

The reference paper uses the Personal Loan Modeling dataset and evaluates:

- Perceptron
- Logistic Regression
- K-Means
- Support Vector Classifier (SVC)
- Multi-Layer Perceptron (MLP)

The paper reports approximately **98% accuracy** and **0.99 ROC-AUC** for its MLP approach with MinMax scaling.

This project uses the paper as a methodological reference but implements the pipeline independently.

---

## 4. Dataset

### Dataset Source

The project uses the **Personal Loan Modeling** dataset from Kaggle:

https://www.kaggle.com/datasets/teertha/personal-loan-modeling

### Dataset Characteristics

- **Records:** 5,000
- **Raw columns:** 14
- **Identifier:** `ID`
- **Input features:** 12
- **Target:** `Personal Loan`
- **Positive class:** 480 customers (9.6%)
- **Negative class:** 4,520 customers (90.4%)

### Input Features

1. Age
2. Experience
3. Income
4. ZIP Code
5. Family
6. CCAvg
7. Education
8. Mortgage
9. Securities Account
10. CD Account
11. Online
12. CreditCard

The raw dataset is **not committed to GitHub**. Download the dataset from the Kaggle source above and place:

```text
Bank_Personal_Loan_Modelling.xlsx
