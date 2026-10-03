# Personal Loan Acceptance Prediction

## Problem Statement

This project focuses on predicting whether a bank customer will accept a personal loan offer based on the customer's demographic, financial, and banking-related information.

The reference paper identifies this as an imbalanced binary classification problem, where only a small proportion of customers accept the loan. This makes accurate identification of potential loan acceptors more challenging and makes accuracy alone an insufficient evaluation measure.

## Objective

Develop and evaluate machine learning models that predict whether a customer will accept a personal loan.

The project aims to:

- Reproduce a baseline based on the reference paper.
- Analyze the limitations of the baseline approach.
- Develop and evaluate an improved approach.
- Compare the baseline and proposed approach using appropriate classification metrics.

## Reference Paper

This project uses the following paper as its primary reference:

> **"Predicting Acceptance of Personal Loans Using Machine Learning Algorithms"**  
> *AJ Rossman and Conor Casey*

### Key Insights from the Reference Paper

- **Dataset:** Kaggle "Personal Loan Modeling" dataset containing 5,000 customer records and 12 customer features.
- **Target:** Personal Loan acceptance represented as a binary outcome (0 or 1).
- **Class Distribution:** The dataset is highly imbalanced, with approximately 10% of customers accepting the loan.
- **Algorithms Evaluated:** Perceptron, Logistic Regression, K-Means, Support Vector Classifier (SVC), and Multi-Layer Perceptron (MLP).
- **Evaluation:** The paper discusses the limitations of accuracy for imbalanced classification and uses ROC curves and AUC for additional evaluation.
- **Reported Result:** The paper reports its MLP approach with MinMaxScaler as its strongest-performing tested approach, reporting approximately 98% accuracy and 0.99 ROC-AUC.

## Our Approach

The project will be developed in stages.

### 1. Baseline Reproduction

Reproduce the core experimental setup from the reference paper using our own implementation, including:

- Dataset preparation
- Removal of the unnecessary index column
- Train/test split
- Feature scaling where appropriate
- Baseline machine learning models

### 2. Limitation Analysis

Analyze the baseline with particular attention to:

- Severe class imbalance
- Misleading accuracy
- False positives and false negatives
- Model sensitivity to preprocessing and parameter settings

### 3. Proposed Improvement

Based on the findings from the baseline experiments, we will implement an improved methodology.

Possible areas to investigate include:

- Class-balancing techniques such as SMOTE or under-sampling
- Cost-sensitive learning
- Alternative machine learning algorithms
- Hyperparameter tuning
- Ensemble-based approaches

The final proposed approach will be selected after the baseline experiments and justified based on the problem characteristics and experimental results.

## Dataset

### Source

Kaggle - Personal Loan Modeling dataset.

### Dataset Overview

- **Records:** 5,000 customers
- **Input Features:** 12
- **Target:** Personal Loan (0 = No, 1 = Yes)
- **Class Distribution:** Approximately 90% negative and 10% positive

### Features

The dataset contains customer attributes including:

- Age
- Experience
- Income
- ZIP Code
- Family size
- Average credit-card spending
- Education
- Mortgage
- Securities account
- CD account
- Online banking
- Credit card

Detailed preprocessing and feature handling will be documented as the implementation progresses.

## Project Structure

```text
personal-loan-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── .gitkeep
│
├── results/
│   ├── figures/
│   └── metrics/
│
└── docs/
    └── methodology.md