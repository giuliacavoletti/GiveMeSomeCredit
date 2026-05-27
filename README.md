# Machine Learning Models for Credit Scoring

This repository contains the code and methodology for evaluating various machine learning models to predict credit default. In the financial sector, credit scoring algorithms assess the probability that a borrower will fail to meet their legal debt obligations. 

This project explores a progression of models—from a baseline Logistic Regression to advanced, customized ensemble methodologies and it was made during the course of Numerical Analysis for Machine Learning at PoliMi.

##  Dataset Overview
The project uses the "Give Me Some Credit" dataset from Kaggle.
* **Size:** The raw training set consists of 150,000 samples and 10 features. After data cleaning, outlier removal, and imputation, the training set was reduced to 147,250 samples.
* **Target Variable:** `SeriousDlqin2yrs` (binary classification). 
* **Class Imbalance:** The dataset is highly imbalanced, with approximately 93.4% of customers labeled as non-default and only 6.6% as default. 

##  Methodology & Feature Engineering

### Data Preprocessing
* Missing values in `MonthlyIncome` and `NumberOfDependents` were imputed using their respective training set medians.
* Anomalous and corrupted samples within delinquency columns were removed, and specific extreme outliers in age and debt ratio were capped or cut.

### Feature Engineering
Three new domain-motivated features were derived to improve model performance:
1. **IncomePerPerson:** Normalizes monthly income by household size.
2. **TotalLate:** Aggregates three delinquency indicators into a single count of late-payment events.
3. **CreditUtilizationAge:** Combines credit-load intensity with borrower maturity into a single ratio.

##  Models Implemented
The project evaluates the following algorithms:
* **Logistic Regression:** Serves as the baseline model.
* **Feed-Forward Neural Network (ANN):** Built entirely from scratch using the **JAX** library to allow for automatic differentiation and JIT compilation. Hyperparameters were systematically tuned using **Optuna**.
* **Tree-Based Ensembles:** Random Forest, XGBoost, and LightGBM.
* **CatBoost:** Utilized to address prediction shift bias through ordered boosting.

##  Advanced Stacking Ensemble
The final stage of the project implements a custom stacking framework:
* **Meta-Feature Generation:** The JAX-based ANN acts as the meta-feature generator. Out-of-fold (OOF) predictions from the ANN are appended as an additional feature column (`ann_meta`) to capture smooth, high-order interactions that tree-based models miss.
* **Downstream Learners:** XGBoost and CatBoost are retrained using the augmented feature matrix.
* **Final Blend:** The predictions of the augmented models, alongside LightGBM and the ANN, are combined using two principled methods:
    1. A **Nelder-Mead** optimized blending algorithm.
    2. A **Logistic Regression** meta-learner operating in logit space.

## Key Results
Models were evaluated primarily on their AUC score and specificity, keeping sensitivity constrained to at least 70%.

| Model | Test AUC |
| :--- | :--- |
| Logistic Regression (Baseline) | 0.8573 |
| Random Forest | 0.8559 |
| XGBoost | 0.8656 |
| Custom ANN (JAX + Optuna) | 0.8684 |
| **CatBoost + ANN Meta-feature** | **0.8690** |
| **Final Ensemble (Nelder-Mead / LR)** | **0.8689** |

*Note: While CatBoost + ANN meta achieved the highest strict test-split AUC, the final ensembles provide the most robust and unbiased estimate of true generalization performance as their weights were optimized exclusively on out-of-fold predictions.*

##  Tech Stack & Requirements
Based on the methodologies used, ensure the following libraries are installed to run the notebooks:
* `Python 3.x`
* `JAX`
* `scikit-learn`
* `Optuna`
* `xgboost`
* `catboost`
* `lightgbm`

## ✍️ Authors
* Giovanni Carpenedo
* Giulia Cavoletti
* *February 2026*
