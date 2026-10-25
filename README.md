# -Thesis-Data-Driven-Student-Behavior-Modeling-and-Dropout-Prediction-Using-Advanced-ML-Techniques

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)

## 📌 Executive Summary
Student dropout is a critical challenge in higher education, leading to wasted institutional resources and lost student potential. Traditional machine learning models often suffer from data leakage, severe class imbalances, and generate excessive false alarms, which overwhelm academic advisors.

This repository contains an end-to-end, explainable machine learning pipeline that introduces a **Customized Gated-MLP Hybrid Model**. By utilizing a learnable feature-gating layer to filter noise and strict temporal validation to prevent data leakage, this system achieves **97.62% accuracy** and significantly reduces false alarms, providing a highly actionable early-warning system for universities.

---

## 🚨 The Problem We Solved
1. **Data Leakage:** Random train-test splits mix historical and future data, yielding overly optimistic results. This project uses strict temporal validation (training on 2018+2021, testing on unseen 2022 data).
2. **Class Imbalance:** Dropout cases are a minority (~7%). We applied SMOTE strictly to training folds to balance the data without contaminating the test set.
3. **False Alarms:** Standard MLPs struggle with noisy data, resulting in low precision. Our Customized Gated-MLP suppresses noisy features and amplifies critical signals, cutting false alarms by 265 students.

---

## 📊 Key Results & Impact
Our model was benchmarked against 5 classical algorithms (Logistic Regression, SVM, Decision Tree, KNN, standard MLP).

| Metric | Baseline (Standard MLP) | **Proposed (Gated-MLP)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 95.20% | **97.62%** | +2.42% |
| **Dropout Precision** | 65.73% | **89.19%** | **+23.46%** |
| **Dropout F1-Score** | 0.6861 | **0.8252** | +13.91 pp |
| **AUC-ROC** | 0.9040 | **0.9229** | +1.89% |

**Real-World Impact:** 
*   **Saved Counselor Time:** Reduced false alarms from 559 down to 294 (a reduction of 265 unnecessary interventions).
*   **High Reliability:** Achieved an excellent Brier Score of 0.0230, meaning the predicted risk probabilities are highly calibrated and trustworthy.
*   **Efficiency:** Trains in just 62.92 seconds (compared to SVM which takes 2062 seconds), making it highly deployable on standard CPUs.

---

## 📂 Repository Structure (Pipeline)
The project is divided into 4 modular Jupyter Notebooks following a standard ML lifecycle:

1. 📁 **`01_UPV_Data_Audit.ipynb`**
   *   Performs initial data exploration, missing value analysis, and strict leakage control checks.
2. 📁 **`02_UPV_Preprocessing.ipynb`**
   *   Handles categorical encoding, numerical scaling, median imputation, and applies SMOTE to handle class imbalance.
3. 📁 **`03_UPV_Feature_Engineering.ipynb`**
   *   Focuses on feature selection, consistency checks across academic years, and preparing the final dataset for modeling.
4. 📁 **`04_UPV_Modeling.ipynb`**
   *   The core of the project. Contains baseline model training, Customized Gated-MLP architecture (PyTorch), statistical validation (McNemar's test, Bootstrap), and SHAP explainability analysis.

---

## 🛠️ Tech Stack & Tools
*   **Data Processing:** Pandas, NumPy, Scikit-learn
*   **Deep Learning:** PyTorch (Custom Neural Network architecture)
*   **Explainability:** SHAP (SHapley Additive exPlanations)
*   **Environment:** Google Colab / Jupyter Notebook

---

## 👨‍💻 Author
**[Mohammad Jahid Hassan]**
*BSc in Computer Science and Engineering*
*Green University of Bangladesh*
* 🔗 [https://www.linkedin.com/in/mohammad-jahid-hassan-71349a1aa/]
* 📧 [mrjahidhassan0@gmail.com]

---
*Keywords: Machine Learning, Deep Learning, Student Dropout Prediction, Gated-MLP, SHAP Explainability, PyTorch*
