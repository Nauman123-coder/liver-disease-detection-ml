# 🏥 Early Liver Disease Detection using Machine Learning

## Project Overview

This project demonstrates the application of supervised machine learning techniques to a crucial clinical problem: the early detection of liver disease using routine laboratory biomarkers. This solution is designed to support clinical decision-making, reduce diagnostic inconsistency, and improve patient outcomes in high-risk populations.

## 💡 Clinical Scenario

As a Clinical Data Analyst at a regional hospital network, the goal was to develop an automated screening tool. Currently, physicians manually review Liver Function Tests (LFTs), leading to inconsistencies and missed opportunities for early intervention. This ML classifier addresses that by analyzing patient laboratory data and flagging high-risk cases for immediate follow-up.

## 🚀 Model Performance Summary

The final model used in this project is the **Logistic Regression Classifier**, which serves as a highly interpretable baseline for binary classification. The initial results highlighted a key trade-off between Sensitivity and Specificity common in imbalanced diagnostic datasets.

| Metric | Logistic Regression (Baseline) | Clinical Significance |
|--------|-------------------------------|----------------------|
| **Area Under the Curve (AUC)** | Moderate (≈ 0.75) | Overall discriminative power of the model. |
| **Accuracy** | Acceptable (≈ 73.7%) | Overall correctness of predictions. |
| **Sensitivity (Recall)** | High (≈ 95.2%) | Crucial: Ability to correctly identify patients with liver disease (minimizing False Negatives/missed cases). |
| **Specificity** | Low (≈ 20.0%) | Ability to correctly identify healthy patients (high rate of False Positives/unnecessary referrals). |

### Visualizations

The key outputs below demonstrate the model's reliability:

- **Confusion Matrix**: Shows the breakdown of True Positives, True Negatives, False Positives, and False Negatives.
- **ROC Curve**: Illustrates the trade-off between Sensitivity and Specificity across all decision thresholds. A curve close to the top-left corner and an AUC near 1.0 indicate strong performance.

## 🛠️ Project Goals & Tasks

This project fulfills the core requirements of a robust machine learning workflow in a healthcare context.

### Learning Objectives

- **Data Analysis**: Understand the structure and clinical meaning of healthcare datasets.
- **Model Training**: Apply supervised learning algorithms (Logistic Regression).
- **Clinical Evaluation**: Measure and interpret performance using clinically relevant metrics (Sensitivity, Specificity).
- **Optimization**: Recognize the need for techniques like class weighting and threshold tuning to balance the Sensitivity-Specificity trade-off (a recommended next step).

### Tasks Completed

- Analyze the Dataset: Loaded and inspected the data structure.
- Identify Features & Target: Defined the 9 clinical features and the binary Outcome target.
- Model Training: Trained the Logistic Regression classifier.
- Model Evaluation & Visualization: Calculated metrics, plotted the Confusion Matrix, and the ROC-AUC curve.

## 📊 Dataset Details

The model was built using the **Indian Liver Patient Dataset (ILPD)**.

### Dataset Composition

| Metric | Value |
|--------|-------|
| **Total Records** | 583 |
| **Positive Cases (Liver Disease)** | 416 (71.4%) |
| **Negative Cases (No Liver Disease)** | 167 (28.6%) |
| **Format** | CSV (Comma Separated Values) |

### Clinical Features

This dataset utilizes standard Liver Function Test (LFT) biomarkers.

| Feature Name | Description | Clinical Significance |
|--------------|-------------|----------------------|
| **Age** | Patient age in years | General risk factor. |
| **Total_Bilirubin** | Serum Bilirubin levels | Liver function indicator. |
| **Direct_Bilirubin** | Conjugated Bilirubin | Specific marker for liver damage/obstruction. |
| **Alkaline_Phosphotase (Alkphos)** | Enzyme concentration | Indicator of liver/bile duct function. |
| **Alamine_Aminotransferase (ALT/SGPT)** | Liver enzyme | Highly specific marker for cellular damage. |
| **Aspartate_Aminotransferase (AST/SGOT)** | Liver enzyme | Indicator of tissue damage (less specific than ALT). |
| **Total_Protiens** | Total protein concentration | Overall liver synthetic function. |
| **Albumin** | Albumin protein levels | Liver-produced protein; low levels indicate chronic damage. |
| **Albumin_and_Globulin_Ratio (A/G Ratio)** | Ratio of proteins | Protein balance indicator. |
| **Target (Outcome)** | Binary (1 = Liver Disease, 0 = No Liver Disease) | Classification goal. |

## ⚙️ Environment and Dependencies

The project code is written in Python and requires the following libraries:

| Library | Purpose |
|---------|---------|
| **pandas** | Data loading and manipulation. |
| **numpy** | Numerical operations (especially for threshold optimization). |
| **sklearn** | Machine learning models (Logistic Regression) and metrics. |
| **matplotlib** | General plotting (used for ROC curve). |
| **seaborn** | Statistical data visualization (used for Confusion Matrix heatmap). |

### Setup
```bash
# Clone the repository
git clone [YOUR-REPO-URL]
cd early-liver-disease-detection

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 🧑‍💻 How to Run the Code

1. Ensure the `liver_dataset.xlsx - in.csv` file is in the root directory.

2. Run the main Python script:
```bash
python liver_detection_visualization.py
```

3. The output will print the final metrics and display the Confusion Matrix and ROC-AUC plots.

---
