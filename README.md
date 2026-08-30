# BITS MBA ZG582 – Assignment 2 / Phase 2
## Telecom Customer Churn Prediction

This repository is ready to upload to GitHub and run in Google Colab/Jupyter. It implements the Phase-2 requirements: preprocessing, feature engineering, Logistic Regression, Random Forest, Gradient Boosting, hyperparameter tuning, evaluation, model comparison, feature importance and business risk scoring.

### 1. Quickest method – Google Colab
1. Open Google Colab.
2. Upload `main.py` and `requirements.txt` from this folder.
3. Run:
```bash
!pip install -r requirements.txt
!python main.py
```
4. The script automatically downloads the IBM Telco Customer Churn CSV if it is not already in `data/`.
5. Open the `outputs/` folder to see the model comparison, tuning results, confusion matrix, ROC curve, precision-recall curve, feature importance and customer risk scores.

### 2. Run locally
```bash
pip install -r requirements.txt
python main.py
```

### 3. Dataset
The code uses the IBM public GitHub copy of the Telco Customer Churn dataset:
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

### 4. GitHub upload steps
```bash
git init
git add .
git commit -m "BITS MBA ZG582 Assignment 2 Phase 2 - Churn Prediction"
git branch -M main
git remote add origin https://github.com/<USERNAME>/BITS-MBA-ZG582-Telco-Churn-Phase2.git
git push -u origin main
```
Then paste the final repository URL into the assignment report.

### 5. Main outputs
- `outputs/model_comparison.csv` – Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC.
- `outputs/tuning_results.csv` – best parameters and cross-validation ROC-AUC.
- `outputs/confusion_matrix.png` – final model classification errors.
- `outputs/roc_curve.png` – model discrimination comparison.
- `outputs/precision_recall_curve.png` – churn-focused performance.
- `outputs/feature_importance.png` – business feature importance using permutation importance.
- `outputs/churn_risk_scored_customers.csv` – customer-level probability, risk band and recommended action.
- `outputs/business_summary.txt` – management-friendly summary.
- `outputs/final_model.joblib` – saved best model.
