"""
BITS MBA ZG582 - Applied Machine Learning & Deep Learning
Assignment 2 / Phase 2: Telecom Customer Churn Prediction

This script is designed to run without code changes in Google Colab, Jupyter,
or a normal Python environment with internet access. It downloads the IBM
Telco Customer Churn dataset automatically if the CSV is not present locally.

Outputs are saved in the 'outputs' folder:
- model_comparison.csv
- classification_reports.txt
- confusion_matrix.png
- roc_curve.png
- precision_recall_curve.png
- feature_importance.png
- churn_risk_scored_customers.csv
- tuning_results.csv
- business_summary.txt
- final_model.joblib
"""

from pathlib import Path
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.inspection import permutation_importance
from joblib import dump
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_FILE = DATA_DIR / "Telco-Customer-Churn.csv"


def load_data():
    """Load the IBM Telco dataset; download it automatically if needed."""
    if not DATA_FILE.exists():
        print("Dataset not found locally. Downloading from IBM's public GitHub repository...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def prepare_data(df):
    """Clean the dataset and add a small number of business-friendly features."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Business features: early-tenure status and monthly revenue proxy.
    df["EarlyTenure"] = (df["tenure"] <= 12).astype(int)
    df["TenureBand"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"]
    ).astype(str)
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
    df["AvgMonthlySpend"] = df["AvgMonthlySpend"].fillna(df["MonthlyCharges"])

    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    X = df.drop(columns=["Churn", "customerID"])
    return X, y, df


def build_preprocessor(X):
    numeric = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ], remainder="drop")
    return preprocessor


def evaluate(model, X_test, y_test, name):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
        "PR-AUC": average_precision_score(y_test, prob),
    }, pred, prob


def main():
    print("\n=== BITS MBA ZG582 | Phase 2 Churn Modelling ===\n")
    df = load_data()
    X, y, df_clean = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Training rows: {len(X_train):,}; Test rows: {len(X_test):,}")
    print(f"Churn rate: {y.mean():.2%}")

    preprocessor = build_preprocessor(X)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # 1. Interpretable baseline
    lr = ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ])

    # 2. Bagging model
    rf_base = ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ])
    rf_grid = {
        "model__n_estimators": [150, 250],
        "model__max_depth": [8, 14, None],
        "model__min_samples_split": [2, 5]
    }
    rf_search = GridSearchCV(rf_base, rf_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)

    # 3. Complex boosting algorithm requested in Phase 1 feedback
    gb_base = ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])
    gb_grid = {
        "model__n_estimators": [80, 120],
        "model__learning_rate": [0.05, 0.10],
        "model__max_depth": [2, 3],
        "model__subsample": [0.8, 1.0]
    }
    gb_search = GridSearchCV(gb_base, gb_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)

    print("\nTraining Logistic Regression...")
    lr.fit(X_train, y_train)
    print("Tuning Random Forest...")
    rf_search.fit(X_train, y_train)
    print("Tuning Gradient Boosting...")
    gb_search.fit(X_train, y_train)

    models = {
        "Logistic Regression": lr,
        "Random Forest (Tuned)": rf_search.best_estimator_,
        "Gradient Boosting (Tuned)": gb_search.best_estimator_
    }

    rows=[]
    predictions={}
    for name, model in models.items():
        row, pred, prob = evaluate(model, X_test, y_test, name)
        rows.append(row)
        predictions[name]=(pred,prob)
        print(f"{name}: AUC={row['ROC-AUC']:.3f}, F1={row['F1-Score']:.3f}, Recall={row['Recall']:.3f}")

    results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    results.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    tuning = pd.DataFrame({
        "Model": ["Random Forest", "Gradient Boosting"],
        "Best CV ROC-AUC": [rf_search.best_score_, gb_search.best_score_],
        "Best Parameters": [str(rf_search.best_params_), str(gb_search.best_params_)]
    })
    tuning.to_csv(OUT_DIR / "tuning_results.csv", index=False)

    best_name = results.iloc[0]["Model"]
    best_model = models[best_name]
    best_pred, best_prob = predictions[best_name]

    # Classification reports
    with open(OUT_DIR / "classification_reports.txt", "w", encoding="utf-8") as f:
        f.write("BITS MBA ZG582 - Classification Reports\n\n")
        for name, model in models.items():
            pred, prob = predictions[name]
            f.write(f"===== {name} =====\n")
            f.write(classification_report(y_test, pred, target_names=["Retained", "Churned"], zero_division=0))
            f.write(f"ROC-AUC: {roc_auc_score(y_test, prob):.4f}\n\n")

    # Confusion matrix
    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix - {best_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0,1], ["Retained", "Churned"])
    ax.set_yticks([0,1], ["Retained", "Churned"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=220)
    plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(7,5))
    for name, (pred, prob) in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curve.png", dpi=220)
    plt.close(fig)

    # Precision-recall curve for final model
    precision, recall, _ = precision_recall_curve(y_test, best_prob)
    ap = average_precision_score(y_test, best_prob)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(recall, precision, label=f"{best_name} (PR-AUC={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "precision_recall_curve.png", dpi=220)
    plt.close(fig)

    # Permutation importance on raw test features. This is model-agnostic and business-friendly.
    sample_n = min(1200, len(X_test))
    X_imp = X_test.sample(sample_n, random_state=RANDOM_STATE)
    y_imp = y_test.loc[X_imp.index]
    perm = permutation_importance(
        best_model, X_imp, y_imp, scoring="roc_auc", n_repeats=5,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    imp = pd.Series(perm.importances_mean, index=X_imp.columns).sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8,5.5))
    imp.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(f"Top Business Features - {best_name}")
    ax.set_xlabel("Mean decrease in ROC-AUC after permutation")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=220)
    plt.close(fig)
    imp.to_csv(OUT_DIR / "feature_importance.csv", header=["Importance"])

    # Score all customers for business use.
    all_prob = best_model.predict_proba(X)[:,1]
    scored = df_clean.copy()
    scored["PredictedChurnProbability"] = all_prob
    scored["RiskBand"] = pd.cut(
        all_prob, bins=[-0.01,0.30,0.60,1.01], labels=["Low","Medium","High"]
    )
    scored["EstimatedMonthlyRevenueAtRisk"] = scored["MonthlyCharges"] * all_prob
    scored["PriorityAction"] = np.select(
        [scored["RiskBand"].eq("High"), scored["RiskBand"].eq("Medium")],
        ["Priority retention call / targeted offer", "Personalised plan or service follow-up"],
        default="Normal service and monitoring"
    )
    scored.sort_values("PredictedChurnProbability", ascending=False).to_csv(
        OUT_DIR / "churn_risk_scored_customers.csv", index=False
    )

    high = scored[scored["RiskBand"]=="High"]
    medium = scored[scored["RiskBand"]=="Medium"]
    business = [
        f"Final model selected by test ROC-AUC: {best_name}",
        f"Test ROC-AUC: {results.iloc[0]['ROC-AUC']:.4f}",
        f"Test F1-score: {results.iloc[0]['F1-Score']:.4f}",
        f"Test recall: {results.iloc[0]['Recall']:.4f}",
        f"High-risk customers: {len(high):,} ({len(high)/len(scored):.1%})",
        f"Medium-risk customers: {len(medium):,} ({len(medium)/len(scored):.1%})",
        f"Estimated monthly revenue at risk (probability-weighted): ₹{scored['EstimatedMonthlyRevenueAtRisk'].sum():,.0f}",
        "Recommended use: rank customers by churn probability and combine risk with customer value before approving retention incentives."
    ]
    (OUT_DIR / "business_summary.txt").write_text("\n".join(business), encoding="utf-8")

    dump(best_model, OUT_DIR / "final_model.joblib")

    print("\n=== Best model ===")
    print(best_name)
    print(results.to_string(index=False, formatters={c: "{:.3f}".format for c in ["Accuracy","Precision","Recall","F1-Score","ROC-AUC","PR-AUC"]}))
    print("\nOutputs saved to:", OUT_DIR)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
