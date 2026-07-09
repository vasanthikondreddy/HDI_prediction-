"""
train_model.py
---------------
End-to-end training pipeline for the HDI Category classifier.

Steps performed:
    1. Load the dataset (HDI.csv) and auto-detect the target column.
    2. Clean the data (handle missing values, remove duplicates).
    3. Encode the target label (LabelEncoder).
    4. Scale the numeric features (StandardScaler).
    5. Split into train/test sets.
    6. Train multiple classification algorithms.
    7. Evaluate every model (Accuracy, Precision, Recall, F1).
    8. Automatically pick the best model based on F1 score.
    9. Save model.pkl, scaler.pkl and label_encoder.pkl with joblib.

Run with:
    python train_model.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# --------------------------------------------------------------------------
# 1. Load dataset & auto-detect target column
# --------------------------------------------------------------------------
DATA_PATH = "HDI.csv"
CANDIDATE_TARGET_NAMES = ["HDI Category", "hdi_category", "Category", "Target"]
FEATURE_COLUMNS = [
    "Life Expectancy",
    "Mean Years of Schooling",
    "Expected Years of Schooling",
    "GNI Per Capita",
]


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV and auto-detect the target column so this script keeps
    working even if a different HDI dataset (with slightly different
    column naming) is dropped in."""
    df = pd.read_csv(path)

    target_col = None
    for candidate in CANDIDATE_TARGET_NAMES:
        if candidate in df.columns:
            target_col = candidate
            break
    if target_col is None:
        # fall back: any column that looks categorical with <= 6 unique values
        for col in df.columns:
            if df[col].dtype == object and df[col].nunique() <= 6:
                target_col = col
                break
    if target_col is None:
        raise ValueError("Could not auto-detect the target column in the dataset.")

    if target_col != "HDI Category":
        df = df.rename(columns={target_col: "HDI Category"})

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and duplicate rows."""
    df = df.drop_duplicates().reset_index(drop=True)

    # Only keep columns we actually use
    keep_cols = [c for c in FEATURE_COLUMNS if c in df.columns] + ["HDI Category"]
    df = df[keep_cols]

    # Fill numeric missing values with the median of their column
    for col in FEATURE_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=["HDI Category"]).reset_index(drop=True)
    return df


def main():
    print("=" * 60)
    print("HDI CATEGORY CLASSIFIER - TRAINING PIPELINE")
    print("=" * 60)

    df = load_dataset(DATA_PATH)
    print(f"\nLoaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    df = clean_dataset(df)
    print(f"After cleaning: {df.shape[0]} rows")

    X = df[FEATURE_COLUMNS]
    y_raw = df["HDI Category"]

    # ----------------------------------------------------------------
    # 2. Encode target
    # ----------------------------------------------------------------
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    print(f"\nClasses found: {list(label_encoder.classes_)}")

    # ----------------------------------------------------------------
    # 3. Scale features
    # ----------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ----------------------------------------------------------------
    # 4. Train/test split
    # ----------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # ----------------------------------------------------------------
    # 5. Define candidate models
    # ----------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            random_state=42, eval_metric="mlogloss", use_label_encoder=False
        )

    # ----------------------------------------------------------------
    # 6. Train & evaluate every model
    # ----------------------------------------------------------------
    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append(
            {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}
        )
        trained_models[name] = model

        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by F1 Score)")
    print("=" * 60)
    print(results_df.to_string(index=False))
    results_df.to_csv("model_comparison.csv", index=False)

    # ----------------------------------------------------------------
    # 7. Pick best model automatically
    # ----------------------------------------------------------------
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\nBest model selected: {best_model_name}")

    y_pred_best = best_model.predict(X_test)
    print("\nClassification Report (Best Model):")
    print(
        classification_report(
            y_test, y_pred_best, target_names=label_encoder.classes_, zero_division=0
        )
    )
    print("Confusion Matrix (Best Model):")
    print(confusion_matrix(y_test, y_pred_best))

    # ----------------------------------------------------------------
    # 8. Save artifacts
    # ----------------------------------------------------------------
    joblib.dump(best_model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")

    with open("best_model_name.txt", "w") as f:
        f.write(best_model_name)

    print("\nSaved: model.pkl, scaler.pkl, label_encoder.pkl")
    print("Training complete.")


if __name__ == "__main__":
    main()
