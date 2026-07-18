"""
train.py
--------
Trains all Machine Learning models used by the AI Resume Builder:

1. Resume Classifier  -> predicts job category from resume text
   Compares: Logistic Regression, Random Forest, SVM, Naive Bayes
   Saves the best performer as models/classifier.pkl

2. ATS Score Regressor -> predicts an ATS-friendliness score (0-100)
   Compares: Linear Regression, Random Forest Regressor, Gradient Boosting
   Saves the best performer as models/ats_model.pkl

3. Shared TF-IDF Vectorizer -> models/tfidf.pkl
   (fit on the classification corpus; reused for job-matching similarity too)

Also saves:
   models/label_encoder.pkl    - maps category names <-> class indices
   models/metrics.json         - all evaluation metrics for the Streamlit
                                  "Model Performance" page
   models/confusion_matrix.npy - confusion matrix of the best classifier

Usage:
    python train.py
"""

import json
import os
import time
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from utils.text_processing import preprocess_for_model

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RANDOM_STATE = 42


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load resume classification and ATS regression datasets, generating
    them first if they don't exist yet."""
    resume_path = os.path.join(DATASET_DIR, "resume_dataset.csv")
    ats_path = os.path.join(DATASET_DIR, "ats_dataset.csv")

    if not os.path.exists(resume_path) or not os.path.exists(ats_path):
        print("Datasets not found. Generating synthetic dataset first...")
        from dataset.generate_dataset import build_datasets
        build_datasets(n_per_category=120)

    resume_df = pd.read_csv(resume_path)
    ats_df = pd.read_csv(ats_path)
    return resume_df, ats_df


def train_classification_models(resume_df: pd.DataFrame) -> Dict:
    print("\n" + "=" * 60)
    print("STEP 1: Resume Classification Model Training")
    print("=" * 60)

    print("Cleaning and preprocessing resume text...")
    resume_df = resume_df.dropna(subset=["Resume", "Category"]).copy()
    resume_df["clean_text"] = resume_df["Resume"].apply(preprocess_for_model)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(resume_df["Category"])

    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    X = tfidf.fit_transform(resume_df["clean_text"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Support Vector Machine": CalibratedClassifierCV(
            SVC(kernel="linear", random_state=RANDOM_STATE), ensemble=False
        ),
        "Naive Bayes": MultinomialNB(),
    }

    results = {}
    best_model = None
    best_model_name = None
    best_f1 = -1
    best_preds = None

    for name, model in candidates.items():
        start = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        elapsed = time.time() - start

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "train_time_sec": round(elapsed, 3),
        }
        print(f"  {name:26s} | Acc: {acc:.4f} | Prec: {prec:.4f} | "
              f"Rec: {rec:.4f} | F1: {f1:.4f} | Time: {elapsed:.2f}s")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name
            best_preds = preds

    print(f"\nBest classification model: {best_model_name} (F1={best_f1:.4f})")

    cm = confusion_matrix(y_test, best_preds)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "classifier.pkl"))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf.pkl"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    np.save(os.path.join(MODELS_DIR, "confusion_matrix.npy"), cm)

    with open(os.path.join(MODELS_DIR, "classes.json"), "w") as f:
        json.dump(list(label_encoder.classes_), f)

    return {
        "results": results,
        "best_model": best_model_name,
        "confusion_matrix": cm.tolist(),
        "classes": list(label_encoder.classes_),
    }


def train_ats_regressor(ats_df: pd.DataFrame) -> Dict:
    print("\n" + "=" * 60)
    print("STEP 2: ATS Score Regression Model Training")
    print("=" * 60)

    ats_df = ats_df.dropna(subset=["Resume", "ATS_Score"]).copy()
    ats_df["clean_text"] = ats_df["Resume"].apply(preprocess_for_model)

    # Reuse a dedicated TF-IDF for the ATS regressor (smaller, numeric-friendly)
    ats_tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), min_df=2)
    X_text = ats_tfidf.fit_transform(ats_df["clean_text"]).toarray()

    exp_years = ats_df["ExperienceYears"].fillna(0).values.reshape(-1, 1)
    X = np.hstack([X_text, exp_years])
    y = ats_df["ATS_Score"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    results = {}
    best_model = None
    best_model_name = None
    best_r2 = -np.inf

    for name, model in candidates.items():
        start = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        elapsed = time.time() - start

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = {
            "mae": round(float(mae), 3),
            "rmse": round(float(rmse), 3),
            "r2_score": round(float(r2), 4),
            "train_time_sec": round(elapsed, 3),
        }
        print(f"  {name:28s} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | "
              f"R2: {r2:.4f} | Time: {elapsed:.2f}s")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name

    print(f"\nBest ATS regression model: {best_model_name} (R2={best_r2:.4f})")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "ats_model.pkl"))
    joblib.dump(ats_tfidf, os.path.join(MODELS_DIR, "ats_tfidf.pkl"))

    return {"results": results, "best_model": best_model_name}


def main():
    print("AI Resume Builder - Model Training Pipeline")
    resume_df, ats_df = load_datasets()
    print(f"Loaded {len(resume_df)} resumes for classification.")
    print(f"Loaded {len(ats_df)} samples for ATS scoring.")

    clf_results = train_classification_models(resume_df)
    ats_results = train_ats_regressor(ats_df)

    metrics = {
        "classification": clf_results,
        "regression": ats_results,
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete. Models saved to /models")
    print("=" * 60)
    for fname in sorted(os.listdir(MODELS_DIR)):
        print(f"  - models/{fname}")


if __name__ == "__main__":
    main()
