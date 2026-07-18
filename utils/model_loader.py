"""
utils/model_loader.py
-----------------------
Central place that loads all trained model artifacts from /models.
app.py wraps this with @st.cache_resource so models load once per
Streamlit session, not on every rerun.
"""

import json
import os
from typing import Dict, Optional, Tuple

import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class ModelBundle:
    """Holds every trained artifact the app needs at inference time."""

    def __init__(self):
        self.classifier = None
        self.tfidf = None
        self.label_encoder = None
        self.ats_model = None
        self.ats_tfidf = None
        self.metrics: Optional[Dict] = None
        self.confusion_matrix: Optional[np.ndarray] = None
        self.classes = None
        self.loaded = False
        self.load_error: Optional[str] = None

    def load(self):
        try:
            self.classifier = joblib.load(os.path.join(MODELS_DIR, "classifier.pkl"))
            self.tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf.pkl"))
            self.label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
            self.ats_model = joblib.load(os.path.join(MODELS_DIR, "ats_model.pkl"))
            self.ats_tfidf = joblib.load(os.path.join(MODELS_DIR, "ats_tfidf.pkl"))

            metrics_path = os.path.join(MODELS_DIR, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    self.metrics = json.load(f)

            cm_path = os.path.join(MODELS_DIR, "confusion_matrix.npy")
            if os.path.exists(cm_path):
                self.confusion_matrix = np.load(cm_path)

            classes_path = os.path.join(MODELS_DIR, "classes.json")
            if os.path.exists(classes_path):
                with open(classes_path) as f:
                    self.classes = json.load(f)
            else:
                self.classes = list(self.label_encoder.classes_)

            self.loaded = True
        except Exception as e:
            # Catches missing files AND any deserialization failure (e.g. a
            # committed .pkl pickled with a different numpy/scikit-learn
            # version than what got installed on this deployment). Either
            # way, the caller (app.py) should fall back to retraining fresh
            # in the current environment rather than crashing.
            self.load_error = (
                f"Could not load saved models ({type(e).__name__}: {e}). "
                "Run `python train.py` to (re)train and save the models in this environment."
            )
            self.loaded = False

    def predict_category(self, clean_text: str) -> Tuple[str, float, Dict[str, float]]:
        """Returns (predicted_category, confidence, {category: probability})."""
        X = self.tfidf.transform([clean_text])
        pred_idx = self.classifier.predict(X)[0]
        predicted_category = self.label_encoder.inverse_transform([pred_idx])[0]

        proba_map = {}
        confidence = 1.0
        if hasattr(self.classifier, "predict_proba"):
            probas = self.classifier.predict_proba(X)[0]
            proba_map = {
                cls: float(p) for cls, p in zip(self.label_encoder.classes_, probas)
            }
            confidence = float(max(probas))

        return predicted_category, confidence, proba_map

    def predict_ats_score(self, clean_text: str, experience_years: int = 0) -> float:
        """Returns a predicted ATS score (0-100)."""
        X_text = self.ats_tfidf.transform([clean_text]).toarray()
        X = np.hstack([X_text, np.array([[experience_years]])])
        score = float(self.ats_model.predict(X)[0])
        return max(0.0, min(100.0, score))
