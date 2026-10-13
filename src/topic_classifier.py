"""
News Topic Classification module for NewsInsight AI.
Trains and compares Multinomial Naive Bayes, Logistic Regression, and K-Nearest Neighbors.
Saves the best model and provides real-time inference with explainability.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

from config.settings import MODELS_DIR, RANDOM_SEED, TEST_SIZE, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE
from src.preprocessing import clean_text, remove_stopwords, preprocess_text, extract_top_features


TOPIC_MODEL_PATH = MODELS_DIR / "topic_classifier.joblib"
TOPIC_VECTORIZER_PATH = MODELS_DIR / "topic_vectorizer.joblib"
TOPIC_METADATA_PATH = MODELS_DIR / "topic_model_meta.joblib"


class TopicClassificationEngine:
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[Any] = None
        self.model_name: str = "Unknown"
        self.classes: list = []
        self.all_models_eval: Dict[str, Any] = {}
        self.load_artifacts_if_available()

    def load_artifacts_if_available(self) -> bool:
        """Loads trained artifacts if present on disk."""
        if TOPIC_MODEL_PATH.exists() and TOPIC_VECTORIZER_PATH.exists() and TOPIC_METADATA_PATH.exists():
            try:
                self.model = joblib.load(TOPIC_MODEL_PATH)
                self.vectorizer = joblib.load(TOPIC_VECTORIZER_PATH)
                meta = joblib.load(TOPIC_METADATA_PATH)
                self.model_name = meta.get("model_name", "Best Model")
                self.classes = list(getattr(self.model, "classes_", meta.get("classes", [])))
                self.all_models_eval = meta.get("eval_results", {})
                return True
            except Exception as e:
                print(f"Error loading topic classifier: {e}")
                return False
        return False

    def train_and_compare(
        self,
        df: pd.DataFrame,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_SEED
    ) -> Dict[str, Any]:
        """
        Trains MultinomialNB, LogisticRegression, and KNeighborsClassifier
        on the provided dataframe using a stratified split.
        Selects the best performing model based on Macro-F1 score and saves it.
        """
        # Ensure clean text and non-null labels
        clean_df = df[df["headline"].notna() & df["category"].notna()].copy()
        clean_df["text_clean"] = clean_df["headline"].apply(preprocess_text)
        clean_df = clean_df[clean_df["text_clean"].str.len() > 2].copy()

        # Check label distribution
        class_counts = clean_df["category"].value_counts()
        valid_classes = class_counts[class_counts >= 2].index.tolist()
        clean_df = clean_df[clean_df["category"].isin(valid_classes)].copy()

        if len(valid_classes) < 2 or len(clean_df) < 10:
            raise ValueError("Insufficient data or classes to train topic classifier. Need at least 2 classes and 10 samples.")

        X = np.array(clean_df["text_clean"].tolist(), dtype=str)
        y = np.array(clean_df["category"].tolist(), dtype=str)

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # Fit TF-IDF Vectorizer ONLY on training set to avoid data leakage
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=True
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Initialize candidate models
        candidate_models = {
            "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                C=1.0,
                class_weight="balanced"
            ),
            "K-Nearest Neighbors": KNeighborsClassifier(
                n_neighbors=min(5, len(X_train)),
                metric="cosine"
            )
        }

        eval_results = {}
        best_name = None
        best_score = -1.0
        best_clf = None

        labels = sorted(list(np.unique(y)))

        for name, clf in candidate_models.items():
            clf.fit(X_train_vec, y_train)
            y_pred = clf.predict(X_test_vec)

            acc = float(accuracy_score(y_test, y_pred))
            macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            macro_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
            macro_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
            cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
            report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

            eval_results[name] = {
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "precision": round(macro_prec, 4),
                "recall": round(macro_rec, 4),
                "confusion_matrix": cm,
                "classification_report": report,
                "labels": labels,
                "train_size": len(X_train),
                "test_size": len(X_test)
            }

            # Select based on Macro-F1
            if macro_f1 > best_score:
                best_score = macro_f1
                best_name = name
                best_clf = clf

        # Persist the winning model, vectorizer, and evaluation summary
        self.vectorizer = vectorizer
        self.model = best_clf
        self.model_name = best_name
        self.classes = labels
        self.all_models_eval = eval_results

        joblib.dump(best_clf, TOPIC_MODEL_PATH)
        joblib.dump(vectorizer, TOPIC_VECTORIZER_PATH)
        joblib.dump({
            "model_name": best_name,
            "classes": labels,
            "eval_results": eval_results,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }, TOPIC_METADATA_PATH)

        return {
            "selected_model": best_name,
            "best_macro_f1": best_score,
            "models": eval_results
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predicts news topic for an input text.
        Returns:
            {
                "predicted_category": str,
                "confidence": float,
                "probabilities": dict,
                "top_features": list,
                "model_name": str,
                "status": "success" | "error"
            }
        """
        if self.model is None or self.vectorizer is None:
            return {
                "status": "error",
                "message": "Topic model is not trained yet. Please train models in scripts or Model Evaluation page."
            }

        cleaned = preprocess_text(text)
        if not cleaned or len(cleaned) < 2:
            return {
                "status": "error",
                "message": "Input text is too short or contains only stop words."
            }

        vec = self.vectorizer.transform([cleaned])
        pred_label = self.model.predict(vec)[0]

        # Get confidence score
        probabilities = {}
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            confidence = float(np.max(probs))
            classes = list(self.model.classes_)
            probabilities = {cls: round(float(probs[i]), 4) for i, cls in enumerate(classes)}
        else:
            confidence = 1.0

        top_features = extract_top_features(
            text=text,
            vectorizer=self.vectorizer,
            model=self.model,
            predicted_class=pred_label,
            top_n=5
        )

        return {
            "status": "success",
            "predicted_category": str(pred_label),
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "top_features": top_features,
            "model_name": self.model_name
        }
