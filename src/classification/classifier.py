"""Scikit-learn based threat classification engine.

Provides a fast, lightweight multi-label classifier using TF-IDF features
and ensemble methods as a baseline before transformer-based classification.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from src.ingestion.generator import THREAT_CATEGORIES

logger = logging.getLogger(__name__)


class ThreatClassifier:
    """Multi-label threat report classifier using TF-IDF + SGD.

    This baseline classifier uses TF-IDF vectorization with an SGD
    (linear SVM) backend wrapped in a OneVsRest strategy for multi-label
    classification of threat intelligence reports.

    Attributes:
        categories: List of threat category labels.
        pipeline: Sklearn classification pipeline.
        mlb: Multi-label binarizer for encoding labels.
    """

    def __init__(self, max_features: int = 10000, ngram_range: tuple = (1, 3)):
        self.categories = THREAT_CATEGORIES
        self.mlb = MultiLabelBinarizer(classes=self.categories)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,
                strip_accents="unicode",
                analyzer="word",
                min_df=2,
                max_df=0.95,
            )),
            ("clf", OneVsRestClassifier(
                SGDClassifier(
                    loss="modified_huber",  # Provides probability estimates
                    penalty="l2",
                    alpha=1e-4,
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                ),
            )),
        ])
        self._is_fitted = False

    def train(
        self,
        df: pd.DataFrame,
        text_column: str = "cleaned_text",
        label_column: str = "labels",
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        """Train the classifier on a labeled dataset.

        Args:
            df: DataFrame with text and label columns.
            text_column: Column containing preprocessed text.
            label_column: Column containing lists of category labels.
            test_size: Fraction of data to hold out for evaluation.

        Returns:
            Dictionary containing training metrics and evaluation results.
        """
        logger.info("Training TF-IDF + SGD classifier on %d samples", len(df))

        X = df[text_column].values
        y = self.mlb.fit_transform(df[label_column].values)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
        )

        self.pipeline.fit(X_train, y_train)
        self._is_fitted = True

        y_pred = self.pipeline.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=self.categories,
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "f1_micro": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "classification_report": report,
        }

        logger.info(
            "Training complete. F1-micro: %.3f, F1-macro: %.3f",
            metrics["f1_micro"], metrics["f1_macro"],
        )
        return metrics

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Classify threat intelligence texts.

        Args:
            texts: List of preprocessed text strings.

        Returns:
            List of prediction dictionaries with labels and probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be trained before prediction. Call train() first.")

        predictions = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)

        results = []
        for pred_row, prob_row in zip(predictions, probabilities):
            labels = self.mlb.inverse_transform(pred_row.reshape(1, -1))[0]
            label_probs = {
                cat: float(prob)
                for cat, prob in zip(self.categories, prob_row)
            }
            results.append({
                "predicted_labels": list(labels) if labels else ["unknown"],
                "probabilities": label_probs,
                "top_category": max(label_probs, key=label_probs.get),
                "confidence": float(max(prob_row)),
            })
        return results

    def predict_single(self, text: str) -> dict[str, Any]:
        """Classify a single text."""
        return self.predict([text])[0]

    def save_model(self, path: str | Path) -> None:
        """Save the trained model artifacts to disk."""
        import joblib
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pipeline, path / "pipeline.joblib")
        joblib.dump(self.mlb, path / "mlb.joblib")

        metadata = {
            "categories": self.categories,
            "is_fitted": self._is_fitted,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model saved to %s", path)

    def load_model(self, path: str | Path) -> None:
        """Load model artifacts from disk."""
        import joblib
        path = Path(path)

        self.pipeline = joblib.load(path / "pipeline.joblib")
        self.mlb = joblib.load(path / "mlb.joblib")

        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        self.categories = metadata["categories"]
        self._is_fitted = metadata["is_fitted"]

        logger.info("Model loaded from %s", path)
