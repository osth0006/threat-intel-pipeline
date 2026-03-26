"""Tests for the threat classification engine."""

import pytest

from src.classification.classifier import ThreatClassifier
from src.ingestion.generator import ThreatIntelGenerator
from src.processing.preprocessor import TextPreprocessor


class TestThreatClassifier:
    """Test suite for ThreatClassifier."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Generate and preprocess training data."""
        generator = ThreatIntelGenerator(seed=42)
        preprocessor = TextPreprocessor()
        df = generator.generate_batch(n=200, seed=42)
        self.df = preprocessor.process_dataframe(df)
        self.classifier = ThreatClassifier()

    def test_train_returns_metrics(self):
        metrics = self.classifier.train(self.df)
        assert "f1_micro" in metrics
        assert "f1_macro" in metrics
        assert "n_train" in metrics
        assert "n_test" in metrics
        assert metrics["f1_micro"] > 0

    def test_predict_returns_labels(self):
        self.classifier.train(self.df)
        results = self.classifier.predict(["APT28 phishing campaign targeting government"])
        assert len(results) == 1
        assert "predicted_labels" in results[0]
        assert "probabilities" in results[0]
        assert "confidence" in results[0]

    def test_predict_single(self):
        self.classifier.train(self.df)
        result = self.classifier.predict_single(
            "Ransomware encrypted 500 systems demanding bitcoin payment"
        )
        assert "predicted_labels" in result
        assert len(result["predicted_labels"]) >= 1

    def test_predict_before_training_raises(self):
        clf = ThreatClassifier()
        with pytest.raises(RuntimeError, match="must be trained"):
            clf.predict(["test text"])

    def test_predict_batch(self):
        self.classifier.train(self.df)
        texts = [
            "APT group espionage campaign",
            "New malware variant detected",
            "Phishing emails targeting finance",
        ]
        results = self.classifier.predict(texts)
        assert len(results) == 3

    def test_probabilities_sum_reasonable(self):
        self.classifier.train(self.df)
        result = self.classifier.predict_single("Critical zero-day vulnerability in Apache")
        probs = result["probabilities"]
        # Modified Huber loss gives calibrated probabilities
        assert all(0 <= p <= 1 for p in probs.values())

    def test_f1_score_reasonable(self):
        """Classifier should achieve reasonable F1 on synthetic data."""
        metrics = self.classifier.train(self.df)
        # Synthetic data has distinctive vocabulary per category
        assert metrics["f1_micro"] > 0.3
