"""Tests for the synthetic threat intelligence data generator."""

import pandas as pd
import pytest

from src.ingestion.generator import (
    THREAT_CATEGORIES,
    ThreatIntelGenerator,
)


class TestThreatIntelGenerator:
    """Test suite for ThreatIntelGenerator."""

    def setup_method(self):
        self.generator = ThreatIntelGenerator(seed=42)

    def test_generate_single_report(self):
        report = self.generator.generate_report()
        assert "report_id" in report
        assert "text" in report
        assert "category" in report
        assert "labels" in report
        assert "severity" in report
        assert report["category"] in THREAT_CATEGORIES

    def test_generate_report_with_category(self):
        report = self.generator.generate_report(category="apt")
        assert report["category"] == "apt"
        assert "apt" in report["labels"]

    def test_generate_report_invalid_category(self):
        with pytest.raises(ValueError, match="Invalid category"):
            self.generator.generate_report(category="nonexistent")

    def test_generate_batch(self):
        df = self.generator.generate_batch(n=50)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert "report_id" in df.columns
        assert "text" in df.columns
        assert "category" in df.columns

    def test_generate_batch_reproducible(self):
        df1 = ThreatIntelGenerator(seed=123).generate_batch(n=10, seed=123)
        df2 = ThreatIntelGenerator(seed=123).generate_batch(n=10, seed=123)
        assert df1["text"].tolist() == df2["text"].tolist()

    def test_report_has_all_fields(self):
        report = self.generator.generate_report()
        required_fields = [
            "report_id", "timestamp", "title", "text", "category",
            "labels", "severity", "confidence", "source", "tlp",
        ]
        for field in required_fields:
            assert field in report, f"Missing field: {field}"

    def test_report_text_not_empty(self):
        for _ in range(20):
            report = self.generator.generate_report()
            assert len(report["text"]) > 50

    def test_all_categories_generated(self):
        df = self.generator.generate_batch(n=500)
        generated_categories = set(df["category"].unique())
        assert generated_categories == set(THREAT_CATEGORIES)

    def test_report_ids_unique(self):
        df = self.generator.generate_batch(n=100)
        assert df["report_id"].nunique() == 100
