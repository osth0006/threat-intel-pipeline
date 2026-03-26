"""Tests for the DuckDB storage layer."""

import tempfile
from pathlib import Path

import pytest

from src.ingestion.generator import ThreatIntelGenerator
from src.processing.preprocessor import TextPreprocessor
from src.storage.database import ThreatDatabase


class TestThreatDatabase:
    """Test suite for ThreatDatabase."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up test database."""
        self.db_path = tmp_path / "test.duckdb"
        self.db = ThreatDatabase(self.db_path)

        generator = ThreatIntelGenerator(seed=42)
        preprocessor = TextPreprocessor()
        self.df = preprocessor.process_dataframe(generator.generate_batch(n=20, seed=42))

        yield
        self.db.close()

    def test_schema_creation(self):
        stats = self.db.get_stats()
        assert stats["total_reports"] == 0

    def test_store_reports(self):
        n = self.db.store_reports(self.df)
        assert n == 20
        assert self.db.get_stats()["total_reports"] == 20

    def test_get_reports(self):
        self.db.store_reports(self.df)
        reports = self.db.get_reports(limit=10)
        assert len(reports) == 10

    def test_get_severity_distribution(self):
        self.db.store_reports(self.df)
        dist = self.db.get_severity_distribution()
        assert not dist.empty
        assert "severity" in dist.columns
        assert "count" in dist.columns

    def test_pipeline_run_tracking(self):
        self.db.start_pipeline_run("test-run-1", "tfidf_sgd")
        self.db.complete_pipeline_run("test-run-1", 20, 20, 100, {"f1": 0.8})
        runs = self.db.get_pipeline_runs()
        assert len(runs) == 1
        assert runs.iloc[0]["status"] == "completed"

    def test_context_manager(self, tmp_path):
        db_path = tmp_path / "ctx.duckdb"
        with ThreatDatabase(db_path) as db:
            stats = db.get_stats()
            assert stats["total_reports"] == 0
