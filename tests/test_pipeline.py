"""Integration tests for the full pipeline."""

import tempfile
from pathlib import Path

import pytest

from src.pipeline import ThreatIntelPipeline


class TestThreatIntelPipeline:
    """Integration test suite for the end-to-end pipeline."""

    def test_full_pipeline_run(self, tmp_path):
        """Test complete pipeline execution with small dataset."""
        db_path = tmp_path / "test.duckdb"
        output_dir = tmp_path / "charts"

        with ThreatIntelPipeline(
            db_path=str(db_path),
            output_dir=str(output_dir),
            seed=42,
        ) as pipeline:
            results = pipeline.run(
                n_reports=50,
                generate_charts=True,
                verbose=False,
            )

        assert results["status"] == "completed"
        assert results["n_reports"] == 50
        assert results["n_classified"] == 50
        assert results["total_entities"] > 0
        assert results["total_iocs"] > 0
        assert results["train_metrics"]["f1_micro"] > 0

    def test_pipeline_no_charts(self, tmp_path):
        """Test pipeline without chart generation."""
        db_path = tmp_path / "test.duckdb"

        with ThreatIntelPipeline(
            db_path=str(db_path),
            output_dir=str(tmp_path / "charts"),
            seed=42,
        ) as pipeline:
            results = pipeline.run(
                n_reports=30,
                generate_charts=False,
                verbose=False,
            )

        assert results["status"] == "completed"
        assert "charts" not in results

    def test_pipeline_db_stats(self, tmp_path):
        """Test that pipeline populates database correctly."""
        db_path = tmp_path / "test.duckdb"

        with ThreatIntelPipeline(
            db_path=str(db_path),
            output_dir=str(tmp_path / "charts"),
            seed=42,
        ) as pipeline:
            results = pipeline.run(n_reports=30, generate_charts=False, verbose=False)
            stats = results["db_stats"]

        assert stats["total_reports"] == 30
        assert stats["total_classifications"] == 30
        assert stats["total_entities"] > 0
        assert stats["pipeline_runs"] == 1
