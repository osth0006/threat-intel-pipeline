"""DuckDB storage layer for threat intelligence data.

Provides persistent storage for processed threat reports, classification
results, extracted entities, and pipeline run metadata. Uses DuckDB for
fast analytical queries on threat data.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS threat_reports (
    report_id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP,
    title VARCHAR,
    text TEXT,
    cleaned_text TEXT,
    category VARCHAR,
    labels JSON,
    severity VARCHAR,
    confidence DOUBLE,
    source VARCHAR,
    tlp VARCHAR,
    token_count INTEGER,
    ioc_count INTEGER,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS classifications (
    id INTEGER PRIMARY KEY,
    report_id VARCHAR REFERENCES threat_reports(report_id),
    classifier_type VARCHAR,
    predicted_labels JSON,
    probabilities JSON,
    top_category VARCHAR,
    prediction_confidence DOUBLE,
    classified_at TIMESTAMP DEFAULT current_timestamp
);

CREATE SEQUENCE IF NOT EXISTS classifications_id_seq START 1;

CREATE TABLE IF NOT EXISTS extracted_entities (
    id INTEGER PRIMARY KEY,
    report_id VARCHAR REFERENCES threat_reports(report_id),
    entity_type VARCHAR,
    entity_value VARCHAR,
    start_pos INTEGER,
    end_pos INTEGER,
    confidence DOUBLE DEFAULT 1.0,
    extracted_at TIMESTAMP DEFAULT current_timestamp
);

CREATE SEQUENCE IF NOT EXISTS entities_id_seq START 1;

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id VARCHAR PRIMARY KEY,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    n_reports INTEGER,
    n_classified INTEGER,
    n_entities_extracted INTEGER,
    classifier_type VARCHAR,
    metrics JSON,
    status VARCHAR DEFAULT 'running'
);
"""


class ThreatDatabase:
    """DuckDB-backed storage for threat intelligence pipeline data.

    Manages schema creation, data insertion, and analytical queries
    for processed threat reports and classification results.
    """

    def __init__(self, db_path: str | Path = "data/threat_intel.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        for statement in SCHEMA_SQL.split(";"):
            statement = statement.strip()
            if statement:
                self.conn.execute(statement)
        logger.info("Database schema initialized at %s", self.db_path)

    def store_reports(self, df: pd.DataFrame) -> int:
        """Store processed threat reports.

        Args:
            df: DataFrame with report data including cleaned_text and IOC counts.

        Returns:
            Number of reports stored.
        """
        records = []
        for _, row in df.iterrows():
            records.append({
                "report_id": row["report_id"],
                "timestamp": row["timestamp"],
                "title": row["title"],
                "text": row["text"],
                "cleaned_text": row.get("cleaned_text", row["text"]),
                "category": row["category"],
                "labels": json.dumps(row["labels"]),
                "severity": row["severity"],
                "confidence": row["confidence"],
                "source": row["source"],
                "tlp": row["tlp"],
                "token_count": int(row.get("token_count", 0)),
                "ioc_count": int(row.get("ioc_count", 0)),
            })

        insert_df = pd.DataFrame(records)
        self.conn.execute("""
            INSERT OR REPLACE INTO threat_reports
            (report_id, timestamp, title, text, cleaned_text, category,
             labels, severity, confidence, source, tlp, token_count, ioc_count)
            SELECT * FROM insert_df
        """)

        logger.info("Stored %d threat reports", len(records))
        return len(records)

    def store_classifications(
        self,
        report_ids: list[str],
        predictions: list[dict[str, Any]],
        classifier_type: str = "tfidf_sgd",
    ) -> int:
        """Store classification results.

        Args:
            report_ids: List of report IDs.
            predictions: List of prediction dictionaries.
            classifier_type: Identifier for the classifier used.

        Returns:
            Number of classifications stored.
        """
        for rid, pred in zip(report_ids, predictions):
            self.conn.execute("""
                INSERT INTO classifications
                (id, report_id, classifier_type, predicted_labels,
                 probabilities, top_category, prediction_confidence)
                VALUES (nextval('classifications_id_seq'), ?, ?, ?, ?, ?, ?)
            """, [
                rid,
                classifier_type,
                json.dumps(pred["predicted_labels"]),
                json.dumps(pred["probabilities"]),
                pred["top_category"],
                pred["confidence"],
            ])

        logger.info("Stored %d classifications", len(predictions))
        return len(predictions)

    def store_entities(
        self,
        report_id: str,
        entities: dict[str, list],
    ) -> int:
        """Store extracted entities for a report.

        Args:
            report_id: Report identifier.
            entities: Dictionary mapping entity types to entity lists.

        Returns:
            Number of entities stored.
        """
        count = 0
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                self.conn.execute("""
                    INSERT INTO extracted_entities
                    (id, report_id, entity_type, entity_value, start_pos, end_pos, confidence)
                    VALUES (nextval('entities_id_seq'), ?, ?, ?, ?, ?, ?)
                """, [
                    report_id,
                    entity_type,
                    entity.text,
                    entity.start,
                    entity.end,
                    entity.confidence,
                ])
                count += 1

        return count

    def start_pipeline_run(self, run_id: str, classifier_type: str) -> None:
        """Record the start of a pipeline run."""
        self.conn.execute("""
            INSERT INTO pipeline_runs (run_id, started_at, classifier_type, status)
            VALUES (?, ?, ?, 'running')
        """, [run_id, datetime.now(), classifier_type])

    def complete_pipeline_run(
        self,
        run_id: str,
        n_reports: int,
        n_classified: int,
        n_entities: int,
        metrics: dict,
    ) -> None:
        """Record completion of a pipeline run."""
        self.conn.execute("""
            UPDATE pipeline_runs
            SET completed_at = ?, n_reports = ?, n_classified = ?,
                n_entities_extracted = ?, metrics = ?, status = 'completed'
            WHERE run_id = ?
        """, [datetime.now(), n_reports, n_classified, n_entities, json.dumps(metrics), run_id])

    # --- Query methods ---

    def get_reports(self, limit: int = 100) -> pd.DataFrame:
        """Retrieve stored threat reports."""
        return self.conn.execute(
            "SELECT * FROM threat_reports ORDER BY timestamp DESC LIMIT ?", [limit]
        ).fetchdf()

    def get_classification_summary(self) -> pd.DataFrame:
        """Get classification distribution summary."""
        return self.conn.execute("""
            SELECT
                top_category,
                COUNT(*) as count,
                ROUND(AVG(prediction_confidence), 3) as avg_confidence
            FROM classifications
            GROUP BY top_category
            ORDER BY count DESC
        """).fetchdf()

    def get_severity_distribution(self) -> pd.DataFrame:
        """Get severity level distribution."""
        return self.conn.execute("""
            SELECT severity, COUNT(*) as count
            FROM threat_reports
            GROUP BY severity
            ORDER BY count DESC
        """).fetchdf()

    def get_timeline(self) -> pd.DataFrame:
        """Get threat report timeline data."""
        return self.conn.execute("""
            SELECT
                DATE_TRUNC('day', CAST(timestamp AS TIMESTAMP)) as date,
                category,
                COUNT(*) as count
            FROM threat_reports
            GROUP BY date, category
            ORDER BY date
        """).fetchdf()

    def get_entity_summary(self) -> pd.DataFrame:
        """Get summary of extracted entities."""
        return self.conn.execute("""
            SELECT
                entity_type,
                entity_value,
                COUNT(*) as occurrences
            FROM extracted_entities
            GROUP BY entity_type, entity_value
            ORDER BY occurrences DESC
            LIMIT 50
        """).fetchdf()

    def get_pipeline_runs(self) -> pd.DataFrame:
        """Get history of pipeline runs."""
        return self.conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY started_at DESC"
        ).fetchdf()

    def get_stats(self) -> dict[str, int]:
        """Get database statistics."""
        reports = self.conn.execute("SELECT COUNT(*) FROM threat_reports").fetchone()[0]
        classifications = self.conn.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
        entities = self.conn.execute("SELECT COUNT(*) FROM extracted_entities").fetchone()[0]
        runs = self.conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
        return {
            "total_reports": reports,
            "total_classifications": classifications,
            "total_entities": entities,
            "pipeline_runs": runs,
        }

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
