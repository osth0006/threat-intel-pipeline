"""End-to-end pipeline orchestrator for threat intelligence processing.

Coordinates data generation, NLP processing, classification, storage,
and visualization into a single executable pipeline.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.classification.classifier import ThreatClassifier
from src.ingestion.generator import ThreatIntelGenerator
from src.processing.entity_extractor import ThreatEntityExtractor
from src.processing.preprocessor import TextPreprocessor
from src.storage.database import ThreatDatabase
from src.visualization.charts import ThreatVisualizer

logger = logging.getLogger(__name__)
console = Console()


class ThreatIntelPipeline:
    """Orchestrates the full threat intelligence processing pipeline.

    Stages:
        1. Ingestion: Generate or load threat intelligence reports
        2. Preprocessing: Clean and normalize text, extract IOCs
        3. Entity Extraction: Identify threat actors, malware, techniques
        4. Classification: Multi-label categorization of reports
        5. Storage: Persist results to DuckDB
        6. Visualization: Generate analytical charts

    Example:
        >>> pipeline = ThreatIntelPipeline()
        >>> results = pipeline.run(n_reports=500)
        >>> print(results["metrics"])
    """

    def __init__(
        self,
        db_path: str = "data/threat_intel.duckdb",
        output_dir: str = "output/charts",
        seed: int = 42,
    ):
        self.generator = ThreatIntelGenerator(seed=seed)
        self.preprocessor = TextPreprocessor()
        self.entity_extractor = ThreatEntityExtractor()
        self.classifier = ThreatClassifier()
        self.db = ThreatDatabase(db_path)
        self.visualizer = ThreatVisualizer(output_dir)
        self.seed = seed

    def run(
        self,
        n_reports: int = 500,
        classifier_type: str = "tfidf_sgd",
        generate_charts: bool = True,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Execute the full pipeline.

        Args:
            n_reports: Number of synthetic reports to generate.
            classifier_type: Classifier to use ("tfidf_sgd" or "transformer").
            generate_charts: Whether to generate visualization charts.
            verbose: Print progress to console.

        Returns:
            Dictionary containing pipeline results and metrics.
        """
        run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.db.start_pipeline_run(run_id, classifier_type)

        results: dict[str, Any] = {"run_id": run_id}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not verbose,
        ) as progress:

            # Stage 1: Ingestion
            task = progress.add_task("Generating synthetic threat reports...", total=None)
            df = self.generator.generate_batch(n=n_reports, seed=self.seed)
            results["n_reports"] = len(df)
            progress.update(task, completed=True, description=f"[green]Generated {len(df)} reports")

            # Stage 2: Preprocessing
            task = progress.add_task("Preprocessing text and extracting IOCs...", total=None)
            df = self.preprocessor.process_dataframe(df)
            results["avg_token_count"] = float(df["token_count"].mean())
            results["total_iocs"] = int(df["ioc_count"].sum())
            progress.update(task, completed=True, description=f"[green]Preprocessed {len(df)} reports, found {results['total_iocs']} IOCs")

            # Stage 3: Entity Extraction
            task = progress.add_task("Extracting threat entities...", total=None)
            total_entities = 0
            entity_records = []
            for _, row in df.iterrows():
                entities = self.entity_extractor.extract_all(row["text"])
                entity_count = sum(len(v) for v in entities.values())
                total_entities += entity_count
                entity_records.append(entities)
                self.db.store_entities(row["report_id"], entities)

            df["entities"] = entity_records
            results["total_entities"] = total_entities
            progress.update(task, completed=True, description=f"[green]Extracted {total_entities} entities")

            # Stage 4: Classification
            task = progress.add_task("Training classifier and predicting...", total=None)
            train_metrics = self.classifier.train(df)
            predictions = self.classifier.predict(df["cleaned_text"].tolist())
            results["train_metrics"] = train_metrics
            results["n_classified"] = len(predictions)
            progress.update(task, completed=True, description=f"[green]Classified {len(predictions)} reports (F1={train_metrics['f1_micro']:.3f})")

            # Stage 5: Storage
            task = progress.add_task("Storing results in DuckDB...", total=None)
            self.db.store_reports(df)
            self.db.store_classifications(
                df["report_id"].tolist(), predictions, classifier_type
            )
            progress.update(task, completed=True, description="[green]Results stored in DuckDB")

            # Stage 6: Visualization
            if generate_charts:
                task = progress.add_task("Generating visualizations...", total=None)
                charts = self._generate_charts()
                results["charts"] = charts
                progress.update(task, completed=True, description=f"[green]Generated {len(charts)} charts")

        # Complete pipeline run
        self.db.complete_pipeline_run(
            run_id=run_id,
            n_reports=len(df),
            n_classified=len(predictions),
            n_entities=total_entities,
            metrics=train_metrics,
        )

        results["status"] = "completed"
        results["db_stats"] = self.db.get_stats()

        if verbose:
            self._print_summary(results)

        return results

    def _generate_charts(self) -> list[str]:
        """Generate all visualization charts."""
        chart_files = []

        classification_df = self.db.get_classification_summary()
        if not classification_df.empty:
            fig = self.visualizer.classification_distribution(classification_df)
            paths = self.visualizer.save_figure(fig, "classification_distribution")
            chart_files.extend(str(p) for p in paths)

        severity_df = self.db.get_severity_distribution()
        if not severity_df.empty:
            fig = self.visualizer.severity_breakdown(severity_df)
            paths = self.visualizer.save_figure(fig, "severity_breakdown")
            chart_files.extend(str(p) for p in paths)

        timeline_df = self.db.get_timeline()
        if not timeline_df.empty:
            fig = self.visualizer.threat_timeline(timeline_df)
            paths = self.visualizer.save_figure(fig, "threat_timeline")
            chart_files.extend(str(p) for p in paths)

        entity_df = self.db.get_entity_summary()
        if not entity_df.empty:
            fig = self.visualizer.entity_frequency(entity_df)
            paths = self.visualizer.save_figure(fig, "entity_frequency")
            chart_files.extend(str(p) for p in paths)

        # Dashboard
        if not classification_df.empty and not severity_df.empty:
            fig = self.visualizer.pipeline_dashboard(
                classification_df, severity_df, timeline_df, entity_df,
            )
            paths = self.visualizer.save_figure(fig, "dashboard")
            chart_files.extend(str(p) for p in paths)

        return chart_files

    def _print_summary(self, results: dict[str, Any]) -> None:
        """Print a formatted summary of pipeline results."""
        console.print()
        console.rule("[bold cyan]Pipeline Results Summary")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Run ID", results["run_id"])
        table.add_row("Reports Generated", str(results["n_reports"]))
        table.add_row("Reports Classified", str(results["n_classified"]))
        table.add_row("Entities Extracted", str(results["total_entities"]))
        table.add_row("IOCs Found", str(results["total_iocs"]))
        table.add_row("Avg Token Count", f"{results['avg_token_count']:.1f}")

        metrics = results.get("train_metrics", {})
        table.add_row("F1 (micro)", f"{metrics.get('f1_micro', 0):.4f}")
        table.add_row("F1 (macro)", f"{metrics.get('f1_macro', 0):.4f}")
        table.add_row("F1 (weighted)", f"{metrics.get('f1_weighted', 0):.4f}")

        stats = results.get("db_stats", {})
        table.add_row("DB Reports", str(stats.get("total_reports", 0)))
        table.add_row("DB Classifications", str(stats.get("total_classifications", 0)))
        table.add_row("DB Entities", str(stats.get("total_entities", 0)))

        console.print(table)
        console.print()

    def close(self) -> None:
        """Clean up resources."""
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
