"""CLI interface for the Threat Intelligence Pipeline.

Provides commands for running the full pipeline, generating data,
training classifiers, and querying stored results.
"""

import json
import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

console = Console()


def setup_logging(verbose: bool) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
@click.version_option(version="0.1.0", prog_name="threat-intel-pipeline")
def cli(verbose: bool) -> None:
    """Threat Intelligence Pipeline - ML-powered threat analysis.

    An end-to-end pipeline for generating, processing, classifying,
    and visualizing threat intelligence data.
    """
    setup_logging(verbose)


@cli.command()
@click.option("-n", "--n-reports", default=500, show_default=True, help="Number of reports to generate.")
@click.option("--classifier", type=click.Choice(["tfidf_sgd", "transformer"]), default="tfidf_sgd", show_default=True, help="Classifier type.")
@click.option("--no-charts", is_flag=True, help="Skip chart generation.")
@click.option("--db-path", default="data/threat_intel.duckdb", show_default=True, help="Database path.")
@click.option("--output-dir", default="output/charts", show_default=True, help="Chart output directory.")
@click.option("--seed", default=42, show_default=True, help="Random seed for reproducibility.")
def run(n_reports: int, classifier: str, no_charts: bool, db_path: str, output_dir: str, seed: int) -> None:
    """Run the full threat intelligence pipeline.

    Generates synthetic threat data, processes it through the NLP pipeline,
    classifies reports, stores results, and generates visualizations.
    """
    from src.pipeline import ThreatIntelPipeline

    console.print("[bold cyan]Threat Intelligence Pipeline[/bold cyan]")
    console.print(f"Processing {n_reports} reports with {classifier} classifier\n")

    with ThreatIntelPipeline(db_path=db_path, output_dir=output_dir, seed=seed) as pipeline:
        results = pipeline.run(
            n_reports=n_reports,
            classifier_type=classifier,
            generate_charts=not no_charts,
        )

    if results.get("charts"):
        console.print(f"[green]Charts saved to {output_dir}/[/green]")

    console.print("[bold green]Pipeline complete![/bold green]")


@cli.command()
@click.option("-n", "--n-reports", default=100, show_default=True, help="Number of reports to generate.")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--output", "-o", default=None, help="Output file path (JSON).")
def generate(n_reports: int, seed: int, output: str | None) -> None:
    """Generate synthetic threat intelligence data.

    Creates realistic threat reports with embedded IOCs, threat actor
    references, and MITRE ATT&CK technique identifiers.
    """
    from src.ingestion.generator import ThreatIntelGenerator

    generator = ThreatIntelGenerator(seed=seed)
    df = generator.generate_batch(n=n_reports)

    console.print(f"[green]Generated {len(df)} threat reports[/green]")

    # Show category distribution
    table = Table(title="Category Distribution")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")

    for cat, count in df["category"].value_counts().items():
        table.add_row(cat, str(count))
    console.print(table)

    if output:
        records = df.to_dict(orient="records")
        with open(output, "w") as f:
            json.dump(records, f, indent=2, default=str)
        console.print(f"[green]Saved to {output}[/green]")


@cli.command()
@click.argument("text")
@click.option("--classifier", type=click.Choice(["tfidf_sgd", "transformer"]), default="tfidf_sgd", show_default=True)
@click.option("--db-path", default="data/threat_intel.duckdb", show_default=True)
def classify(text: str, classifier: str, db_path: str) -> None:
    """Classify a single threat intelligence text.

    Requires a trained model (run the pipeline first).
    """
    from src.classification.classifier import ThreatClassifier
    from src.processing.entity_extractor import ThreatEntityExtractor
    from src.processing.preprocessor import TextPreprocessor

    preprocessor = TextPreprocessor()
    extractor = ThreatEntityExtractor()

    cleaned = preprocessor.clean_text(text)
    entities = extractor.extract_summary(text)

    console.print("[bold]Extracted Entities:[/bold]")
    for etype, values in entities.items():
        console.print(f"  [cyan]{etype}:[/cyan] {', '.join(values)}")

    iocs = preprocessor.extract_iocs(text)
    if iocs:
        console.print("\n[bold]IOCs:[/bold]")
        for ioc_type, values in iocs.items():
            console.print(f"  [yellow]{ioc_type}:[/yellow] {', '.join(values)}")

    if classifier == "tfidf_sgd":
        clf = ThreatClassifier()
        try:
            clf.load_model("models/tfidf_sgd")
            result = clf.predict_single(cleaned)
            console.print(f"\n[bold]Classification:[/bold] {', '.join(result['predicted_labels'])}")
            console.print(f"[bold]Confidence:[/bold] {result['confidence']:.3f}")
            console.print("\n[bold]All Probabilities:[/bold]")
            for cat, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
                bar = "#" * int(prob * 40)
                console.print(f"  {cat:20s} {prob:.3f} {bar}")
        except Exception:
            console.print("\n[yellow]No trained model found. Run the pipeline first: threat-intel run[/yellow]")


@cli.command()
@click.option("--db-path", default="data/threat_intel.duckdb", show_default=True)
def stats(db_path: str) -> None:
    """Show database statistics and pipeline run history."""
    from src.storage.database import ThreatDatabase

    try:
        db = ThreatDatabase(db_path)
    except Exception as e:
        console.print(f"[red]Could not open database: {e}[/red]")
        sys.exit(1)

    db_stats = db.get_stats()

    table = Table(title="Database Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    for key, value in db_stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    console.print(table)

    runs = db.get_pipeline_runs()
    if not runs.empty:
        console.print("\n[bold]Recent Pipeline Runs:[/bold]")
        run_table = Table()
        run_table.add_column("Run ID", style="cyan")
        run_table.add_column("Status")
        run_table.add_column("Reports", justify="right")
        run_table.add_column("Classified", justify="right")
        run_table.add_column("Entities", justify="right")

        for _, run in runs.head(10).iterrows():
            status_color = "green" if run["status"] == "completed" else "yellow"
            run_table.add_row(
                str(run["run_id"])[:40],
                f"[{status_color}]{run['status']}[/{status_color}]",
                str(run.get("n_reports", "")),
                str(run.get("n_classified", "")),
                str(run.get("n_entities_extracted", "")),
            )
        console.print(run_table)

    # Classification summary
    summary = db.get_classification_summary()
    if not summary.empty:
        console.print("\n[bold]Classification Summary:[/bold]")
        cls_table = Table()
        cls_table.add_column("Category", style="cyan")
        cls_table.add_column("Count", justify="right")
        cls_table.add_column("Avg Confidence", justify="right")

        for _, row in summary.iterrows():
            cls_table.add_row(
                str(row["top_category"]),
                str(row["count"]),
                f"{row['avg_confidence']:.3f}",
            )
        console.print(cls_table)

    db.close()


@cli.command()
@click.option("--db-path", default="data/threat_intel.duckdb", show_default=True)
@click.option("--output-dir", default="output/charts", show_default=True)
def visualize(db_path: str, output_dir: str) -> None:
    """Generate visualization charts from stored data."""
    from src.storage.database import ThreatDatabase
    from src.visualization.charts import ThreatVisualizer

    try:
        db = ThreatDatabase(db_path)
    except Exception as e:
        console.print(f"[red]Could not open database: {e}[/red]")
        sys.exit(1)

    viz = ThreatVisualizer(output_dir)

    classification_df = db.get_classification_summary()
    severity_df = db.get_severity_distribution()
    timeline_df = db.get_timeline()
    entity_df = db.get_entity_summary()

    charts_saved = []

    if not classification_df.empty:
        fig = viz.classification_distribution(classification_df)
        charts_saved.extend(viz.save_figure(fig, "classification_distribution"))

    if not severity_df.empty:
        fig = viz.severity_breakdown(severity_df)
        charts_saved.extend(viz.save_figure(fig, "severity_breakdown"))

    if not timeline_df.empty:
        fig = viz.threat_timeline(timeline_df)
        charts_saved.extend(viz.save_figure(fig, "threat_timeline"))

    if not entity_df.empty:
        fig = viz.entity_frequency(entity_df)
        charts_saved.extend(viz.save_figure(fig, "entity_frequency"))

    if not classification_df.empty and not severity_df.empty:
        fig = viz.pipeline_dashboard(classification_df, severity_df, timeline_df, entity_df)
        charts_saved.extend(viz.save_figure(fig, "dashboard"))

    console.print(f"[green]Saved {len(charts_saved)} chart files to {output_dir}/[/green]")
    for path in charts_saved:
        console.print(f"  {path}")

    db.close()


if __name__ == "__main__":
    cli()
