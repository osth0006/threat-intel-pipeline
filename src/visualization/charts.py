"""Plotly-based visualizations for threat intelligence analysis.

Generates interactive charts showing classification distributions,
threat timelines, entity networks, and pipeline performance metrics.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Consistent color scheme for threat categories
CATEGORY_COLORS = {
    "apt": "#e74c3c",
    "malware": "#e67e22",
    "phishing": "#f39c12",
    "vulnerability": "#2ecc71",
    "ransomware": "#9b59b6",
    "supply_chain": "#3498db",
    "insider_threat": "#1abc9c",
    "ddos": "#34495e",
    "data_exfiltration": "#e84393",
    "zero_day": "#d63031",
}

SEVERITY_COLORS = {
    "critical": "#e74c3c",
    "high": "#e67e22",
    "medium": "#f39c12",
    "low": "#2ecc71",
    "informational": "#3498db",
}


class ThreatVisualizer:
    """Creates analytical visualizations of threat intelligence data.

    All methods return Plotly figure objects that can be displayed
    interactively or exported as static images.
    """

    def __init__(self, output_dir: str | Path = "output/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def classification_distribution(
        self,
        df: pd.DataFrame,
        title: str = "Threat Classification Distribution",
    ) -> go.Figure:
        """Bar chart of classification category distribution.

        Args:
            df: DataFrame with 'top_category' and 'count' columns.
            title: Chart title.

        Returns:
            Plotly Figure object.
        """
        colors = [CATEGORY_COLORS.get(cat, "#95a5a6") for cat in df["top_category"]]

        fig = go.Figure(data=[
            go.Bar(
                x=df["top_category"],
                y=df["count"],
                marker_color=colors,
                text=df["count"],
                textposition="auto",
            )
        ])
        fig.update_layout(
            title=title,
            xaxis_title="Threat Category",
            yaxis_title="Number of Reports",
            template="plotly_dark",
            font=dict(size=12),
            height=500,
        )
        return fig

    def severity_breakdown(
        self,
        df: pd.DataFrame,
        title: str = "Threat Severity Distribution",
    ) -> go.Figure:
        """Pie chart of severity level distribution.

        Args:
            df: DataFrame with 'severity' and 'count' columns.
            title: Chart title.

        Returns:
            Plotly Figure object.
        """
        colors = [SEVERITY_COLORS.get(s, "#95a5a6") for s in df["severity"]]

        fig = go.Figure(data=[
            go.Pie(
                labels=df["severity"],
                values=df["count"],
                marker=dict(colors=colors),
                hole=0.4,
                textinfo="label+percent",
            )
        ])
        fig.update_layout(
            title=title,
            template="plotly_dark",
            font=dict(size=12),
            height=500,
        )
        return fig

    def threat_timeline(
        self,
        df: pd.DataFrame,
        title: str = "Threat Activity Timeline",
    ) -> go.Figure:
        """Area chart showing threat activity over time by category.

        Args:
            df: DataFrame with 'date', 'category', and 'count' columns.
            title: Chart title.

        Returns:
            Plotly Figure object.
        """
        fig = px.area(
            df,
            x="date",
            y="count",
            color="category",
            color_discrete_map=CATEGORY_COLORS,
            title=title,
            template="plotly_dark",
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Report Count",
            font=dict(size=12),
            height=500,
            legend_title="Category",
        )
        return fig

    def confidence_distribution(
        self,
        df: pd.DataFrame,
        title: str = "Classification Confidence Distribution",
    ) -> go.Figure:
        """Histogram of prediction confidence scores.

        Args:
            df: DataFrame with 'prediction_confidence' column.
            title: Chart title.

        Returns:
            Plotly Figure object.
        """
        fig = go.Figure(data=[
            go.Histogram(
                x=df["prediction_confidence"],
                nbinsx=30,
                marker_color="#3498db",
                opacity=0.8,
            )
        ])
        fig.update_layout(
            title=title,
            xaxis_title="Confidence Score",
            yaxis_title="Count",
            template="plotly_dark",
            font=dict(size=12),
            height=400,
        )
        return fig

    def entity_frequency(
        self,
        df: pd.DataFrame,
        title: str = "Top Extracted Entities",
        n_top: int = 20,
    ) -> go.Figure:
        """Horizontal bar chart of most frequently extracted entities.

        Args:
            df: DataFrame with 'entity_type', 'entity_value', 'occurrences'.
            title: Chart title.
            n_top: Number of top entities to show.

        Returns:
            Plotly Figure object.
        """
        top_df = df.head(n_top).sort_values("occurrences")

        fig = go.Figure(data=[
            go.Bar(
                y=top_df["entity_value"].str[:40],
                x=top_df["occurrences"],
                orientation="h",
                marker_color=top_df["entity_type"].map({
                    "threat_actors": "#e74c3c",
                    "malware": "#e67e22",
                    "mitre_techniques": "#3498db",
                    "ioc_ipv4": "#2ecc71",
                    "ioc_domain": "#9b59b6",
                    "ioc_cve": "#f39c12",
                }).fillna("#95a5a6"),
                text=top_df["entity_type"],
                textposition="auto",
            )
        ])
        fig.update_layout(
            title=title,
            xaxis_title="Occurrences",
            yaxis_title="Entity",
            template="plotly_dark",
            font=dict(size=11),
            height=600,
            margin=dict(l=200),
        )
        return fig

    def pipeline_dashboard(
        self,
        classification_df: pd.DataFrame,
        severity_df: pd.DataFrame,
        timeline_df: pd.DataFrame,
        entity_df: pd.DataFrame,
    ) -> go.Figure:
        """Combined dashboard with multiple chart panels.

        Args:
            classification_df: Classification distribution data.
            severity_df: Severity distribution data.
            timeline_df: Timeline data.
            entity_df: Entity frequency data.

        Returns:
            Plotly Figure with subplots.
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Classification Distribution",
                "Severity Breakdown",
                "Threat Timeline",
                "Top Entities",
            ),
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "bar"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        # Classification distribution
        colors = [CATEGORY_COLORS.get(cat, "#95a5a6") for cat in classification_df["top_category"]]
        fig.add_trace(
            go.Bar(
                x=classification_df["top_category"],
                y=classification_df["count"],
                marker_color=colors,
                showlegend=False,
            ),
            row=1, col=1,
        )

        # Severity pie
        sev_colors = [SEVERITY_COLORS.get(s, "#95a5a6") for s in severity_df["severity"]]
        fig.add_trace(
            go.Pie(
                labels=severity_df["severity"],
                values=severity_df["count"],
                marker=dict(colors=sev_colors),
                hole=0.4,
                showlegend=True,
            ),
            row=1, col=2,
        )

        # Timeline
        if not timeline_df.empty:
            for category in timeline_df["category"].unique():
                cat_data = timeline_df[timeline_df["category"] == category]
                fig.add_trace(
                    go.Scatter(
                        x=cat_data["date"],
                        y=cat_data["count"],
                        name=category,
                        mode="lines+markers",
                        line=dict(color=CATEGORY_COLORS.get(category, "#95a5a6")),
                        showlegend=False,
                    ),
                    row=2, col=1,
                )

        # Top entities
        top_entities = entity_df.head(10).sort_values("occurrences")
        if not top_entities.empty:
            fig.add_trace(
                go.Bar(
                    y=top_entities["entity_value"].str[:30],
                    x=top_entities["occurrences"],
                    orientation="h",
                    marker_color="#3498db",
                    showlegend=False,
                ),
                row=2, col=2,
            )

        fig.update_layout(
            title="Threat Intelligence Pipeline Dashboard",
            template="plotly_dark",
            height=900,
            width=1400,
            font=dict(size=11),
        )
        return fig

    def save_figure(self, fig: go.Figure, filename: str, formats: list[str] | None = None) -> list[Path]:
        """Save a figure to disk in specified formats.

        Args:
            fig: Plotly Figure to save.
            filename: Base filename (without extension).
            formats: List of formats (html, png, json). Defaults to [html, json].

        Returns:
            List of saved file paths.
        """
        if formats is None:
            formats = ["html", "json"]

        saved = []
        for fmt in formats:
            path = self.output_dir / f"{filename}.{fmt}"
            if fmt == "html":
                fig.write_html(str(path), include_plotlyjs="cdn")
            elif fmt == "png":
                fig.write_image(str(path), scale=2)
            elif fmt == "json":
                fig.write_json(str(path))
            saved.append(path)
            logger.info("Saved chart: %s", path)

        return saved
