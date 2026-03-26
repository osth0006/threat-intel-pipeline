"""Threat classification engine with ML and transformer models."""

from src.classification.classifier import ThreatClassifier

__all__ = ["ThreatClassifier", "TransformerClassifier"]


def __getattr__(name: str):
    if name == "TransformerClassifier":
        from src.classification.transformer_classifier import TransformerClassifier
        return TransformerClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
