"""Transformer-based threat classification using HuggingFace models.

Uses DistilBERT as a lightweight transformer for multi-label
classification of threat intelligence reports. Supports both
zero-shot classification and fine-tuning workflows.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from src.ingestion.generator import THREAT_CATEGORIES

logger = logging.getLogger(__name__)

# Labels formatted for zero-shot classification
CATEGORY_DESCRIPTIONS = {
    "apt": "advanced persistent threat state-sponsored espionage campaign",
    "malware": "malicious software virus trojan backdoor rootkit",
    "phishing": "phishing social engineering credential theft spear-phishing",
    "vulnerability": "software vulnerability CVE exploit security flaw",
    "ransomware": "ransomware encryption extortion data hostage",
    "supply_chain": "supply chain compromise software dependency attack",
    "insider_threat": "insider threat unauthorized access data theft employee",
    "ddos": "distributed denial of service DDoS network flood attack",
    "data_exfiltration": "data exfiltration theft stolen information leakage",
    "zero_day": "zero day exploit unknown vulnerability unpatched",
}


class TransformerClassifier:
    """Transformer-based multi-label threat classifier.

    Provides two classification modes:
    1. Zero-shot classification using NLI models (no training needed)
    2. Fine-tuned classification using DistilBERT (requires training data)

    The zero-shot mode is useful for rapid prototyping and handles
    unseen categories, while fine-tuned mode provides higher accuracy
    on known threat categories.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        mode: str = "zero-shot",
        device: str | None = None,
    ):
        """Initialize the transformer classifier.

        Args:
            model_name: HuggingFace model identifier.
            mode: Classification mode - "zero-shot" or "fine-tuned".
            device: Device for inference ("cpu", "cuda", "mps").
        """
        self.model_name = model_name
        self.mode = mode
        self.categories = THREAT_CATEGORIES
        self.category_descriptions = CATEGORY_DESCRIPTIONS

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._pipeline = None
        self._tokenizer = None
        self._model = None

    def _init_zero_shot(self) -> None:
        """Initialize the zero-shot classification pipeline."""
        logger.info("Initializing zero-shot classifier with facebook/bart-large-mnli")
        self._pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device if self.device != "cpu" else -1,
        )

    def _init_fine_tuned(self) -> None:
        """Initialize tokenizer and model for fine-tuned classification."""
        logger.info("Initializing fine-tuned classifier with %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.categories),
            problem_type="multi_label_classification",
        )
        self._model.to(self.device)
        self._model.eval()

    def predict_zero_shot(
        self,
        texts: list[str],
        threshold: float = 0.3,
        multi_label: bool = True,
    ) -> list[dict[str, Any]]:
        """Classify texts using zero-shot NLI-based approach.

        Args:
            texts: List of text strings to classify.
            threshold: Minimum confidence threshold for label assignment.
            multi_label: Allow multiple labels per text.

        Returns:
            List of prediction dictionaries.
        """
        if self._pipeline is None:
            self._init_zero_shot()

        candidate_labels = list(self.category_descriptions.values())
        label_map = dict(zip(candidate_labels, self.categories))

        results = []
        for text in texts:
            # Truncate to model max length
            truncated = text[:1024]
            output = self._pipeline(
                truncated,
                candidate_labels=candidate_labels,
                multi_label=multi_label,
            )

            label_scores = {}
            for label, score in zip(output["labels"], output["scores"]):
                category = label_map[label]
                label_scores[category] = float(score)

            predicted = [cat for cat, score in label_scores.items() if score >= threshold]
            if not predicted:
                predicted = [max(label_scores, key=label_scores.get)]

            results.append({
                "predicted_labels": predicted,
                "probabilities": label_scores,
                "top_category": max(label_scores, key=label_scores.get),
                "confidence": float(max(label_scores.values())),
            })

        return results

    def predict_fine_tuned(
        self,
        texts: list[str],
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Classify texts using fine-tuned transformer model.

        Args:
            texts: List of text strings to classify.
            threshold: Sigmoid threshold for label assignment.

        Returns:
            List of prediction dictionaries.
        """
        if self._tokenizer is None or self._model is None:
            self._init_fine_tuned()

        results = []
        # Process in batches
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.sigmoid(outputs.logits).cpu().numpy()

            for prob_row in probs:
                label_scores = {
                    cat: float(prob)
                    for cat, prob in zip(self.categories, prob_row)
                }
                predicted = [
                    cat for cat, prob in label_scores.items()
                    if prob >= threshold
                ]
                if not predicted:
                    predicted = [max(label_scores, key=label_scores.get)]

                results.append({
                    "predicted_labels": predicted,
                    "probabilities": label_scores,
                    "top_category": max(label_scores, key=label_scores.get),
                    "confidence": float(max(label_scores.values())),
                })

        return results

    def predict(self, texts: list[str], **kwargs) -> list[dict[str, Any]]:
        """Classify texts using the configured mode.

        Args:
            texts: List of text strings to classify.
            **kwargs: Additional arguments passed to the mode-specific method.

        Returns:
            List of prediction dictionaries.
        """
        if self.mode == "zero-shot":
            return self.predict_zero_shot(texts, **kwargs)
        elif self.mode == "fine-tuned":
            return self.predict_fine_tuned(texts, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def predict_single(self, text: str, **kwargs) -> dict[str, Any]:
        """Classify a single text."""
        return self.predict([text], **kwargs)[0]

    def get_model_info(self) -> dict[str, str]:
        """Return model configuration details."""
        return {
            "model_name": self.model_name,
            "mode": self.mode,
            "device": self.device,
            "n_categories": str(len(self.categories)),
            "categories": ", ".join(self.categories),
        }
