"""Text preprocessing pipeline for threat intelligence reports.

Handles tokenization, normalization, and feature extraction from
unstructured threat intelligence text.
"""

import re
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing steps."""
    lowercase: bool = True
    remove_urls: bool = False  # URLs are IOCs - preserve by default
    normalize_whitespace: bool = True
    remove_special_chars: bool = False  # Preserve technical indicators
    min_token_length: int = 2
    max_token_length: int = 100


# Patterns for threat intelligence indicators
IOC_PATTERNS = {
    "ipv4": re.compile(
        r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
    ),
    "ipv4_port": re.compile(
        r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?):\d{1,5}\b'
    ),
    "domain": re.compile(
        r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+(?:com|net|org|gov|mil|edu|xyz|info|cc|top|io|ru|cn)\b'
    ),
    "url": re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+'
    ),
    "sha256": re.compile(r'\b[a-fA-F0-9]{64}\b'),
    "sha1": re.compile(r'\b[a-fA-F0-9]{40}\b'),
    "md5": re.compile(r'\b[a-fA-F0-9]{32}\b'),
    "cve": re.compile(r'CVE-\d{4}-\d{4,7}'),
    "email": re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
    "mitre_technique": re.compile(r'\bT\d{4}(?:\.\d{3})?\b'),
}


class TextPreprocessor:
    """Preprocesses threat intelligence text for downstream NLP tasks.

    Performs text normalization while preserving technical indicators
    (IP addresses, hashes, CVEs, MITRE technique IDs) that are critical
    for threat analysis.
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()

    def extract_iocs(self, text: str) -> dict[str, list[str]]:
        """Extract indicators of compromise from text.

        Args:
            text: Raw threat intelligence text.

        Returns:
            Dictionary mapping IOC types to lists of extracted values.
        """
        iocs: dict[str, list[str]] = {}
        for ioc_type, pattern in IOC_PATTERNS.items():
            matches = list(set(pattern.findall(text)))
            if matches:
                iocs[ioc_type] = sorted(matches)
        return iocs

    def normalize_text(self, text: str) -> str:
        """Normalize text while preserving technical indicators.

        Args:
            text: Raw text to normalize.

        Returns:
            Normalized text string.
        """
        if self.config.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()

        if self.config.lowercase:
            # Preserve case for known technical identifiers
            preserved = {}
            for ioc_type, pattern in IOC_PATTERNS.items():
                for match in pattern.finditer(text):
                    key = f"__IOC_{ioc_type}_{len(preserved)}__"
                    preserved[key] = match.group()
                    text = text.replace(match.group(), key, 1)

            text = text.lower()

            for key, original in preserved.items():
                text = text.replace(key.lower(), original)

        return text

    def clean_text(self, text: str) -> str:
        """Clean text for ML model input.

        Applies full preprocessing pipeline: normalization,
        whitespace cleanup, and optional character filtering.

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned text ready for model input.
        """
        text = self.normalize_text(text)

        if self.config.remove_urls:
            text = IOC_PATTERNS["url"].sub("[URL]", text)

        if self.config.remove_special_chars:
            text = re.sub(r'[^\w\s.\-:/]', '', text)

        return text.strip()

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Process a DataFrame of threat reports.

        Adds columns for cleaned text, extracted IOCs, and token count.

        Args:
            df: DataFrame containing threat reports.
            text_column: Name of the column containing report text.

        Returns:
            DataFrame with additional processed columns.
        """
        result = df.copy()
        result["cleaned_text"] = result[text_column].apply(self.clean_text)
        result["iocs"] = result[text_column].apply(self.extract_iocs)
        result["token_count"] = result["cleaned_text"].apply(lambda x: len(x.split()))
        result["ioc_count"] = result["iocs"].apply(
            lambda x: sum(len(v) for v in x.values())
        )
        return result
