"""Tests for the text preprocessing pipeline."""

import pandas as pd
import pytest

from src.processing.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Test suite for TextPreprocessor."""

    def setup_method(self):
        self.preprocessor = TextPreprocessor()

    def test_extract_ipv4(self):
        text = "C2 server at 192.168.1.100 and 10.0.0.1"
        iocs = self.preprocessor.extract_iocs(text)
        assert "ipv4" in iocs
        assert "192.168.1.100" in iocs["ipv4"]
        assert "10.0.0.1" in iocs["ipv4"]

    def test_extract_cve(self):
        text = "Exploiting CVE-2023-44228 and CVE-2021-34527"
        iocs = self.preprocessor.extract_iocs(text)
        assert "cve" in iocs
        assert "CVE-2023-44228" in iocs["cve"]
        assert "CVE-2021-34527" in iocs["cve"]

    def test_extract_hash(self):
        text = f"SHA256: {'a' * 64}"
        iocs = self.preprocessor.extract_iocs(text)
        assert "sha256" in iocs

    def test_extract_mitre_technique(self):
        text = "Using T1566 for initial access and T1059.001 for execution"
        iocs = self.preprocessor.extract_iocs(text)
        assert "mitre_technique" in iocs
        assert "T1566" in iocs["mitre_technique"]
        assert "T1059.001" in iocs["mitre_technique"]

    def test_normalize_text(self):
        text = "Multiple   spaces   and\nnewlines\there"
        result = self.preprocessor.normalize_text(text)
        assert "  " not in result
        assert "\n" not in result

    def test_clean_text_preserves_iocs(self):
        text = "Attack from 192.168.1.1 using CVE-2023-12345"
        cleaned = self.preprocessor.clean_text(text)
        assert "192.168.1.1" in cleaned
        assert "CVE-2023-12345" in cleaned

    def test_process_dataframe(self):
        df = pd.DataFrame({
            "text": [
                "APT28 attacked via 10.0.0.1 using T1566",
                "Ransomware CVE-2023-1234 exploited",
            ]
        })
        result = self.preprocessor.process_dataframe(df)
        assert "cleaned_text" in result.columns
        assert "iocs" in result.columns
        assert "token_count" in result.columns
        assert "ioc_count" in result.columns
        assert all(result["token_count"] > 0)
