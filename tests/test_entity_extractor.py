"""Tests for the threat entity extractor."""

import pytest

from src.processing.entity_extractor import ThreatEntityExtractor


class TestThreatEntityExtractor:
    """Test suite for ThreatEntityExtractor."""

    def setup_method(self):
        self.extractor = ThreatEntityExtractor()

    def test_extract_threat_actors(self):
        text = "APT28 and Lazarus Group conducted coordinated attacks"
        actors = self.extractor.extract_threat_actors(text)
        actor_names = [a.text for a in actors]
        assert "APT28" in actor_names
        assert "Lazarus Group" in actor_names

    def test_extract_malware(self):
        text = "Deployed Emotet and Cobalt Strike for persistence"
        malware = self.extractor.extract_malware(text)
        malware_names = [m.text for m in malware]
        assert "Emotet" in malware_names
        assert "Cobalt Strike" in malware_names

    def test_extract_mitre_techniques(self):
        text = "Initial access via T1566 followed by T1059 execution"
        techniques = self.extractor.extract_mitre_techniques(text)
        assert len(techniques) >= 2
        technique_ids = [t.text for t in techniques]
        assert any("T1566" in t for t in technique_ids)

    def test_extract_iocs(self):
        text = "C2 at 192.168.1.1 exploiting CVE-2023-44228"
        iocs = self.extractor.extract_iocs(text)
        ioc_values = [i.text for i in iocs]
        assert "192.168.1.1" in ioc_values
        assert "CVE-2023-44228" in ioc_values

    def test_extract_all(self):
        text = (
            "APT29 deployed Cobalt Strike via T1566 phishing. "
            "C2 server at 10.0.0.1 exploiting CVE-2023-12345."
        )
        result = self.extractor.extract_all(text)
        assert "threat_actors" in result
        assert "malware" in result
        assert "mitre_techniques" in result
        assert "iocs" in result

    def test_extract_summary_deduplicates(self):
        text = "APT28 used APT28 tactics. APT28 is persistent."
        summary = self.extractor.extract_summary(text)
        assert len(summary.get("threat_actors", [])) == 1

    def test_entity_positions(self):
        text = "Attack by APT28"
        actors = self.extractor.extract_threat_actors(text)
        assert len(actors) == 1
        assert actors[0].start == text.index("APT28")
        assert actors[0].end == text.index("APT28") + len("APT28")

    def test_empty_text(self):
        result = self.extractor.extract_all("")
        for entities in result.values():
            assert len(entities) == 0
