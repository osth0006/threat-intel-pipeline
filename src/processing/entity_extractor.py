"""Named entity extraction for threat intelligence domain.

Extracts threat actors, malware families, MITRE ATT&CK techniques,
and other domain-specific entities from threat intelligence text.
"""

import re
from dataclasses import dataclass

from src.ingestion.generator import (
    MALWARE_FAMILIES,
    MITRE_TECHNIQUES,
    THREAT_ACTORS,
)
from src.processing.preprocessor import IOC_PATTERNS


@dataclass
class ExtractedEntity:
    """A single extracted entity with metadata."""
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float = 1.0


class ThreatEntityExtractor:
    """Extracts domain-specific entities from threat intelligence text.

    Uses pattern matching and dictionary lookup to identify threat actors,
    malware families, MITRE techniques, and technical indicators.
    This serves as a baseline extractor; production systems would layer
    transformer-based NER on top.
    """

    def __init__(self):
        # Build lookup sets for fast matching
        self._threat_actors = set(THREAT_ACTORS)
        self._malware_families = set(MALWARE_FAMILIES)
        self._mitre_lookup = {tid: name for tid, name in MITRE_TECHNIQUES}

        # Build regex patterns for multi-word entity matching
        self._actor_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(a) for a in sorted(THREAT_ACTORS, key=len, reverse=True)) + r')\b',
            re.IGNORECASE,
        )
        self._malware_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(m) for m in sorted(MALWARE_FAMILIES, key=len, reverse=True)) + r')\b',
            re.IGNORECASE,
        )

    def extract_threat_actors(self, text: str) -> list[ExtractedEntity]:
        """Extract threat actor names from text."""
        entities = []
        for match in self._actor_pattern.finditer(text):
            entities.append(ExtractedEntity(
                text=match.group(),
                entity_type="threat_actor",
                start=match.start(),
                end=match.end(),
            ))
        return entities

    def extract_malware(self, text: str) -> list[ExtractedEntity]:
        """Extract malware family names from text."""
        entities = []
        for match in self._malware_pattern.finditer(text):
            entities.append(ExtractedEntity(
                text=match.group(),
                entity_type="malware",
                start=match.start(),
                end=match.end(),
            ))
        return entities

    def extract_mitre_techniques(self, text: str) -> list[ExtractedEntity]:
        """Extract MITRE ATT&CK technique identifiers from text."""
        entities = []
        for match in IOC_PATTERNS["mitre_technique"].finditer(text):
            tid = match.group()
            technique_name = self._mitre_lookup.get(tid, "Unknown Technique")
            entities.append(ExtractedEntity(
                text=f"{tid} ({technique_name})",
                entity_type="mitre_technique",
                start=match.start(),
                end=match.end(),
            ))
        return entities

    def extract_iocs(self, text: str) -> list[ExtractedEntity]:
        """Extract technical indicators of compromise from text."""
        entities = []
        for ioc_type in ["cve", "ipv4", "sha256", "md5", "domain", "email"]:
            pattern = IOC_PATTERNS.get(ioc_type)
            if pattern:
                for match in pattern.finditer(text):
                    entities.append(ExtractedEntity(
                        text=match.group(),
                        entity_type=f"ioc_{ioc_type}",
                        start=match.start(),
                        end=match.end(),
                    ))
        return entities

    def extract_all(self, text: str) -> dict[str, list[ExtractedEntity]]:
        """Extract all entity types from text.

        Args:
            text: Threat intelligence report text.

        Returns:
            Dictionary mapping entity types to lists of extracted entities.
        """
        return {
            "threat_actors": self.extract_threat_actors(text),
            "malware": self.extract_malware(text),
            "mitre_techniques": self.extract_mitre_techniques(text),
            "iocs": self.extract_iocs(text),
        }

    def extract_summary(self, text: str) -> dict[str, list[str]]:
        """Extract a deduplicated summary of entities from text.

        Args:
            text: Threat intelligence report text.

        Returns:
            Dictionary mapping entity types to deduplicated entity strings.
        """
        all_entities = self.extract_all(text)
        return {
            entity_type: sorted(set(e.text for e in entities))
            for entity_type, entities in all_entities.items()
            if entities
        }
