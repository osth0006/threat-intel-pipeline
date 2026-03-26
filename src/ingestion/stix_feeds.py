"""STIX/TAXII feed simulator for generating structured threat intelligence objects.

Produces STIX 2.1 compliant objects for realistic threat intelligence ingestion
scenarios without requiring connectivity to live threat feeds.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Any

from stix2 import (
    AttackPattern,
    Bundle,
    Identity,
    Indicator,
    IntrusionSet,
    Malware,
    Relationship,
    ThreatActor,
)

from src.ingestion.generator import (
    MALWARE_FAMILIES,
    MITRE_TECHNIQUES,
    TARGET_SECTORS,
    THREAT_ACTORS,
    _random_domain,
    _random_ip,
)


class STIXFeedSimulator:
    """Simulates a STIX/TAXII threat intelligence feed.

    Generates STIX 2.1 bundles containing realistic threat actor profiles,
    indicators of compromise, malware descriptions, and attack patterns
    suitable for pipeline ingestion testing.
    """

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
        self._identity = Identity(
            name="Threat Intel Pipeline",
            identity_class="system",
            description="Automated threat intelligence collection and analysis platform",
        )

    def _create_threat_actor(self) -> ThreatActor:
        name = random.choice(THREAT_ACTORS)
        return ThreatActor(
            name=name,
            description=f"Nation-state threat actor group tracked as {name}",
            threat_actor_types=[random.choice([
                "nation-state", "crime-syndicate", "activist", "insider",
            ])],
            sophistication=random.choice([
                "expert", "advanced", "intermediate",
            ]),
            resource_level=random.choice([
                "government", "organization", "individual",
            ]),
            primary_motivation=random.choice([
                "espionage", "financial-gain", "disruption", "destruction",
            ]),
            created_by_ref=self._identity.id,
        )

    def _create_malware(self) -> Malware:
        name = random.choice(MALWARE_FAMILIES)
        return Malware(
            name=name,
            description=f"Malware family: {name}",
            malware_types=[random.choice([
                "backdoor", "ransomware", "trojan", "worm", "rootkit",
                "spyware", "remote-access-trojan", "dropper",
            ])],
            is_family=True,
            created_by_ref=self._identity.id,
        )

    def _create_indicator(self) -> Indicator:
        indicator_type = random.choice(["ipv4-addr", "domain-name", "file"])
        if indicator_type == "ipv4-addr":
            ip = _random_ip()
            pattern = f"[ipv4-addr:value = '{ip}']"
            desc = f"Malicious IP address: {ip}"
        elif indicator_type == "domain-name":
            domain = _random_domain()
            pattern = f"[domain-name:value = '{domain}']"
            desc = f"Malicious domain: {domain}"
        else:
            hash_val = uuid.uuid4().hex
            pattern = f"[file:hashes.'SHA-256' = '{hash_val}']"
            desc = f"Malicious file hash: {hash_val}"

        valid_from = datetime.now() - timedelta(days=random.randint(1, 90))
        return Indicator(
            name=desc,
            description=desc,
            pattern=pattern,
            pattern_type="stix",
            valid_from=valid_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            indicator_types=[random.choice([
                "malicious-activity", "anomalous-activity", "compromised",
            ])],
            created_by_ref=self._identity.id,
        )

    def _create_attack_pattern(self) -> AttackPattern:
        technique_id, technique_name = random.choice(MITRE_TECHNIQUES)
        return AttackPattern(
            name=technique_name,
            description=f"MITRE ATT&CK technique {technique_id}: {technique_name}",
            external_references=[{
                "source_name": "mitre-attack",
                "external_id": technique_id,
                "url": f"https://attack.mitre.org/techniques/{technique_id.replace('.', '/')}/",
            }],
            created_by_ref=self._identity.id,
        )

    def _create_intrusion_set(self) -> IntrusionSet:
        actor = random.choice(THREAT_ACTORS)
        sector = random.choice(TARGET_SECTORS)
        return IntrusionSet(
            name=f"Operation targeting {sector}",
            description=(
                f"Intrusion set attributed to {actor} targeting "
                f"{sector} sector organizations"
            ),
            first_seen=datetime.now() - timedelta(days=random.randint(90, 730)),
            goals=[f"Compromise {sector} infrastructure"],
            resource_level=random.choice(["government", "organization"]),
            primary_motivation="espionage",
            created_by_ref=self._identity.id,
        )

    def generate_bundle(self, n_objects: int = 50) -> Bundle:
        """Generate a STIX 2.1 bundle with interrelated threat objects.

        Args:
            n_objects: Approximate number of STIX objects to generate.

        Returns:
            STIX Bundle containing threat intelligence objects.
        """
        objects: list[Any] = [self._identity]
        relationships: list[Relationship] = []

        generators = [
            self._create_threat_actor,
            self._create_malware,
            self._create_indicator,
            self._create_attack_pattern,
            self._create_intrusion_set,
        ]

        for _ in range(n_objects):
            gen = random.choice(generators)
            obj = gen()
            objects.append(obj)

        # Create relationships between objects
        threat_actors = [o for o in objects if o.type == "threat-actor"]
        malwares = [o for o in objects if o.type == "malware"]
        indicators = [o for o in objects if o.type == "indicator"]
        attack_patterns = [o for o in objects if o.type == "attack-pattern"]

        for actor in threat_actors[:5]:
            if malwares:
                mal = random.choice(malwares)
                relationships.append(Relationship(
                    relationship_type="uses",
                    source_ref=actor.id,
                    target_ref=mal.id,
                    created_by_ref=self._identity.id,
                ))
            if attack_patterns:
                ap = random.choice(attack_patterns)
                relationships.append(Relationship(
                    relationship_type="uses",
                    source_ref=actor.id,
                    target_ref=ap.id,
                    created_by_ref=self._identity.id,
                ))

        for indicator in indicators[:10]:
            if malwares:
                mal = random.choice(malwares)
                relationships.append(Relationship(
                    relationship_type="indicates",
                    source_ref=indicator.id,
                    target_ref=mal.id,
                    created_by_ref=self._identity.id,
                ))

        objects.extend(relationships)
        return Bundle(objects=objects)

    def generate_feed(self, n_bundles: int = 5, objects_per_bundle: int = 30) -> list[Bundle]:
        """Generate multiple STIX bundles simulating a feed collection.

        Args:
            n_bundles: Number of bundles to generate.
            objects_per_bundle: Objects per bundle.

        Returns:
            List of STIX Bundles.
        """
        return [self.generate_bundle(objects_per_bundle) for _ in range(n_bundles)]
