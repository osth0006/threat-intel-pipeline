"""Synthetic threat intelligence data generator.

Produces realistic threat reports with embedded indicators of compromise (IOCs),
threat actor references, and tactic/technique/procedure (TTP) descriptions.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

# --- Threat vocabulary for realistic synthetic generation ---

THREAT_ACTORS = [
    "APT28", "APT29", "APT41", "Lazarus Group", "Sandworm", "Turla",
    "Fancy Bear", "Cozy Bear", "Equation Group", "DarkSide", "REvil",
    "Conti", "LockBit", "BlackCat", "Kimsuky", "Charming Kitten",
    "MuddyWater", "OilRig", "Hafnium", "Nobelium", "FIN7", "FIN12",
    "Scattered Spider", "Volt Typhoon", "Salt Typhoon",
]

MALWARE_FAMILIES = [
    "Emotet", "TrickBot", "Cobalt Strike", "Mimikatz", "QakBot",
    "IcedID", "BazarLoader", "Ryuk", "Conti Ransomware", "BlackMatter",
    "DarkSide Ransomware", "SolarWinds SUNBURST", "Pegasus", "PlugX",
    "ShadowPad", "BADFLICK", "CHOPSTICK", "WannaCry", "NotPetya",
    "Snake Malware", "Industroyer", "Triton", "BlackEnergy",
]

MITRE_TECHNIQUES = [
    ("T1566", "Phishing"),
    ("T1059", "Command and Scripting Interpreter"),
    ("T1053", "Scheduled Task/Job"),
    ("T1548", "Abuse Elevation Control Mechanism"),
    ("T1134", "Access Token Manipulation"),
    ("T1087", "Account Discovery"),
    ("T1098", "Account Manipulation"),
    ("T1583", "Acquire Infrastructure"),
    ("T1595", "Active Scanning"),
    ("T1071", "Application Layer Protocol"),
    ("T1560", "Archive Collected Data"),
    ("T1557", "Adversary-in-the-Middle"),
    ("T1110", "Brute Force"),
    ("T1059.001", "PowerShell"),
    ("T1059.003", "Windows Command Shell"),
    ("T1078", "Valid Accounts"),
    ("T1190", "Exploit Public-Facing Application"),
    ("T1133", "External Remote Services"),
    ("T1486", "Data Encrypted for Impact"),
    ("T1027", "Obfuscated Files or Information"),
    ("T1055", "Process Injection"),
    ("T1021", "Remote Services"),
    ("T1082", "System Information Discovery"),
    ("T1003", "OS Credential Dumping"),
    ("T1569", "System Services"),
    ("T1547", "Boot or Logon Autostart Execution"),
]

THREAT_CATEGORIES = [
    "apt", "malware", "phishing", "vulnerability", "ransomware",
    "supply_chain", "insider_threat", "ddos", "data_exfiltration",
    "zero_day",
]

SEVERITY_LEVELS = ["critical", "high", "medium", "low", "informational"]

TARGET_SECTORS = [
    "government", "defense", "energy", "finance", "healthcare",
    "technology", "telecommunications", "manufacturing", "education",
    "transportation", "critical_infrastructure",
]

TARGET_REGIONS = [
    "North America", "Western Europe", "Eastern Europe", "East Asia",
    "Southeast Asia", "Middle East", "South Asia", "Africa",
    "South America", "Oceania",
]

# --- Report templates ---

REPORT_TEMPLATES = {
    "apt": [
        "Advanced persistent threat group {actor} has been observed conducting a sophisticated "
        "cyber espionage campaign targeting {sector} organizations in {region}. The operation "
        "leverages {malware} for initial access via {technique}. Post-compromise activity "
        "includes lateral movement using {technique2} and data staging for exfiltration. "
        "The campaign has been active since {date} and affects an estimated {count} entities. "
        "IOC analysis reveals C2 infrastructure at {ip} communicating over port {port}.",

        "Intelligence assessment indicates {actor} has shifted operational focus to {sector} "
        "targets across {region}. This campaign, tracked internally as Operation {op_name}, "
        "employs a multi-stage infection chain beginning with {technique}. Secondary payloads "
        "include {malware} variants with enhanced evasion capabilities. Network indicators "
        "show beaconing to {domain} with certificate SHA256 hash {hash}. TTPs align with "
        "previously attributed {actor} operations from {date}.",
    ],
    "malware": [
        "New variant of {malware} detected in the wild, exhibiting polymorphic behavior and "
        "advanced anti-analysis techniques. Initial delivery vector is {technique} with "
        "secondary payload deployment via {technique2}. The malware establishes persistence "
        "through {technique3} and communicates with C2 servers at {ip}:{port}. Binary analysis "
        "shows {hash} with modified encryption routines compared to previous versions. "
        "Affected systems span {sector} organizations in {region}.",

        "Threat researchers have identified a previously undocumented malware family, "
        "designated {malware}, being deployed by {actor} against {sector} targets. The malware "
        "features modular architecture with plugins for keylogging, screen capture, and "
        "credential harvesting via {technique}. Command and control communications use "
        "encrypted DNS tunneling to {domain}. Samples analyzed: {hash}.",
    ],
    "phishing": [
        "Large-scale phishing campaign attributed to {actor} targeting {sector} personnel "
        "in {region}. Attack chain uses {technique} with lure documents themed around "
        "{lure_theme}. Credential harvesting infrastructure hosted at {domain} mimics "
        "legitimate {sector} portals. Campaign has compromised approximately {count} accounts "
        "since {date}. Email headers reveal origin IP {ip} with SPF/DKIM bypass techniques.",

        "Spear-phishing operation targeting senior {sector} officials detected. Messages "
        "contain weaponized attachments exploiting {cve} to deploy {malware}. The adversary, "
        "assessed as {actor}, uses {technique} for initial execution. Phishing infrastructure "
        "at {domain} uses Let's Encrypt certificates issued {date}. Telemetry shows "
        "{count} delivery attempts across {region}.",
    ],
    "vulnerability": [
        "Critical vulnerability {cve} discovered in widely deployed {software} affecting "
        "{sector} organizations globally. The flaw allows remote code execution via "
        "{technique} without authentication. CVSS score: {cvss}. Exploitation in the wild "
        "confirmed, with {actor} actively leveraging the vulnerability. Patch available from "
        "vendor as of {date}. Affected versions: {versions}. Recommended immediate patching "
        "and IOC scanning for {hash}.",

        "Zero-day vulnerability {cve} in {software} being actively exploited by {actor} "
        "targeting {sector} in {region}. The vulnerability enables {technique} leading to "
        "full system compromise. No vendor patch available. Temporary mitigations include "
        "network segmentation and monitoring for connections to {ip}. CVSS: {cvss}. "
        "Estimated {count} vulnerable instances identified via internet scanning.",
    ],
    "ransomware": [
        "Ransomware incident attributed to {actor} affiliate program impacting {sector} "
        "organization in {region}. Initial access gained via {technique} followed by "
        "deployment of {malware}. Encryption of {count} systems completed within {hours} "
        "hours of initial compromise. Ransom demand: {ransom} in cryptocurrency. Data "
        "exfiltration confirmed prior to encryption, with {size}GB staged to {ip}. "
        "Recovery operations ongoing since {date}.",

        "{actor} ransomware group claims attack against {sector} entity in {region}. "
        "Attack leveraged {technique} for initial access and {technique2} for privilege "
        "escalation. {malware} payload deployed via {technique3}. Double extortion model "
        "with data leak site posting on {date}. Affected infrastructure includes {count} "
        "endpoints. TTPs consistent with {actor} playbook tracked since {date2}.",
    ],
    "supply_chain": [
        "Supply chain compromise detected affecting {software} used by {sector} organizations "
        "in {region}. Malicious code injection attributed to {actor} introduces {malware} "
        "backdoor via {technique}. Affected versions: {versions}. Compromise enables "
        "remote access and credential harvesting. C2 communications observed to {domain}. "
        "Impact assessment indicates {count} downstream organizations potentially affected "
        "since {date}.",
    ],
    "insider_threat": [
        "Insider threat incident detected at {sector} organization in {region}. Privileged "
        "user conducted unauthorized data access and exfiltration of {size}GB of sensitive "
        "data via {technique}. Activity spanned {days} days beginning {date}. Data staged "
        "to external storage at {domain}. Behavioral indicators included after-hours access "
        "patterns and bulk download of {count} classified documents.",
    ],
    "ddos": [
        "Distributed denial of service attack targeting {sector} infrastructure in {region}. "
        "Attack attributed to {actor} using {technique} amplification vectors. Peak traffic "
        "volume reached {bandwidth} Gbps sustained over {hours} hours beginning {date}. "
        "Source IPs primarily from botnet infrastructure previously associated with {malware}. "
        "Mitigation engaged at network edge. {count} service endpoints affected.",
    ],
    "data_exfiltration": [
        "Data exfiltration campaign by {actor} targeting {sector} organizations in {region}. "
        "Adversary leveraged {technique} for initial access and deployed {malware} for "
        "persistent access. Data collection focused on {data_type} with {size}GB exfiltrated "
        "to infrastructure at {ip} over {days} days. Network traffic analysis shows DNS "
        "tunneling to {domain} using {technique2}.",
    ],
    "zero_day": [
        "Active exploitation of zero-day vulnerability {cve} in {software} detected. "
        "Campaign attributed to {actor} targeting {sector} in {region}. Exploitation chain "
        "uses {technique} to achieve remote code execution with SYSTEM privileges. No patch "
        "available. {malware} deployed post-exploitation for persistence via {technique2}. "
        "C2 traffic observed to {ip}:{port}. CVSS: {cvss}. Estimated {count} organizations "
        "at risk. Emergency mitigations published {date}.",
    ],
}

LURE_THEMES = [
    "budget planning documents", "security clearance renewal",
    "organizational restructuring", "COVID-19 policy updates",
    "annual performance reviews", "classified briefing schedules",
    "vendor contract renewals", "emergency response protocols",
    "IT system maintenance notices", "executive travel itineraries",
]

SOFTWARE_TARGETS = [
    "Microsoft Exchange Server", "Apache Log4j", "Fortinet FortiOS",
    "Citrix ADC", "VMware vCenter", "Pulse Secure VPN",
    "SolarWinds Orion", "Atlassian Confluence", "Ivanti EPMM",
    "MOVEit Transfer", "Barracuda ESG", "Cisco IOS XE",
]

DATA_TYPES = [
    "classified documents", "personnel records", "financial data",
    "intellectual property", "operational plans", "communications metadata",
    "network architecture diagrams", "credentials databases",
    "research data", "diplomatic cables",
]

OP_NAMES = [
    "GHOSTNET", "AURORA", "NIGHTDRAGON", "DUSTYSTORM", "IRONLOTUS",
    "SHADOWFORGE", "CRYSTALVIPER", "THUNDERSTRIKE", "FROSTBITE",
    "SILENTBREAKER", "DARKNEXUS", "COBALTEDGE", "STEELPHOENIX",
]


def _random_ip() -> str:
    return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def _random_domain() -> str:
    tlds = [".com", ".net", ".org", ".xyz", ".info", ".cc", ".top"]
    words = [
        "cloud", "service", "update", "secure", "portal", "cdn", "api",
        "auth", "sync", "data", "relay", "proxy", "node", "edge",
    ]
    return f"{random.choice(words)}-{random.choice(words)}{random.randint(1,999)}{random.choice(tlds)}"


def _random_hash() -> str:
    return uuid.uuid4().hex + uuid.uuid4().hex[:32]


def _random_cve() -> str:
    year = random.randint(2020, 2025)
    num = random.randint(1000, 50000)
    return f"CVE-{year}-{num}"


class ThreatIntelGenerator:
    """Generates synthetic threat intelligence reports with realistic content.

    Each report includes narrative text, structured metadata, and embedded
    indicators of compromise suitable for NLP extraction tasks.
    """

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
        self._report_counter = 0

    def _fill_template(self, category: str) -> str:
        templates = REPORT_TEMPLATES[category]
        template = random.choice(templates)

        techniques = random.sample(MITRE_TECHNIQUES, min(3, len(MITRE_TECHNIQUES)))
        past_date = datetime.now() - timedelta(days=random.randint(1, 365))
        past_date2 = datetime.now() - timedelta(days=random.randint(366, 730))

        placeholders: dict[str, Any] = {
            "actor": random.choice(THREAT_ACTORS),
            "malware": random.choice(MALWARE_FAMILIES),
            "sector": random.choice(TARGET_SECTORS),
            "region": random.choice(TARGET_REGIONS),
            "technique": f"{techniques[0][0]} ({techniques[0][1]})",
            "technique2": f"{techniques[1][0]} ({techniques[1][1]})" if len(techniques) > 1 else "",
            "technique3": f"{techniques[2][0]} ({techniques[2][1]})" if len(techniques) > 2 else "",
            "ip": _random_ip(),
            "port": random.choice([443, 8443, 8080, 4443, 53, 80, 9090]),
            "domain": _random_domain(),
            "hash": _random_hash(),
            "date": past_date.strftime("%Y-%m-%d"),
            "date2": past_date2.strftime("%Y-%m-%d"),
            "count": random.randint(5, 5000),
            "cve": _random_cve(),
            "cvss": round(random.uniform(7.0, 10.0), 1),
            "software": random.choice(SOFTWARE_TARGETS),
            "versions": f"{random.randint(1,15)}.{random.randint(0,9)}.x through {random.randint(1,15)}.{random.randint(0,9)}.x",
            "lure_theme": random.choice(LURE_THEMES),
            "hours": random.randint(2, 72),
            "days": random.randint(1, 180),
            "ransom": f"${random.randint(100, 5000)}K",
            "size": random.randint(1, 500),
            "bandwidth": random.randint(50, 2000),
            "data_type": random.choice(DATA_TYPES),
            "op_name": random.choice(OP_NAMES),
        }

        return template.format(**placeholders)

    def generate_report(self, category: str | None = None) -> dict[str, Any]:
        """Generate a single synthetic threat intelligence report.

        Args:
            category: Threat category. Random if not specified.

        Returns:
            Dictionary containing report text, metadata, and IOCs.
        """
        if category is None:
            category = random.choice(THREAT_CATEGORIES)
        elif category not in THREAT_CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Must be one of {THREAT_CATEGORIES}")

        self._report_counter += 1
        report_id = f"TIR-{datetime.now().strftime('%Y%m%d')}-{self._report_counter:05d}"
        timestamp = datetime.now() - timedelta(
            days=random.randint(0, 90),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )

        text = self._fill_template(category)

        # Assign secondary labels for multi-label scenarios
        all_categories = [category]
        if random.random() < 0.3:
            secondary = random.choice([c for c in THREAT_CATEGORIES if c != category])
            all_categories.append(secondary)

        return {
            "report_id": report_id,
            "timestamp": timestamp.isoformat(),
            "title": f"Threat Report: {category.replace('_', ' ').title()} Activity - {report_id}",
            "text": text,
            "category": category,
            "labels": all_categories,
            "severity": random.choice(SEVERITY_LEVELS),
            "confidence": round(random.uniform(0.5, 1.0), 2),
            "source": random.choice(["OSINT", "SIGINT", "HUMINT", "TECHINT", "PARTNER"]),
            "tlp": random.choice(["WHITE", "GREEN", "AMBER", "RED"]),
        }

    def generate_batch(self, n: int = 100, seed: int | None = None) -> pd.DataFrame:
        """Generate a batch of synthetic threat reports.

        Args:
            n: Number of reports to generate.
            seed: Random seed for reproducibility.

        Returns:
            DataFrame containing all generated reports.
        """
        if seed is not None:
            random.seed(seed)

        reports = [self.generate_report() for _ in range(n)]
        return pd.DataFrame(reports)
