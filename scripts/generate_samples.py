#!/usr/bin/env python3
"""Generate sample data for the threat intelligence pipeline.

Creates synthetic threat reports and STIX bundles for demonstration
and testing purposes. All data is entirely synthetic.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.generator import ThreatIntelGenerator


def main():
    """Generate sample datasets."""
    output_dir = Path("data/samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = ThreatIntelGenerator(seed=42)

    # Generate a small sample set
    print("Generating sample threat reports...")
    df = generator.generate_batch(n=25, seed=42)

    # Save as JSON
    records = df.to_dict(orient="records")
    with open(output_dir / "sample_reports.json", "w") as f:
        json.dump(records, f, indent=2, default=str)

    print(f"Saved {len(records)} sample reports to {output_dir / 'sample_reports.json'}")

    # Print category distribution
    print("\nCategory distribution:")
    for cat, count in df["category"].value_counts().items():
        print(f"  {cat}: {count}")

    # Print sample report
    print("\n--- Sample Report ---")
    sample = records[0]
    print(f"ID: {sample['report_id']}")
    print(f"Title: {sample['title']}")
    print(f"Category: {sample['category']}")
    print(f"Severity: {sample['severity']}")
    print(f"Text: {sample['text'][:200]}...")
    print("---")


if __name__ == "__main__":
    main()
