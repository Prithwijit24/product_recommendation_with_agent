#!/usr/bin/env python3
"""Live end-to-end smoke: orchestrate for fixed profiles, dump routine JSON."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project_folder.agentic import orchestrate

PROFILES = [
    (
        {
            "age_range": "25-34",
            "sex": {"value": "F", "confidence": 0.87},
            "race": {"value": "Asian", "confidence": 0.74},
        },
        {
            "skin_type": "combination",
            "primary_concern": "Uneven tone & pigmentation",
            "sun_exposure": "Moderate outdoor",
            "sensitivities": ["None"],
            "budget": "medium",
            "pregnant": "no",
        },
    ),
    (
        {
            "age_range": "55-64",
            "sex": {"value": "M", "confidence": 0.9},
            "race": {"value": "Black", "confidence": 0.8},
        },
        {
            "skin_type": "oily",
            "primary_concern": "Fine lines & aging",
            "sun_exposure": "Significant outdoor",
            "sensitivities": ["None"],
            "budget": "high",
            "pregnant": "no",
        },
    ),
]


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    out_dir = Path("data") / session
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (demo, answers) in enumerate(PROFILES):
        result = orchestrate(demo, answers, session_id=f"{session}-{i}")
        with open(out_dir / f"routine_{i}.json", "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"profile {i}: {len(result['routine'])} products")
        for p in result["routine"]:
            print(f"  - {p['product_name']} | {p['price']} | {p['url']}")
        print(f"  concerns: {result['concerns_addressed']}")


if __name__ == "__main__":
    main()