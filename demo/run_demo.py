#!/usr/bin/env python3
"""
MedGuard Triage Copilot – Demo Script
=======================================
Runs triage on sample cases and prints formatted results.

Usage:
    python demo/run_demo.py
"""

import json
import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from core.router import MedGuardRouter


DEMO_CASES = [
    {
        "label": "Cardiac Emergency",
        "input": (
            "58-year-old male with crushing chest pain radiating to left arm "
            "for 45 minutes. Smoker with hypertension. Diaphoretic."
        ),
    },
    {
        "label": "Stroke – FAST Positive",
        "input": (
            "72-year-old female with sudden right-sided weakness and facial droop "
            "starting 30 minutes ago. Slurred speech. AFib history."
        ),
    },
    {
        "label": "Minor URI",
        "input": (
            "28-year-old female with runny nose, mild sore throat, and dry cough "
            "for 3 days. No fever. No shortness of breath."
        ),
    },
    {
        "label": "Abdominal Pain – Moderate",
        "input": (
            "45-year-old female with right lower quadrant pain for 12 hours, "
            "nausea, low-grade fever 37.8°C. No prior surgeries."
        ),
    },
]


def main():
    print("\n" + "=" * 60)
    print("  🏥 MedGuard Triage Copilot – Demo")
    print("=" * 60)

    router = MedGuardRouter()

    for case in DEMO_CASES:
        print(f"\n{'─' * 60}")
        print(f"  📋 Case: {case['label']}")
        print(f"  Input: {case['input'][:80]}...")
        print(f"{'─' * 60}")

        try:
            result = router.triage(case["input"])
            triage = result.get("triage_output", {})

            urgency = triage.get("urgency_level", "?")
            label = triage.get("urgency_label", "?")
            confidence = triage.get("confidence_score", "?")
            flags = triage.get("red_flags", [])

            print(f"  Urgency:    Level {urgency} – {label}")
            print(f"  Confidence: {confidence}")

            if flags:
                print(f"  Red Flags:")
                for rf in flags[:3]:
                    if isinstance(rf, dict):
                        print(f"    • [{rf.get('category')}] {rf.get('flag')}")
                    else:
                        print(f"    • {rf}")

            actions = triage.get("recommended_actions", [])
            if actions:
                print(f"  Actions:")
                for a in actions[:3]:
                    print(f"    • {a}")

            overrides = result.get("safety_overrides_applied", [])
            if overrides:
                print(f"  Safety Overrides: {overrides}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n{'=' * 60}")
    print("  ⚕️  All outputs require review by qualified professionals.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
