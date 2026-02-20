#!/usr/bin/env python3
"""
MedGuard Triage Copilot – Main Entrypoint
===========================================
Usage:
    python main.py "58yo male with crushing chest pain for 45min, smoker"
    python main.py --file data/samples/sample_text_cases.json
    python main.py --interactive

Importable convenience function:
    from main import run_triage
    result = run_triage(patient_dict_or_text)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

from core.router import MedGuardRouter
from core.logging_utils import setup_logging

# Module-level singleton router (lazy-initialised on first call)
_router: MedGuardRouter | None = None


def _get_router(config_path: str | None = None) -> MedGuardRouter:
    global _router
    if _router is None:
        _router = MedGuardRouter(config_path=config_path)
    return _router


def run_triage(
    patient,
    input_type: str = "auto",
    config_path: str | None = None,
) -> dict:
    """
    Run the MedGuard triage pipeline and return a clean final output.

    Parameters
    ----------
    patient : str | dict | Path
        • str  – free-text intake  (e.g. "58yo male, chest pain 45 min")
        • dict – patient object with keys like ``chief_complaint``, ``symptoms``,
                 ``age``, ``sex``, ``risk_factors``, ``duration``
        • Path – path to an audio file (for the voice pipeline)
    input_type : str
        "auto" (default), "text", "voice", or "structured"
    config_path : str, optional
        Path to a custom ``model_config.yaml``.

    Returns
    -------
    dict with keys:
        id, timestamp, input_type,
        triage_output  (urgency_level, urgency_label, red_flags,
                        recommended_actions, follow_up_questions,
                        confidence_score, confidence_band, safety_note,
                        reasoning_summary),
        safety_overrides_applied, disclaimer, pipeline_errors
    """
    router = _get_router(config_path)
    return router.triage(patient, input_type=input_type)


# ── Presentation helpers ────────────────────────────────────────────────────

def print_result(result: dict, verbose: bool = False):
    """Pretty-print a triage result to stdout."""
    triage = result.get("triage_output", {})

    COLORS = {5: "\033[91m", 4: "\033[93m", 3: "\033[33m", 2: "\033[94m", 1: "\033[92m"}
    RESET = "\033[0m"

    urgency = triage.get("urgency_level", "?")
    color = COLORS.get(urgency, "")

    print(f"\n{'='*60}")
    print(f"  MEDGUARD TRIAGE RESULT  |  ID: {result.get('id', '?')}")
    print(f"{'='*60}")
    print(f"  Urgency:    {color}Level {urgency} – {triage.get('urgency_label', '?')}{RESET}")
    conf = triage.get('confidence_score', '?')
    band = triage.get('confidence_band', '?')
    print(f"  Confidence: {conf} ({band})")
    reasoning = triage.get("reasoning_summary", "")
    if reasoning:
        print(f"  Rationale:  {reasoning}")
    print(f"{'─'*60}")

    # Red flags
    red_flags = triage.get("red_flags", [])
    if red_flags:
        print(f"  🚩 RED FLAGS ({len(red_flags)}):")
        for rf in red_flags:
            if isinstance(rf, dict):
                print(f"     • [{rf.get('category', '?')}] {rf.get('flag', '?')}")
                print(f"       → {rf.get('recommended_action', '')}")
            else:
                print(f"     • {rf}")

    # Recommended actions
    actions = triage.get("recommended_actions", [])
    if actions:
        print(f"\n  📋 RECOMMENDED ACTIONS:")
        for a in actions:
            print(f"     • {a}")

    # Follow-up questions
    followups = triage.get("follow_up_questions", [])
    if followups:
        print(f"\n  ❓ FOLLOW-UP QUESTIONS:")
        for q in followups:
            print(f"     • {q}")

    # Safety note (includes low-confidence guardrail message when applicable)
    safety = triage.get("safety_note", "")
    if safety:
        print(f"\n  ⚠️  SAFETY NOTE: {safety}")

    # Safety overrides
    overrides = result.get("safety_overrides_applied", [])
    if overrides:
        print(f"\n  🔒 SAFETY OVERRIDES:")
        for o in overrides:
            print(f"     • {o}")

    if verbose:
        print(f"\n  📊 Full triage JSON:")
        print(json.dumps(triage, indent=2, default=str))

    print(f"\n{'─'*60}")
    print(f"  ⚕️  {result.get('disclaimer', '')}")
    print(f"{'='*60}\n")


def run_interactive(router: MedGuardRouter):
    """Interactive mode – type patient descriptions."""
    print("\n🏥 MedGuard Triage Copilot – Interactive Mode")
    print("Type patient intake text. Type 'quit' to exit.\n")

    while True:
        try:
            text = input("Patient > ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue

            result = router.triage(text)
            print_result(result)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

    print("\nGoodbye.")


def main():
    parser = argparse.ArgumentParser(description="MedGuard Triage Copilot")
    parser.add_argument("text", nargs="?", help="Patient intake text")
    parser.add_argument("--file", "-f", help="Path to JSON test case file")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--config", "-c", help="Path to model config YAML")

    args = parser.parse_args()

    setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

    router = MedGuardRouter(config_path=args.config)

    if args.interactive:
        run_interactive(router)
    elif args.file:
        with open(args.file) as f:
            data = json.load(f)
        cases = data if isinstance(data, list) else data.get("cases", [data])
        for case in cases:
            patient_input = case.get("input", case.get("patient", case))
            result = router.triage(patient_input)
            print_result(result, verbose=args.verbose)
    elif args.text:
        result = router.triage(args.text)
        print_result(result, verbose=args.verbose)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
