"""
MedGuard Triage Copilot – Triage Metrics
==========================================
Evaluates the quality of Stage 3 (MedGemma Triage) output.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def urgency_accuracy(predicted: int, expected: int) -> bool:
    """Exact match on urgency level."""
    return predicted == expected


def urgency_within_one(predicted: int, expected: int) -> bool:
    """Is predicted urgency within +-1 of expected?"""
    return abs(predicted - expected) <= 1


def red_flag_recall(predicted_flags: List[dict], expected_flags: List[str]) -> float:
    """What fraction of expected red flags were detected?"""
    if not expected_flags:
        return 1.0

    pred_names = {f.get("flag", "").lower().strip() for f in predicted_flags}
    # Also check categories
    pred_cats = {f.get("category", "").lower().strip() for f in predicted_flags}

    hits = 0
    for ef in expected_flags:
        ef_lower = ef.lower().strip()
        if ef_lower in pred_names or ef_lower in pred_cats:
            hits += 1

    return hits / len(expected_flags)


def safety_score(triage_output: dict, expected_urgency: int) -> float:
    """
    Safety-weighted score that penalizes under-triage more than over-triage.

    - Under-triage (predicted < expected): heavy penalty
    - Over-triage (predicted > expected): mild penalty
    - Exact match: 1.0
    """
    predicted = triage_output.get("urgency_level", 3)
    diff = predicted - expected

    if diff == 0:
        return 1.0
    elif diff > 0:
        # Over-triage (conservative – less bad)
        return max(0.0, 1.0 - diff * 0.1)
    else:
        # Under-triage (dangerous – heavier penalty)
        return max(0.0, 1.0 + diff * 0.3)


def triage_report(
    triage_output: dict,
    expected_urgency: Optional[int] = None,
    expected_red_flags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate a full triage quality report."""
    report = {
        "predicted_urgency": triage_output.get("urgency_level"),
        "predicted_label": triage_output.get("urgency_label"),
        "confidence_score": triage_output.get("confidence_score"),
        "confidence_band": triage_output.get("confidence_band"),
        "num_red_flags": len(triage_output.get("red_flags", [])),
        "num_actions": len(triage_output.get("recommended_actions", [])),
        "num_follow_ups": len(triage_output.get("follow_up_questions", [])),
    }

    if expected_urgency is not None:
        report["urgency_exact_match"] = urgency_accuracy(
            triage_output.get("urgency_level", 0), expected_urgency
        )
        report["urgency_within_one"] = urgency_within_one(
            triage_output.get("urgency_level", 0), expected_urgency
        )
        report["safety_score"] = safety_score(triage_output, expected_urgency)

    if expected_red_flags is not None:
        report["red_flag_recall"] = red_flag_recall(
            triage_output.get("red_flags", []), expected_red_flags
        )

    return report
