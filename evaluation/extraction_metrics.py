"""
MedGuard Triage Copilot – Extraction Metrics
==============================================
Evaluates the quality of Stage 2 (Gemma Structurer) output.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def extraction_completeness(structured: dict, required_fields: Optional[List[str]] = None) -> float:
    """
    Fraction of required fields that are non-empty.

    Parameters
    ----------
    structured : dict
        Output from Stage 2.
    required_fields : list[str], optional
        Dot-separated field names. Defaults to core fields.

    Returns
    -------
    float – 0.0 to 1.0
    """
    if required_fields is None:
        required_fields = [
            "chief_complaint",
            "symptoms",
            "demographics.age",
            "demographics.sex",
        ]

    filled = 0
    for field in required_fields:
        val = _get_nested(structured, field)
        if val is not None and val != "" and val != []:
            filled += 1

    return filled / len(required_fields) if required_fields else 1.0


def symptom_recall(predicted: List[dict], expected: List[str]) -> float:
    """
    What fraction of expected symptoms were extracted?

    Parameters
    ----------
    predicted : list[dict]
        Extracted symptoms (each with a "name" key).
    expected : list[str]
        Ground-truth symptom names (case-insensitive match).
    """
    if not expected:
        return 1.0

    pred_names = {s.get("name", "").lower().strip() for s in predicted}
    pred_normalized = {s.get("normalized_name", "").lower().strip() for s in predicted if s.get("normalized_name")}
    all_pred = pred_names | pred_normalized

    hits = sum(1 for e in expected if e.lower().strip() in all_pred)
    return hits / len(expected)


def symptom_precision(predicted: List[dict], expected: List[str]) -> float:
    """What fraction of extracted symptoms are correct?"""
    if not predicted:
        return 1.0 if not expected else 0.0

    expected_lower = {e.lower().strip() for e in expected}
    hits = 0
    for s in predicted:
        name = s.get("name", "").lower().strip()
        normalized = (s.get("normalized_name") or "").lower().strip()
        if name in expected_lower or normalized in expected_lower:
            hits += 1

    return hits / len(predicted)


def negation_accuracy(predicted: List[dict], expected: List[dict]) -> float:
    """
    Accuracy of negation detection.

    Parameters
    ----------
    predicted, expected : list[dict]
        Each with "name" and "negation" keys.
    """
    if not expected:
        return 1.0

    expected_map = {e["name"].lower().strip(): e.get("negation", "affirmed") for e in expected}
    correct = 0
    total = 0

    for p in predicted:
        name = p.get("name", "").lower().strip()
        if name in expected_map:
            total += 1
            if p.get("negation", "affirmed") == expected_map[name]:
                correct += 1

    return correct / total if total > 0 else 1.0


def extraction_report(
    structured: dict,
    expected_symptoms: Optional[List[str]] = None,
    expected_negations: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Generate a full extraction quality report."""
    report = {
        "completeness": extraction_completeness(structured),
        "extraction_confidence": structured.get("extraction_confidence", None),
        "num_symptoms": len(structured.get("symptoms", [])),
        "num_missing_fields": len(structured.get("missing_information", [])),
        "num_uncertainty_notes": len(structured.get("uncertainty_notes", [])),
    }

    if expected_symptoms:
        report["symptom_recall"] = symptom_recall(structured.get("symptoms", []), expected_symptoms)
        report["symptom_precision"] = symptom_precision(structured.get("symptoms", []), expected_symptoms)

    if expected_negations:
        report["negation_accuracy"] = negation_accuracy(structured.get("symptoms", []), expected_negations)

    return report


def _get_nested(data: dict, dotted_key: str):
    keys = dotted_key.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return None
    return current
