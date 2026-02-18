"""
MedGuard Triage Copilot – Uncertainty Engine
=============================================
Analyzes structured clinical data for sources of uncertainty,
propagates confidence scores, and flags gaps.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


# ── Required fields for a complete clinical picture ─────────────────────────

REQUIRED_FIELDS = [
    "chief_complaint",
    "demographics.age",
    "demographics.sex",
    "symptoms",
]

RECOMMENDED_FIELDS = [
    "vitals.heart_rate",
    "vitals.blood_pressure_systolic",
    "vitals.spo2",
    "vitals.temperature_c",
    "medical_history",
    "medications",
    "allergies",
]


def _get_nested(data: dict, dotted_key: str):
    """Retrieve a nested value using dot notation."""
    keys = dotted_key.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return None
    return current


class UncertaintyEngine:
    """Assess and propagate uncertainty through structured clinical data."""

    def __init__(
        self,
        low_confidence_threshold: float = 0.4,
        medium_confidence_threshold: float = 0.65,
    ):
        self.low_threshold = low_confidence_threshold
        self.medium_threshold = medium_confidence_threshold

    def analyze(self, structured_data: dict) -> dict:
        """
        Analyze structured data for uncertainty sources.

        Returns an enriched copy with updated uncertainty_notes,
        missing_information, and an adjusted extraction_confidence.
        """
        notes: List[dict] = list(structured_data.get("uncertainty_notes", []))
        missing: List[str] = list(structured_data.get("missing_information", []))

        # Check required fields
        for field in REQUIRED_FIELDS:
            val = _get_nested(structured_data, field)
            if val is None or val == "" or val == []:
                missing.append(f"Missing required field: {field}")
                notes.append({
                    "field": field,
                    "reason": "Required field not provided",
                    "impact": "high",
                })

        # Check recommended fields
        for field in RECOMMENDED_FIELDS:
            val = _get_nested(structured_data, field)
            if val is None:
                missing.append(f"Missing recommended field: {field}")
                notes.append({
                    "field": field,
                    "reason": "Recommended field not provided",
                    "impact": "medium",
                })

        # Check per-symptom confidence
        for i, symptom in enumerate(structured_data.get("symptoms", [])):
            conf = symptom.get("confidence", 1.0)
            if conf < self.low_threshold:
                notes.append({
                    "field": f"symptoms[{i}].{symptom.get('name', '?')}",
                    "reason": f"Low extraction confidence ({conf:.2f})",
                    "impact": "high",
                })
            elif conf < self.medium_threshold:
                notes.append({
                    "field": f"symptoms[{i}].{symptom.get('name', '?')}",
                    "reason": f"Medium extraction confidence ({conf:.2f})",
                    "impact": "medium",
                })

        # Check for uncertain negation
        for i, symptom in enumerate(structured_data.get("symptoms", [])):
            if symptom.get("negation") == "uncertain":
                notes.append({
                    "field": f"symptoms[{i}].{symptom.get('name', '?')}",
                    "reason": "Negation status unclear",
                    "impact": "high",
                })

        # Compute adjusted confidence
        high_impact = sum(1 for n in notes if n.get("impact") == "high")
        medium_impact = sum(1 for n in notes if n.get("impact") == "medium")

        base_conf = structured_data.get("extraction_confidence", 1.0)
        penalty = high_impact * 0.15 + medium_impact * 0.05
        adjusted = max(0.0, min(1.0, base_conf - penalty))

        # Deduplicate
        seen_missing = set()
        unique_missing = []
        for m in missing:
            if m not in seen_missing:
                seen_missing.add(m)
                unique_missing.append(m)

        result = dict(structured_data)
        result["uncertainty_notes"] = notes
        result["missing_information"] = unique_missing
        result["extraction_confidence"] = round(adjusted, 3)

        return result
