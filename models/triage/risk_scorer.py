"""
MedGuard Triage Copilot – Risk Scorer
======================================
Combines model-based triage output with rule-based red flag detection
to produce a final, safety-adjusted risk score.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_ESCALATION_PATH = Path(__file__).resolve().parents[2] / "configs" / "escalation_policy.yaml"
_DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "configs" / "safety_rules.yaml"


class RiskScorer:
    """Produces final risk assessment by merging model + rule-based signals."""

    def __init__(
        self,
        escalation_path: Optional[str] = None,
        rules_path: Optional[str] = None,
    ):
        esc_path = Path(escalation_path) if escalation_path else _DEFAULT_ESCALATION_PATH
        rules_path_ = Path(rules_path) if rules_path else _DEFAULT_RULES_PATH

        self.escalation_policy = {}
        if esc_path.exists():
            with open(esc_path) as f:
                self.escalation_policy = yaml.safe_load(f)

        self.confidence_config = {}
        if rules_path_.exists():
            with open(rules_path_) as f:
                cfg = yaml.safe_load(f)
                self.confidence_config = cfg.get("confidence", {})

    def score(
        self,
        triage_output: dict,
        rule_flags: List[dict],
        rule_min_urgency: int,
        extraction_confidence: float,
    ) -> dict:
        """
        Compute final scored triage output.

        Parameters
        ----------
        triage_output : dict
            Raw output from MedGemma triage reasoner.
        rule_flags : list[dict]
            Red flags from rule-based engine.
        rule_min_urgency : int
            Minimum urgency level from rule engine.
        extraction_confidence : float
            Confidence from the extraction stage.

        Returns
        -------
        dict – enriched triage output with safety overrides applied.
        """
        result = dict(triage_output)
        overrides: List[str] = []

        model_urgency = result.get("urgency_level", 3)
        final_urgency = model_urgency

        # ── Rule-based urgency override ─────────────────────────────────
        if rule_min_urgency > model_urgency:
            final_urgency = rule_min_urgency
            overrides.append(
                f"Urgency upgraded {model_urgency} → {rule_min_urgency} by red flag rules"
            )

        # ── Merge red flags (deduplicating by lowercase flag name) ────────
        existing_flags = result.get("red_flags", [])
        seen_flags: set = set()
        deduped_flags = []
        for f in existing_flags:
            key = f.get("flag", "").strip().lower()
            if key and key not in seen_flags:
                seen_flags.add(key)
                deduped_flags.append(f)

        for rf in rule_flags:
            key = rf["flag"].strip().lower()
            if key not in seen_flags:
                deduped_flags.append(rf)
                seen_flags.add(key)
                overrides.append(f"Red flag added by rule engine: {rf['flag']}")

        result["red_flags"] = deduped_flags

        # ── Confidence adjustment ───────────────────────────────────────
        model_confidence = result.get("confidence_score", 0.5)
        # Blend with extraction confidence
        blended = round(model_confidence * 0.6 + extraction_confidence * 0.4, 3)
        result["confidence_score"] = blended

        low_thresh = self.confidence_config.get("low_threshold", 0.4)
        med_thresh = self.confidence_config.get("medium_threshold", 0.65)
        high_thresh = self.confidence_config.get("high_threshold", 0.85)

        if blended >= high_thresh:
            result["confidence_band"] = "high"
        elif blended >= med_thresh:
            result["confidence_band"] = "medium"
        elif blended >= low_thresh:
            result["confidence_band"] = "low"
        else:
            result["confidence_band"] = "uncertain"
            overrides.append("Low confidence – flagged for senior review")

        # ── Low-confidence safety bump ──────────────────────────────────
        if blended < low_thresh and final_urgency < 4:
            final_urgency = max(final_urgency, 3)
            overrides.append("Urgency bumped to ≥3 due to low confidence")

        # ── Multiple red flags escalation ───────────────────────────────
        if len(existing_flags) >= 2 and final_urgency < 5:
            final_urgency = min(final_urgency + 1, 5)
            overrides.append("Urgency +1 due to multiple red flags")

        result["urgency_level"] = final_urgency

        # ── Attach urgency label from escalation policy ─────────────────
        levels = self.escalation_policy.get("urgency_levels", {})
        level_info = levels.get(final_urgency, {})
        result["urgency_label"] = level_info.get("label", f"Level {final_urgency}")

        # ── Attach safety note ──────────────────────────────────────────
        safety_parts = [result.get("safety_note", "")]
        if overrides:
            safety_parts.append("SAFETY OVERRIDES: " + "; ".join(overrides))
        result["safety_note"] = " ".join(p for p in safety_parts if p).strip()
        # ── Low-confidence guardrail ─────────────────────────────────
        if result.get("confidence_score", 1.0) < 0.5:
            result["safety_note"] = (
                result["safety_note"].rstrip(". ") +
                " Low confidence — clinician review strongly recommended."
            ).strip()
        return result, overrides
