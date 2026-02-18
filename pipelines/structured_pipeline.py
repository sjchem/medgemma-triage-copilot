"""
MedGuard Triage Copilot – Structured Pipeline
===============================================
Pipeline: Structured clinical data → Triage (Stage 3 only)
Used when clinical data is already structured (e.g., from EHR or manual entry).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

from models.triage.medgemma_reasoner import MedGemmaReasoner
from models.triage.red_flag_engine import RedFlagEngine
from models.triage.risk_scorer import RiskScorer

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This is an AI-assisted triage tool. It does NOT provide medical diagnoses. "
    "All outputs must be reviewed by qualified healthcare professionals."
)


class StructuredPipeline:
    """Triage reasoning on pre-structured clinical data."""

    def __init__(
        self,
        reasoner: Optional[MedGemmaReasoner] = None,
        red_flag_engine: Optional[RedFlagEngine] = None,
        risk_scorer: Optional[RiskScorer] = None,
        **kwargs,
    ):
        self.reasoner = reasoner or MedGemmaReasoner(
            **kwargs.get("triage_config", {})
        )
        self.red_flag_engine = red_flag_engine or RedFlagEngine()
        self.risk_scorer = risk_scorer or RiskScorer()

    def run(self, structured_data: dict) -> dict:
        """
        Run triage reasoning on structured clinical data.

        Returns PipelineResult-compatible dict.
        """
        errors = []
        overrides = []

        # Rule-based red flag scan (always runs)
        logger.info("Structured pipeline: scanning for red flags")
        rule_flags = self.red_flag_engine.scan(structured_data)
        rule_min_urgency = self.red_flag_engine.get_min_urgency(rule_flags)

        # Model-based triage reasoning
        logger.info("Structured pipeline: running triage reasoner")
        try:
            triage_raw = self.reasoner.reason(structured_data)
        except Exception as e:
            logger.error("Triage model failed: %s", e)
            triage_raw = self.reasoner.reason_safe(structured_data)
            errors.append(f"Triage model error (fallback used): {e}")

        # Risk scoring: merge model + rules
        extraction_conf = structured_data.get("extraction_confidence", 0.5)
        triage_final, applied_overrides = self.risk_scorer.score(
            triage_output=triage_raw,
            rule_flags=rule_flags,
            rule_min_urgency=rule_min_urgency,
            extraction_confidence=extraction_conf,
        )
        overrides.extend(applied_overrides)

        return {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat(),
            "input_type": "structured",
            "asr_output": None,
            "structured_data": structured_data,
            "triage_output": triage_final,
            "safety_overrides_applied": overrides,
            "pipeline_errors": errors,
            "disclaimer": DISCLAIMER,
        }
