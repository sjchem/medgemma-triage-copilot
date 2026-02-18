"""
MedGuard Triage Copilot – Text Pipeline
=========================================
Pipeline: Free text → Structure → Triage
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from models.extraction.gemma_structurer import GemmaStructurer
from models.extraction.uncertainty_engine import UncertaintyEngine
from pipelines.structured_pipeline import StructuredPipeline

logger = logging.getLogger(__name__)


class TextPipeline:
    """Text intake → structured extraction → triage reasoning."""

    def __init__(
        self,
        structurer: Optional[GemmaStructurer] = None,
        uncertainty_engine: Optional[UncertaintyEngine] = None,
        structured_pipeline: Optional[StructuredPipeline] = None,
        **kwargs,
    ):
        self.structurer = structurer or GemmaStructurer(
            **kwargs.get("structurer_config", {})
        )
        self.uncertainty_engine = uncertainty_engine or UncertaintyEngine()
        self.structured_pipeline = structured_pipeline or StructuredPipeline(**kwargs)

    def run(self, patient_text: str | dict) -> dict:
        """
        Run text → structure → triage pipeline.

        Parameters
        ----------
        patient_text : str or dict
            Free-text patient intake or a semi-structured dict.

        Returns
        -------
        dict – PipelineResult-compatible.
        """
        errors = []

        # Stage 2: Extraction
        logger.info("Text pipeline: extracting structured data")
        try:
            structured = self.structurer.structure(patient_text)
        except Exception as e:
            logger.error("Structurer failed: %s", e)
            structured = self.structurer.structure_safe(patient_text)
            errors.append(f"Structurer error (fallback used): {e}")

        # Uncertainty analysis
        structured = self.uncertainty_engine.analyze(structured)

        # Stage 3: Triage
        result = self.structured_pipeline.run(structured)
        result["input_type"] = "text"
        result["structured_data"] = structured
        result["pipeline_errors"] = errors + result.get("pipeline_errors", [])

        return result
