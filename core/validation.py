"""
MedGuard Triage Copilot – Validation Utilities
================================================
Validates pipeline inputs and outputs against Pydantic schemas.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from pydantic import ValidationError

from models.extraction.schema_definition import (
    StructuredClinicalData,
    TriageOutput,
    PipelineResult,
)

logger = logging.getLogger(__name__)


def validate_structured_data(data: dict) -> Tuple[Optional[StructuredClinicalData], List[str]]:
    """
    Validate extracted clinical data against the schema.

    Returns (validated_model, errors_list).
    """
    errors = []
    try:
        validated = StructuredClinicalData(**data)
        return validated, []
    except ValidationError as e:
        for err in e.errors():
            field = ".".join(str(loc) for loc in err["loc"])
            errors.append(f"Validation error at '{field}': {err['msg']}")
        logger.warning("Structured data validation failed: %d errors", len(errors))
        return None, errors


def validate_triage_output(data: dict) -> Tuple[Optional[TriageOutput], List[str]]:
    """
    Validate triage output against the schema.

    Returns (validated_model, errors_list).
    """
    errors = []
    try:
        validated = TriageOutput(**data)
        return validated, []
    except ValidationError as e:
        for err in e.errors():
            field = ".".join(str(loc) for loc in err["loc"])
            errors.append(f"Validation error at '{field}': {err['msg']}")
        logger.warning("Triage output validation failed: %d errors", len(errors))
        return None, errors


def validate_pipeline_result(data: dict) -> Tuple[Optional[PipelineResult], List[str]]:
    """
    Validate the full pipeline result.

    Returns (validated_model, errors_list).
    """
    errors = []
    try:
        validated = PipelineResult(**data)
        return validated, []
    except ValidationError as e:
        for err in e.errors():
            field = ".".join(str(loc) for loc in err["loc"])
            errors.append(f"Validation error at '{field}': {err['msg']}")
        logger.warning("Pipeline result validation failed: %d errors", len(errors))
        return None, errors


def coerce_urgency(value: Any) -> int:
    """Safely coerce urgency to int in [1, 5]."""
    try:
        v = int(value)
        return max(1, min(5, v))
    except (TypeError, ValueError):
        logger.warning("Could not parse urgency '%s' – defaulting to 3", value)
        return 3


def coerce_confidence(value: Any) -> float:
    """Safely coerce confidence to float in [0, 1]."""
    try:
        v = float(value)
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        logger.warning("Could not parse confidence '%s' – defaulting to 0.5", value)
        return 0.5
