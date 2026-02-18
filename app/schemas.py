"""
Backward-compatible re-exports from the new schema module.
See models/extraction/schema_definition.py for the full definitions.
"""
from models.extraction.schema_definition import (  # noqa: F401
    StructuredClinicalData,
    TriageOutput,
    PipelineResult,
    Symptom,
    VitalSigns,
    PatientDemographics,
    RedFlag,
    ASROutput,
)
