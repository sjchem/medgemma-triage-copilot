"""
MedGuard Triage Copilot – Schema Definitions
=============================================
Pydantic models for the three-stage pipeline:
  Stage 1 output: Raw transcript
  Stage 2 output: Structured clinical data (from Gemma structurer)
  Stage 3 output: Triage decision (from MedGemma reasoner)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Enums ───────────────────────────────────────────────────────────────────


class UrgencyLevel(int, Enum):
    NON_URGENT = 1
    LOW = 2
    MODERATE = 3
    URGENT = 4
    CRITICAL = 5


class ConfidenceBand(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class NegationStatus(str, Enum):
    AFFIRMED = "affirmed"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"


# ── Stage 1: ASR Output ────────────────────────────────────────────────────


class TranscriptSegment(BaseModel):
    """A single segment from speech-to-text."""
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: Optional[float] = None


class ASROutput(BaseModel):
    """Complete output from MedASR (Stage 1)."""
    transcript: str
    segments: List[TranscriptSegment] = Field(default_factory=list)
    language: str = "en"
    duration_s: Optional[float] = None
    model_id: str = "google/medasr"


# ── Stage 2: Structured Clinical Data ──────────────────────────────────────


class Symptom(BaseModel):
    """A single clinical symptom with metadata."""
    name: str
    normalized_name: Optional[str] = None
    body_region: Optional[str] = None
    severity: Optional[str] = None           # mild / moderate / severe
    onset: Optional[str] = None              # acute / gradual / sudden
    duration: Optional[str] = None
    negation: NegationStatus = NegationStatus.AFFIRMED
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class VitalSigns(BaseModel):
    """Vital signs extracted from intake."""
    heart_rate: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    respiratory_rate: Optional[int] = None
    temperature_c: Optional[float] = None
    spo2: Optional[float] = None
    gcs: Optional[int] = None


class PatientDemographics(BaseModel):
    """Basic patient demographics."""
    age: Optional[int] = None
    sex: Optional[str] = None
    weight_kg: Optional[float] = None
    pregnant: Optional[bool] = None


class ClinicalEntity(BaseModel):
    """A generic clinical entity (condition, medication, allergy, etc.)."""
    name: str
    category: str                            # condition / medication / allergy / procedure
    negation: NegationStatus = NegationStatus.AFFIRMED
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    details: Optional[str] = None


class UncertaintyNote(BaseModel):
    """Tracks a specific source of uncertainty in extraction."""
    field: str
    reason: str
    impact: str = "low"                      # low / medium / high


class StructuredClinicalData(BaseModel):
    """Complete output from Gemma Structurer (Stage 2)."""
    chief_complaint: str
    symptoms: List[Symptom] = Field(default_factory=list)
    vitals: Optional[VitalSigns] = None
    demographics: PatientDemographics = Field(default_factory=PatientDemographics)
    medical_history: List[ClinicalEntity] = Field(default_factory=list)
    medications: List[ClinicalEntity] = Field(default_factory=list)
    allergies: List[ClinicalEntity] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    uncertainty_notes: List[UncertaintyNote] = Field(default_factory=list)
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    raw_input: Optional[str] = None


# ── Stage 3: Triage Output ─────────────────────────────────────────────────


class RedFlag(BaseModel):
    """A detected red flag with source and action."""
    flag: str
    category: str                            # cardiac / stroke / respiratory / ...
    source: str                              # which symptom/data triggered this
    recommended_action: str


class TriageOutput(BaseModel):
    """Complete output from MedGemma Triage Reasoner (Stage 3)."""
    urgency_level: int = Field(ge=1, le=5)
    urgency_label: str
    red_flags: List[RedFlag] = Field(default_factory=list)
    risk_categories: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_band: ConfidenceBand = ConfidenceBand.MEDIUM
    safety_note: str = ""
    reasoning_summary: Optional[str] = None


# ── Full Pipeline Result ────────────────────────────────────────────────────


class PipelineResult(BaseModel):
    """End-to-end result wrapping all three stages."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    input_type: str = "text"                 # text / voice / structured
    asr_output: Optional[ASROutput] = None
    structured_data: Optional[StructuredClinicalData] = None
    triage_output: Optional[TriageOutput] = None
    safety_overrides_applied: List[str] = Field(default_factory=list)
    pipeline_errors: List[str] = Field(default_factory=list)
    disclaimer: str = (
        "This is an AI-assisted triage tool. It does NOT provide medical diagnoses. "
        "All outputs must be reviewed by qualified healthcare professionals."
    )
