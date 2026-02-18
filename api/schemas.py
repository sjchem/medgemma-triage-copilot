"""
MedGuard Triage Copilot – API Schemas
=======================================
Pydantic models for the REST API request/response contracts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request Models ──────────────────────────────────────────────────────────


class TextTriageRequest(BaseModel):
    """Request body for text-based triage."""
    patient_text: str = Field(..., description="Free-text patient intake description")
    metadata: Optional[Dict[str, Any]] = None


class StructuredTriageRequest(BaseModel):
    """Request body for pre-structured clinical data."""
    chief_complaint: str
    symptoms: List[Dict[str, Any]] = Field(default_factory=list)
    demographics: Optional[Dict[str, Any]] = None
    vitals: Optional[Dict[str, Any]] = None
    medical_history: Optional[List[Dict[str, Any]]] = None
    medications: Optional[List[Dict[str, Any]]] = None
    allergies: Optional[List[Dict[str, Any]]] = None
    risk_factors: Optional[List[str]] = None


# ── Response Models ─────────────────────────────────────────────────────────


class RedFlagResponse(BaseModel):
    flag: str
    category: str
    source: str
    recommended_action: str


class TriageResponse(BaseModel):
    """Triage result returned by the API."""
    id: str
    timestamp: str
    urgency_level: int
    urgency_label: str
    red_flags: List[RedFlagResponse] = Field(default_factory=list)
    risk_categories: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    confidence_score: float
    confidence_band: str
    safety_note: str
    safety_overrides_applied: List[str] = Field(default_factory=list)
    disclaimer: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    models: Dict[str, str] = Field(default_factory=dict)
