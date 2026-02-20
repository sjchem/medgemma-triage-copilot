"""
MedGuard Triage Copilot – FastAPI Application
===============================================
REST API for the three-stage triage pipeline.

Endpoints:
  POST /triage/text      – Free-text patient intake → triage
  POST /triage/structured – Pre-structured data → triage
  GET  /health           – Health check
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    TextTriageRequest,
    StructuredTriageRequest,
    TriageResponse,
    HealthResponse,
)
from core.router import MedGuardRouter
from core.logging_utils import setup_logging

# ── Setup ───────────────────────────────────────────────────────────────────

setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MedGuard Triage Copilot",
    description=(
        "An agentic, safety-first AI triage system using "
        "Gemma (structurer) + MedGemma (reasoner) for clinical prioritization."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize router (lazy – models loaded on first call)
router = MedGuardRouter()


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        models={
            "structurer": router.config.get("structurer", {}).get("model_id", "google/gemma-7b-it"),
            "triage": router.config.get("triage", {}).get("model_id", "google/medgemma-4b-it"),
        },
    )


@app.post("/triage/text", response_model=TriageResponse)
async def triage_text(request: TextTriageRequest):
    """Run triage on free-text patient intake."""
    try:
        result = router.triage(request.patient_text, input_type="text")
        return _build_response(result)
    except Exception as e:
        logger.error("Triage text endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/triage/structured", response_model=TriageResponse)
async def triage_structured(request: StructuredTriageRequest):
    """Run triage on pre-structured clinical data."""
    try:
        data = request.model_dump(exclude_none=True)
        result = router.triage(data, input_type="structured")
        return _build_response(result)
    except Exception as e:
        logger.error("Triage structured endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_response(result: dict) -> TriageResponse:
    """Convert pipeline result to API response."""
    triage = result.get("triage_output", {})
    return TriageResponse(
        id=result.get("id", "unknown"),
        timestamp=result.get("timestamp", ""),
        urgency_level=triage.get("urgency_level", 3),
        urgency_label=triage.get("urgency_label", "Unknown"),
        red_flags=triage.get("red_flags", []),
        risk_categories=triage.get("risk_categories", []),
        recommended_actions=triage.get("recommended_actions", []),
        follow_up_questions=triage.get("follow_up_questions", []),
        confidence_score=triage.get("confidence_score", 0.0),
        confidence_band=triage.get("confidence_band", "uncertain"),
        safety_note=triage.get("safety_note", ""),
        safety_overrides_applied=result.get("safety_overrides_applied", []),
        disclaimer=result.get("disclaimer", ""),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
