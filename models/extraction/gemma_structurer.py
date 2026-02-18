"""
MedGuard Triage Copilot – Gemma Clinical Structurer
====================================================
Stage 2: Uses Gemma 2-2B Instruct to extract structured clinical entities
from free-text patient intake. Handles:
  - Symptom extraction & normalization
  - Negation detection
  - Risk factor identification
  - Missing information tracking
  - Uncertainty metadata propagation
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Structurer System Prompt ────────────────────────────────────────────────

STRUCTURER_SYSTEM_PROMPT = """You are a clinical data structurer. Your ONLY job is to extract and organize clinical information from patient intake text into a precise JSON structure.

RULES:
1. Extract ONLY what is explicitly stated. Do NOT infer diagnoses.
2. Mark negated symptoms clearly (e.g., "denies nausea" → negation: "negated").
3. If information is missing or ambiguous, list it in "missing_information".
4. Assign an extraction_confidence score (0.0–1.0) based on clarity of input.
5. Normalize symptom names to standard clinical terminology where possible.
6. Track uncertainty in "uncertainty_notes" for any ambiguous extractions.

OUTPUT FORMAT – Return ONLY valid JSON with this exact structure:
{
  "chief_complaint": "string",
  "symptoms": [
    {
      "name": "string",
      "normalized_name": "string or null",
      "body_region": "string or null",
      "severity": "mild|moderate|severe or null",
      "onset": "acute|gradual|sudden or null",
      "duration": "string or null",
      "negation": "affirmed|negated|uncertain",
      "confidence": 0.0-1.0
    }
  ],
  "vitals": {
    "heart_rate": null, "blood_pressure_systolic": null,
    "blood_pressure_diastolic": null, "respiratory_rate": null,
    "temperature_c": null, "spo2": null, "gcs": null
  },
  "demographics": {"age": null, "sex": null, "weight_kg": null, "pregnant": null},
  "medical_history": [{"name": "string", "category": "condition", "negation": "affirmed", "confidence": 1.0, "details": null}],
  "medications": [{"name": "string", "category": "medication", "negation": "affirmed", "confidence": 1.0, "details": null}],
  "allergies": [{"name": "string", "category": "allergy", "negation": "affirmed", "confidence": 1.0, "details": null}],
  "risk_factors": ["string"],
  "missing_information": ["string"],
  "uncertainty_notes": [{"field": "string", "reason": "string", "impact": "low|medium|high"}],
  "extraction_confidence": 0.0-1.0
}"""

STRUCTURER_USER_PROMPT = """Extract structured clinical data from the following patient intake:

---
{patient_input}
---

Return ONLY the JSON object. No explanations, no markdown."""


class GemmaStructurer:
    """Clinical structurer — calls your dedicated HF Inference Endpoint."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        timeout: int = 120,
        **_,  # absorb legacy kwargs (model_id, use_inference_api, etc.)
    ):
        self.endpoint_url = endpoint_url or os.environ.get("STRUCTURER_ENDPOINT_URL")
        if not self.endpoint_url:
            raise EnvironmentError(
                "STRUCTURER_ENDPOINT_URL is not set. "
                "Add it to your .env file or pass endpoint_url= to GemmaStructurer()."
            )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout = timeout

    # ── Model call ──────────────────────────────────────────────────────

    def _call_model(self, prompt: str) -> str:
        """Call the dedicated Gemma endpoint."""
        import requests

        token = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            raise EnvironmentError("Set HF_API_TOKEN or HF_TOKEN in your .env file.")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "do_sample": False,
                "return_full_text": False,
            },
        }

        logger.info("Calling structurer endpoint: %s", self.endpoint_url)
        response = requests.post(
            self.endpoint_url, headers=headers, json=payload, timeout=self.timeout
        )
        response.raise_for_status()

        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return str(result)

    # ── JSON extraction ─────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract the first valid JSON object from model output."""
        # Try direct parse first
        text = text.strip()
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # Try to find JSON block in markdown code fence
        md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if md_match:
            try:
                return json.loads(md_match.group(1))
            except json.JSONDecodeError:
                pass

        # Brace-matching fallback
        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in model output: {text[:200]}")

        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        continue

        raise ValueError(f"Could not parse valid JSON from model output: {text[:300]}")

    # ── Public API ──────────────────────────────────────────────────────

    def structure(self, patient_input: str | dict) -> dict:
        """
        Extract structured clinical data from patient intake text.

        Parameters
        ----------
        patient_input : str or dict
            Free-text description or a dict (will be serialized).

        Returns
        -------
        dict matching the StructuredClinicalData schema.
        """
        if isinstance(patient_input, dict):
            input_text = json.dumps(patient_input, indent=2)
        else:
            input_text = patient_input

        prompt = (
            f"{STRUCTURER_SYSTEM_PROMPT}\n\n"
            f"{STRUCTURER_USER_PROMPT.format(patient_input=input_text)}"
        )

        raw_output = self._call_model(prompt)
        logger.debug("Structurer raw output: %s", raw_output[:500])

        structured = self._extract_json(raw_output)

        # Attach raw input for traceability
        structured["raw_input"] = input_text

        return structured

    def structure_safe(self, patient_input: str | dict) -> dict:
        """
        Like structure() but returns a fallback on failure instead of raising.
        """
        try:
            return self.structure(patient_input)
        except Exception as e:
            logger.error("Structurer failed: %s", e)
            input_text = json.dumps(patient_input) if isinstance(patient_input, dict) else patient_input
            return {
                "chief_complaint": input_text[:200],
                "symptoms": [],
                "vitals": None,
                "demographics": {},
                "medical_history": [],
                "medications": [],
                "allergies": [],
                "risk_factors": [],
                "missing_information": ["Structurer extraction failed – raw input forwarded"],
                "uncertainty_notes": [
                    {"field": "all", "reason": str(e), "impact": "high"}
                ],
                "extraction_confidence": 0.0,
                "raw_input": input_text,
            }
