"""
MedGuard Triage Copilot – MedGemma Triage Reasoner
====================================================
Stage 3: Uses MedGemma 4B-IT for safety-focused triage reasoning.
Operates ONLY on structured clinical data (from Stage 2).
Handles:
  - Risk stratification
  - Urgency scoring (1–5)
  - Safety note generation
  - Follow-up question suggestions
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Triage System Prompt ────────────────────────────────────────────────────

TRIAGE_SYSTEM_PROMPT = """You are a medical triage reasoning engine. You receive STRUCTURED clinical data and output a TRIAGE ASSESSMENT only.

STRICT RULES:
1. Do NOT produce diagnoses, critiques, explanations, or prose.
2. Do NOT repeat or evaluate information — only assess urgency and risk.
3. Prioritize patient safety. When in doubt, triage UP.
4. Each red flag must be unique — do NOT list duplicates.
5. Each recommended action must be concise and non-redundant.
6. Assign a confidence score (0.0–1.0) reflecting certainty in urgency level.
7. Output ONLY the JSON object below — no markdown, no preamble, no commentary.

URGENCY LEVELS:
  1 = Non-Urgent       (routine, 24–72 h)
  2 = Low Urgency      (minor, 4–24 h)
  3 = Moderate         (potentially serious, 1–4 h)
  4 = Urgent           (serious, 15–60 min)
  5 = Critical         (immediate resuscitation)

REQUIRED OUTPUT — valid JSON only:
{
  "urgency_level": <int 1-5>,
  "urgency_label": "<string>",
  "red_flags": [
    {"flag": "<unique finding>", "category": "<cardiac|stroke|respiratory|hemorrhage|neurological|sepsis|psychiatric|pediatric|other>", "source": "<symptom|history|demography>", "recommended_action": "<single concise action>"}
  ],
  "risk_categories": ["<string>"],
  "recommended_actions": ["<concise, non-redundant action>"],
  "follow_up_questions": ["<specific gap-filling question>"],
  "confidence_score": <0.0-1.0>,
  "confidence_band": "<high|medium|low|uncertain>",
  "safety_note": "<one sentence for clinician>",
  "reasoning_summary": "<one sentence summary of triage rationale>"
}"""

TRIAGE_USER_PROMPT = """Triage the following structured clinical data. Return ONLY the JSON object — no other text.

{structured_data}"""


class MedGemmaReasoner:
    """Triage reasoner — calls your dedicated MedGemma HF Inference Endpoint."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        timeout: int = 120,
        **_,  # absorb legacy kwargs (model_id, use_inference_api, etc.)
    ):
        self.endpoint_url = endpoint_url or os.environ.get("TRIAGE_ENDPOINT_URL")
        if not self.endpoint_url:
            raise EnvironmentError(
                "TRIAGE_ENDPOINT_URL is not set. "
                "Add it to your .env file or pass endpoint_url= to MedGemmaReasoner()."
            )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout = timeout

    # ── Model call ──────────────────────────────────────────────────────

    def _call_model(self, prompt: str) -> str:
        """Call the dedicated MedGemma endpoint."""
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

        logger.info("Calling triage endpoint: %s", self.endpoint_url)
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
        text = text.strip()
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if md_match:
            try:
                return json.loads(md_match.group(1))
            except json.JSONDecodeError:
                pass

        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in triage output: {text[:200]}")

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

        raise ValueError(f"Could not parse valid JSON from triage output: {text[:300]}")

    # ── Public API ──────────────────────────────────────────────────────

    def reason(self, structured_data: dict) -> dict:
        """
        Perform triage reasoning on structured clinical data.

        Parameters
        ----------
        structured_data : dict
            Output from Stage 2 (GemmaStructurer).

        Returns
        -------
        dict matching the TriageOutput schema.
        """
        # Remove raw_input before sending to triage model (not needed for reasoning)
        data_for_model = {k: v for k, v in structured_data.items() if k != "raw_input"}

        prompt = (
            f"{TRIAGE_SYSTEM_PROMPT}\n\n"
            f"{TRIAGE_USER_PROMPT.format(structured_data=json.dumps(data_for_model, indent=2))}"
        )

        raw_output = self._call_model(prompt)
        logger.debug("Triage raw output: %s", raw_output[:500])

        triage = self._extract_json(raw_output)
        return triage

    def reason_safe(self, structured_data: dict) -> dict:
        """Like reason() but returns a safe fallback on failure."""
        try:
            return self.reason(structured_data)
        except Exception as e:
            logger.error("Triage reasoning failed: %s", e)
            return {
                "urgency_level": 4,
                "urgency_label": "Urgent – Triage Engine Error",
                "red_flags": [],
                "risk_categories": ["system_error"],
                "recommended_actions": [
                    "SYSTEM ERROR: Triage model failed. Route to clinician immediately.",
                    "Treat as potentially urgent until human review.",
                ],
                "follow_up_questions": [],
                "confidence_score": 0.0,
                "confidence_band": "uncertain",
                "safety_note": f"Automated triage failed ({e}). Default to urgent. Clinician must assess.",
                "reasoning_summary": "Model error – defaulting to urgent for safety.",
            }
