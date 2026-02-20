"""
MedGuard Triage Copilot – MedGemma Triage Reasoner
====================================================
Stage 3: Medical triage reasoning. Supports two backends:

  - "hf_endpoint"  : Dedicated HuggingFace Inference Endpoint (legacy, slow)
  - "vertex_ai"    : Google Cloud Vertex AI Model Garden — medgemma-27b-text-it
                     No cold starts, SLA-backed, ~3–8 s latency

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
_LIST_FIELDS = (
    "red_flags", "risk_categories", "recommended_actions",
    "follow_up_questions", "symptoms", "medical_history",
    "medications", "allergies", "risk_factors",
    "missing_information", "uncertainty_notes",
)


def _sanitize_triage_output(data: dict) -> dict:
    """Strip None values from list fields in triage output."""
    for key in _LIST_FIELDS:
        val = data.get(key)
        if isinstance(val, list):
            data[key] = [item for item in val if item is not None]
        elif val is None:
            data[key] = []
    return data
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
        return _sanitize_triage_output(triage)

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


# ═══════════════════════════════════════════════════════════════════════════════
# Vertex AI Backend — google/medgemma-27b-text-it (recommended)
# ═══════════════════════════════════════════════════════════════════════════════

class VertexAIReasoner:
    """
    Triage reasoner backed by Google Cloud Vertex AI.

    Supports two Vertex AI call paths — auto-selected based on config:

    1. DEDICATED ENDPOINT  (endpoint_id is set)
       Uses aiplatform.Endpoint.predict() — for models you have already
       deployed in the Vertex AI console (e.g. one-click deploy).
       → Your current setup: medgemma-1.5-4b-it in europe-west4

    2. MODEL GARDEN API  (no endpoint_id)
       Uses vertexai.GenerativeModel — calls Model Garden directly,
       no dedicated deployment needed (pay-per-token).
       → Useful for medgemma-27b-text-it without deploying it yourself.

    Setup (one-time):
      1. Enable Vertex AI API in your GCP project
      2. Accept MedGemma terms in Model Garden (if not already done)
      3. Authenticate:
            gcloud auth application-default login
         OR set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
      4. pip install google-cloud-aiplatform

    Required env vars:
      GOOGLE_CLOUD_PROJECT   — your GCP project ID
      GOOGLE_CLOUD_LOCATION  — region matching your endpoint (e.g. europe-west4)
    """

    # Gemma2/MedGemma chat template tokens
    _TURN_START = "<start_of_turn>"
    _TURN_END   = "<end_of_turn>"

    def __init__(
        self,
        model: str = "medgemma-1.5-4b-it",
        endpoint_id: Optional[str] = None,
        project: Optional[str] = None,
        location: str = "us-central1",
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
        **_,
    ):
        try:
            import vertexai  # noqa: F401 — just verify the package is installed
            from google.cloud import aiplatform  # noqa: F401
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform is required for the Vertex AI backend.\n"
                "Install it with:  pip install google-cloud-aiplatform"
            )

        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self.project:
            raise EnvironmentError(
                "GOOGLE_CLOUD_PROJECT is not set. "
                "Add it to your .env file or pass project= to VertexAIReasoner()."
            )
        self.location = (
            location
            or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        )
        self.model_name        = model
        self.endpoint_id       = endpoint_id or os.environ.get("VERTEX_ENDPOINT_ID")
        self.max_output_tokens = max_output_tokens
        self.temperature       = temperature

        # Lazy — actual SDK objects are created on first use to avoid
        # crashing the whole app at startup if credentials aren't ready yet.
        self._endpoint           = None
        self._generative_model   = None
        self._sdk_initialised    = False

        logger.info(
            "VertexAIReasoner configured — model=%s endpoint_id=%s location=%s "
            "(SDK will connect on first triage call)",
            self.model_name, self.endpoint_id or "none (Model Garden)", self.location,
        )

    # ── Lazy SDK initialisation ─────────────────────────────────────────

    def _ensure_sdk(self):
        """Initialise Vertex AI SDK and endpoint/model on first use."""
        if self._sdk_initialised:
            return

        import vertexai
        from google.cloud import aiplatform

        vertexai.init(project=self.project, location=self.location)

        if self.endpoint_id:
            # ── Path 1: dedicated deployed endpoint ─────────────────────
            endpoint_name = (
                f"projects/{self.project}/locations/{self.location}"
                f"/endpoints/{self.endpoint_id}"
            )
            self._endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)
            logger.info(
                "VertexAIReasoner → dedicated endpoint %s (%s) model=%s",
                self.endpoint_id, self.location, self.model_name,
            )
        else:
            # ── Path 2: Model Garden GenerativeModel ────────────────────
            from vertexai.generative_models import GenerativeModel, GenerationConfig
            self._generative_model = GenerativeModel(
                model_name=self.model_name,
                system_instruction=TRIAGE_SYSTEM_PROMPT,
            )
            self._gen_config = GenerationConfig(
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
            )
            logger.info(
                "VertexAIReasoner → Model Garden model=%s project=%s location=%s",
                self.model_name, self.project, self.location,
            )

        self._sdk_initialised = True

    # ── Model call ─────────────────────────────────────────────────────

    def _format_prompt(self, user_message: str) -> str:
        """
        Build Gemma2 / MedGemma chat-template prompt.
        System instruction is prepended to the user turn since Gemma2
        instruction-tuned models expect:
          <start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n
        """
        return (
            f"{self._TURN_START}user\n"
            f"{TRIAGE_SYSTEM_PROMPT}\n\n"
            f"{user_message}"
            f"{self._TURN_END}\n"
            f"{self._TURN_START}model\n"
        )

    def _call_dedicated_endpoint(self, prompt: str) -> str:
        """Call the deployed Vertex AI endpoint via predict()."""
        self._ensure_sdk()
        instances = [{
            "prompt": self._format_prompt(prompt),
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }]
        logger.info(
            "Calling dedicated Vertex endpoint %s …", self.endpoint_id
        )
        response = self._endpoint.predict(instances=instances)

        # Handle both string and dict prediction formats
        pred = response.predictions[0]
        if isinstance(pred, dict):
            return pred.get("content", pred.get("output", str(pred)))
        return str(pred)

    def _call_model_garden(self, prompt: str) -> str:
        """Call the Model Garden GenerativeModel."""
        self._ensure_sdk()
        logger.info(
            "Calling Vertex AI Model Garden model=%s …", self.model_name
        )
        response = self._generative_model.generate_content(
            prompt,
            generation_config=self._gen_config,
        )
        return response.text

    def _call_model(self, prompt: str) -> str:
        if self.endpoint_id:
            return self._call_dedicated_endpoint(prompt)
        return self._call_model_garden(prompt)

    # ── Reuse JSON extractor and public API from MedGemmaReasoner ───────

    @staticmethod
    def _extract_json(text: str) -> dict:
        return MedGemmaReasoner._extract_json(text)

    def reason(self, structured_data: dict) -> dict:
        data_for_model = {k: v for k, v in structured_data.items() if k != "raw_input"}
        prompt = TRIAGE_USER_PROMPT.format(
            structured_data=json.dumps(data_for_model, indent=2)
        )
        raw_output = self._call_model(prompt)
        logger.debug("VertexAI triage raw output: %s", raw_output[:500])
        return _sanitize_triage_output(self._extract_json(raw_output))

    def reason_safe(self, structured_data: dict) -> dict:
        try:
            return self.reason(structured_data)
        except Exception as e:
            logger.error("VertexAI triage reasoning failed: %s", e)
            return {
                "urgency_level": 4,
                "urgency_label": "Urgent – Triage Engine Error",
                "red_flags": [],
                "risk_categories": ["system_error"],
                "recommended_actions": [
                    "SYSTEM ERROR: Vertex AI triage failed. Route to clinician immediately.",
                    "Treat as potentially urgent until human review.",
                ],
                "follow_up_questions": [],
                "confidence_score": 0.0,
                "confidence_band": "uncertain",
                "safety_note": f"Vertex AI triage failed ({e}). Default to urgent. Clinician must assess.",
                "reasoning_summary": "Model error – defaulting to urgent for safety.",
            }



# ═══════════════════════════════════════════════════════════════════════════════
# Factory — selects backend from config
# ═══════════════════════════════════════════════════════════════════════════════

def create_reasoner(config: dict) -> "MedGemmaReasoner | VertexAIReasoner":
    """
    Factory that returns the correct triage reasoner based on config["backend"].

    Supported values for config["backend"]:
      "vertex_ai"    →  VertexAIReasoner   (Google Cloud, recommended)
                        If vertex_endpoint_id is set → dedicated deployed endpoint
                        Otherwise               → Model Garden API
      "hf_endpoint"  →  MedGemmaReasoner   (HuggingFace Endpoint, legacy)

    Example model_config.yaml entries:

      # Vertex AI — dedicated deployed endpoint (your current setup)
      triage:
        backend: vertex_ai
        vertex_model: medgemma-1.5-4b-it
        vertex_location: europe-west4
        vertex_endpoint_id: mg-endpoint-300c3b17-8504-4683-9d8d-5a806ea762db

      # Vertex AI — Model Garden API (no dedicated deployment)
      triage:
        backend: vertex_ai
        vertex_model: medgemma-27b-text-it
        vertex_location: us-central1

      # HF Endpoint (legacy)
      triage:
        backend: hf_endpoint
        endpoint_url: null                 # loaded from TRIAGE_ENDPOINT_URL
    """
    backend = config.get("backend", "hf_endpoint")

    if backend == "vertex_ai":
        return VertexAIReasoner(
            model=config.get("vertex_model", "medgemma-1.5-4b-it"),
            endpoint_id=config.get("vertex_endpoint_id"),
            project=config.get("vertex_project"),
            location=config.get("vertex_location", "us-central1"),
            max_output_tokens=config.get("max_new_tokens", 1024),
            temperature=config.get("temperature", 0.2),
        )

    # default: HuggingFace dedicated endpoint (legacy)
    return MedGemmaReasoner(
        endpoint_url=config.get("endpoint_url"),
        max_new_tokens=config.get("max_new_tokens", 1024),
        temperature=config.get("temperature", 0.2),
        timeout=config.get("timeout", 120),
    )
