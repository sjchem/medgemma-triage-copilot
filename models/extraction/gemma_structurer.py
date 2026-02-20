"""
MedGuard Triage Copilot – Gemma Clinical Structurer
====================================================
Stage 2: Uses Gemma 2-2B Instruct to extract structured clinical entities
from free-text patient intake. Supports two backends:

  - "hf_endpoint"  : Dedicated HuggingFace Inference Endpoint (cloud, slower)
  - "ollama"       : Local Ollama server via LangChain (fast, fully offline)

Handles:
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

STRUCTURER_SYSTEM_PROMPT = """You are a clinical data structurer. Extract clinical information from patient intake text into JSON.

RULES:
1. Extract ONLY what is explicitly stated. Do NOT infer diagnoses.
2. Mark negated symptoms (e.g., "denies nausea" → negation: "negated").
3. List missing/ambiguous info in "missing_information".
4. Assign extraction_confidence (0.0–1.0) based on input clarity.

Return ONLY valid JSON:
{"chief_complaint": "str", "symptoms": [{"name": "str", "severity": "mild|moderate|severe|null", "duration": "str|null", "negation": "affirmed|negated|uncertain", "confidence": 0.0}], "vitals": {"heart_rate": null, "blood_pressure_systolic": null, "blood_pressure_diastolic": null, "respiratory_rate": null, "temperature_c": null, "spo2": null}, "demographics": {"age": null, "sex": null}, "medical_history": [{"name": "str", "negation": "affirmed"}], "medications": [{"name": "str"}], "allergies": [{"name": "str"}], "risk_factors": ["str"], "missing_information": ["str"], "extraction_confidence": 0.0}"""

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

    # ── Sanitise LLM output ──────────────────────────────────────────────

    @staticmethod
    def _sanitize_output(data: dict) -> dict:
        """Remove None values from list fields to prevent downstream join() errors."""
        _LIST_FIELDS = (
            "symptoms", "medical_history", "medications",
            "allergies", "risk_factors", "missing_information",
            "uncertainty_notes", "recommended_actions",
            "follow_up_questions", "risk_categories",
        )
        for key in _LIST_FIELDS:
            val = data.get(key)
            if isinstance(val, list):
                data[key] = [item for item in val if item is not None]
            elif val is None:
                data[key] = []
        return data

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
        structured = self._sanitize_output(structured)

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


# ═══════════════════════════════════════════════════════════════════════════════
# Ollama Backend (local, LangChain-powered)
# ═══════════════════════════════════════════════════════════════════════════════

class OllamaStructurer:
    """
    Clinical structurer backed by a local Ollama server via LangChain.

    Requires:
      - Ollama running locally:  https://ollama.com/download
      - Model pulled:            ollama pull gemma2:2b
      - Package:                 pip install langchain-ollama

    By default connects to http://localhost:11434.
    Override via OLLAMA_BASE_URL env var or base_url= kwarg.

    Speed advantage over HF endpoint:
      - No network latency (fully local)
      - No cold-start on shared-inference infrastructure
      - Typical latency: 2–8 s on CPU, <2 s on GPU
    """

    def __init__(
        self,
        model: str = "gemma2:2b",
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        **_,  # absorb extra kwargs from config dicts
    ):
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama is required for the Ollama backend.\n"
                "Install it with:  pip install langchain-ollama"
            )

        self.model = model
        self._llm = ChatOllama(
            model=model,
            base_url=base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
            num_predict=max_tokens,
            num_ctx=2048,       # Smaller context window = faster on CPU
        )
        logger.info("OllamaStructurer initialised — model=%s", model)

    # ── Model call ──────────────────────────────────────────────────────

    def _call_model(self, prompt: str) -> str:
        """Send prompt to local Ollama via LangChain."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=STRUCTURER_SYSTEM_PROMPT),
            HumanMessage(content=STRUCTURER_USER_PROMPT.format(patient_input=prompt)),
        ]
        response = self._llm.invoke(messages)
        return response.content

    # ── reuse JSON extractor and public API from GemmaStructurer ────────

    @staticmethod
    def _extract_json(text: str) -> dict:
        return GemmaStructurer._extract_json(text)

    def structure(self, patient_input: str | dict) -> dict:
        if isinstance(patient_input, dict):
            input_text = json.dumps(patient_input, indent=2)
        else:
            input_text = patient_input

        raw_output = self._call_model(input_text)
        logger.debug("OllamaStructurer raw output: %s", raw_output[:500])

        structured = self._extract_json(raw_output)
        structured = GemmaStructurer._sanitize_output(structured)
        structured["raw_input"] = input_text
        return structured

    def structure_safe(self, patient_input: str | dict) -> dict:
        try:
            return self.structure(patient_input)
        except Exception as e:
            logger.error("OllamaStructurer failed: %s", e)
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
                "missing_information": ["OllamaStructurer extraction failed – raw input forwarded"],
                "uncertainty_notes": [
                    {"field": "all", "reason": str(e), "impact": "high"}
                ],
                "extraction_confidence": 0.0,
                "raw_input": input_text,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# Factory — selects backend from config
# ═══════════════════════════════════════════════════════════════════════════════

def create_structurer(config: dict) -> "GemmaStructurer | OllamaStructurer":
    """
    Factory that returns the correct structurer based on config["backend"].

    Supported values for config["backend"]:
      "hf_endpoint"  →  GemmaStructurer  (HuggingFace Inference Endpoint, cloud)
      "ollama"       →  OllamaStructurer (local Ollama via LangChain, fast)

    Example model_config.yaml entries:

      # cloud (default)
      structurer:
        backend: hf_endpoint
        endpoint_url: null        # loaded from STRUCTURER_ENDPOINT_URL

      # local Ollama
      structurer:
        backend: ollama
        ollama_model: gemma2:2b   # or llama3.2:3b, mistral, phi3, etc.
        ollama_base_url: http://localhost:11434
        parameters:
          temperature: 0.1
          max_new_tokens: 1024
    """
    backend = config.get("backend", "hf_endpoint")

    if backend == "ollama":
        return OllamaStructurer(
            model=config.get("ollama_model", "gemma2:2b"),
            base_url=config.get("ollama_base_url"),
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_new_tokens", 512),
        )

    # default: HuggingFace dedicated endpoint
    return GemmaStructurer(
        endpoint_url=config.get("endpoint_url"),
        max_new_tokens=config.get("max_new_tokens", 1024),
        temperature=config.get("temperature", 0.1),
        timeout=config.get("timeout", 120),
    )
