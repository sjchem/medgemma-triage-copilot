"""
MedGuard Triage Copilot – Router
=================================
Top-level entrypoint that auto-detects input type and dispatches
to the appropriate pipeline (voice / text / structured).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import yaml

from pipelines.voice_pipeline import VoicePipeline
from pipelines.text_pipeline import TextPipeline
from pipelines.structured_pipeline import StructuredPipeline

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "model_config.yaml"


class MedGuardRouter:
    """One-call entrypoint for MedGuard Triage Copilot."""

    def __init__(self, config_path: Optional[str] = None):
        cfg_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

        self.config: dict = {}
        if cfg_path.exists():
            with open(cfg_path) as f:
                self.config = yaml.safe_load(f) or {}

        # Build pipeline configs from YAML
        structurer_cfg = self._build_structurer_config()
        triage_cfg = self._build_triage_config()
        asr_cfg = self._build_asr_config()

        kwargs = {
            "structurer_config": structurer_cfg,
            "triage_config": triage_cfg,
            "asr_config": asr_cfg,
        }

        self.voice_pipeline = VoicePipeline(**kwargs)
        self.text_pipeline = TextPipeline(**kwargs)
        self.structured_pipeline = StructuredPipeline(**kwargs)

    def _build_structurer_config(self) -> dict:
        s = self.config.get("structurer", {})
        params = s.get("parameters", {})
        return {
            "model_id": s.get("model_id", "google/gemma-7b-it"),
            "endpoint_url": s.get("endpoint_url"),
            "use_inference_api": s.get("use_inference_api", True),
            "max_new_tokens": params.get("max_new_tokens", 1024),
            "temperature": params.get("temperature", 0.1),
            "timeout": s.get("timeout", 120),
        }

    def _build_triage_config(self) -> dict:
        t = self.config.get("triage", {})
        params = t.get("parameters", {})
        return {
            "model_id": t.get("model_id", "google/medgemma-4b-it"),
            "endpoint_url": t.get("endpoint_url"),
            "use_inference_api": t.get("use_inference_api", True),
            "max_new_tokens": params.get("max_new_tokens", 1024),
            "temperature": params.get("temperature", 0.2),
            "timeout": t.get("timeout", 120),
        }

    def _build_asr_config(self) -> dict:
        a = self.config.get("asr", {})
        return {
            "model_id": a.get("model_id", "google/medasr"),
            "device": a.get("device", "auto"),
            "torch_dtype": a.get("torch_dtype", "float32"),
            "chunk_length_s": a.get("chunk_length_s", 20),
            "stride_length_s": a.get("stride_length_s", 2),
            "sample_rate": a.get("sample_rate", 16_000),
            "use_pipeline": a.get("use_pipeline", True),
            "local_dir": a.get("local_dir"),
        }

    # ── Public API ──────────────────────────────────────────────────────

    def triage(self, input_data: Union[str, dict, Path], input_type: str = "auto") -> dict:
        """
        Run the full triage pipeline.

        Parameters
        ----------
        input_data : str, dict, or Path
            - str: free-text patient intake OR path to audio file
            - dict: pre-structured clinical data
            - Path: audio file path
        input_type : str
            "auto" (default), "text", "voice", or "structured"

        Returns
        -------
        dict – PipelineResult
        """
        if input_type == "auto":
            input_type = self._detect_type(input_data)

        logger.info("MedGuard routing to %s pipeline", input_type)

        if input_type == "voice":
            return self.voice_pipeline.run(input_data)
        elif input_type == "structured":
            return self.structured_pipeline.run(input_data)
        else:
            return self.text_pipeline.run(input_data)

    @staticmethod
    def _detect_type(input_data) -> str:
        """Auto-detect input type."""
        if isinstance(input_data, dict):
            # If it has clinical schema keys, treat as structured
            if "chief_complaint" in input_data and "symptoms" in input_data:
                return "structured"
            return "text"  # dict without schema → serialize to text

        if isinstance(input_data, Path):
            return "voice"

        if isinstance(input_data, str):
            # Check if it's a path to an audio file
            p = Path(input_data)
            if p.exists() and p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
                return "voice"
            return "text"

        return "text"
