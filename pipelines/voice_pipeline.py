"""
MedGuard Triage Copilot – Voice Pipeline
==========================================
Full pipeline: Audio → ASR → Structure → Triage
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from models.asr.medasr_wrapper import MedASRWrapper
from pipelines.text_pipeline import TextPipeline

logger = logging.getLogger(__name__)


class VoicePipeline:
    """End-to-end voice → triage pipeline."""

    def __init__(
        self,
        asr: Optional[MedASRWrapper] = None,
        text_pipeline: Optional[TextPipeline] = None,
        **kwargs,
    ):
        self.asr = asr or MedASRWrapper(**kwargs.get("asr_config", {}))
        self.text_pipeline = text_pipeline or TextPipeline(**kwargs)

    def run(self, audio_path: Union[str, Path]) -> dict:
        """
        Run full pipeline from audio file.

        Returns PipelineResult-compatible dict.
        """
        logger.info("Voice pipeline: transcribing %s", audio_path)

        # Stage 1: ASR
        asr_output = self.asr.transcribe(audio_path)
        transcript = asr_output.get("transcript", "")

        if not transcript.strip():
            return {
                "input_type": "voice",
                "asr_output": asr_output,
                "structured_data": None,
                "triage_output": None,
                "pipeline_errors": ["ASR produced empty transcript"],
            }

        # Stages 2 + 3 via text pipeline
        result = self.text_pipeline.run(transcript)
        result["input_type"] = "voice"
        result["asr_output"] = asr_output

        return result
