"""
MedGuard Triage Copilot – MedASR Wrapper
=========================================
Stage 1: Speech-to-text using Google MedASR (Conformer CTC, 105M params).
Specialised for medical dictation — preserves clinical terminology,
medication names, and radiology terms out of the box.

Model: https://huggingface.co/google/medasr
Architecture: Conformer CTC  |  Input: mono 16 kHz  |  Output: text
Requires: transformers ≥ 5.0.0 (or nightly), librosa, torch
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Default local cache location (set via config or env)
_DEFAULT_LOCAL_DIR = Path(__file__).resolve().parents[2] / "models" / "asr" / "medasr_local"


class MedASRWrapper:
    """Wraps Google MedASR for medical speech-to-text transcription.

    Two loading modes:
      1. Pipeline API  (simpler, handles chunking automatically)
      2. Direct model   (AutoModelForCTC + AutoProcessor — more control)

    Both load the model locally the first time and cache it.
    Pass ``local_dir`` to point at a pre-downloaded copy.
    """

    def __init__(
        self,
        model_id: str = "google/medasr",
        device: str = "auto",
        torch_dtype: str = "float32",
        chunk_length_s: int = 20,
        stride_length_s: int = 2,
        sample_rate: int = 16_000,
        use_pipeline: bool = True,
        local_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.sample_rate = sample_rate
        self.use_pipeline = use_pipeline
        self.local_dir = local_dir         # path to pre-downloaded weights

        # Lazy-loaded objects
        self._pipe = None
        self._model = None
        self._processor = None

    # ── Resolve device ──────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    # ── Resolve model source (local dir or HF hub id) ──────────────────

    def _model_source(self) -> str:
        """Return local path if available, otherwise the HF model id."""
        if self.local_dir and Path(self.local_dir).exists():
            return self.local_dir
        return self.model_id

    # ── Pipeline API (recommended) ──────────────────────────────────────

    def _load_pipeline(self):
        """Load MedASR via the HF ``pipeline`` API (lazy)."""
        if self._pipe is not None:
            return

        try:
            from transformers import pipeline as hf_pipeline

            source = self._model_source()
            device = self._resolve_device()

            logger.info("Loading MedASR pipeline from %s on %s …", source, device)
            self._pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=source,
                device=device,
            )
            logger.info("MedASR pipeline ready.")
        except ImportError as exc:
            raise ImportError(
                "MedASR requires `transformers >= 5.0.0`, `torch`, and `librosa`. "
                "Install transformers nightly if needed:\n"
                "  pip install git+https://github.com/huggingface/transformers.git\n"
                "  pip install librosa torch"
            ) from exc

    def _transcribe_pipeline(self, audio_path: str) -> dict:
        """Transcribe using the pipeline API."""
        self._load_pipeline()
        result = self._pipe(
            audio_path,
            chunk_length_s=self.chunk_length_s,
            stride_length_s=self.stride_length_s,
        )
        return result

    # ── Direct model (more control) ─────────────────────────────────────

    def _load_model(self):
        """Load AutoModelForCTC + AutoProcessor (lazy)."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor

            source = self._model_source()
            device = self._resolve_device()

            logger.info("Loading MedASR model from %s on %s …", source, device)
            self._processor = AutoProcessor.from_pretrained(source)
            self._model = AutoModelForCTC.from_pretrained(source).to(device)
            self._model.eval()
            logger.info("MedASR model ready (%s).", device)
        except ImportError as exc:
            raise ImportError(
                "MedASR requires `transformers >= 5.0.0`, `torch`, and `librosa`. "
                "Install transformers nightly if needed:\n"
                "  pip install git+https://github.com/huggingface/transformers.git\n"
                "  pip install librosa torch"
            ) from exc

    def _transcribe_direct(self, audio_path: str) -> dict:
        """Transcribe using the direct model API."""
        import librosa
        import torch

        self._load_model()
        device = self._resolve_device()

        # Load audio at 16 kHz mono (MedASR requirement)
        speech, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration_s = len(speech) / sr

        inputs = self._processor(
            speech,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = self._model.generate(**inputs)

        decoded = self._processor.batch_decode(outputs)[0]

        return {"text": decoded, "duration_s": duration_s}

    # ── Public API ──────────────────────────────────────────────────────

    def transcribe(self, audio_input: Union[str, Path]) -> dict:
        """
        Transcribe an audio file to text.

        Parameters
        ----------
        audio_input : str or Path
            Path to the audio file (.wav, .mp3, .flac, etc.).
            Must be mono-channel; will be resampled to 16 kHz internally.

        Returns
        -------
        dict with keys: transcript, segments, language, model_id, duration_s
        """
        audio_path = str(audio_input)

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Transcribing: %s (mode=%s)", audio_path,
                     "pipeline" if self.use_pipeline else "direct")

        if self.use_pipeline:
            raw = self._transcribe_pipeline(audio_path)
        else:
            raw = self._transcribe_direct(audio_path)

        # Normalize output
        segments = []
        if "chunks" in raw:
            for chunk in raw["chunks"]:
                ts = chunk.get("timestamp", [None, None])
                segments.append({
                    "text": chunk.get("text", ""),
                    "start_time": ts[0] if ts else None,
                    "end_time": ts[1] if ts else None,
                })

        return {
            "transcript": raw.get("text", ""),
            "segments": segments,
            "language": "en",
            "model_id": self.model_id,
            "duration_s": raw.get("duration_s"),
        }

    def transcribe_from_bytes(self, audio_bytes: bytes, filename: str = "input.wav") -> dict:
        """Transcribe from raw audio bytes (writes to a temp file first)."""
        import tempfile

        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.transcribe(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ── Convenience: download model for offline use ─────────────────────

    @staticmethod
    def download(
        local_dir: Optional[str] = None,
        model_id: str = "google/medasr",
    ) -> str:
        """
        Download MedASR weights to a local directory for offline use.

        Parameters
        ----------
        local_dir : str, optional
            Where to save. Defaults to ``models/asr/medasr_local/``.
        model_id : str
            HuggingFace model ID.

        Returns
        -------
        str – path to the downloaded model directory.

        Usage
        -----
        >>> MedASRWrapper.download()              # downloads to default loc
        >>> MedASRWrapper.download("models/asr/medasr_local")
        >>> asr = MedASRWrapper(local_dir="models/asr/medasr_local")
        """
        from huggingface_hub import snapshot_download

        dest = local_dir or str(_DEFAULT_LOCAL_DIR)
        logger.info("Downloading %s → %s …", model_id, dest)

        token = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
        path = snapshot_download(
            repo_id=model_id,
            local_dir=dest,
            token=token,
        )
        logger.info("Download complete: %s", path)
        return path
