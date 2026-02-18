"""
Backward-compatible re-exports of the triage pipeline.
See core/router.py for the unified entrypoint.
"""
from core.router import MedGuardRouter  # noqa: F401
from pipelines.text_pipeline import TextPipeline  # noqa: F401
from pipelines.structured_pipeline import StructuredPipeline  # noqa: F401
