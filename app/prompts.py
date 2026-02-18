"""
Backward-compatible re-exports of prompt constants.
See models/extraction/gemma_structurer.py and models/triage/medgemma_reasoner.py.
"""
from models.extraction.gemma_structurer import (  # noqa: F401
    STRUCTURER_SYSTEM_PROMPT,
    STRUCTURER_USER_PROMPT,
)
from models.triage.medgemma_reasoner import (  # noqa: F401
    TRIAGE_SYSTEM_PROMPT,
    TRIAGE_USER_PROMPT,
)
