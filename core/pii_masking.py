"""
MedGuard Triage Copilot – PII Masking
======================================
Detects and masks Personally Identifiable Information (PII) in
transcribed text before it enters the clinical pipeline.

Masked PII categories:
  - Person names          → [NAME]
  - Phone numbers         → [PHONE]
  - Email addresses       → [EMAIL]
  - SSN / national IDs    → [SSN]
  - Street addresses      → [ADDRESS]
  - Dates of birth        → [DOB]
  - Credit card numbers   → [CARD]
  - IP addresses          → [IP]

Uses pure regex — no external NLP models or network calls required.
All masking happens locally before any data leaves the pipeline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PIIMaskResult:
    """Result of PII masking."""
    masked_text: str
    original_text: str
    detections: list[dict] = field(default_factory=list)
    pii_found: bool = False

    @property
    def summary(self) -> str:
        if not self.detections:
            return "No PII detected."
        types = sorted({d["type"] for d in self.detections})
        return f"Masked {len(self.detections)} PII item(s): {', '.join(types)}"


# ── Regex patterns for PII detection ────────────────────────────────────────

# Phone numbers: various formats (US, IN, international)
_PHONE_PATTERNS = [
    r'\+?\d{1,3}[-.\.\s]?\(?\d{2,5}\)?[-.\.\s]?\d{3,5}[-.\.\s]?\d{3,5}',  # +91-98765-43210
    r'\b\d{10,12}\b',                                                   # 9876543210
    r'\(\d{3}\)\s?\d{3}[-.\.\s]?\d{4}',                                  # (800) 555-1234
]

# Email addresses
_EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

# SSN (US) / Aadhaar (IN) / national IDs
_SSN_PATTERNS = [
    r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',       # US SSN: 123-45-6789
    r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',       # Aadhaar: 1234 5678 9012
]

# Credit card numbers (basic 16-digit patterns)
_CARD_PATTERN = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'

# Date of birth patterns (spoken / written)
_DOB_PATTERNS = [
    # "date of birth is 15th March 1990" / "dob 03/15/1990"
    r'(?:date\s+of\s+birth|d\.?o\.?b\.?|born\s+on)\s*(?:is\s+)?(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
    r'(?:date\s+of\s+birth|d\.?o\.?b\.?|born\s+on)\s*(?:is\s+)?(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{2,4})',
]

# IP addresses
_IP_PATTERN = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'

# Street addresses (heuristic: number + street name + suffix)
_ADDRESS_PATTERN = r'\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Road|Rd|Lane|Ln|Court|Ct|Circle|Cir|Way|Place|Pl)\b'

# Common name intro patterns (spoken: "my name is ...", "this is ...")
_NAME_INTRO_PATTERNS = [
    r"(?:my\s+name\s+is|i\s+am|i'm|this\s+is|patient\s+name(?:\s+is)?|name\s*:\s*)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
    r"(?:call\s+me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
]


class PIIMasker:
    """Detects and masks PII in text using regex patterns.

    Parameters
    ----------
    enabled : bool
        Master switch. When False, ``mask()`` returns text unchanged.
    categories : set[str] | None
        Which PII types to mask. None = all. Options:
        ``phone``, ``email``, ``ssn``, ``card``, ``dob``, ``ip``,
        ``address``, ``name``.
    """

    ALL_CATEGORIES = {"phone", "email", "ssn", "card", "dob", "ip", "address", "name"}

    def __init__(
        self,
        enabled: bool = True,
        categories: Optional[set[str]] = None,
    ):
        self.enabled = enabled
        self.categories = categories or self.ALL_CATEGORIES

    def mask(self, text: str) -> PIIMaskResult:
        """Mask PII in *text* and return a ``PIIMaskResult``."""
        if not self.enabled or not text:
            return PIIMaskResult(masked_text=text, original_text=text)

        original = text
        detections: list[dict] = []

        # Order matters: mask longer / more specific patterns first.

        # 1. Email (before phone — emails contain numbers)
        if "email" in self.categories:
            text, dets = self._mask_pattern(text, _EMAIL_PATTERN, "EMAIL")
            detections.extend(dets)

        # 2. Credit card (before phone — 16 digits)
        if "card" in self.categories:
            text, dets = self._mask_pattern(text, _CARD_PATTERN, "CARD")
            detections.extend(dets)

        # 3. SSN / national ID
        if "ssn" in self.categories:
            for pat in _SSN_PATTERNS:
                text, dets = self._mask_pattern(text, pat, "SSN")
                detections.extend(dets)

        # 4. Date of birth (contextual — requires intro phrase)
        if "dob" in self.categories:
            for pat in _DOB_PATTERNS:
                text, dets = self._mask_dob(text, pat)
                detections.extend(dets)

        # 5. Phone numbers
        if "phone" in self.categories:
            for pat in _PHONE_PATTERNS:
                text, dets = self._mask_pattern(text, pat, "PHONE")
                detections.extend(dets)

        # 6. IP addresses
        if "ip" in self.categories:
            text, dets = self._mask_pattern(text, _IP_PATTERN, "IP")
            detections.extend(dets)

        # 7. Street addresses
        if "address" in self.categories:
            text, dets = self._mask_pattern(text, _ADDRESS_PATTERN, "ADDRESS")
            detections.extend(dets)

        # 8. Names (contextual — requires intro phrase)
        if "name" in self.categories:
            for pat in _NAME_INTRO_PATTERNS:
                text, dets = self._mask_name(text, pat)
                detections.extend(dets)

        if detections:
            logger.info("PII masking: %d item(s) redacted", len(detections))

        return PIIMaskResult(
            masked_text=text,
            original_text=original,
            detections=detections,
            pii_found=bool(detections),
        )

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _mask_pattern(text: str, pattern: str, label: str) -> tuple[str, list[dict]]:
        """Replace all matches of *pattern* with ``[LABEL]``."""
        detections = []
        for m in re.finditer(pattern, text, re.IGNORECASE):
            detections.append({
                "type": label,
                "start": m.start(),
                "end": m.end(),
                "matched": m.group(),
            })
        masked = re.sub(pattern, f"[{label}]", text, flags=re.IGNORECASE)
        return masked, detections

    @staticmethod
    def _mask_dob(text: str, pattern: str) -> tuple[str, list[dict]]:
        """Mask date of birth that follows an intro phrase."""
        detections = []
        for m in re.finditer(pattern, text, re.IGNORECASE):
            date_val = m.group(1) if m.lastindex else m.group(0)
            detections.append({
                "type": "DOB",
                "start": m.start(),
                "end": m.end(),
                "matched": date_val,
            })
        masked = re.sub(pattern, lambda m: m.group(0).replace(
            m.group(1) if m.lastindex else m.group(0), "[DOB]"
        ), text, flags=re.IGNORECASE)
        return masked, detections

    @staticmethod
    def _mask_name(text: str, pattern: str) -> tuple[str, list[dict]]:
        """Mask person names that follow an intro phrase like 'my name is'."""
        detections = []
        for m in re.finditer(pattern, text, re.IGNORECASE):
            name_val = m.group(1)
            detections.append({
                "type": "NAME",
                "start": m.start(1),
                "end": m.end(1),
                "matched": name_val,
            })
        masked = re.sub(pattern, lambda m: m.group(0).replace(
            m.group(1), "[NAME]"
        ), text, flags=re.IGNORECASE)
        return masked, detections
