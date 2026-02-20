"""
MedGuard Triage Copilot – Red Flag Detection Engine
=====================================================
Rule-based red flag detection that runs independently of (and in addition to)
the model-based triage. Ensures critical patterns are NEVER missed.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "configs" / "safety_rules.yaml"


class RedFlagEngine:
    """Deterministic red flag scanner using keyword rules from safety_rules.yaml."""

    def __init__(self, rules_path: Optional[str] = None):
        path = Path(rules_path) if rules_path else _DEFAULT_RULES_PATH

        if not path.exists():
            logger.warning("Safety rules not found at %s – using empty ruleset", path)
            self.rules: dict = {}
        else:
            with open(path) as f:
                config = yaml.safe_load(f)
            self.rules = config.get("red_flags", {})

    def scan(self, structured_data: dict) -> List[dict]:
        """
        Scan structured clinical data for red flags.

        Returns a list of dicts:
          {"flag": str, "category": str, "source": str, "recommended_action": str}
        """
        flags: List[dict] = []
        text_corpus = self._build_searchable_text(structured_data)

        for category, rule in self.rules.items():
            keywords = rule.get("keywords", [])
            escalation = rule.get("escalation", "Review required")

            for keyword in keywords:
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                if pattern.search(text_corpus):
                    flags.append({
                        "flag": keyword,
                        "category": category,
                        "source": "rule_engine",
                        "recommended_action": escalation,
                    })

        # Deduplicate by flag text
        seen = set()
        unique = []
        for f in flags:
            if f["flag"] not in seen:
                seen.add(f["flag"])
                unique.append(f)

        if unique:
            logger.warning("Red flags detected: %s", [f["flag"] for f in unique])

        return unique

    def get_min_urgency(self, flags: List[dict]) -> int:
        """Return the minimum urgency level required by detected flags."""
        if not flags:
            return 1

        min_urgency = 1
        for flag in flags:
            category = flag.get("category", "")
            rule = self.rules.get(category, {})
            min_urgency = max(min_urgency, rule.get("min_urgency", 1))

        return min_urgency

    @staticmethod
    def _build_searchable_text(data: dict) -> str:
        """Flatten structured data into searchable text.

        Also includes raw_input as a safety net — if the structurer missed
        extracting a symptom, the keyword scan can still catch it from the
        original patient text.
        """
        parts: list[str] = []

        # Primary: structured fields (most reliable)
        parts.append(str(data.get("chief_complaint", "") or ""))

        for symptom in data.get("symptoms", []) or []:
            if not isinstance(symptom, dict):
                parts.append(str(symptom))
                continue
            if symptom.get("negation") != "negated":
                parts.append(str(symptom.get("name", "") or ""))
                parts.append(str(symptom.get("normalized_name", "") or ""))
                parts.append(str(symptom.get("severity", "") or ""))

        for rf in data.get("risk_factors", []) or []:
            if rf is not None:
                parts.append(str(rf))

        for entity in data.get("medical_history", []) or []:
            if not isinstance(entity, dict):
                parts.append(str(entity) if entity else "")
                continue
            if entity.get("negation") != "negated":
                parts.append(str(entity.get("name", "") or ""))

        # Safety net: also scan the original raw text in case structurer
        # missed or mis-classified a critical symptom
        raw = data.get("raw_input", "")
        if isinstance(raw, str):
            parts.append(raw)

        return " ".join(str(p) for p in parts if p is not None).lower()
