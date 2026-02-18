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
        """Flatten structured data into searchable text."""
        parts = []

        parts.append(data.get("chief_complaint", ""))

        for symptom in data.get("symptoms", []):
            if symptom.get("negation") != "negated":
                parts.append(symptom.get("name", ""))
                parts.append(symptom.get("normalized_name", "") or "")
                parts.append(symptom.get("severity", "") or "")

        for rf in data.get("risk_factors", []):
            parts.append(rf)

        for entity in data.get("medical_history", []):
            if entity.get("negation") != "negated":
                parts.append(entity.get("name", ""))

        return " ".join(parts).lower()
