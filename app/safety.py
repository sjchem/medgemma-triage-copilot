"""
Backward-compatible re-exports of safety components.
See models/triage/red_flag_engine.py and models/triage/risk_scorer.py.
"""
from models.triage.red_flag_engine import RedFlagEngine  # noqa: F401
from models.triage.risk_scorer import RiskScorer  # noqa: F401
