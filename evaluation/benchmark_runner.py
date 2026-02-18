"""
MedGuard Triage Copilot – Benchmark Runner
============================================
Runs evaluation across a set of test cases and produces aggregate metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from evaluation.extraction_metrics import extraction_report
from evaluation.triage_metrics import triage_report

logger = logging.getLogger(__name__)


def load_test_cases(path: str) -> List[dict]:
    """Load test cases from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if "cases" in data:
        return data["cases"]
    raise ValueError(f"Expected a list or {{cases: [...]}} in {path}")


def run_benchmark(
    pipeline,
    test_cases: List[dict],
    verbose: bool = True,
) -> dict:
    """
    Run the pipeline on a list of test cases and collect metrics.

    Parameters
    ----------
    pipeline : object
        Any pipeline with a .run() method that accepts patient input.
    test_cases : list[dict]
        Each case should have:
          - "input": str or dict (patient intake)
          - "expected_urgency": int (optional)
          - "expected_red_flags": list[str] (optional)
          - "expected_symptoms": list[str] (optional)
    verbose : bool
        Print per-case results.

    Returns
    -------
    dict with aggregate metrics.
    """
    results = []
    errors = []

    for i, case in enumerate(test_cases):
        case_id = case.get("id", f"case_{i}")
        patient_input = case.get("input", case.get("patient", ""))

        try:
            result = pipeline.run(patient_input)

            # Extraction metrics
            ext_report = {}
            if result.get("structured_data"):
                ext_report = extraction_report(
                    result["structured_data"],
                    expected_symptoms=case.get("expected_symptoms"),
                )

            # Triage metrics
            tri_report = {}
            if result.get("triage_output"):
                tri_report = triage_report(
                    result["triage_output"],
                    expected_urgency=case.get("expected_urgency"),
                    expected_red_flags=case.get("expected_red_flags"),
                )

            case_result = {
                "case_id": case_id,
                "status": "success",
                "extraction_metrics": ext_report,
                "triage_metrics": tri_report,
                "pipeline_errors": result.get("pipeline_errors", []),
                "safety_overrides": result.get("safety_overrides_applied", []),
            }

            if verbose:
                urgency = tri_report.get("predicted_urgency", "?")
                confidence = tri_report.get("confidence_score", "?")
                logger.info(
                    "Case %s: urgency=%s confidence=%s",
                    case_id, urgency, confidence,
                )

        except Exception as e:
            case_result = {
                "case_id": case_id,
                "status": "error",
                "error": str(e),
            }
            errors.append(case_id)
            logger.error("Case %s failed: %s", case_id, e)

        results.append(case_result)

    # Aggregate
    successful = [r for r in results if r["status"] == "success"]
    aggregate = _compute_aggregate(successful)
    aggregate["total_cases"] = len(test_cases)
    aggregate["successful_cases"] = len(successful)
    aggregate["failed_cases"] = len(errors)
    aggregate["per_case_results"] = results

    return aggregate


def _compute_aggregate(successful_results: List[dict]) -> dict:
    """Compute aggregate metrics from successful results."""
    if not successful_results:
        return {}

    # Collect triage metrics
    urgency_exact = []
    urgency_within_one = []
    safety_scores = []
    red_flag_recalls = []
    completeness_scores = []
    confidence_scores = []

    for r in successful_results:
        tri = r.get("triage_metrics", {})
        ext = r.get("extraction_metrics", {})

        if "urgency_exact_match" in tri:
            urgency_exact.append(tri["urgency_exact_match"])
        if "urgency_within_one" in tri:
            urgency_within_one.append(tri["urgency_within_one"])
        if "safety_score" in tri:
            safety_scores.append(tri["safety_score"])
        if "red_flag_recall" in tri:
            red_flag_recalls.append(tri["red_flag_recall"])
        if "completeness" in ext:
            completeness_scores.append(ext["completeness"])
        if tri.get("confidence_score") is not None:
            confidence_scores.append(tri["confidence_score"])

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "urgency_exact_accuracy": _mean([float(x) for x in urgency_exact]),
        "urgency_within_one_accuracy": _mean([float(x) for x in urgency_within_one]),
        "mean_safety_score": _mean(safety_scores),
        "mean_red_flag_recall": _mean(red_flag_recalls),
        "mean_extraction_completeness": _mean(completeness_scores),
        "mean_confidence": _mean(confidence_scores),
    }
