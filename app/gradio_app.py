"""
MedGuard Triage Copilot – Gradio Frontend
==========================================
Agentic Multimodal Safety Layer — Never miss a clinical red flag.

Layout
------
  ┌──────────────────────────────────────┐
  │  HEADER  (Branding + Safety tagline) │
  └──────────────────────────────────────┘
  ┌──────────────────┬───────────────────┐
  │  INPUT PANEL     │  RESULTS PANEL    │
  │  (Text/Voice     │  (Urgency badge,  │
  │   + Structured)  │   Red flags, …)   │
  └──────────────────┴───────────────────┘
  ┌──────────────────────────────────────┐
  │  FOLLOW-UP INTERACTION PANEL         │
  └──────────────────────────────────────┘
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# ── ensure project root is on sys.path ──────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(override=True)

import gradio as gr

from core.router import MedGuardRouter

logger = logging.getLogger(__name__)

# ── Lazy-loaded router (shared across all calls) ─────────────────────────────
_router: MedGuardRouter | None = None


def get_router() -> MedGuardRouter:
    global _router
    if _router is None:
        logger.info("Initialising MedGuardRouter …")
        _router = MedGuardRouter()
    return _router


# ── Colour / label maps ──────────────────────────────────────────────────────

# Internal engine uses 1=lowest priority → 5=highest/critical
# (matches escalation_policy.yaml and safety_rules.yaml min_urgency values)
URGENCY_META = {
    1: {"label": "Non-Urgent",            "color": "#2980B9", "text": "#FFFFFF", "icon": "🔵"},
    2: {"label": "Low Urgency",           "color": "#27AE60", "text": "#FFFFFF", "icon": "🟢"},
    3: {"label": "Moderate Urgency",      "color": "#F1C40F", "text": "#1A1A2E", "icon": "🟡"},
    4: {"label": "Urgent",               "color": "#E67E22", "text": "#FFFFFF", "icon": "🟠"},
    5: {"label": "Critical / Immediate",  "color": "#C0392B", "text": "#FFFFFF", "icon": "🔴"},
}

ESCALATION_META = {
    "immediate": {"color": "#C0392B", "icon": "🚨", "label": "IMMEDIATE"},
    "urgent":    {"color": "#E67E22", "icon": "⚠️",  "label": "URGENT"},
    "review":    {"color": "#F1C40F", "icon": "🔍", "label": "REVIEW REQUIRED"},
    "routine":   {"color": "#7F8C8D", "icon": "✅", "label": "ROUTINE"},
}

CONFIDENCE_COLOR = {
    "high":       "#27AE60",
    "moderate":   "#F1C40F",
    "low":        "#E67E22",
    "uncertain":  "#C0392B",
}

URGENCY_LEGEND = """
| Level | Label              | Meaning                                      |
|-------|--------------------|----------------------------------------------|
| 1     | Non-Urgent         | No resources needed                          |
| 2     | Low Urgency        | Needs one resource — can safely wait         |
| 3     | Moderate Urgency   | Stable but needs timely evaluation           |
| 4     | Urgent             | Serious — risk of deterioration              |
| 5     | Critical/Immediate | Life-threatening — resuscitation required    |
"""


# ── HTML helpers ─────────────────────────────────────────────────────────────

def _badge_html(urgency_level: int) -> str:
    m = URGENCY_META.get(urgency_level, URGENCY_META[3])
    return f"""
<div style="
  background:{m['color']};color:{m['text']};
  border-radius:12px;padding:18px 24px;text-align:center;
  font-family:'Segoe UI',sans-serif;margin-bottom:6px;
  box-shadow:0 4px 14px rgba(0,0,0,0.25);">
  <div style="font-size:2.6rem;line-height:1.1;">{m['icon']}</div>
  <div style="font-size:1.45rem;font-weight:700;letter-spacing:0.5px;">
    Priority Level {urgency_level} – {m['label']}
  </div>
  <div style="font-size:0.82rem;margin-top:4px;opacity:0.88;">
    Clinical Urgency (1 = routine → 5 = critical)
  </div>
</div>
"""


def _escalation_html(band: str) -> str:
    key = band.lower() if band.lower() in ESCALATION_META else "routine"
    m   = ESCALATION_META[key]
    return f"""
<div style="
  background:{m['color']}22;border:2px solid {m['color']};
  border-radius:10px;padding:12px 18px;text-align:center;
  font-family:'Segoe UI',sans-serif;margin-bottom:6px;">
  <span style="font-size:1.25rem;">{m['icon']}</span>
  <span style="font-weight:700;font-size:1.05rem;color:{m['color']};
    margin-left:8px;letter-spacing:0.5px;">
    Escalation: {m['label']}
  </span>
</div>
"""


def _confidence_html(score: float, band: str) -> str:
    pct  = int(score * 100)
    col  = CONFIDENCE_COLOR.get(band.lower(), "#7F8C8D")
    bar_w = max(4, pct)
    return f"""
<div style="font-family:'Segoe UI',sans-serif;padding:10px 0 4px;">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
    <span style="font-size:0.92rem;color:#555;">Confidence</span>
    <span style="font-weight:700;font-size:1.1rem;color:{col};">{pct}%</span>
    <span style="background:{col}22;color:{col};border-radius:6px;
      padding:2px 10px;font-size:0.82rem;font-weight:600;">{band.capitalize()}</span>
  </div>
  <div style="background:#E0E0E0;border-radius:6px;height:8px;overflow:hidden;">
    <div style="background:{col};width:{bar_w}%;height:100%;
      border-radius:6px;transition:width 0.4s ease;"></div>
  </div>
</div>
"""


def _red_flags_html(flags: list) -> str:
    if not flags:
        return """
<div style="border:1.5px dashed #27AE6044;border-radius:10px;padding:14px 18px;
  font-family:'Segoe UI',sans-serif;font-size:0.93rem;background:#F0FFF0;">
  <span style="font-size:1.15rem;">✅</span>
  <span style="color:#27AE60;font-weight:600;">No critical red flags detected.</span>
  <div style="color:#7F8C8D;font-size:0.82rem;margin-top:4px;">
    Standard clinical assessment still recommended.
  </div>
</div>"""
    _CAT_COLORS = {
        "cardiac": "#E74C3C", "stroke": "#9B59B6", "respiratory": "#2980B9",
        "hemorrhage": "#C0392B", "neurological": "#8E44AD", "sepsis": "#E67E22",
        "psychiatric": "#16A085", "pediatric": "#F39C12", "other": "#7F8C8D",
    }
    items = ""
    for f in flags:
        if isinstance(f, dict):
            flag_text = f.get("flag", str(f))
            category  = f.get("category", "").lower()
            action    = f.get("recommended_action", "")
            cat_col   = _CAT_COLORS.get(category, "#7F8C8D")
            cat_badge = (
                f'<span style="background:{cat_col}18;color:{cat_col};'
                f'border-radius:4px;padding:1px 7px;font-size:0.72rem;'
                f'font-weight:600;margin-left:6px;text-transform:uppercase;">'
                f'{category}</span>'
            ) if category else ""
            action_line = (
                f'<div style="color:#555;font-size:0.82rem;margin:2px 0 0 18px;">'
                f'↳ <em>{action}</em></div>'
            ) if action else ""
        else:
            flag_text   = str(f)
            cat_badge   = ""
            action_line = ""
        items += f"""
<li style="margin-bottom:10px;line-height:1.45;">
  <span style="color:#C0392B;font-weight:600;">⚑</span>
  &nbsp;<strong>{flag_text}</strong>{cat_badge}
  {action_line}
</li>"""
    count = len(flags)
    return f"""
<div style="border:2px solid #E74C3C33;border-left:4px solid #C0392B;
  background:#FFF5F5;border-radius:10px;padding:14px 18px;
  font-family:'Segoe UI',sans-serif;">
  <div style="font-weight:700;color:#C0392B;margin-bottom:10px;font-size:1rem;
    display:flex;align-items:center;gap:8px;">
    🚩 Red Flags Identified
    <span style="background:#C0392B;color:#fff;border-radius:50%;
      width:22px;height:22px;display:inline-flex;align-items:center;
      justify-content:center;font-size:0.78rem;font-weight:700;">{count}</span>
  </div>
  <ul style="margin:0;padding-left:20px;color:#2C3E50;font-size:0.93rem;">
    {items}
  </ul>
</div>"""


def _actions_html(actions: list) -> str:
    if not actions:
        return """
<div style="border:1.5px dashed #2980B944;border-radius:10px;padding:14px 18px;
  font-family:'Segoe UI',sans-serif;font-size:0.93rem;background:#F0F8FF;">
  <span style="font-size:1.15rem;">🧭</span>
  <span style="color:#1A6B9A;font-weight:600;">Awaiting triage assessment.</span>
  <div style="color:#7F8C8D;font-size:0.82rem;margin-top:4px;">
    Recommended actions will appear once triage is complete.
  </div>
</div>"""
    _PRIO_ICONS = ["🔴", "🟠", "🟡", "🔵", "⚪"]
    items = ""
    for i, a in enumerate(actions):
        icon = _PRIO_ICONS[i] if i < len(_PRIO_ICONS) else "▸"
        items += f'<li style="margin-bottom:8px;color:#2C3E50;line-height:1.4;">{icon} {a}</li>'
    return f"""
<div style="border:1.5px solid #2980B933;background:#F0F8FF;border-radius:10px;
  padding:14px 18px;font-family:'Segoe UI',sans-serif;margin-top:4px;">
  <div style="font-weight:700;color:#1A6B9A;margin-bottom:10px;font-size:1rem;">
    🧭 Recommended Next Steps ({len(actions)})
  </div>
  <ol style="margin:0;padding-left:20px;font-size:0.93rem;list-style:none;">
    {items}
  </ol>
</div>"""


def _safety_html(note: str, overrides: list, disclaimer: str) -> str:
    override_block = ""
    if overrides:
        override_items = "".join(
            f'<li style="color:#5D4037;font-size:0.85rem;">{o}</li>'
            for o in overrides
        )
        override_block = f"""
<div style="margin-top:10px;background:#FFF3E0;border-radius:8px;padding:10px 14px;">
  <span style="font-weight:600;font-size:0.88rem;color:#E65100;">
    ⚙️ Safety Overrides Applied:
  </span>
  <ul style="margin:6px 0 0;padding-left:20px;">{override_items}</ul>
</div>"""
    return f"""
<div style="border:1.5px solid #BDC3C7;border-radius:10px;padding:14px 18px;
  font-family:'Segoe UI',sans-serif;margin-top:4px;background:#FAFAFA;">
  <div style="font-weight:700;color:#555;margin-bottom:8px;font-size:0.95rem;">
    🛡️ Safety Note
  </div>
  <p style="margin:0;color:#444;font-size:0.9rem;line-height:1.5;">{note}</p>
  {override_block}
  <div style="margin-top:10px;color:#888;font-size:0.78rem;font-style:italic;">
    {disclaimer}
  </div>
</div>"""


def _export_report(result: dict) -> str:
    """Return a plain-text triage summary."""
    t = result.get("triage_output", {})
    lines = [
        "═══════════════════════════════════════════════════",
        "        MedGuard Triage Copilot — Report",
        f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "═══════════════════════════════════════════════════",
        f"  Case ID        : {result.get('id', 'N/A')}",
        f"  Input type     : {result.get('input_type', 'N/A')}",
        f"  Urgency Level  : {t.get('urgency_level', '?')} — {t.get('urgency_label', '')}",
        f"  Confidence     : {int(t.get('confidence_score', 0)*100)}%"
        f"  ({t.get('confidence_band', '')})",
        "",
        "RED FLAGS:",
    ]
    for f in t.get("red_flags", []):
        flag = f.get("flag", f) if isinstance(f, dict) else f
        lines.append(f"  • {flag}")
    lines += ["", "RECOMMENDED ACTIONS:"]
    for a in t.get("recommended_actions", []):
        lines.append(f"  • {a}")
    lines += ["", "FOLLOW-UP QUESTIONS:"]
    for q in t.get("follow_up_questions", []):
        lines.append(f"  • {q}")
    lines += [
        "",
        "SAFETY NOTE:",
        f"  {t.get('safety_note', '')}",
        "",
        "DISCLAIMER:",
        f"  {result.get('disclaimer', '')}",
        "═══════════════════════════════════════════════════",
    ]
    return "\n".join(lines)


# ── Pipeline helpers ─────────────────────────────────────────────────────────

def _run_text_pipeline(patient_text: str) -> dict:
    return get_router().triage(patient_text, input_type="text")


def _run_structured_pipeline(
    age, sex, chief_complaint, symptoms, duration, risk_factors
) -> dict:
    data = {
        "chief_complaint": chief_complaint or "Not specified",
        "demographics": {"age": age, "sex": sex},
        "symptoms": [{"name": s.strip()} for s in symptoms.split(",") if s.strip()],
        "risk_factors": [r.strip() for r in risk_factors.split(",") if r.strip()],
        "duration": duration,
    }
    return get_router().triage(data, input_type="structured")


def _parse_result(result: dict):
    t = result.get("triage_output", {})
    return (
        t.get("urgency_level", 3),
        t.get("confidence_band", "uncertain"),
        t.get("confidence_score", 0.0),
        t.get("red_flags", []),
        t.get("recommended_actions", []),
        t.get("follow_up_questions", []),
        t.get("safety_note", ""),
        result.get("safety_overrides_applied", []),
        result.get("disclaimer", ""),
    )


# ── State ─────────────────────────────────────────────────────────────────────

# Phase: "idle" → "intake" (asking clarifying questions) → "results"
_initial_state: dict = {
    "phase": "idle",          # idle | intake | results
    "initial_text": "",
    "intake_questions": [],
    "intake_qa": [],          # [{question, answer}, …]
    "current_q_idx": 0,
    "result": {},
}


# ── Intake question generator ────────────────────────────────────────────────

_INTAKE_PROMPT = """You are a medical intake assistant. A patient said:
"{text}"

Generate exactly 4 concise intake questions to gather missing clinical information before triage.
Focus on: age & sex (if not stated), key vital measurements, symptom duration/severity, and any relevant medical history.
Return ONLY a valid JSON array of 4 question strings, nothing else.
Example: ["How old are you and what is your sex?", "How long have you had this symptom?", ...]"""


def generate_intake_questions(text: str) -> list[str]:
    """
    Generate 4-5 targeted intake questions for the initial complaint.
    Tries Gemma endpoint first; falls back to smart rule-based questions.
    """
    # ── Try Gemma endpoint ────────────────────────────────────────────
    endpoint = os.environ.get("STRUCTURER_ENDPOINT_URL")
    token    = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
    if endpoint and token and not _DEMO_MODE:
        try:
            import requests as _req
            prompt = _INTAKE_PROMPT.format(text=text)
            resp = _req.post(
                endpoint,
                headers={"Authorization": f"Bearer {token}",
                         "Content-Type": "application/json"},
                json={"inputs": prompt,
                      "parameters": {"max_new_tokens": 256,
                                     "temperature": 0.1,
                                     "return_full_text": False}},
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()
            raw_text = raw[0].get("generated_text", "") if isinstance(raw, list) else str(raw)
            # Extract JSON array
            import re as _re
            m = _re.search(r'\[.*?\]', raw_text, _re.DOTALL)
            if m:
                questions = json.loads(m.group())
                if isinstance(questions, list) and len(questions) >= 2:
                    return [str(q) for q in questions[:5]]
        except Exception as e:
            logger.warning("Gemma intake question generation failed: %s", e)

    # ── Smart rule-based fallback ─────────────────────────────────────
    tl = text.lower()
    questions: list[str] = []

    # 1. Age / sex — almost always needed
    if not any(k in tl for k in ["year old", "years old", " yo ", "male", "female", "man", "woman", "boy", "girl"]):
        questions.append("What is your age and sex?")

    # 2. Temperature / vital measurement
    if any(k in tl for k in ["fever", "temperature", "hot", "chills"]):
        if "degree" not in tl and "°" not in tl:
            questions.append("What is your current body temperature (if measured)?")

    # 3. Symptom-specific depth
    if any(k in tl for k in ["pain", "ache", "hurt", "sore"]):
        questions.append("Where exactly is the pain, and how severe is it on a scale of 1–10?")
    if any(k in tl for k in ["breath", "breathing", "short", "wheez", "cough"]):
        questions.append("Are you short of breath at rest, or only during activity?")
    if any(k in tl for k in ["dizz", "faint", "lightheaded", "headache"]):
        questions.append("Did the symptom come on suddenly or gradually?")
    if any(k in tl for k in ["chest", "heart", "palpitat"]):
        questions.append("Do you feel chest tightness, pressure, or palpitations?")

    # 4. Duration — almost always needed
    if not any(k in tl for k in ["hour", "day", "week", "month", "since", "started", "began", "ago"]):
        questions.append("When did the symptoms start, and have they been getting better or worse?")

    # 5. Other symptoms
    questions.append("Do you have any other symptoms (e.g. nausea, vomiting, rash, fatigue)?")

    # 6. Medical history / meds
    if not any(k in tl for k in ["diabete", "hypertension", "asthma", "medication", "allerg", "condition"]):
        questions.append("Do you have any chronic conditions or take regular medications?")

    # Return at most 5, at least 3
    if len(questions) < 3:
        questions += [
            "Are you currently taking any medications or have known allergies?",
            "Have you experienced similar symptoms before?",
        ]
    return questions[:5]


# ── Question HTML helper ─────────────────────────────────────────────────────

def _question_html(idx: int, total: int, question: str, phase: str = "intake") -> str:
    phase_label = "Intake" if phase == "intake" else "Follow-up"
    return f"""
<div style="font-family:'Inter','Segoe UI',sans-serif;padding:12px 4px;">
  <div style="font-size:0.78rem;color:#64748B;margin-bottom:6px;font-weight:600;
    text-transform:uppercase;letter-spacing:0.6px;">
    {phase_label} · Question {idx} of {total}
  </div>
  <div style="font-size:1.12rem;font-weight:700;color:#1A1A2E;line-height:1.55;
    padding:12px 16px;background:#F0F7FF;border-left:4px solid #0F3460;
    border-radius:0 8px 8px 0;">
    {question}
  </div>
</div>"""


def _intake_pending_html(total: int) -> str:
    """Shown in the results panel while intake is in progress."""
    return f"""
<div style="text-align:center;padding:40px 20px;font-family:'Inter','Segoe UI',sans-serif;">
  <div style="font-size:2.5rem;margin-bottom:12px;">🩺</div>
  <div style="font-size:1.2rem;font-weight:700;color:#0F3460;margin-bottom:8px;">
    Intake Interview in Progress
  </div>
  <div style="font-size:0.92rem;color:#64748B;max-width:320px;margin:0 auto;
    line-height:1.6;">
    Please answer the {total} questions below.<br>
    Once complete, the full AI triage analysis will run automatically.
  </div>
</div>"""


# ── Start assessment (Tab 1 — intake-first flow) ─────────────────────────────

def start_assessment(patient_text: str, state: dict):
    """Analyze button → generate intake questions, show interview panel."""
    if not patient_text.strip():
        return (
            state,
            "<p style='color:#888;font-style:italic;padding:30px;text-align:center;'>"
            "Enter your complaint above and click Start Assessment.</p>",
            "", "", "", "", "", "",
            gr.update(visible=False),  # intake panel
            gr.update(visible=True),   # results_col
            gr.update(visible=False),
            gr.update(value=""),
        )

    questions = generate_intake_questions(patient_text)
    state = {
        "phase": "intake",
        "initial_text": patient_text,
        "intake_questions": questions,
        "intake_qa": [],
        "current_q_idx": 0,
        "result": {},
    }
    q_html = _question_html(1, len(questions), questions[0], "intake")

    return (
        state,
        "", "", "", "", "", "", "",           # results (hidden)
        gr.update(visible=True),               # show intake panel
        gr.update(visible=False),              # hide results_col
        gr.update(value=q_html, visible=True), # first question
        gr.update(value=""),                   # clear answer box
    )


# ── Structured form triage (direct — no intake needed) ───────────────────────

def run_structured_triage(age, sex, chief_complaint, symptoms, duration, risk_factors, state: dict):
    if not chief_complaint.strip():
        return (
            state,
            "<p style='color:#888;font-style:italic;padding:30px;text-align:center;'>"
            "Please fill in the Chief Complaint above.</p>",
            "", "", "", "", "", "",
            gr.update(visible=False),  # intake panel
            gr.update(visible=True),   # results_col
            gr.update(visible=False),
            gr.update(value=""),
        )
    try:
        result = _run_structured_pipeline(age, sex, chief_complaint, symptoms, duration, risk_factors)
    except Exception as e:
        logger.error("Structured triage error: %s", e)
        result = _mock_result()

    return _emit_results(result, state, hide_panel=True)


# ── Submit answer (intake phase) ─────────────────────────────────────────────

def submit_intake_answer(answer: str, state: dict):
    """
    Save the answer, advance to next question.
    On the last answer → build enriched text → run full pipeline → emit results.
    """
    questions = state.get("intake_questions", [])
    idx       = state.get("current_q_idx", 0)
    qa        = state.get("intake_qa", [])

    # Save current answer
    if idx < len(questions):
        qa.append({"question": questions[idx], "answer": answer.strip()})

    next_idx = idx + 1
    state = {**state, "intake_qa": qa, "current_q_idx": next_idx}

    # ── More questions remain ─────────────────────────────────────────
    if next_idx < len(questions):
        q_html = _question_html(next_idx + 1, len(questions), questions[next_idx], "intake")
        return (
            state,
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(visible=True),           # keep intake panel shown
            gr.update(visible=False),          # keep results_col hidden
            gr.update(value=q_html, visible=True),
            gr.update(value=""),
        )

    # ── All answered → run full pipeline ─────────────────────────────
    initial_text = state.get("initial_text", "")
    enriched     = _build_enriched_text_from_intake(initial_text, qa)
    try:
        result = _run_text_pipeline(enriched)
    except Exception as e:
        logger.error("Pipeline error after intake: %s", e, exc_info=True)
        result = _mock_result()

    return _emit_results(result, state, hide_panel=True)


# ── Shared result emitter ────────────────────────────────────────────────────

def _emit_results(result: dict, state: dict, hide_panel: bool = False):
    """Build all result HTML and return the 12-tuple expected by _result_outputs."""
    urgency_level, conf_band, conf_score, red_flags, actions, _, safety, overrides, disclaimer = (
        _parse_result(result)
    )
    state = {**state, "phase": "results", "result": result}

    return (
        state,
        _badge_html(urgency_level),
        _escalation_html(conf_band),
        _confidence_html(conf_score, conf_band),
        _red_flags_html(red_flags),
        _actions_html(actions),
        _safety_html(safety, overrides, disclaimer),
        _export_report(result),
        gr.update(visible=False),   # hide intake panel
        gr.update(visible=True),    # show results_col
        gr.update(value="", visible=False),
        gr.update(value=""),
    )


# ── Build enriched text from intake Q&A ─────────────────────────────────────

def _build_enriched_text_from_intake(initial_text: str, qa: list) -> str:
    parts = [initial_text.strip()]
    for item in qa:
        q = item.get("question", "").strip()
        a = item.get("answer",   "").strip()
        if a:
            parts.append(f"{q} {a}")
    return " ".join(parts)


# ── Mock result for offline/demo mode ────────────────────────────────────────

def _mock_result() -> dict:
    """Return a plausible demo result when models are unavailable."""
    return {
        "id": "demo-0001",
        "timestamp": datetime.utcnow().isoformat(),
        "input_type": "text",
        "structured_data": {},
        "triage_output": {
            "urgency_level": 5,
            "urgency_label": "Critical / Immediate",
            "confidence_score": 0.78,
            "confidence_band": "moderate",
            "red_flags": [
                {"flag": "Chest pain with diaphoresis", "category": "Cardiac",
                 "source": "rule", "recommended_action": "ECG immediately"},
                {"flag": "Shortness of breath at rest", "category": "Respiratory",
                 "source": "rule", "recommended_action": "Oxygen saturation monitoring"},
            ],
            "risk_categories": ["cardiac", "respiratory"],
            "recommended_actions": [
                "Immediate ECG",
                "IV access establishment",
                "Continuous cardiac monitoring",
                "Notify attending physician",
            ],
            "follow_up_questions": [
                "When did the chest pain begin?",
                "Does the pain radiate to your arm or jaw?",
                "Do you have a history of cardiac disease?",
            ],
            "safety_note": (
                "Symptoms are consistent with possible acute coronary syndrome. "
                "Immediate evaluation by a physician is required."
            ),
        },
        "safety_overrides_applied": [],
        "disclaimer": (
            "This is an AI-assisted triage tool. It does NOT provide medical diagnoses. "
            "All outputs must be reviewed by qualified healthcare professionals."
        ),
    }


# ── Logo path helper ─────────────────────────────────────────────────────────

_LOGO_PATH = str(Path(__file__).resolve().parents[1] / "assets" / "logo.png")
_LOGO_EXISTS = Path(_LOGO_PATH).exists()


def _logo_img_tag() -> str:
    """Return an <img> tag for the logo if present, else an SVG shield fallback."""
    if _LOGO_EXISTS:
        import base64
        with open(_LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="height:72px;width:72px;object-fit:contain;border-radius:10px;" '
            f'alt="MedGuard logo" />'
        )
    # Fallback: inline SVG shield + heart-pulse
    return """
<svg width="72" height="72" viewBox="0 0 72 72" fill="none"
     xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="sg" x1="0" y1="0" x2="72" y2="72" gradientUnits="userSpaceOnUse">
      <stop offset="0%"  stop-color="#00E5FF"/>
      <stop offset="100%" stop-color="#7B2FFF"/>
    </linearGradient>
  </defs>
  <!-- Shield -->
  <path d="M36 4 L64 16 L64 38 C64 54 50 66 36 70 C22 66 8 54 8 38 L8 16 Z"
        fill="none" stroke="url(#sg)" stroke-width="2.5" opacity="0.9"/>
  <!-- Heart -->
  <path d="M28 30 C28 26,22 22,18 27 C14 32,20 38,28 44 C36 38,42 32,38 27 C34 22,28 26,28 30Z"
        fill="none" stroke="#00E5FF" stroke-width="1.8" transform="translate(8,2) scale(0.9)"/>
  <!-- ECG line -->
  <polyline points="14,42 22,42 26,34 30,50 34,38 38,42 58,42"
            fill="none" stroke="#FF4F6A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;700&family=Inter:wght@400;600;700;800&display=swap');

/* ── global ── */
body, .gradio-container {
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
  background: #F0F2F5 !important;
}

/* ── header ── */
#mg-header {
  background: linear-gradient(135deg,#0A0E27 0%,#0D1B3E 55%,#0A2A5E 100%);
  border-radius: 16px;
  padding: 24px 32px;
  margin-bottom: 20px;
  border: 1px solid rgba(0,229,255,0.12);
  box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

/* ── Google G multicolor ── */
.google-g {
  font-family: 'Google Sans', 'Inter', sans-serif;
  font-weight: 900;
  font-size: 1.9rem;
  line-height: 1;
  background: linear-gradient(
    135deg,
    #4285F4 0%,
    #4285F4 28%,
    #EA4335 28%,
    #EA4335 50%,
    #FBBC05 50%,
    #FBBC05 72%,
    #34A853 72%,
    #34A853 100%
  );
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  display: inline-block;
  filter: drop-shadow(0 1px 3px rgba(0,0,0,0.4));
}

/* ── panels ── */
.mg-card {
  background: #FFFFFF;
  border-radius: 14px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  padding: 22px;
}

/* ── tabs ── */
.tab-nav button {
  font-weight: 600 !important;
  font-size: 0.92rem !important;
}

/* ── buttons ── */
#btn-analyze, #btn-submit-structured {
  background: #0F3460 !important;
  color: #FFFFFF !important;
  border: none !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  border-radius: 8px !important;
  padding: 10px 28px !important;
  cursor: pointer !important;
  transition: background 0.2s ease;
}
#btn-analyze:hover, #btn-submit-structured:hover {
  background: #16213E !important;
}
#btn-followup-submit {
  background: #27AE60 !important;
  color: #FFFFFF !important;
  border: none !important;
  font-weight: 700 !important;
  border-radius: 8px !important;
  cursor: pointer !important;
  transition: background 0.2s ease;
}
#btn-followup-submit:hover { background: #1E8449 !important; }

/* ── accordion ── */
.mg-accordion { margin-top: 6px; }

/* ── textarea ── */
textarea { border-radius: 8px !important; }

/* ── export ── */
#export-box textarea {
  font-family: 'Courier New', monospace !important;
  font-size: 0.82rem !important;
  background: #1E1E1E !important;
  color: #D4D4D4 !important;
}
"""

def _build_header_html() -> str:
    logo_img = _logo_img_tag()
    return f"""
<div id="mg-header">
  <div style="display:flex;justify-content:space-between;align-items:center;
    flex-wrap:wrap;gap:16px;">

    <!-- LEFT: logo + title -->
    <div style="display:flex;align-items:center;gap:16px;">
      {logo_img}
      <div>
        <div style="color:#FFFFFF;font-size:1.65rem;font-weight:800;
          letter-spacing:0.2px;line-height:1.15;font-family:'Inter',sans-serif;">
          MedGuard Triage Copilot
        </div>
        <div style="color:#7DD3FC;font-size:0.87rem;margin-top:5px;
          letter-spacing:0.3px;">
          Agentic Multimodal Safety Layer &nbsp;·&nbsp; Never miss a clinical red flag
        </div>
      </div>
    </div>

    <!-- RIGHT: Google-style G branding + disclaimer -->
    <div style="text-align:right;">
      <div style="display:flex;align-items:center;justify-content:flex-end;
        gap:6px;margin-bottom:6px;">
        <span style="color:#CBD5E1;font-size:0.88rem;font-weight:500;">Powered by</span>
        <!-- Google G -->
        <span class="google-g">G</span>
        <span style="color:#FFFFFF;font-size:1.15rem;font-weight:800;
          letter-spacing:0.3px;font-family:'Inter',sans-serif;">MedGemma</span>
      </div>
      <!-- Google-brand coloured underline bar -->
      <div style="display:flex;justify-content:flex-end;gap:3px;margin-bottom:8px;">
        <div style="height:3px;width:22px;border-radius:2px;background:#4285F4;"></div>
        <div style="height:3px;width:22px;border-radius:2px;background:#EA4335;"></div>
        <div style="height:3px;width:22px;border-radius:2px;background:#FBBC05;"></div>
        <div style="height:3px;width:22px;border-radius:2px;background:#34A853;"></div>
      </div>
      <div style="color:#94A3B8;font-size:0.74rem;max-width:260px;
        line-height:1.5;text-align:right;">
        ⚠️ For clinical decision support only.<br>
        Not a substitute for professional medical judgement.
      </div>
    </div>

  </div>
</div>
"""


# Build at import time (logo embedded as base64 if present)
_HEADER_HTML: str = _build_header_html()


# ── Transcription helper ─────────────────────────────────────────────────────

# Track whether we're running in demo mode (set by launch_ui.py)
_DEMO_MODE: bool = False


def transcribe_audio(audio_path: str | None) -> tuple[str, str]:
    """
    Transcribe *audio_path* to text.

    Priority chain (first success wins):
      1. Google MedASR (local weights)  — medical vocab, fully offline, no token needed
      2. openai-whisper tiny            — local, offline, general vocab
      3. SpeechRecognition + Google     — network fallback

    Returns (transcript_text, status_html)
    """
    if not audio_path:
        return "", ""

    # ── Stage 1: MedASR local model (best medical accuracy, fully offline) ──
    # Always attempted first — model is pre-downloaded to models/asr/medasr_local/
    # No HF token needed; local_files_only prevents any network calls.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not _DEMO_MODE:
        try:
            from models.asr.medasr_wrapper import MedASRWrapper, _DEFAULT_LOCAL_DIR
            logger.info("Transcribing with MedASR (local) …")
            asr = MedASRWrapper(local_dir=str(_DEFAULT_LOCAL_DIR))
            result = asr.transcribe(audio_path)
            text = result.get("transcript", "").strip()
            if text:
                status = _transcribe_status_html(
                    "✅ Transcribed via MedGemma MedASR (medical-grade)", "#0F3460"
                )
                return text, status
        except Exception as e:
            logger.warning("MedASR transcription failed: %s", e)

    # ── Stage 2: Whisper tiny (local, offline) ─────────────────────────
    try:
        import whisper  # openai-whisper
        logger.info("Transcribing with Whisper (tiny) …")
        model = whisper.load_model("tiny")   # ~39 MB, fast on CPU
        result = model.transcribe(audio_path, fp16=False)
        text = result.get("text", "").strip()
        if text:
            status = _transcribe_status_html(
                "⚠️ Transcribed via Whisper (MedASR unavailable)", "#27AE60"
            )
            return text, status
    except Exception as e:
        logger.warning("Whisper transcription failed: %s", e)

    # ── Stage 3: SpeechRecognition + Google free API ───────────────────
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        if text:
            status = _transcribe_status_html(
                "✅ Transcribed via Google Speech API", "#27AE60"
            )
            return text, status
    except Exception as e:
        logger.warning("SpeechRecognition fallback failed: %s", e)

    status = _transcribe_status_html(
        "⚠️ Could not transcribe — type your complaint manually, "
        "or set HF_TOKEN in .env for MedASR.",
        "#E67E22",
    )
    return "", status


def _transcribe_status_html(msg: str, color: str) -> str:
    return (
        f'<div style="font-family:\'Segoe UI\',sans-serif;font-size:0.85rem;'
        f'color:{color};padding:4px 0 2px;">'
        f'{msg}</div>'
    )


# ── Build Gradio App ──────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="MedGuard Triage Copilot") as demo:

        # ── shared state ──────────────────────────────────────────────────
        state = gr.State(value={
            "result": {},
            "qa_pairs": [],
            "current_q_idx": 0,
            "follow_up_questions": [],
        })

        # ═══════════════════════════════════════════════════════════════════
        # HEADER
        # ═══════════════════════════════════════════════════════════════════
        gr.HTML(_HEADER_HTML)

        # ═══════════════════════════════════════════════════════════════════
        # MAIN LAYOUT  (left input | right results)
        # ═══════════════════════════════════════════════════════════════════
        with gr.Row(equal_height=False):

            # ── LEFT: Input Panel ─────────────────────────────────────────
            with gr.Column(scale=4, elem_classes="mg-card"):
                gr.Markdown("## 🩺 Patient Intake")

                with gr.Tabs() as input_tabs:

                    # ────── Tab 1: Free Text / Voice ──────────────────────
                    with gr.Tab("📝 Free Text / Voice"):
                        txt_input = gr.Textbox(
                            label="Patient Complaint",
                            placeholder=(
                                "Describe the patient's symptoms, history, and concerns "
                                "in plain language …\n\n"
                                "Example: 65-year-old male with sudden onset chest pain "
                                "radiating to left arm, diaphoresis, and shortness of breath."
                            ),
                            lines=7,
                            max_lines=14,
                        )
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="🎙️ Voice input — press record, speak, then stop",
                            visible=True,
                        )
                        asr_status = gr.HTML(value="", visible=True)
                        gr.Markdown(
                            "_Audio is transcribed locally by Whisper and pasted "
                            "into the text box above. Review before clicking Analyze._",
                            elem_classes="mg-hint",
                        )
                        btn_analyze = gr.Button(
                            "🩺 Start Assessment →",
                            variant="primary",
                            elem_id="btn-analyze",
                        )

                    # ────── Tab 2: Structured Intake Form ─────────────────
                    with gr.Tab("🗂️ Structured Form"):
                        with gr.Row():
                            age_in       = gr.Number(label="Age", minimum=0, maximum=120, value=None)
                            sex_in       = gr.Dropdown(
                                label="Sex",
                                choices=["Male", "Female", "Other / Unknown"],
                                value="Other / Unknown",
                            )
                        cc_in           = gr.Textbox(
                            label="Chief Complaint *",
                            placeholder="e.g. Chest pain with shortness of breath",
                        )
                        symptoms_in     = gr.Textbox(
                            label="Symptoms (comma-separated)",
                            placeholder="e.g. chest pain, dyspnea, diaphoresis",
                        )
                        duration_in     = gr.Textbox(
                            label="Duration",
                            placeholder="e.g. 2 hours",
                        )
                        risk_factors_in = gr.Textbox(
                            label="Risk Factors (comma-separated)",
                            placeholder="e.g. hypertension, diabetes, smoking",
                        )
                        btn_structured  = gr.Button(
                            "🔍 Submit",
                            variant="primary",
                            elem_id="btn-submit-structured",
                        )

            # ── RIGHT: Results Panel ──────────────────────────────────────
            with gr.Column(scale=5, elem_classes="mg-card"):

                # ── Intake interview (shown while collecting answers) ─────
                with gr.Column(visible=False) as followup_panel:
                    gr.HTML("""
<div style="background:#F0F7FF;border:1px solid #C7DEFF;border-radius:12px;
  padding:20px 20px 8px 20px;margin-bottom:8px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
    <span style="font-size:1.6rem;">🩺</span>
    <div>
      <div style="font-size:1.15rem;font-weight:800;color:#0F3460;
        font-family:'Inter','Segoe UI',sans-serif;">Intake Interview</div>
      <div style="font-size:0.82rem;color:#64748B;margin-top:2px;">
        Answer each question — triage runs automatically after the last one.
      </div>
    </div>
  </div>
</div>""")
                    q_display_out = gr.HTML(visible=True)
                    with gr.Row():
                        followup_answer = gr.Textbox(
                            label="Your Answer",
                            placeholder="Type your answer and press Submit …",
                            scale=5,
                        )
                        btn_followup = gr.Button(
                            "Submit →",
                            variant="primary",
                            elem_id="btn-followup-submit",
                            scale=1,
                        )

                # ── Results cards (hidden while intake is running) ─────────
                with gr.Column(visible=True) as results_col:
                    gr.Markdown("## 📊 Triage Results")

                    # Urgency badge
                    urgency_out = gr.HTML(
                        value="<div style='color:#AAA;font-style:italic;"
                              "text-align:center;padding:30px 0;'>"
                              "Results will appear here after analysis.</div>"
                    )

                    # Escalation badge
                    escalation_out = gr.HTML()

                    # Confidence
                    confidence_out = gr.HTML()

                    # Red flags accordion
                    with gr.Accordion("🚩 Red Flags", open=True, elem_classes="mg-accordion"):
                        red_flags_out = gr.HTML()

                    # Actions accordion
                    with gr.Accordion("🧭 Recommended Actions", open=True, elem_classes="mg-accordion"):
                        actions_out = gr.HTML()

                    # Safety note + disclaimer accordion
                    with gr.Accordion("🛡️ Safety Note & Disclaimer", open=False, elem_classes="mg-accordion"):
                        safety_out = gr.HTML()

                    # Export (above the legend so it's easily accessible)
                    with gr.Accordion("📄 Export Summary", open=False, elem_classes="mg-accordion"):
                        export_out = gr.Textbox(
                            label="Structured Triage Report",
                            lines=20,
                            interactive=False,
                            elem_id="export-box",
                        )

                    # ESI legend
                    with gr.Accordion("📖 What do these levels mean?", open=False, elem_classes="mg-accordion"):
                        gr.Markdown(URGENCY_LEGEND)

        # ── common outputs tuple ──────────────────────────────────────────
        # shared between text and structured callbacks  (12 outputs)
        _result_outputs = [
            state,
            urgency_out,
            escalation_out,
            confidence_out,
            red_flags_out,
            actions_out,
            safety_out,
            export_out,
            followup_panel,   # intake interview panel
            results_col,      # results cards column
            q_display_out,
            followup_answer,
        ]

        # ── Wire: audio → auto-transcribe into textbox ───────────────────
        audio_input.stop_recording(
            fn=transcribe_audio,
            inputs=[audio_input],
            outputs=[txt_input, asr_status],
        )
        # Also trigger on upload (if user uploads a file)
        audio_input.upload(
            fn=transcribe_audio,
            inputs=[audio_input],
            outputs=[txt_input, asr_status],
        )

        # ── Wire: Start Assessment (Tab 1) → intake interview ────────────
        btn_analyze.click(
            fn=start_assessment,
            inputs=[txt_input, state],
            outputs=_result_outputs,
        )

        # ── Wire: structured form → direct triage (all info provided) ────
        btn_structured.click(
            fn=run_structured_triage,
            inputs=[age_in, sex_in, cc_in, symptoms_in, duration_in, risk_factors_in, state],
            outputs=_result_outputs,
        )

        # ── Wire: intake answer → next question or final pipeline run ─────
        btn_followup.click(
            fn=submit_intake_answer,
            inputs=[followup_answer, state],
            outputs=_result_outputs,
        )
        # Also allow pressing Enter in the answer box
        followup_answer.submit(
            fn=submit_intake_answer,
            inputs=[followup_answer, state],
            outputs=_result_outputs,
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MedGuard Triage Copilot – Gradio UI")
    parser.add_argument("--host",        default="0.0.0.0",  help="Bind host")
    parser.add_argument("--port",        type=int, default=7860, help="Port (default 7860)")
    parser.add_argument("--share",       action="store_true", help="Create public Gradio share link")
    parser.add_argument("--demo-mode",   action="store_true", help="Use mock backend (no GPU needed)")
    args = parser.parse_args()

    if args.demo_mode:
        # Monkey-patch pipelines with mock
        import app.gradio_app as _self
        _self._run_text_pipeline       = lambda _t: _mock_result()
        _self._run_structured_pipeline = lambda *_a: _mock_result()
        print("[DEMO MODE] Using mock backend — no models loaded.")

    logging.basicConfig(level=logging.INFO)
    app_instance = build_app()
    app_instance.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=_CSS,
        inbrowser=True,
    )
