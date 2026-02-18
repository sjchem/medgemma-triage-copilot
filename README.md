# 🏥 MedGuard Triage Copilot

**An agentic, multimodal AI that prioritizes patients safely by detecting risk, red flags, and urgency — before care begins.**

> Built for the [Google MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge/overview)

---

## Architecture

MedGuard uses a **three-stage, safety-first pipeline** built on Google's Health AI ecosystem:

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Stage 1   │     │     Stage 2      │     │      Stage 3        │
│   MedASR    │────▶│  Gemma 7B-IT     │────▶│  MedGemma 4B-IT     │
│  (MedASR)   │     │  Structurer      │     │  Triage Reasoner    │
│             │     │                  │     │                     │
│ Voice → Text│     │ Text → Struct    │     │ Struct → Triage     │
└─────────────┘     └──────────────────┘     └─────────────────────┘
                            │                         │
                    ┌───────┴────────┐        ┌───────┴────────┐
                    │  Uncertainty   │        │  Red Flag      │
                    │  Engine        │        │  Engine        │
                    └────────────────┘        │  Risk Scorer   │
                                              └────────────────┘
```

| Stage | Model | Role |
|-------|-------|------|
| **1. MedASR** | Google MedASR (105M) | Medical speech-to-text, Conformer CTC architecture |
| **2. Structurer** | Gemma 2-2B | Clinical entity extraction, negation detection, normalization |
| **3. Triage** | MedGemma 4B-IT | Safety-focused triage reasoning, red flag detection, risk scoring |

### Key Design Principles

- **Safety-first**: Rule-based red flag engine runs independently of models — critical patterns are NEVER missed
- **Uncertainty propagation**: Confidence scores flow from extraction through triage and blend with rule-based signals
- **Decoupled reasoning**: Language understanding (Stage 2) is separated from medical reasoning (Stage 3)
- **Fail-safe defaults**: On any model failure, system defaults to **urgent** and routes to clinician

---

## Project Structure

```
medguard-triage-copilot/
│
├── main.py                          # CLI entrypoint
├── requirements.txt
├── .env.example
│
├── configs/
│   ├── model_config.yaml            # Model IDs, parameters, endpoints
│   ├── safety_rules.yaml            # Red flag keywords & thresholds
│   └── escalation_policy.yaml       # Urgency → action mapping
│
├── models/
│   ├── asr/
│   │   └── medasr_wrapper.py        # Stage 1: Google MedASR
│   ├── extraction/
│   │   ├── gemma_structurer.py      # Stage 2: Gemma clinical structurer
│   │   ├── schema_definition.py     # Pydantic schemas for all stages
│   │   └── uncertainty_engine.py    # Confidence & gap analysis
│   └── triage/
│       ├── medgemma_reasoner.py     # Stage 3: MedGemma triage
│       ├── red_flag_engine.py       # Rule-based red flag scanner
│       └── risk_scorer.py           # Final risk scoring & overrides
│
├── pipelines/
│   ├── voice_pipeline.py            # Audio → ASR → Structure → Triage
│   ├── text_pipeline.py             # Text → Structure → Triage
│   └── structured_pipeline.py       # Structured data → Triage only
│
├── core/
│   ├── router.py                    # Auto-routing entrypoint
│   ├── validation.py                # Schema validation utilities
│   └── logging_utils.py             # Structured logging
│
├── evaluation/
│   ├── extraction_metrics.py        # Completeness, recall, negation accuracy
│   ├── triage_metrics.py            # Urgency accuracy, safety score
│   └── benchmark_runner.py          # Batch evaluation runner
│
├── api/
│   ├── main.py                      # FastAPI REST API
│   └── schemas.py                   # API request/response models
│
├── data/
│   ├── samples/                     # Demo input data
│   └── test_cases/                  # Evaluation test suites
│       ├── high_risk_cases.json
│       ├── low_risk_cases.json
│       └── edge_cases.json
│
└── notebooks/
    └── triage_copilot_v1.ipynb      # Development notebook
```

---

## Quick Start

### 1. Setup

```bash
# Clone and create virtual environment
git clone <repo-url>
cd medguard-triage-copilot
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure your HuggingFace token
cp .env.example .env
# Edit .env and add your HF_API_TOKEN
```

### 2. Run Triage (CLI)

```bash
# Single patient
python main.py "58yo male with crushing chest pain for 45min, smoker, diaphoretic"

# From test file
python main.py --file data/samples/sample_text_cases.json

# Interactive mode
python main.py --interactive
```

### 3. Run API Server

```bash
uvicorn api.main:app --reload --port 8000
# Then POST to /triage/text or /triage/structured
```

### 4. Run Evaluation

```python
from evaluation.benchmark_runner import load_test_cases, run_benchmark
from pipelines.text_pipeline import TextPipeline

pipeline = TextPipeline()
cases = load_test_cases("data/test_cases/high_risk_cases.json")
results = run_benchmark(pipeline, cases)
print(results)
```

---

## Safety Features

| Feature | Description |
|---------|-------------|
| **Rule-based Red Flags** | Keyword-driven detection runs independently of AI models |
| **Urgency Override** | Red flags automatically upgrade urgency level |
| **Fail-safe Default** | Model failures default to Level 4 (Urgent) |
| **Confidence Blending** | Extraction + triage confidence scores are blended |
| **Low-confidence Escalation** | Scores < 0.4 trigger automatic senior review flag |
| **Disclaimer** | Every output carries a non-diagnosis disclaimer |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_API_TOKEN` | Yes | HuggingFace API token |
| `STRUCTURER_ENDPOINT_URL` | No | Custom endpoint for Gemma structurer |
| `TRIAGE_ENDPOINT_URL` | No | Custom endpoint for MedGemma triage |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

---

## Disclaimer

⚠️ **This is an AI-assisted triage tool for the Google MedGemma Impact Challenge. It does NOT provide medical diagnoses. All outputs must be reviewed by qualified healthcare professionals. In emergencies, call emergency services immediately.**
