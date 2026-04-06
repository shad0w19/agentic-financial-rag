# Financial Agentic RAG (Production)

Production entrypoint is `app_2.py`.

## What This App Does

- Secure tax/finance assistant with three-lane routing.
- Global security precheck before routing.
- Model-based intent/domain routing.
- Confidence and quality gating before final answer.
- Phase D latency optimizations:
  - response caching
  - parallel retrieval strategy for multi-domain coverage

## Project Structure (Runtime)

```
.
├── app_2.py
├── requirements.txt
├── ARCHITECTURE.md
├── .env.example
├── data/
│   ├── raw/
│   ├── chunks/          # generated (gitignored)
│   └── vector_store/    # generated (gitignored)
└── src/
    ├── agents/
    ├── api/
    ├── classifiers/
    ├── confidence/
    ├── config/
    ├── core/
    ├── interfaces/
    ├── orchestration/
    ├── provenance/
    ├── retrieval/
    ├── security/
    ├── services/
    └── import_map.py
```

## Setup

1. Create and activate virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Configure environment.

```powershell
Copy-Item .env.example .env
```

Set at minimum:

```
OPENROUTER_API_KEY=your_key_here
```

## Build Retrieval Artifacts (First Time / After Data Change)

Artifacts are intentionally not tracked in Git.

```powershell
python -m src.data_pipeline.run_pipeline
```

This generates:

- `data/chunks/chunks.json`
- `data/vector_store/*`

## Run App

```powershell
streamlit run app_2.py
```

## Notes

- `app_2.py` is the only production UI entrypoint.
- Generated folders (`data/chunks`, `data/vector_store`) are excluded from Git and must be regenerated locally.
- Logs and caches are excluded from Git.

Files:

src/evaluation/ragas_eval.py
src/evaluation/security_tests.py

Evaluation metrics:

faithfulness
answer relevance
context precision

Security metrics:

prompt injection defense rate
data leakage detection
defense success rate
Development Order (Critical)

Never generate files out of order.

1. types.py
2. interfaces/*
3. import_map.py
4. ARCHITECTURE.md

5. data_pipeline/*
6. retrieval/*
7. services/*
8. security/*
9. provenance/*
10. agents/*
11. orchestration/*
12. api/*
13. evaluation/*

Rule:

A module can only be generated after all of its dependencies exist.
Expected System After 2 Weeks

Features:

Hybrid RAG retrieval
Federated knowledge nodes
Multi-hop agent reasoning
Secure query gatekeeper
Deterministic financial calculations
Provenance DAG audit trail