# Financial Agentic RAG - Production Architecture

## Scope

This document reflects the active production architecture used by `app_2.py`.

Implemented capabilities included here:

- Phase A: orchestrator-first security gate
- Phase B: intent/domain model routing
- Phase C: quality assessment and confidence gating
- Phase D: response cache and parallel retrieval strategy

Out-of-scope here:

- historical planning notes
- removed test/evaluation scaffolding
- experimental/deprecated entrypoints

## Runtime Entry and Flow

Primary entrypoint:

- `app_2.py`

High-level path:

1. User query enters Streamlit UI (`app_2.py`).
2. Query is sent to `QueryOrchestrator.run_query`.
3. Security precheck runs first (block/pass decision).
4. Cache lookup is attempted for fast return.
5. Intent routing chooses lane.
6. Tax lane uses workflow with agents and retrieval.
7. Quality and confidence gates finalize response policy.
8. Successful responses are cached.

## Detailed Request Pipeline

### 1) UI Layer

- File: `app_2.py`
- Responsibilities:
  - collect user query
  - invoke orchestrator
  - render answer, metadata, route/quality/security context

### 2) Orchestrator Layer

- File: `src/services/query_orchestrator.py`
- Responsibilities:
  - global security precheck (mandatory)
  - cache check and cache writeback
  - intent and domain routing
  - lane dispatch (`trivial`, `general_finance`, `tax_rag`)
  - response normalization and metadata enrichment

### 3) Security Layer (Phase A)

- Files:
  - `src/security/security_gatekeeper.py`
  - `src/security/input_validator.py`
  - `src/security/injection_detector.py`
  - `src/security/adversarial_classifier.py`
- Behavior:
  - fail-closed for unsafe queries
  - blocked queries do not proceed to retrieval/reasoning

### 4) Routing Layer (Phase B)

- Files:
  - `src/classifiers/intent_classifier.py`
  - `src/classifiers/domain_classifier.py`
- Behavior:
  - semantic intent classification
  - domain classification for tax queries
  - fallback policy on low confidence

### 5) Workflow/Agent Layer

- Files:
  - `src/api/server.py` (workflow initialization)
  - `src/orchestration/workflow.py`
  - `src/orchestration/nodes.py`
  - `src/orchestration/graph_state.py`
  - `src/agents/planner_agent.py`
  - `src/agents/retrieval_agent.py`
  - `src/agents/reasoning_agent.py`
  - `src/agents/verification_agent.py`
- Behavior:
  - planner -> retrieval -> reasoning -> verification
  - defense-in-depth checks at workflow level

### 6) Retrieval Layer

- Files:
  - `src/retrieval/federated_router.py`
  - `src/retrieval/hybrid_retriever.py`
  - `src/retrieval/vector_index.py`
  - `src/retrieval/bm25_index.py`
  - `src/retrieval/embedding_model.py`
  - `src/retrieval/reranker.py`
- Behavior:
  - federated domain routing
  - vector + lexical retrieval
  - result merge/rerank

### 7) Phase D Latency Layer

- Files:
  - `src/services/response_cache.py`
  - `src/retrieval/parallel_retriever.py`
  - `src/agents/retrieval_agent.py` (parallel/sequential strategy selection)
- Behavior:
  - normalized-query cache with TTL + LRU
  - parallel retrieval path for multi-domain/low-confidence scenarios
  - sequential fallback on errors

### 8) Confidence and Quality Layer (Phase C)

- Files:
  - `src/confidence/quality_assessment.py`
  - `src/confidence/answer_quality_evaluator.py`
  - `src/confidence/confidence_composer.py`
- Behavior:
  - compose retrieval/reasoning/verification quality signals
  - gate or abstain on low-confidence unsafe outputs

### 9) Provenance Layer

- Files:
  - `src/provenance/provenance_graph.py`
  - `src/provenance/dag_builder.py`
- Behavior:
  - step-level traceability for retrieval/reasoning/verification events

## Data and Artifacts

### Tracked inputs

- `data/raw/` (source documents where distribution is allowed)

### Generated artifacts (not tracked in Git)

- `data/chunks/chunks.json`
- `data/vector_store/*`

Generated via:

- `python -m src.data_pipeline.run_pipeline`

## Configuration

- Primary config: `src/config/settings.py`
- Environment file: `.env` (not tracked)
- Template: `.env.example` (tracked)

Required key:

- `OPENROUTER_API_KEY`

## Production Repository Policy

### Kept

- runtime source code (`src/`)
- production entrypoint (`app_2.py`)
- `requirements.txt`
- `README.md`
- this architecture document
- `.env.example`

### Removed/Excluded

- tests and evaluation scaffolding from public runtime repo
- logs and temporary outputs
- caches and pycache artifacts
- generated retrieval artifacts
- duplicate legacy UI entrypoint

## Minimal Runtime Tree

```
.
├── app_2.py
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── .env.example
├── data/
│   └── raw/
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

## Operational Notes

1. First run requires building retrieval artifacts.
2. Without generated indices/chunks, tax RAG retrieval will be degraded or unavailable.
3. Cache significantly reduces repeated-query latency.
4. Security precheck is always executed before lane routing.
