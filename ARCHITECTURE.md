# Financial Agentic RAG - Production Architecture

## Scope

This document reflects the active production runtime used by app_2.py.

Implemented capabilities:
- Phase A: orchestrator-first security gate
- Phase B: intent and domain routing
- Phase C: quality assessment and confidence gating
- Phase D: cache and retrieval latency controls

Out of scope:
- historical planning notes
- removed evaluation scaffolding
- deprecated entrypoints

## Runtime Entry and Flow

Primary UI entrypoint:
- app_2.py

Core runtime entrypoint:
- src/services/query_orchestrator.py (run_query)

High-level request path:
1. Streamlit collects query and deep-mode toggle (default is fast mode when toggle is off).
2. app_2.py calls run_query(query, mode, chat_history).
3. Orchestrator performs mandatory security precheck.
4. Orchestrator attempts cache lookup.
5. Intent/domain routing selects lane.
6. Lane executes (trivial, out_of_scope, ambiguous_in_domain, general_finance, tax_rag).
7. Deep mode applies quality and confidence gating.
8. Successful responses are cached.

## Detailed Request Pipeline

### 1) UI Layer
- File: app_2.py
- Responsibilities:
  - collect user input
  - pass mode (fast/deep) and chat_history
  - render answer, confidence, route, timings, sources

### 2) Orchestrator Layer
- File: src/services/query_orchestrator.py
- Responsibilities:
  - mandatory global security precheck
  - cache lookup/writeback (keyed on execution query after optional context prepend)
  - optional recent chat-history context prepend
  - intent and domain classification
  - lane dispatch and response normalization

### 3) Security Layer (Phase A)
- Files:
  - src/security/security_gatekeeper.py
  - src/security/input_validator.py
  - src/security/injection_detector.py
  - src/security/adversarial_classifier.py
- Behavior:
  - fail-closed on unsafe input
  - blocked queries do not proceed to retrieval/reasoning

### 4) Routing Layer (Phase B)
- Files:
  - src/classifiers/intent_classifier.py
  - src/classifiers/domain_classifier.py
- Behavior:
  - intent lane routing
  - domain detection for tax queries
  - tax rescue heuristic for likely tax asks

### 5) Lane Model

#### a) trivial
- instant static response

#### b) out_of_scope
- immediate finance-only deflection

#### c) ambiguous_in_domain
- clarifying question before heavy execution

#### d) general_finance
- direct LLM answer
- no RAG retrieval and no workflow pipeline

#### e) tax_rag
- mode=fast: direct retrieval plus one LLM call
- mode=deep: full workflow pipeline

### 6) Fast Tax Mode (mode="fast")
- Implemented in _run_fast_tax_lane in src/services/query_orchestrator.py
- Behavior:
  1. direct federated retrieval (top_k=3)
  2. optional domain_hint passthrough to router
  3. bounded context assembly (2000-char cap)
  4. single LLM answer call
- Skips planner and verification by design

### 7) Deep Tax Mode (mode="deep")
- Implemented via _run_tax_rag_lane(mode="deep") in src/services/query_orchestrator.py
- Calls AgentWorkflow.run in src/orchestration/workflow.py
- Stage order:
  - security
  - planner
  - retrieval
  - reasoning
  - verification
- Uses stage budgets and timeout/degradation metadata

### 8) Workflow and Agents
- Files:
  - src/api/server.py (workflow initialization)
  - src/orchestration/workflow.py
  - src/orchestration/nodes.py
  - src/orchestration/graph_state.py
  - src/agents/planner_agent.py
  - src/agents/retrieval_agent.py
  - src/agents/reasoning_agent.py
  - src/agents/verification_agent.py
- Behavior:
  - planner fallback if planning fails/times out
  - per-stage timing and degraded flags

### 9) Retrieval Layer
- Files:
  - src/retrieval/federated_router.py
  - src/retrieval/hybrid_retriever.py
  - src/retrieval/vector_index.py
  - src/retrieval/bm25_index.py
  - src/retrieval/embedding_model.py
  - src/retrieval/reranker.py
  - src/retrieval/parallel_retriever.py
- Behavior:
  - federated domain routing
  - vector + lexical retrieval
  - optional parallel strategy in retrieval agent path
  - domain_hint short-circuit in federated router search when valid

### 10) Confidence and Quality Layer (Phase C)
- Files:
  - src/confidence/quality_assessment.py
  - src/confidence/answer_quality_evaluator.py
  - src/confidence/confidence_composer.py
- Behavior:
  - compose retrieval/reasoning/verification quality signals
  - confidence gating and override policy for low-confidence responses

### 11) Cache Layer (Phase D Tier 1)
- File: src/services/response_cache.py
- Behavior:
  - normalized query keying
  - TTL-aware entries
  - LRU eviction
  - checked before lane execution

## Data and Artifacts

Tracked source inputs:
- data/raw/

Generated artifacts (not tracked in Git):
- data/chunks/chunks.json
- data/vector_store/*

Build command:
- python -m src.data_pipeline.run_pipeline

## Configuration

- Primary config: src/config/settings.py
- Env file: .env (not tracked)
- Template: .env.example

Required key:
- OPENROUTER_API_KEY

Primary runtime settings:
- DEFAULT_TIMEOUT = 120
- REASONING_MODEL = deepseek/deepseek-r1
- GENERAL_MODEL = qwen/qwen3-30b-a3b

## Operational Notes

1. First run can be slower due to warmup/index loading.
2. Tax RAG quality depends on built retrieval artifacts.
3. Fast mode is default in orchestrator; deep mode is opt-in from UI toggle.
4. app_2.py passes mode and chat_history into run_query.
5. Security precheck executes before lane routing.
