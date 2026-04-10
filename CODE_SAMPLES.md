# Financial Agentic RAG - Code Samples and Runtime Signatures

## Purpose

This file lists current, practical signatures and usage patterns matching the active runtime.

## 1) Main Runtime Entry

File:
- src/services/query_orchestrator.py

Signature:
- run_query(query: str, timeout_seconds: int = 120, mode: str = "fast", chat_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]

Notes:
- mode defaults to fast
- chat_history can be injected for conversational context
- security precheck always runs first

Lane coverage in current runtime:
- trivial
- out_of_scope
- ambiguous_in_domain
- general_finance
- tax_rag

## 2) UI Call Pattern

File:
- app_2.py

Pattern:
- deep_mode = st.toggle(...)
- mode = "deep" if deep_mode else "fast"
- res = demo.run_query(final_query.strip(), mode=mode, chat_history=st.session_state.messages)

## 3) Tax Lane Dispatcher

File:
- src/services/query_orchestrator.py

Method:
- _run_tax_rag_lane(query, query_id, timeout_seconds=120, mode="fast", domain_hint=None)

Behavior:
- mode="fast" -> _run_fast_tax_lane
- mode="deep" -> full workflow path
- route remains tax_rag, while fast-lane metadata marks lane as tax_rag_fast

## 4) Fast Tax Lane

File:
- src/services/query_orchestrator.py

Method:
- _run_fast_tax_lane(query, query_id, domain_hint, timeout_seconds)

Behavior:
1. router.search(query, k=FAST_LANE_TOP_K, domain_hint=domain_hint)
2. context compaction
3. single LLM call
4. return response with fast metadata

Key constants:
- FAST_LANE_TOP_K = 3

## 5) Deep Tax Lane and Workflow

Files:
- src/services/query_orchestrator.py
- src/orchestration/workflow.py

Workflow call:
- self.workflow.run(query, initial_metadata, stage_budgets, retrieval_k)

Current workflow signature:
- run(query: str, initial_metadata: dict | None = None, stage_budgets: dict | None = None, retrieval_k: int = 5) -> GraphState

Important note:
- workflow currently does not expose mode/skip_planner parameters

Deep mode stage budgets (orchestrator constants):
- planner: 20.0
- retrieval: 20.0
- reasoning: 60.0
- verification: 20.0

Other constants:
- TAX_RAG_TOP_K = 6

## 6) Federated Router Search

File:
- src/retrieval/federated_router.py

Signature:
- search(query: str, k: int = 5, filters: Dict[str, Any] | None = None, domain_hint: Optional[str] = None) -> RetrievalResult

Behavior:
- valid domain_hint can bypass route_hybrid detection path
- otherwise falls back to standard routing

## 7) Retrieval Agent Strategy Selection

File:
- src/agents/retrieval_agent.py

Relevant methods:
- execute(query, plan, k=5, force_parallel=False, parent_node_id=None)
- _should_use_parallel(query, force_parallel=False)

Behavior:
- selects sequential or parallel retrieval using domain-confidence policy
- applies chunk cap for bounded downstream latency

## 8) General Finance Lane

File:
- src/services/query_orchestrator.py

Method:
- _run_general_finance_lane(query, timeout_seconds=20)

Behavior:
- one direct LLM call
- no tax workflow planner/retrieval/reasoning/verification stages
- suited for conceptual finance questions

## 9) Configuration Snapshot

File:
- src/config/settings.py

Current key settings:
- DEFAULT_TIMEOUT = 120
- REASONING_MODEL = deepseek/deepseek-r1
- GENERAL_MODEL = qwen/qwen3-30b-a3b
- OPENROUTER_BASE_URL = https://openrouter.ai/api/v1

## 10) Typical Response Shape

Top-level keys:
- query
- blocked
- answer
- confidence
- retrieved_docs_count
- plan_steps
- metadata
- timings
- route

Mode differences:
- fast tax: retrieval_time_ms and llm_time_ms timing fields
- deep tax: planner/retrieval/reasoning/verification stage timing metadata
