# Financial Agentic RAG - System Logic Guide

## Purpose

This document describes the current runtime execution logic from user query to final response.

## Entry Points

- UI entrypoint: app_2.py
- Runtime entrypoint: QueryOrchestrator.run_query in src/services/query_orchestrator.py

Current run_query signature in use:
- run_query(query, timeout_seconds=120, mode="fast", chat_history=None)

Default mode behavior:
- mode defaults to fast unless UI explicitly sets deep

## End-to-End Execution Flow

1. Query normalization and empty-query guard.
2. Optional clarification-follow-up expansion.
3. Optional prepend of recent chat_history context to execution query.
4. Mandatory security gate on user query.
5. Cache lookup on execution_query (after optional chat-history prepend).
6. Intent classification and lane selection.
7. Tax rescue heuristic for likely tax queries.
8. Domain classification for tax lane.
9. Lane execution.
10. Metadata enrichment and cache writeback.

## Lane Routing Logic

Possible lanes:
- trivial
- out_of_scope
- ambiguous_in_domain
- general_finance
- tax_rag

Special routes:
- blocked (security block)
- error (execution failure)

### trivial lane
- returns immediate static response

### out_of_scope lane
- returns finance-only boundary response

### ambiguous_in_domain lane
- returns clarification prompt

### general_finance lane
- direct LLM answer
- bypasses tax workflow stages (planner/retrieval/reasoning/verification pipeline)

### tax_rag lane
Dispatcher behavior:
- mode="fast" -> _run_fast_tax_lane
- mode="deep" -> full workflow path in _run_tax_rag_lane

## Fast Tax Path (mode="fast")

Location:
- _run_fast_tax_lane in src/services/query_orchestrator.py

Steps:
1. Ensure workflow/router initialized.
2. Federated retrieval with top_k=3 and optional domain_hint.
3. Compact context to bounded prompt size.
4. Single LLM call for final answer.
5. Return normalized response.

Characteristics:
- no GraphState orchestration path
- no planner stage
- no verification stage
- confidence currently fixed to 0.70 in this path

Fallback:
- retrieval/router failure falls back to deep path

## Deep Tax Path (mode="deep")

Location:
- _run_tax_rag_lane(mode="deep") in src/services/query_orchestrator.py
- workflow execution in src/orchestration/workflow.py

Workflow order:
1. security node
2. planner node
3. retrieval node
4. reasoning node
5. verification node

Control model:
- per-stage timeout budgets
- degraded flags and timeout_stage metadata
- planner fallback if planning fails/times out

Post-workflow steps:
- tax grounding guard
- quality assessment
- confidence composition
- gating/override decisions

## Retrieval Logic

### Federated router
File:
- src/retrieval/federated_router.py

Key behavior:
- search(query, k, filters, domain_hint)
- if valid domain_hint is provided and indexed, router skips route_hybrid detection path
- otherwise uses hybrid routing logic

### Retrieval strategy inside agent
File:
- src/agents/retrieval_agent.py

Key behavior:
- chooses parallel vs sequential strategy with confidence-based policy
- can call parallel retriever for multi-domain coverage
- sequential fallback on failures
- caps per-result chunk count for bounded downstream latency

## Cache Logic (Phase D Tier 1)

File:
- src/services/response_cache.py

Behavior:
- query normalization
- LRU storage with TTL
- checked before lane execution
- writeback for successful responses

## Confidence and Quality Logic (Phase C)

Files:
- src/confidence/quality_assessment.py
- src/confidence/confidence_composer.py

Behavior:
- evaluate quality signals
- compose confidence level
- apply gating action (allow/caveat/clarify/admit uncertainty)

## UI Integration Details

File:
- app_2.py

Runtime behavior:
- deep mode toggle controls mode parameter
- mode is passed to run_query as "deep" or "fast"
- st.session_state.messages passed as chat_history

## Observability Metadata

Common response fields:
- answer
- confidence
- route
- metadata
- timings
- plan_steps

Deep mode metadata includes stage timings and degradation markers.
Fast mode includes retrieval_time_ms and llm_time_ms timing fields (without deep workflow stage timing breakdown).
