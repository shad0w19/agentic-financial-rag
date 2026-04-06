"""
File: src/orchestration/nodes.py

Purpose:
Execution nodes for each pipeline step.
Pure functions that update GraphState.

Dependencies:
from src.import_map import ISecurityGatekeeper, Query
from src.orchestration.graph_state import GraphState
from src.agents.planner_agent import PlannerAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.verification_agent import VerificationAgent

Implements Interface:
None (orchestration functions)

Notes:
- Pure functions (no class)
- Each takes state + dependency
- Returns updated state
"""

import logging
from typing import Any, Callable

from src.agents.planner_agent import PlannerAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.verification_agent import VerificationAgent
from src.import_map import ISecurityGatekeeper, Query
from src.orchestration.graph_state import GraphState


logger = logging.getLogger(__name__)


def run_security(
    state: GraphState,
    gatekeeper: ISecurityGatekeeper,
) -> GraphState:
    """
    Run security validation node.
    
    OPTIMIZATION: Skip if orchestrator already prechecked (gatekeeper_stage marker).
    This is defense-in-depth; orchestrator gate is primary, workflow gate is backup.
    
    Args:
        state: Current GraphState
        gatekeeper: ISecurityGatekeeper instance
    
    Returns:
        Updated GraphState
    """
    # Check if orchestrator already validated (QueryOrchestrator.run_query)
    precheck_stage = state.metadata.get("gatekeeper_stage")
    if precheck_stage == "orchestrator_pre_routing_passed":
        logger.debug("Security already checked by orchestrator, skipping workflow-level check")
        state.validated_query = state.query
        return state
    
    try:
        query_obj = Query(text=state.query)
        passed, result = gatekeeper.check_query(query_obj)

        if passed:
            state.validated_query = state.query
            logger.info("Security check passed (workflow-level)")
        else:
            logger.warning(f"Security check failed: {result.threat_detected}")
            state.metadata["security_blocked"] = True
            if hasattr(result, 'threat_detected'):
                state.metadata["threat"] = result.threat_detected

    except Exception as e:
        logger.error(f"Security check error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        state.metadata["security_error"] = str(e)
        # On error, we should block the query for safety
        state.metadata["security_blocked"] = True

    return state


def run_planner(
    state: GraphState,
    planner: PlannerAgent,
) -> GraphState:
    """
    Run planner node.
    
    Args:
        state: Current GraphState
        planner: PlannerAgent instance
    
    Returns:
        Updated GraphState
    """
    try:
        if not state.validated_query:
            logger.warning("Query not validated, using original")
            query = state.query
        else:
            query = state.validated_query

        plan = planner.plan(query)
        state.plan = plan
        logger.info(f"Generated plan with {len(plan)} steps")

    except Exception as e:
        logger.error(f"Planning error: {e}")
        state.metadata["planning_error"] = str(e)

    return state


def run_retrieval(
    state: GraphState,
    retriever: RetrievalAgent,
) -> GraphState:
    """
    Run retrieval node.
    
    Args:
        state: Current GraphState
        retriever: RetrievalAgent instance
    
    Returns:
        Updated GraphState
    """
    try:
        if not state.plan:
            logger.warning("No plan available for retrieval")
            return state

        query = state.validated_query or state.query
        results = retriever.execute(query, state.plan, k=5)
        state.retrieved_docs = results

        total_chunks = sum(len(r.chunks) for r in results)
        logger.info(f"Retrieved {total_chunks} chunks from {len(results)} sources")

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        state.metadata["retrieval_error"] = str(e)

    return state


def run_reasoning(
    state: GraphState,
    reasoner: ReasoningAgent,
) -> GraphState:
    """
    Run reasoning node.
    
    Args:
        state: Current GraphState
        reasoner: ReasoningAgent instance
    
    Returns:
        Updated GraphState
    """
    try:
        if not state.retrieved_docs:
            logger.warning("No retrieved documents for reasoning")
            state.answer = "No relevant information found."
            return state

        query = state.validated_query or state.query
        answer = reasoner.reason(
            query=query,
            retrieved_docs=state.retrieved_docs,
            plan=state.plan or [],
        )
        state.answer = answer
        logger.info("Generated answer")

    except Exception as e:
        logger.error(f"Reasoning error: {e}")
        state.metadata["reasoning_error"] = str(e)
        state.answer = "Error generating answer."

    return state


def run_verification(
    state: GraphState,
    verifier: VerificationAgent,
) -> GraphState:
    """
    Run verification node.
    
    Args:
        state: Current GraphState
        verifier: VerificationAgent instance
    
    Returns:
        Updated GraphState
    """
    try:
        if not state.answer:
            logger.warning("No answer to verify")
            return state

        result = verifier.verify(
            answer=state.answer,
            retrieved_docs=state.retrieved_docs or [],
            plan=state.plan or [],
        )
        state.verification = result

        if result["is_valid"]:
            logger.info(f"Answer verified (confidence: {result['confidence']:.2f})")
        else:
            logger.warning(f"Answer verification issues: {result['issues']}")

    except Exception as e:
        logger.error(f"Verification error: {e}")
        state.metadata["verification_error"] = str(e)

    return state
