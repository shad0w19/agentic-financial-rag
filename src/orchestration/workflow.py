"""
File: src/orchestration/workflow.py

Purpose:
Main orchestrator executing full agent pipeline.
Coordinates all agents through deterministic workflow.

Dependencies:
from src.import_map import ISecurityGatekeeper
from src.orchestration.graph_state import GraphState
from src.orchestration.nodes import (
    run_security, run_planner, run_retrieval,
    run_reasoning, run_verification
)
from src.agents.planner_agent import PlannerAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.verification_agent import VerificationAgent

Implements Interface:
None (orchestrator)

Notes:
- Deterministic sequential flow
- No async, no retries, no branching
- Full dependency injection
"""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

from src.agents.planner_agent import PlannerAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.verification_agent import VerificationAgent
from src.import_map import ISecurityGatekeeper
from src.orchestration.graph_state import GraphState
from src.orchestration.nodes import (
    run_planner,
    run_reasoning,
    run_retrieval,
    run_security,
    run_verification,
)


logger = logging.getLogger(__name__)


class AgentWorkflow:
    """
    Main orchestrator for agent pipeline execution.
    """

    DEFAULT_STAGE_BUDGETS = {
        "planner": 10.0,
        "retrieval": 10.0,
        "reasoning": 60.0,
        "verification": 20.0,
    }
    
    # Phase 2: Planner stabilization metrics
    _planner_timeout_count = 0
    _planner_fallback_count = 0
    _planner_total_runs = 0

    def __init__(
        self,
        gatekeeper: ISecurityGatekeeper,
        planner: PlannerAgent,
        retriever: RetrievalAgent,
        reasoner: ReasoningAgent,
        verifier: VerificationAgent,
    ) -> None:
        """
        Initialize workflow with all dependencies.
        
        Args:
            gatekeeper: ISecurityGatekeeper instance
            planner: PlannerAgent instance
            retriever: RetrievalAgent instance
            reasoner: ReasoningAgent instance
            verifier: VerificationAgent instance
        """
        self.gatekeeper = gatekeeper
        self.planner = planner
        self.retriever = retriever
        self.reasoner = reasoner
        self.verifier = verifier
        self.logger = logging.getLogger(__name__)

    def _run_stage_with_timeout(
        self,
        state: GraphState,
        stage_name: str,
        stage_fn,
        timeout_seconds: float,
        degrade_on_timeout: bool = False,
    ) -> GraphState:
        started = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(stage_fn, state)
        try:
            updated_state = future.result(timeout=timeout_seconds)
            return updated_state
        except TimeoutError:
            future.cancel()
            state.metadata["timeout_stage"] = stage_name
            state.metadata.setdefault("degraded_flags", []).append(f"{stage_name}_timeout")
            if degrade_on_timeout:
                state.metadata["verification_degraded"] = True
            self.logger.warning("Stage '%s' timed out after %.1fs", stage_name, timeout_seconds)
            return state
        except Exception as exc:
            state.metadata.setdefault("degraded_flags", []).append(f"{stage_name}_error")
            state.metadata[f"{stage_name}_error"] = str(exc)
            self.logger.exception("Stage '%s' failed: %s", stage_name, exc)
            return state
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            stage_ms = (time.perf_counter() - started) * 1000
            state.metadata.setdefault("stage_timings_ms", {})[stage_name] = stage_ms

    def run(
        self,
        query: str,
        initial_metadata: dict | None = None,
        stage_budgets: dict | None = None,
        retrieval_k: int = 5,
    ) -> GraphState:
        """
        Execute full agent pipeline.
        
        Args:
            query: User query string
        
        Returns:
            Final GraphState with all results
        """
        self.logger.info(f"Starting workflow for query: {query[:50]}...")
        workflow_started = time.perf_counter()

        # Initialize state
        state = GraphState(query=query)
        if initial_metadata:
            state.metadata.update(initial_metadata)
        state.metadata.setdefault("query_id", str(uuid.uuid4()))
        state.metadata.setdefault("stage_timings_ms", {})
        state.metadata.setdefault("degraded_flags", [])

        budgets = dict(self.DEFAULT_STAGE_BUDGETS)
        if stage_budgets:
            budgets.update(stage_budgets)

        # Step 1: Security validation
        self.logger.debug("Running security check...")
        state = run_security(state, self.gatekeeper)

        if state.metadata.get("security_blocked"):
            state.is_blocked = True
            self.logger.warning("Query blocked by security gatekeeper")
            state.answer = "Query blocked due to security concerns."
            return state

        # Step 2: Planning
        self.logger.debug("Running planner...")
        self.__class__._planner_total_runs += 1
        
        state = self._run_stage_with_timeout(
            state,
            "planner",
            lambda s: run_planner(s, self.planner),
            timeout_seconds=float(budgets["planner"]),
        )

        if state.metadata.get("timeout_stage") == "planner":
            self.__class__._planner_timeout_count += 1
            state.metadata["planner_timeout_count"] = self.__class__._planner_timeout_count
            state.metadata["planner_timeout_rate"] = (
                self.__class__._planner_timeout_count / max(1, self.__class__._planner_total_runs)
            )
        
        if state.metadata.get("planning_error") or not state.plan:
            try:
                self.__class__._planner_fallback_count += 1
                fallback_query = state.validated_query or state.query
                state.plan = self.planner._generate_fallback_plan(fallback_query)
                
                # Phase 2: Validate fallback plan quality
                fallback_quality = self._validate_fallback_plan_quality(state.plan, fallback_query)
                state.metadata["planner_fallback"] = True
                state.metadata["planner_fallback_quality"] = fallback_quality
                state.metadata["planner_fallback_count"] = self.__class__._planner_fallback_count
                state.metadata["planner_fallback_rate"] = (
                    self.__class__._planner_fallback_count / max(1, self.__class__._planner_total_runs)
                )
                state.metadata.setdefault("degraded_flags", []).append("planner_fallback")
                state.metadata.pop("planning_error", None)
                
                if not fallback_quality.get("is_valid"):
                    self.logger.warning(
                        "Planner fallback applied but quality check failed: %s",
                        fallback_quality.get("issues", [])
                    )
                    state.metadata.setdefault("degraded_flags", []).append("planner_fallback_low_quality")
                else:
                    self.logger.warning("Planner fallback applied with acceptable quality")
            except Exception as exc:
                self.logger.exception("Planner fallback failed: %s", exc)
                state.answer = "Planning error: unable to build fallback plan."
                state.metadata["planning_error"] = str(exc)
                return state

        if state.metadata.get("planning_error"):
            self.logger.warning("Planning failed")
            state.answer = f"Planning error: {state.metadata.get('planning_error')}"
            return state

        # Step 3: Retrieval
        self.logger.debug("Running retrieval...")
        state = self._run_stage_with_timeout(
            state,
            "retrieval",
            lambda s: run_retrieval(s, self.retriever, retrieval_k=retrieval_k),
            timeout_seconds=float(budgets["retrieval"]),
        )

        if state.metadata.get("retrieval_error"):
            self.logger.warning("Retrieval failed")
            state.answer = "Error during document retrieval."
            return state

        retrieved_docs = state.retrieved_docs or []
        total_chunks = sum(len(r.chunks) for r in retrieved_docs)
        total_chars = sum(
            len((chunk.text or "").strip())
            for result in retrieved_docs
            for chunk in result.chunks
        )
        if total_chunks == 0 or total_chars < 160:
            state.metadata["retrieval_weak"] = True
            state.metadata.setdefault("degraded_flags", []).append("weak_retrieval_clarification")
            state.answer = (
                "I need a bit more context to answer accurately. "
                "Please specify whether this is about personal income tax, GST, or corporate tax, and share your scenario."
            )
            self.logger.info("Weak retrieval detected, returning clarification early")
            state.metadata.setdefault("stage_timings_ms", {})["total"] = (time.perf_counter() - workflow_started) * 1000
            return state

        # Step 4: Reasoning
        self.logger.debug("Running reasoning...")
        state = self._run_stage_with_timeout(
            state,
            "reasoning",
            lambda s: run_reasoning(s, self.reasoner),
            timeout_seconds=float(budgets["reasoning"]),
        )

        if state.metadata.get("reasoning_error") or "reasoning_timeout" in state.metadata.get("degraded_flags", []):
            self.logger.warning("Reasoning failed or timed out")
            state.answer = "Answer generation timed out. Please try a more specific question."
            return state

        # Step 5: Verification
        self.logger.debug("Running verification...")
        state = self._run_stage_with_timeout(
            state,
            "verification",
            lambda s: run_verification(s, self.verifier),
            timeout_seconds=float(budgets["verification"]),
            degrade_on_timeout=True,
        )

        if state.metadata.get("verification_error"):
            self.logger.warning("Verification failed")
            state.metadata["verification_degraded"] = True
            state.metadata.setdefault("degraded_flags", []).append("verification_failed")

        state.metadata.setdefault("stage_timings_ms", {})["total"] = (time.perf_counter() - workflow_started) * 1000
        self.logger.info(
            "Workflow completed | query_id=%s | timings_ms=%s | timeout_stage=%s | degraded=%s",
            state.metadata.get("query_id"),
            state.metadata.get("stage_timings_ms"),
            state.metadata.get("timeout_stage"),
            state.metadata.get("degraded_flags", []),
        )

        self.logger.info("Workflow completed successfully")
        return state

    def _validate_fallback_plan_quality(self, plan: list, query: str) -> dict:
        """
        Validate fallback plan quality to ensure it's not degenerate.
        
        Phase 2 stabilization: Ensure fallback plans include retrieval,
        reasoning, and are specific to the query type (not generic).
        
        Args:
            plan: List of PlanStep objects
            query: Original query string
        
        Returns:
            Dict with is_valid (bool) and issues (List[str])
        """
        issues: list = []
        
        # Check for essential steps
        action_types = [step.action_type for step in plan]
        
        if "retrieval" not in action_types:
            issues.append("Missing retrieval step")
        
        if "reasoning" not in action_types:
            issues.append("Missing reasoning step")
        
        # Check for minimum plan length (security + retrieval + reasoning)
        if len(plan) < 3:
            issues.append(f"Plan too short: {len(plan)} steps (expected >= 3)")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "plan_length": len(plan),
            "has_retrieval": "retrieval" in action_types,
            "has_reasoning": "reasoning" in action_types,
        }
    
    @classmethod
    def get_planner_metrics(cls) -> dict:
        """
        Get Phase 2 planner stabilization metrics.
        
        Returns:
            Dict with timeout and fallback counts/rates
        """
        total = max(1, cls._planner_total_runs)
        return {
            "total_runs": cls._planner_total_runs,
            "timeout_count": cls._planner_timeout_count,
            "fallback_count": cls._planner_fallback_count,
            "timeout_rate": cls._planner_timeout_count / total,
            "fallback_rate": cls._planner_fallback_count / total,
        }

    def get_workflow_status(self, state: GraphState) -> dict:
        """
        Get workflow execution status.
        
        Args:
            state: GraphState to check
        
        Returns:
            Status dict
        """
        return {
            "completion": state.get_completion_percentage(),
            "stages": state.get_status(),
            "has_answer": state.answer is not None,
            "is_verified": state.verification is not None,
        }
