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

    def run(self, query: str) -> GraphState:
        """
        Execute full agent pipeline.
        
        Args:
            query: User query string
        
        Returns:
            Final GraphState with all results
        """
        self.logger.info(f"Starting workflow for query: {query[:50]}...")

        # Initialize state
        state = GraphState(query=query)

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
        state = run_planner(state, self.planner)

        if state.metadata.get("planning_error"):
            self.logger.warning("Planning failed")
            state.answer = f"Planning error: {state.metadata.get('planning_error')}"
            return state

        # Step 3: Retrieval
        self.logger.debug("Running retrieval...")
        state = run_retrieval(state, self.retriever)

        if state.metadata.get("retrieval_error"):
            self.logger.warning("Retrieval failed")
            state.answer = "Error during document retrieval."
            return state

        # Step 4: Reasoning
        self.logger.debug("Running reasoning...")
        state = run_reasoning(state, self.reasoner)

        if state.metadata.get("reasoning_error"):
            self.logger.warning("Reasoning failed")
            state.answer = "Error during reasoning."
            return state

        # Step 5: Verification
        self.logger.debug("Running verification...")
        state = run_verification(state, self.verifier)

        if state.metadata.get("verification_error"):
            self.logger.warning("Verification failed")

        self.logger.info("Workflow completed successfully")
        return state

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
