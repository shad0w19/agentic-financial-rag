"""
File: src/orchestration/graph_state.py

Purpose:
Shared state object passed across all agents.
Data container for workflow state.

Dependencies:
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.import_map import PlanStep, RetrievalResult

Implements Interface:
None (data container)

Notes:
- Immutable-style dataclass
- Passed through entire workflow
- No logic, only data
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.import_map import PlanStep, RetrievalResult


@dataclass
class GraphState:
    """
    Shared state object for workflow execution.
    Passed through all agents and orchestration nodes.
    """

    query: str
    validated_query: Optional[str] = None
    plan: Optional[List[PlanStep]] = None
    retrieved_docs: Optional[List[RetrievalResult]] = None
    answer: Optional[str] = None
    verification: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_blocked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to dictionary.
        
        Returns:
            State as dict
        """
        return {
            "query": self.query,
            "validated_query": self.validated_query,
            "plan": self.plan,
            "retrieved_docs": self.retrieved_docs,
            "answer": self.answer,
            "verification": self.verification,
            "metadata": self.metadata,
        }

    def get_status(self) -> Dict[str, bool]:
        """
        Get completion status of workflow stages.
        
        Returns:
            Status dict
        """
        return {
            "query_validated": self.validated_query is not None,
            "plan_created": self.plan is not None,
            "docs_retrieved": self.retrieved_docs is not None,
            "answer_generated": self.answer is not None,
            "answer_verified": self.verification is not None,
        }

    def is_complete(self) -> bool:
        """
        Check if workflow is complete.
        
        Returns:
            True if all stages complete
        """
        status = self.get_status()
        return all(status.values())

    def get_completion_percentage(self) -> float:
        """
        Get workflow completion percentage.
        
        Returns:
            Percentage (0.0-1.0)
        """
        status = self.get_status()
        completed = sum(1 for v in status.values() if v)
        total = len(status)
        return completed / total if total > 0 else 0.0
