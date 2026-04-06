"""
File: src/interfaces/provenance.py

Purpose:
Abstract interface for provenance tracking and audit trail management.
Records all agent decisions and reasoning for auditability.

Dependencies:
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from src.core.types import ProvenanceEvent, AgentState

Implements Interface:
IProvenanceTracker, IProvenanceGraph (abstract base classes)

Notes:
- Provenance must be immutable once recorded
- Enables complete DAG reconstruction of reasoning
- Required for regulatory compliance and debugging
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from src.core.types import ProvenanceEvent, AgentState


class IProvenanceTracker(ABC):
    """
    Abstract interface for recording provenance events.
    
    Tracks all agent decisions, tool calls, and reasoning steps
    for audit trail and compliance purposes.
    """

    @abstractmethod
    def record_event(self, event: ProvenanceEvent) -> str:
        """
        Record a provenance event.
        
        Args:
            event: ProvenanceEvent to record
        
        Returns:
            Event ID for reference
        """
        pass

    @abstractmethod
    def record_agent_action(
        self,
        agent_name: str,
        action: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        parent_event_id: Optional[str] = None,
    ) -> str:
        """
        Record an agent action with inputs and outputs.
        
        Args:
            agent_name: Name of agent performing action
            action: Action description
            input_data: Input to the action
            output_data: Output from the action
            parent_event_id: Parent event for DAG tracking
        
        Returns:
            Event ID
        """
        pass

    @abstractmethod
    def record_retrieval(
        self,
        query: str,
        retrieved_chunks: int,
        strategy: str,
        parent_event_id: Optional[str] = None,
    ) -> str:
        """
        Record a retrieval operation.
        
        Args:
            query: Query text
            retrieved_chunks: Number of chunks retrieved
            strategy: Retrieval strategy used
            parent_event_id: Parent event ID
        
        Returns:
            Event ID
        """
        pass

    @abstractmethod
    def record_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
        success: bool,
        parent_event_id: Optional[str] = None,
    ) -> str:
        """
        Record a tool call (tax calculation, etc.).
        
        Args:
            tool_name: Name of tool
            parameters: Tool parameters
            result: Tool result
            success: Whether call succeeded
            parent_event_id: Parent event ID
        
        Returns:
            Event ID
        """
        pass

    @abstractmethod
    def get_event(self, event_id: str) -> Optional[ProvenanceEvent]:
        """
        Retrieve a recorded event.
        
        Args:
            event_id: ID of event to retrieve
        
        Returns:
            ProvenanceEvent or None if not found
        """
        pass

    @abstractmethod
    def get_trace_for_query(self, session_id: str) -> List[ProvenanceEvent]:
        """
        Get complete event trace for a session/query.
        
        Args:
            session_id: Session ID to trace
        
        Returns:
            List of events in execution order
        """
        pass


class IProvenanceGraph(ABC):
    """
    Abstract interface for provenance graph (DAG) operations.
    
    Builds and queries dependency graph of all reasoning steps.
    """

    @abstractmethod
    def build_dag(self, session_id: str) -> Dict[str, Any]:
        """
        Build directed acyclic graph for a session.
        
        Args:
            session_id: Session ID
        
        Returns:
            DAG representation (nodes, edges, metadata)
        """
        pass

    @abstractmethod
    def get_path_to_result(
        self,
        session_id: str,
        result_event_id: str,
    ) -> List[str]:
        """
        Get path of events leading to a result.
        
        Traces backwards through dependencies to show
        complete reasoning chain.
        
        Args:
            session_id: Session ID
            result_event_id: Event ID of final result
        
        Returns:
            List of event IDs in path
        """
        pass

    @abstractmethod
    def get_event_dependencies(self, event_id: str) -> List[str]:
        """
        Get all events this event depends on.
        
        Args:
            event_id: Event to query
        
        Returns:
            List of dependent event IDs
        """
        pass

    @abstractmethod
    def validate_dag_integrity(self, session_id: str) -> bool:
        """
        Validate DAG integrity (no cycles, consistency).
        
        Args:
            session_id: Session to validate
        
        Returns:
            True if DAG is valid
        """
        pass

    @abstractmethod
    def export_dag(self, session_id: str, format: str = "json") -> str:
        """
        Export DAG in specified format.
        
        Args:
            session_id: Session ID
            format: Export format ("json", "dot", "yaml")
        
        Returns:
            Exported DAG string
        """
        pass

    @abstractmethod
    def get_decision_points(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all decision points in the reasoning chain.
        
        Args:
            session_id: Session ID
        
        Returns:
            List of decision point metadata
        """
        pass
