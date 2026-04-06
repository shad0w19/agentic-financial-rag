"""
File: src/provenance/dag_builder.py

Purpose:
Helper methods for recording execution steps into provenance graph.
Standardizes node creation across system components.

Dependencies:
from typing import Dict, List, Any, Optional
from src.import_map import IProvenanceGraph

Implements Interface:
None (utility builder)

Notes:
- Dependency injection pattern
- Generic and reusable
- No business logic
"""

import logging
from typing import Any, Dict, List, Optional

from src.import_map import IProvenanceGraph


logger = logging.getLogger(__name__)


class DAGBuilder:
    """
    Helper for recording execution steps into provenance graph.
    """

    def __init__(self, graph: IProvenanceGraph) -> None:
        """
        Initialize DAG builder.
        
        Args:
            graph: IProvenanceGraph instance
        """
        self.graph = graph
        self.logger = logging.getLogger(__name__)
        self.step_counter = 0

    def record_retrieval(
        self,
        query: str,
        results: List[Dict[str, Any]],
        source: str,
        parent_node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record retrieval step.
        
        Args:
            query: Query text
            results: Retrieved results
            source: Data source (personal_tax, corporate_tax, gst)
            parent_node_id: Parent node ID
            metadata: Additional metadata
        
        Returns:
            Node ID
        """
        node_id = self._generate_node_id("retrieval")

        node_metadata = {
            "source": source,
            "result_count": len(results),
            **(metadata or {}),
        }

        self.graph.add_node(
            node_id=node_id,
            node_type="retrieval",
            input_data={"query": query},
            output_data={"results": results},
            metadata=node_metadata,
        )

        if parent_node_id:
            self.graph.add_edge(parent_node_id, node_id)

        return node_id

    def record_calculation(
        self,
        calculation_type: str,
        inputs: Dict[str, Any],
        result: Dict[str, Any],
        parent_node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record calculation step.
        
        Args:
            calculation_type: Type (tax, investment, gst)
            inputs: Input parameters
            result: Calculation result
            parent_node_id: Parent node ID
            metadata: Additional metadata
        
        Returns:
            Node ID
        """
        node_id = self._generate_node_id("calculation")

        node_metadata = {
            "calculation_type": calculation_type,
            **(metadata or {}),
        }

        self.graph.add_node(
            node_id=node_id,
            node_type="calculation",
            input_data=inputs,
            output_data=result,
            metadata=node_metadata,
        )

        if parent_node_id:
            self.graph.add_edge(parent_node_id, node_id)

        return node_id

    def record_security_check(
        self,
        check_type: str,
        passed: bool,
        threat_detected: Optional[str],
        parent_node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record security check step.
        
        Args:
            check_type: Type (input_validation, injection, adversarial)
            passed: Whether check passed
            threat_detected: Threat name if detected
            parent_node_id: Parent node ID
            metadata: Additional metadata
        
        Returns:
            Node ID
        """
        node_id = self._generate_node_id("security")

        node_metadata = {
            "check_type": check_type,
            "passed": passed,
            "threat": threat_detected,
            **(metadata or {}),
        }

        self.graph.add_node(
            node_id=node_id,
            node_type="security_check",
            input_data={"check_type": check_type},
            output_data={"passed": passed, "threat": threat_detected},
            metadata=node_metadata,
        )

        if parent_node_id:
            self.graph.add_edge(parent_node_id, node_id)

        return node_id

    def record_agent_step(
        self,
        agent_name: str,
        action: str,
        reasoning: str,
        output: Dict[str, Any],
        parent_node_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record agent reasoning step.
        
        Args:
            agent_name: Agent name (planner, retrieval, reasoning, verification)
            action: Action taken
            reasoning: Reasoning explanation
            output: Agent output
            parent_node_ids: List of parent node IDs
            metadata: Additional metadata
        
        Returns:
            Node ID
        """
        node_id = self._generate_node_id("agent")

        node_metadata = {
            "agent": agent_name,
            "action": action,
            "reasoning": reasoning,
            **(metadata or {}),
        }

        self.graph.add_node(
            node_id=node_id,
            node_type="agent_step",
            input_data={"agent": agent_name, "action": action},
            output_data=output,
            metadata=node_metadata,
        )

        if parent_node_ids:
            for parent_id in parent_node_ids:
                self.graph.add_edge(parent_id, node_id)

        return node_id

    def record_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Dict[str, Any],
        parent_node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record tool call step.
        
        Args:
            tool_name: Tool name
            tool_input: Tool input
            tool_output: Tool output
            parent_node_id: Parent node ID
            metadata: Additional metadata
        
        Returns:
            Node ID
        """
        node_id = self._generate_node_id("tool")

        node_metadata = {
            "tool": tool_name,
            **(metadata or {}),
        }

        self.graph.add_node(
            node_id=node_id,
            node_type="tool_call",
            input_data=tool_input,
            output_data=tool_output,
            metadata=node_metadata,
        )

        if parent_node_id:
            self.graph.add_edge(parent_node_id, node_id)

        return node_id

    def _generate_node_id(self, prefix: str) -> str:
        """Generate unique node ID."""
        self.step_counter += 1
        return f"{prefix}_{self.step_counter}"
