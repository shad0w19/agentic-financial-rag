"""
File: src/provenance/provenance_graph.py

Purpose:
DAG-based provenance tracking for execution traces.
Implements IProvenanceGraph interface.

Dependencies:
from typing import Dict, List, Any, Optional
from src.import_map import IProvenanceGraph

Implements Interface:
IProvenanceGraph

Notes:
- In-memory adjacency list implementation
- No external libraries (no networkx)
- Generic DAG for all system components
"""

import logging
from typing import Any, Dict, List, Optional

from src.import_map import IProvenanceGraph


logger = logging.getLogger(__name__)


class ProvenanceGraph(IProvenanceGraph):
    """
    DAG-based provenance graph for execution traces.
    """

    def __init__(self) -> None:
        """Initialize provenance graph."""
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, List[str]] = {}
        self.reverse_edges: Dict[str, List[str]] = {}
        self.logger = logging.getLogger(__name__)

    def add_node(
        self,
        node_id: str,
        node_type: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """
        Add node to graph.
        
        Args:
            node_id: Unique node identifier
            node_type: Type of node (agent, retrieval, tool, etc.)
            input_data: Input to this node
            output_data: Output from this node
            metadata: Additional metadata
        """
        self.nodes[node_id] = {
            "type": node_type,
            "input": input_data,
            "output": output_data,
            "metadata": metadata or {},
        }
        if node_id not in self.edges:
            self.edges[node_id] = []
        if node_id not in self.reverse_edges:
            self.reverse_edges[node_id] = []

    def add_edge(self, parent_id: str, child_id: str) -> None:
        """
        Add edge (parent → child).
        
        Args:
            parent_id: Parent node ID
            child_id: Child node ID
        """
        if parent_id not in self.edges:
            self.edges[parent_id] = []
        if child_id not in self.reverse_edges:
            self.reverse_edges[child_id] = []

        if child_id not in self.edges[parent_id]:
            self.edges[parent_id].append(child_id)
        if parent_id not in self.reverse_edges[child_id]:
            self.reverse_edges[child_id].append(parent_id)

    def build_dag(self, session_id: str) -> Dict[str, Any]:
        """
        Build DAG representation.
        
        Args:
            session_id: Session identifier
        
        Returns:
            DAG dict with nodes and edges
        """
        return {
            "session_id": session_id,
            "nodes": self.nodes,
            "edges": self.edges,
            "node_count": len(self.nodes),
            "edge_count": sum(len(v) for v in self.edges.values()),
        }

    def get_path_to_result(
        self,
        session_id: str,
        result_node_id: str,
    ) -> List[str]:
        """
        Get path from root to result node.
        
        Args:
            session_id: Session identifier
            result_node_id: Result node ID
        
        Returns:
            List of node IDs in path
        """
        path: List[str] = []
        current = result_node_id

        while current:
            path.insert(0, current)
            parents = self.reverse_edges.get(current, [])
            current = parents[0] if parents else None

        return path

    def get_event_dependencies(self, node_id: str) -> List[str]:
        """
        Get all nodes this node depends on.
        
        Args:
            node_id: Node identifier
        
        Returns:
            List of dependent node IDs
        """
        return self.reverse_edges.get(node_id, [])

    def validate_dag_integrity(self, session_id: str) -> bool:
        """
        Validate DAG has no cycles.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if valid DAG
        """
        visited: set = set()
        rec_stack: set = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for child in self.edges.get(node, []):
                if child not in visited:
                    if has_cycle(child):
                        return True
                elif child in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in self.nodes:
            if node not in visited:
                if has_cycle(node):
                    return False

        return True

    def export_dag(self, session_id: str, format: str = "json") -> str:
        """
        Export DAG in specified format.
        
        Args:
            session_id: Session identifier
            format: Export format (json, dot, yaml)
        
        Returns:
            Exported DAG string
        """
        import json

        dag = self.build_dag(session_id)

        if format == "json":
            return json.dumps(dag, indent=2)
        elif format == "dot":
            return self._to_dot_format()
        else:
            return json.dumps(dag, indent=2)

    def get_decision_points(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all decision points in reasoning chain.
        
        Args:
            session_id: Session identifier
        
        Returns:
            List of decision point metadata
        """
        decisions: List[Dict[str, Any]] = []

        for node_id, node_data in self.nodes.items():
            if node_data.get("type") in ["decision", "branch", "choice"]:
                decisions.append(
                    {
                        "node_id": node_id,
                        "type": node_data["type"],
                        "metadata": node_data.get("metadata", {}),
                    }
                )

        return decisions

    def _to_dot_format(self) -> str:
        """Convert to Graphviz DOT format."""
        lines = ["digraph {"]

        for node_id, node_data in self.nodes.items():
            label = f"{node_id}\\n({node_data['type']})"
            lines.append(f'  "{node_id}" [label="{label}"];')

        for parent, children in self.edges.items():
            for child in children:
                lines.append(f'  "{parent}" -> "{child}";')

        lines.append("}")
        return "\n".join(lines)
