"""
File: src/agents/retrieval_agent.py

Purpose:
Executes retrieval steps from planner output.
Handles all retrieval-related plan steps.
Supports both sequential and parallel (Phase D Tier 2) retrieval strategies.

Dependencies:
from typing import List, Dict, Any, Optional
from src.import_map import IRetriever, PlanStep, RetrievalResult, Query, DomainClassifier
from src.provenance.dag_builder import DAGBuilder
from src.retrieval.parallel_retriever import ParallelRetriever (Phase D)

Implements Interface:
None (agent component)

Notes:
- Dependency injection for retriever
- Phase D: Supports both sequential and parallel strategies
- Stateless execution
- Provenance recording
"""

import logging
from typing import Any, Dict, List, Optional

from src.import_map import IRetriever, PlanStep, Query, RetrievalResult, DomainClassifier, Domain
from src.retrieval.parallel_retriever import ParallelRetriever
from src.provenance.dag_builder import DAGBuilder


logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Retrieval agent for executing retrieval plan steps.
    Supports sequential (default) and parallel (Phase D Tier 2) retrieval.
    """

    def __init__(
        self,
        retriever: IRetriever,
        parallel_retriever: Optional[ParallelRetriever] = None,
        domain_classifier: Optional[DomainClassifier] = None,
        dag_builder: Optional[DAGBuilder] = None,
    ) -> None:
        """
        Initialize retrieval agent.
        
        Args:
            retriever: IRetriever instance (sequential, for backward compatibility)
            parallel_retriever: Optional ParallelRetriever (Phase D Tier 2)
            domain_classifier: Optional DomainClassifier (for routing decision)
            dag_builder: Optional DAGBuilder for provenance
        """
        self.retriever = retriever
        self.parallel_retriever = parallel_retriever
        self.domain_classifier = domain_classifier
        self.dag_builder = dag_builder
        self.logger = logging.getLogger(__name__)
        self.use_parallel = parallel_retriever is not None and domain_classifier is not None

    def execute(
        self,
        query: Query | str,
        plan: List[PlanStep],
        k: int = 5,
        force_parallel: bool = False,
        parent_node_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Execute retrieval steps from plan.
        
        Phase D Tier 2: Intelligently choose between sequential and parallel retrieval.
        
        Args:
            query: Query object or string
            plan: List of PlanStep from planner
            k: Number of results per retrieval
            force_parallel: Force parallel retrieval if available
            parent_node_id: Parent node ID for provenance
        
        Returns:
            List of RetrievalResult
        """
        query_text = query.text if isinstance(query, Query) else query
        results: List[RetrievalResult] = []

        # Filter retrieval steps
        retrieval_steps = [
            step for step in plan if step.action_type == "retrieval"
        ]

        if not retrieval_steps:
            self.logger.debug("No retrieval steps in plan")
            return results

        # Phase D Tier 2: Decide retrieval strategy
        use_parallel_strategy = self._should_use_parallel(query_text, force_parallel)
        
        if use_parallel_strategy:
            self.logger.info(f"Using PARALLEL retrieval for query: {query_text[:50]}...")
        else:
            self.logger.info(f"Using SEQUENTIAL retrieval for query: {query_text[:50]}...")

        for step in retrieval_steps:
            try:
                # Phase D Tier 2: Use parallel retriever if applicable
                if use_parallel_strategy:
                    result = self._execute_parallel_retrieval(query_text, k)
                else:
                    result = self._execute_sequential_retrieval(query_text, k)
                
                results.append(result)

                # Record provenance
                if self.dag_builder and parent_node_id:
                    self.dag_builder.record_retrieval(
                        query=query_text,
                        results=[
                            {
                                "chunk_id": chunk.chunk_id,
                                "text": chunk.text[:100],
                                "source": chunk.source.value,
                            }
                            for chunk in result.chunks
                        ],
                        parent_node_id=parent_node_id,
                    )
                    # Phase D: Record retrieval strategy
                    logger.debug(f"Retrieval recorded with strategy: {result.strategy_used}")

            except Exception as exc:
                self.logger.exception(f"Retrieval execution failed: {exc}")
                # Don't re-raise - continue with remaining steps

        return results

    def _should_use_parallel(self, query: str, force_parallel: bool = False) -> bool:
        """
        Determine if parallel retrieval should be used.
        
        Decision logic:
        - If force_parallel=True: use parallel
        - If domain_classifier detects multi-domain: use parallel
        - If domain_classifier confidence is low: use parallel (hedging)
        - Otherwise: use sequential (faster for single-domain, backward compat)
        """
        if not self.use_parallel:
            return False  # ParallelRetriever not available
        
        if force_parallel:
            return True  # Explicitly requested
        
        # Classify query domain
        try:
            domain_class = self.domain_classifier.classify(query)
            
            # Use parallel if multi-domain or low confidence
            is_multi_domain = domain_class.is_multi_domain
            is_low_confidence = domain_class.confidence < 0.7
            
            if is_multi_domain:
                self.logger.debug(f"Multi-domain detected: {[d.value for d in domain_class.domains_detected]}")
                return True
            
            if is_low_confidence:
                self.logger.debug(f"Low confidence ({domain_class.confidence:.2f}), using parallel for coverage")
                return True
            
            return False
        
        except Exception as e:
            self.logger.warning(f"Domain classification failed ({e}), falling back to sequential")
            return False

    def _execute_sequential_retrieval(self, query: str, k: int) -> RetrievalResult:
        """Execute sequential retrieval using standard retriever."""
        return self.retriever.search(query, k=k)

    def _execute_parallel_retrieval(self, query: str, k: int) -> RetrievalResult:
        """Execute parallel retrieval using ParallelRetriever (Phase D Tier 2)."""
        if not self.parallel_retriever:
            self.logger.warning("ParallelRetriever not available, falling back to sequential")
            return self._execute_sequential_retrieval(query, k)
        
        try:
            # Phase D: Use parallel retriever for multi-domain coverage
            result = self.parallel_retriever.search(
                query=query,
                k=k,
                force_parallel=True,
            )
            self.logger.debug(f"Parallel retrieval retrieved {len(result.chunks)} chunks")
            return result
        
        except Exception as e:
            self.logger.exception(f"Parallel retrieval failed: {e}, falling back to sequential")
            return self._execute_sequential_retrieval(query, k)

    def execute_multi_hop(
        self,
        query: Query | str,
        plan: List[PlanStep],
        num_hops: int = 2,
        k: int = 5,
        parent_node_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Execute multi-hop retrieval.
        
        Args:
            query: Query object or string
            plan: List of PlanStep
            num_hops: Number of retrieval hops
            k: Results per hop
            parent_node_id: Parent node ID for provenance
        
        Returns:
            List of RetrievalResult
        """
        query_text = query.text if isinstance(query, Query) else query
        all_results: List[RetrievalResult] = []

        current_query = query_text

        for hop in range(num_hops):
            results = self.execute(
                current_query,
                plan,
                k=k,
                parent_node_id=parent_node_id,
            )

            all_results.extend(results)

            if results and results[0].chunks:
                current_query = " ".join(
                    [chunk.text[:50] for chunk in results[0].chunks[:2]]
                )

        return all_results

    def get_retrieval_stats(
        self,
        results: List[RetrievalResult],
    ) -> Dict[str, Any]:
        """
        Get statistics about retrieval results.
        
        Args:
            results: List of RetrievalResult
        
        Returns:
            Stats dict
        """
        total_chunks = sum(len(r.chunks) for r in results)
        avg_score = (
            sum(sum(r.scores) for r in results) / total_chunks
            if total_chunks > 0
            else 0.0
        )

        sources = {}
        for result in results:
            for chunk in result.chunks:
                source = chunk.source.value
                sources[source] = sources.get(source, 0) + 1

        return {
            "total_results": len(results),
            "total_chunks": total_chunks,
            "avg_score": avg_score,
            "sources": sources,
        }
