"""
File: src/evaluation/ragas_eval.py

Purpose:
Lightweight evaluation module for RAG performance.
Computes relevance, faithfulness, and context precision.

Dependencies:
from typing import List, Dict, Any
from src.orchestration.workflow import AgentWorkflow

Implements Interface:
None (evaluation utility)

Notes:
- No external libraries
- Simple scoring logic
- RAGAS-inspired metrics
"""

import logging
from typing import Any, Dict, List

from src.orchestration.workflow import AgentWorkflow


logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluator for RAG system performance.
    """

    def __init__(self, workflow: AgentWorkflow) -> None:
        """
        Initialize evaluator.
        
        Args:
            workflow: AgentWorkflow instance
        """
        self.workflow = workflow
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        dataset: List[Dict[str, str]],
    ) -> Dict[str, float]:
        """
        Evaluate workflow on dataset.
        
        Args:
            dataset: List of {"query": "...", "expected": "..."}
        
        Returns:
            Dict with avg_relevance, avg_faithfulness, avg_context_precision
        """
        if not dataset:
            return {
                "avg_relevance": 0.0,
                "avg_faithfulness": 0.0,
                "avg_context_precision": 0.0,
            }

        relevance_scores = []
        faithfulness_scores = []
        context_precision_scores = []

        for sample in dataset:
            query = sample.get("query", "")
            expected = sample.get("expected", "")

            if not query:
                continue

            # Run workflow
            state = self.workflow.run(query)

            # Compute metrics
            relevance = self._compute_relevance(query, state.answer or "")
            faithfulness = self._compute_faithfulness(
                state.answer or "", state.retrieved_docs or []
            )
            context_precision = self._compute_context_precision(
                state.retrieved_docs or [], state.answer or ""
            )

            relevance_scores.append(relevance)
            faithfulness_scores.append(faithfulness)
            context_precision_scores.append(context_precision)

        # Compute averages
        avg_relevance = (
            sum(relevance_scores) / len(relevance_scores)
            if relevance_scores
            else 0.0
        )
        avg_faithfulness = (
            sum(faithfulness_scores) / len(faithfulness_scores)
            if faithfulness_scores
            else 0.0
        )
        avg_context_precision = (
            sum(context_precision_scores) / len(context_precision_scores)
            if context_precision_scores
            else 0.0
        )

        return {
            "avg_relevance": avg_relevance,
            "avg_faithfulness": avg_faithfulness,
            "avg_context_precision": avg_context_precision,
        }

    def _compute_relevance(self, query: str, answer: str) -> float:
        """
        Compute relevance score (query vs answer overlap).
        
        Args:
            query: Query text
            answer: Answer text
        
        Returns:
            Relevance score (0-1)
        """
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        if not query_words:
            return 0.0

        overlap = len(query_words & answer_words)
        return min(overlap / len(query_words), 1.0)

    def _compute_faithfulness(self, answer: str, retrieved_docs: List) -> float:
        """
        Compute faithfulness score (answer vs docs overlap).
        
        Args:
            answer: Answer text
            retrieved_docs: List of RetrievalResult
        
        Returns:
            Faithfulness score (0-1)
        """
        if not retrieved_docs:
            return 0.0

        # Collect doc text
        doc_text = " ".join(
            [chunk.text for r in retrieved_docs for chunk in r.chunks]
        )

        if not doc_text:
            return 0.0

        answer_words = set(answer.lower().split())
        doc_words = set(doc_text.lower().split())

        if not answer_words:
            return 0.0

        overlap = len(answer_words & doc_words)
        return min(overlap / len(answer_words), 1.0)

    def _compute_context_precision(
        self,
        retrieved_docs: List,
        answer: str,
    ) -> float:
        """
        Compute context precision (docs vs answer usage).
        
        Args:
            retrieved_docs: List of RetrievalResult
            answer: Answer text
        
        Returns:
            Context precision score (0-1)
        """
        if not retrieved_docs:
            return 0.0

        total_chunks = sum(len(r.chunks) for r in retrieved_docs)
        if total_chunks == 0:
            return 0.0

        # Count chunks referenced in answer
        answer_lower = answer.lower()
        referenced_chunks = 0

        for result in retrieved_docs:
            for chunk in result.chunks:
                chunk_text = chunk.text.lower()
                if any(
                    word in answer_lower
                    for word in chunk_text.split()[:5]
                ):
                    referenced_chunks += 1

        return min(referenced_chunks / total_chunks, 1.0)

    def get_evaluation_report(
        self,
        metrics: Dict[str, float],
    ) -> str:
        """
        Get human-readable evaluation report.
        
        Args:
            metrics: Metrics dict from evaluate()
        
        Returns:
            Report string
        """
        return (
            f"RAG Evaluation Report:\n"
            f"  Relevance: {metrics['avg_relevance']:.3f}\n"
            f"  Faithfulness: {metrics['avg_faithfulness']:.3f}\n"
            f"  Context Precision: {metrics['avg_context_precision']:.3f}"
        )
