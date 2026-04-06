"""
File: src/retrieval/reranker.py

Purpose:
Reranks retrieval results using cross-encoder model.
Improves relevance of top-k results.

Dependencies:
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from src.import_map import RetrievalResult, DocumentChunk

Implements Interface:
None (utility)

Notes:
- Uses cross-encoder for pairwise relevance scoring
- Reranks results in-place
- Supports batch reranking
"""

import logging
from typing import List, Tuple

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

from sentence_transformers import CrossEncoder

from src.import_map import DocumentChunk, RetrievalResult


logger = logging.getLogger(__name__)


class Reranker:
    """
    Reranks retrieval results using cross-encoder.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        """
        Initialize reranker.
        
        Args:
            model_name: HuggingFace cross-encoder model
        """
        self.model_name = model_name
        self.model: CrossEncoder | None = None
        self.logger = logging.getLogger(__name__)
        self._load_model()

    def _load_model(self) -> None:
        """Load cross-encoder model with GPU support."""
        try:
            # Determine device
            device = "cuda" if CUDA_AVAILABLE else "cpu"
            
            self.model = CrossEncoder(self.model_name, device=device)
            
            if device == "cuda":
                self.logger.info(f"Loaded reranker on GPU (CUDA): {self.model_name}")
            else:
                self.logger.info(f"Loaded reranker on CPU: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load reranker: {e}")
            raise

    def rerank(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int | None = None,
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """
        Rerank chunks by relevance to query.
        
        Args:
            query: Query text
            chunks: List of DocumentChunk
            top_k: Return top-k (None = all)
        
        Returns:
            Tuple of (reranked_chunks, scores)
        """
        if not chunks:
            return [], []

        if not self.model:
            raise RuntimeError("Model not loaded")

        try:
            # Prepare pairs
            pairs = [(query, chunk.text) for chunk in chunks]

            # Score
            scores = self.model.predict(pairs)

            # Sort by score
            ranked = sorted(
                zip(chunks, scores),
                key=lambda x: x[1],
                reverse=True,
            )

            if top_k:
                ranked = ranked[:top_k]

            reranked_chunks = [chunk for chunk, _ in ranked]
            reranked_scores = [float(score) for _, score in ranked]

            return reranked_chunks, reranked_scores

        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            raise

    def rerank_result(
        self,
        query: str,
        result: RetrievalResult,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """
        Rerank a RetrievalResult.
        
        Args:
            query: Query text
            result: RetrievalResult to rerank
            top_k: Return top-k
        
        Returns:
            Reranked RetrievalResult
        """
        reranked_chunks, reranked_scores = self.rerank(
            query, result.chunks, top_k
        )

        return RetrievalResult(
            chunks=reranked_chunks,
            strategy_used=result.strategy_used,
            scores=reranked_scores,
            query_used=query,
        )

    def rerank_batch(
        self,
        queries: List[str],
        results: List[RetrievalResult],
        top_k: int | None = None,
    ) -> List[RetrievalResult]:
        """
        Rerank multiple results.
        
        Args:
            queries: List of queries
            results: List of RetrievalResult
            top_k: Return top-k per result
        
        Returns:
            List of reranked RetrievalResult
        """
        if len(queries) != len(results):
            raise ValueError("Queries and results must match length")

        return [
            self.rerank_result(query, result, top_k)
            for query, result in zip(queries, results)
        ]

    def is_ready(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
