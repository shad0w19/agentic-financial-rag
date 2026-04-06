"""
File: src/retrieval/hybrid_retriever.py

Purpose:
Hybrid retriever combining vector search and BM25 keyword search.
Implements IRetriever interface.

Dependencies:
from typing import List, Dict, Any
from src.import_map import IRetriever, RetrievalResult, RetrievalStrategy, DocumentChunk
from src.retrieval.embedding_model import EmbeddingModel
from src.retrieval.vector_index import VectorIndex

Implements Interface:
IRetriever

Notes:
- Combines semantic (vector) and keyword (BM25) search
- Reranks combined results
- Supports multi-hop retrieval
"""

import logging
from typing import Any, Dict, List

from sentence_transformers import CrossEncoder

from src.import_map import (
    DocumentChunk,
    IRetriever,
    RetrievalResult,
    RetrievalStrategy,
)
from src.retrieval.embedding_model import EmbeddingModel
from src.retrieval.vector_index import VectorIndex
from src.config.settings import (
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_TOP_K,
)


logger = logging.getLogger(__name__)


class HybridRetriever(IRetriever):
    """
    Hybrid retriever combining vector and BM25 search.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_index: VectorIndex,
        chunks: List[DocumentChunk],
    ) -> None:
        """
        Initialize hybrid retriever.
        
        Args:
            embedding_model: EmbeddingModel instance
            vector_index: VectorIndex instance
            chunks: List of DocumentChunk objects
        """
        self.embedding_model = embedding_model
        self.vector_index = vector_index
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.logger = logging.getLogger(__name__)
        
        # Initialize CrossEncoder for reranking (local, no API calls)
        # Force CUDA for faster reranking
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                self.logger.info("Using CUDA for reranker (GPU acceleration enabled)")
            else:
                self.logger.info("CUDA not available, falling back to CPU")
        except ImportError:
            device = "cpu"
        
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=device
        )
        
        self.logger.info(f"CrossEncoder initialized with device: {device}")

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """
        Search using hybrid approach.
        
        Args:
            query: Search query
            k: Number of results
            filters: Optional filters
        
        Returns:
            RetrievalResult
        """
        if not query:
            raise ValueError("Query cannot be empty")

        # Vector search
        query_embedding = self.embedding_model.embed_text(query)
        vector_results = self.vector_index.search(query_embedding, k=k)

        # BM25 search (simple keyword matching)
        bm25_results = self._bm25_search(query, k=k)

        # Stage 1: Combine vector + BM25 (get top 25)
        combined = self._combine_results(vector_results, bm25_results, k=25)

        # Extract chunks for reranking
        candidate_chunks = [
            self.chunks[doc_id]
            for doc_id, score in combined
            if doc_id in self.chunks
        ]

        # Stage 2: Rerank using API (top 5)
        result_chunks = self._rerank_chunks(query, candidate_chunks, k=k)

        # Calculate scores for reranked results
        scores = [1.0] * len(result_chunks)  # Placeholder scores after reranking

        return RetrievalResult(
            chunks=result_chunks,
            strategy_used=RetrievalStrategy.HYBRID,
            scores=scores,
            query_used=query,
        )

    def search_multi_hop(
        self,
        initial_query: str,
        num_hops: int = 2,
        k_per_hop: int = 5,
    ) -> List[RetrievalResult]:
        """
        Multi-hop retrieval.
        
        Args:
            initial_query: Starting query
            num_hops: Number of hops
            k_per_hop: Results per hop
        
        Returns:
            List of RetrievalResult
        """
        results = []
        current_query = initial_query

        for hop in range(num_hops):
            result = self.search(current_query, k=k_per_hop)
            results.append(result)

            if result.chunks:
                current_query = " ".join(
                    [chunk.text[:100] for chunk in result.chunks[:2]]
                )

        return results

    def is_indexed(self) -> bool:
        """Check if index is populated."""
        return self.vector_index.get_vector_count() > 0

    def index_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Index document chunks.
        
        Args:
            chunks: List of DocumentChunk objects
        """
        if not chunks:
            return

        embeddings = self.embedding_model.embed_texts(
            [chunk.text for chunk in chunks]
        )

        ids = [chunk.chunk_id for chunk in chunks]
        metadata = [
            {
                "text": chunk.text,
                "source": chunk.source.value,
                "document": chunk.document_name,
                "page": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

        self.vector_index.add_vectors(embeddings, ids, metadata)
        self.chunks.update({chunk.chunk_id: chunk for chunk in chunks})

    def _bm25_search(self, query: str, k: int = 5) -> List[tuple]:
        """Simple BM25-like keyword search."""
        query_terms = set(query.lower().split())
        scores = []

        for chunk_id, chunk in self.chunks.items():
            chunk_terms = set(chunk.text.lower().split())
            overlap = len(query_terms & chunk_terms)
            if overlap > 0:
                score = overlap / len(query_terms)
                scores.append((chunk_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def _combine_results(
        self,
        vector_results: List[tuple],
        bm25_results: List[tuple],
        k: int,
    ) -> List[tuple]:
        """Combine and deduplicate results."""
        combined = {}

        # Handle vector_results - could be 2 or 3 element tuples
        for result in vector_results:
            if len(result) >= 2:
                doc_id, score = result[0], result[1]
                combined[doc_id] = combined.get(doc_id, 0) + score * 0.7

        # Handle bm25_results - 2 element tuples
        for doc_id, score in bm25_results:
            combined[doc_id] = combined.get(doc_id, 0) + score * 0.3

        sorted_results = sorted(
            combined.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_results[:k]

    def _rerank_chunks(
        self,
        query: str,
        chunks: List[DocumentChunk],
        k: int = 5,
    ) -> List[DocumentChunk]:
        """
        Rerank chunks using HuggingFace CrossEncoder (local, no API calls).
        
        Args:
            query: Search query
            chunks: List of candidate chunks
            k: Number of results to return
        
        Returns:
            Reranked list of DocumentChunks
        """
        if not chunks:
            return []
        
        try:
            # Prepare pairs: (query, chunk_text)
            pairs = [(query, chunk.text) for chunk in chunks]
            
            # Get scores from CrossEncoder
            scores = self.reranker.predict(pairs)
            
            # Sort by score descending
            ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            
            # Return top k
            return [chunk for chunk, score in ranked[:k]]
        
        except Exception as e:
            self.logger.error(f"Reranking error: {e}")
            return chunks[:k]
