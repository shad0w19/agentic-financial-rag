"""
File: src/retrieval/bm25_index.py

Purpose:
BM25 keyword-based retrieval index.
Simple in-memory implementation for keyword search.

Dependencies:
from typing import List, Dict, Any
from src.import_map import DocumentChunk

Implements Interface:
None (utility)

Notes:
- No embeddings required
- No FAISS dependency
- In-memory storage
- BM25 scoring algorithm
"""

import logging
import math
from typing import Any, Dict, List

from src.import_map import DocumentChunk


logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 keyword-based index for retrieval.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """
        Initialize BM25 index.
        
        Args:
            k1: BM25 parameter (saturation)
            b: BM25 parameter (length normalization)
        """
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, DocumentChunk] = {}
        self.doc_freqs: Dict[str, Dict[str, int]] = {}
        self.idf: Dict[str, float] = {}
        self.avg_doc_length = 0.0
        self.logger = logging.getLogger(__name__)

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Add documents to index.
        
        Args:
            chunks: List of DocumentChunk
        """
        if not chunks:
            return

        for chunk in chunks:
            self.documents[chunk.chunk_id] = chunk
            tokens = self._tokenize(chunk.text)
            self.doc_freqs[chunk.chunk_id] = self._count_tokens(tokens)

        self._compute_idf()
        self.logger.info(f"Indexed {len(chunks)} documents")

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> List[tuple]:
        """
        Search using BM25.
        
        Args:
            query: Query text
            k: Number of results
        
        Returns:
            List of (doc_id, score) tuples
        """
        if not query or not self.documents:
            return []

        query_tokens = self._tokenize(query)
        scores: Dict[str, float] = {}

        for doc_id, doc_freqs in self.doc_freqs.items():
            score = self._bm25_score(query_tokens, doc_freqs)
            if score > 0:
                scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        return text.lower().split()

    def _count_tokens(self, tokens: List[str]) -> Dict[str, int]:
        """Count token frequencies."""
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return counts

    def _compute_idf(self) -> None:
        """Compute IDF for all terms."""
        num_docs = len(self.documents)
        if num_docs == 0:
            return

        term_doc_count: Dict[str, int] = {}

        for doc_freqs in self.doc_freqs.values():
            for term in doc_freqs.keys():
                term_doc_count[term] = term_doc_count.get(term, 0) + 1

        for term, count in term_doc_count.items():
            self.idf[term] = math.log(
                (num_docs - count + 0.5) / (count + 0.5) + 1
            )

        total_length = sum(
            sum(freqs.values()) for freqs in self.doc_freqs.values()
        )
        self.avg_doc_length = total_length / num_docs if num_docs > 0 else 0

    def _bm25_score(
        self,
        query_tokens: List[str],
        doc_freqs: Dict[str, int],
    ) -> float:
        """Compute BM25 score."""
        score = 0.0
        doc_length = sum(doc_freqs.values())

        for token in query_tokens:
            if token not in self.idf:
                continue

            freq = doc_freqs.get(token, 0)
            idf = self.idf[token]

            numerator = freq * (self.k1 + 1)
            denominator = (
                freq
                + self.k1
                * (
                    1
                    - self.b
                    + self.b * (doc_length / self.avg_doc_length)
                )
            )

            score += idf * (numerator / denominator)

        return score

    def get_document_count(self) -> int:
        """Get number of indexed documents."""
        return len(self.documents)

    def clear(self) -> None:
        """Clear index."""
        self.documents.clear()
        self.doc_freqs.clear()
        self.idf.clear()
        self.avg_doc_length = 0.0
