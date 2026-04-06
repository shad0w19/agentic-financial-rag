"""
File: src/interfaces/retriever.py

Purpose:
Abstract interface for retrieval operations.
Defines contract for all retrieval implementations (vector, BM25, hybrid, federated).

Dependencies:
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.core.types import Query, RetrievalResult, DocumentChunk

Implements Interface:
IRetriever (abstract base class)

Notes:
- All retrieval implementations must implement this interface
- Enables dependency injection and testing
- Supports multi-strategy retrieval (vector, BM25, hybrid)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from src.core.types import Query, RetrievalResult, DocumentChunk


class IRetriever(ABC):
    """
    Abstract interface for document retrieval.
    
    All retrieval implementations must conform to this interface.
    Supports vector search, BM25, hybrid, and federated strategies.
    """

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 5,
        filters: Dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query text
            k: Number of results to return
            filters: Optional filters (e.g., document source)
        
        Returns:
            RetrievalResult containing matched chunks and scores
        
        Raises:
            ValueError: If query is invalid
        """
        pass

    @abstractmethod
    def search_multi_hop(
        self,
        initial_query: str,
        num_hops: int = 2,
        k_per_hop: int = 5,
    ) -> List[RetrievalResult]:
        """
        Perform multi-hop retrieval.
        
        Args:
            initial_query: Starting query
            num_hops: Number of retrieval hops
            k_per_hop: Results per hop
        
        Returns:
            List of RetrievalResult for each hop
        """
        pass

    @abstractmethod
    def is_indexed(self) -> bool:
        """
        Check if the retriever has indexed data.
        
        Returns:
            True if index exists and is populated
        """
        pass

    @abstractmethod
    def index_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Index a batch of document chunks.
        
        Args:
            chunks: List of DocumentChunk objects to index
        
        Raises:
            ValueError: If chunks are invalid
        """
        pass
