"""
File: src/interfaces/embedding.py

Purpose:
Abstract interface for embedding model operations.
Defines contract for vector embeddings and similarity computations.

Dependencies:
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.core.types import DocumentChunk

Implements Interface:
IEmbeddingModel (abstract base class)

Notes:
- Embeddings should be deterministic for same input
- Vector dimension should be consistent
- Supports batch operations for efficiency
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class IEmbeddingModel(ABC):
    """
    Abstract interface for embedding model operations.
    
    Converts text and documents into vector embeddings
    for similarity search and retrieval.
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        
        Raises:
            ValueError: If text is invalid
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch).
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        
        Raises:
            ValueError: If texts are invalid
        """
        pass

    @abstractmethod
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity score (0.0-1.0)
        """
        pass

    @abstractmethod
    def compute_similarities(
        self,
        embedding: List[float],
        embeddings: List[List[float]],
    ) -> List[float]:
        """
        Compute similarity between one and many embeddings.
        
        Args:
            embedding: Query embedding vector
            embeddings: List of candidate embeddings
        
        Returns:
            List of similarity scores
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get dimension of embedding vectors.
        
        Returns:
            Vector dimension
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get name of embedding model.
        
        Returns:
            Model name/identifier
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if model is loaded and ready.
        
        Returns:
            True if model is ready
        """
        pass


class IVectorIndex(ABC):
    """
    Abstract interface for vector index operations.
    
    Stores and searches embeddings efficiently.
    """

    @abstractmethod
    def add_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: List[Dict[str, Any]] | None = None,
    ) -> None:
        """
        Add vectors to the index.
        
        Args:
            vectors: List of embedding vectors
            ids: Unique IDs for vectors
            metadata: Optional metadata for each vector
        
        Raises:
            ValueError: If vectors or IDs invalid
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filters: Dict[str, Any] | None = None,
    ) -> List[tuple]:
        """
        Search for nearest neighbors.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results
            filters: Optional filters for metadata
        
        Returns:
            List of (id, score, metadata) tuples
        """
        pass

    @abstractmethod
    def search_batch(
        self,
        query_vectors: List[List[float]],
        k: int = 5,
    ) -> List[List[tuple]]:
        """
        Search for nearest neighbors (batch).
        
        Args:
            query_vectors: List of query vectors
            k: Number of results per query
        
        Returns:
            List of result lists
        """
        pass

    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> None:
        """
        Delete vectors from index.
        
        Args:
            ids: IDs of vectors to delete
        """
        pass

    @abstractmethod
    def get_vector_count(self) -> int:
        """
        Get number of vectors in index.
        
        Returns:
            Vector count
        """
        pass

    @abstractmethod
    def save_index(self, path: str) -> None:
        """
        Save index to disk.
        
        Args:
            path: File path to save to
        """
        pass

    @abstractmethod
    def load_index(self, path: str) -> None:
        """
        Load index from disk.
        
        Args:
            path: File path to load from
        """
        pass
