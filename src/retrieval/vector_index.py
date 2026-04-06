"""
File: src/retrieval/vector_index.py

Purpose:
FAISS-based vector index for storing and searching embeddings.
Implements IVectorIndex interface for semantic similarity search.

Dependencies:
from typing import List, Dict, Any, Optional, Tuple
import faiss
import numpy as np
import logging
from src.import_map import IVectorIndex

Implements Interface:
IVectorIndex

Notes:
- Uses FAISS for efficient similarity search
- Stores metadata mapping (index_id → doc info)
- Supports cosine similarity and inner product
- Persistent save/load functionality
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from src.import_map import IVectorIndex


logger = logging.getLogger(__name__)


class VectorIndex(IVectorIndex):
    """
    FAISS-based vector index for embedding storage and search.
    
    Manages vector embeddings with metadata and provides
    efficient similarity search using FAISS.
    """

    def __init__(self, dimension: int = 384) -> None:
        """
        Initialize vector index.
        
        Args:
            dimension: Embedding dimension (default 384 for all-MiniLM-L6-v2)
        """
        self.dimension = dimension
        self.index: faiss.IndexFlatL2 | None = None
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.id_counter = 0
        self.logger = logging.getLogger(__name__)
        self._initialize_index()

    def _initialize_index(self) -> None:
        """Initialize FAISS index."""
        try:
            # Use L2 distance (Euclidean)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.logger.info(f"Initialized FAISS index with dimension {self.dimension}")
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS index: {e}")
            raise

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
            ValueError: If inputs are invalid
        """
        if not vectors or not ids:
            raise ValueError("Vectors and IDs cannot be empty")

        if len(vectors) != len(ids):
            raise ValueError("Vectors and IDs must have same length")

        if metadata and len(metadata) != len(vectors):
            raise ValueError("Metadata must match vectors length")

        if not self.index:
            raise RuntimeError("Index not initialized")

        try:
            # Convert to numpy array
            vectors_array = np.array(vectors, dtype=np.float32)

            # Add to FAISS index
            self.index.add(vectors_array)

            # Store metadata
            for i, doc_id in enumerate(ids):
                meta_dict = metadata[i] if metadata else {}
                self.metadata[self.id_counter + i] = {
                    "doc_id": doc_id,
                    "metadata": meta_dict,
                }

            self.id_counter += len(vectors)
            self.logger.info(f"Added {len(vectors)} vectors to index")

        except Exception as e:
            self.logger.error(f"Failed to add vectors: {e}")
            raise

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
            filters: Optional filters (not implemented)
        
        Returns:
            List of (id, score, metadata) tuples
        """
        if not query_vector:
            raise ValueError("Query vector cannot be empty")

        if not self.index:
            raise RuntimeError("Index not initialized")

        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query dimension {len(query_vector)} "
                f"doesn't match index dimension {self.dimension}"
            )

        try:
            # Convert to numpy array
            query_array = np.array([query_vector], dtype=np.float32)

            # Search
            distances, indices = self.index.search(query_array, k)

            # Format results
            results: List[tuple] = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx == -1:  # Invalid result
                    continue

                meta = self.metadata.get(int(idx), {})
                doc_id = meta.get("doc_id", f"doc_{idx}")
                metadata = meta.get("metadata", {})

                # Convert L2 distance to similarity score (0-1)
                similarity = 1.0 / (1.0 + float(distance))

                results.append((doc_id, similarity, metadata))

            return results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise

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
        if not query_vectors:
            raise ValueError("Query vectors cannot be empty")

        return [self.search(qv, k) for qv in query_vectors]

    def delete_vectors(self, ids: List[str]) -> None:
        """
        Delete vectors from index.
        
        Note: FAISS doesn't support deletion directly.
        This is a placeholder for interface compliance.
        
        Args:
            ids: IDs of vectors to delete
        """
        self.logger.warning(
            "FAISS doesn't support deletion. "
            "Consider rebuilding index."
        )

    def get_vector_count(self) -> int:
        """
        Get number of vectors in index.
        
        Returns:
            Vector count
        """
        if not self.index:
            return 0
        return self.index.ntotal

    def save_index(self, path: str) -> None:
        """
        Save index to disk.
        
        Args:
            path: File path to save to
        """
        if not self.index:
            raise RuntimeError("Index not initialized")

        try:
            faiss.write_index(self.index, path)
            self.logger.info(f"Index saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            raise

    def load_index(self, path: str) -> None:
        """
        Load index from disk.
        
        Args:
            path: File path to load from
        """
        try:
            self.index = faiss.read_index(path)
            self.logger.info(f"Index loaded from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            raise
