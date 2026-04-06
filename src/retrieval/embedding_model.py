"""
File: src/retrieval/embedding_model.py

Purpose:
Local SentenceTransformer-based embedding model using all-MiniLM-L6-v2.
Fast, efficient embeddings optimized for retrieval tasks.

Dependencies:
from typing import List
from sentence_transformers import SentenceTransformer

Implements Interface:
IEmbeddingModel

Notes:
- Uses local SentenceTransformer model
- High-throughput batching (~2000+ sentences/sec)
- Embedding cache to avoid recomputation
- Dimension: 384 (all-MiniLM-L6-v2)
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from src.import_map import IEmbeddingModel


logger = logging.getLogger(__name__)


class EmbeddingModel(IEmbeddingModel):
    """
    Local SentenceTransformer-based embedding model.
    
    Uses all-MiniLM-L6-v2 for fast, efficient embeddings.
    Implements caching to avoid recomputing embeddings for identical texts.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model with SentenceTransformer.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache = {}  # Cache for embedding results
        self.dimension = 384  # MiniLM always outputs 384-dim vectors
        
        logger.info(f"Initialized SentenceTransformer Model: {model_name}")
        logger.info(f"Embedding dimension: {self.dimension}")
        logger.info(f"Model ready for high-throughput batching (~2000+ sentences/sec)")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text with caching.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector (List[float])
        
        Raises:
            ValueError: If text is empty
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be non-empty string")
        
        # Check cache first
        if text in self.cache:
            return self.cache[text]
        
        # Encode with normalization (improves semantic relevance)
        embedding = self.model.encode(text, normalize_embeddings=True).tolist()
        
        # Store in cache
        self.cache[text] = embedding
        
        return embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with high-throughput batching.
        
        Uses SentenceTransformer's native batching (~2000+ sentences/sec).
        Preserves order of input texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors, preserving order
        
        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        logger.info(f"Batch encoding {len(texts)} texts (MiniLM local model)")
        
        # Encode all texts at once with normalization
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        # Convert to list of lists
        embeddings_list = embeddings.tolist()
        
        # Cache results for future queries
        for text, embedding in zip(texts, embeddings_list):
            if text not in self.cache:
                self.cache[text] = embedding
        
        logger.info(f"✓ Encoded {len(embeddings_list)} embeddings (dimension: {self.dimension})")
        
        return embeddings_list

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity score (0.0-1.0)
        """
        if not embedding1 or not embedding2:
            raise ValueError("Embeddings cannot be empty")
        
        import numpy as np
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

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
        if not embedding or not embeddings:
            raise ValueError("Embeddings cannot be empty")
        
        return [
            self.compute_similarity(embedding, emb)
            for emb in embeddings
        ]

    def get_embedding_dimension(self) -> int:
        """
        Get dimension of embedding vectors.
        
        Returns:
            Vector dimension (384 for all-MiniLM-L6-v2)
        """
        return self.dimension

    def get_model_name(self) -> str:
        """
        Get name of embedding model.
        
        Returns:
            Model name/identifier
        """
        return self.model_name

    def is_ready(self) -> bool:
        """
        Check if model is loaded and ready.
        
        Returns:
            True if model is ready
        """
        return self.model is not None