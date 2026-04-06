"""
File: src/data_pipeline/chunker.py

Purpose:
Split cleaned text into semantic chunks with overlapping context.
Preserves document structure and enables effective RAG retrieval.

Dependencies:
from typing import List, Dict, Any
import logging
from src.core.types import DocumentChunk, DocumentSource

Implements Interface:
None (utility module for data pipeline)

Notes:
- Uses semantic chunking with overlap
- Preserves paragraph boundaries
- Maintains metadata and source information
- Generates unique chunk IDs
"""

import logging
from typing import Any, Dict, List

from src.core.types import DocumentChunk, DocumentSource


logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits text into semantic chunks for RAG.
    
    Uses overlapping chunks with configurable strategy
    while preserving semantic boundaries and metadata.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        strategy: str = "paragraph",
    ) -> None:
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target chunk size in tokens/words
            chunk_overlap: Overlap between consecutive chunks
            strategy: Chunking strategy ("paragraph", "sentence", "sliding")
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)

    def chunk_text(
        self,
        text: str,
        document_name: str,
        source: DocumentSource,
        metadata: Dict[str, Any] | None = None,
    ) -> List[DocumentChunk]:
        """
        Split text into semantic chunks.
        
        Args:
            text: Text to chunk
            document_name: Name of source document
            source: DocumentSource enum
            metadata: Optional document-level metadata
        
        Returns:
            List of DocumentChunk objects
        """
        if not text:
            return []

        chunks: List[DocumentChunk] = []
        
        if self.strategy == "paragraph":
            chunks = self._chunk_by_paragraph(
                text, document_name, source, metadata
            )
        elif self.strategy == "sentence":
            chunks = self._chunk_by_sentence(
                text, document_name, source, metadata
            )
        else:  # sliding
            chunks = self._chunk_sliding_window(
                text, document_name, source, metadata
            )

        return chunks

    def _chunk_by_paragraph(
        self,
        text: str,
        document_name: str,
        source: DocumentSource,
        metadata: Dict[str, Any] | None = None,
    ) -> List[DocumentChunk]:
        """Chunk by paragraph boundaries."""
        paragraphs = text.split("\n\n")
        chunks: List[DocumentChunk] = []
        chunk_id = 0
        
        current_text = ""
        page_num = 1

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding paragraph would exceed chunk size, save current
            if (
                current_text
                and len((current_text + " " + para).split()) > self.chunk_size
            ):
                chunk = self._create_chunk(
                    current_text,
                    chunk_id,
                    document_name,
                    source,
                    page_num,
                    metadata,
                )
                chunks.append(chunk)
                chunk_id += 1

                
                # Add overlap (keep recent paragraphs roughly matching overlap size to avoid broken sentences)
                paras = current_text.split("\n\n")
                overlap_words = 0
                overlap_paras = []
                for p in reversed(paras):
                    overlap_paras.insert(0, p)
                    overlap_words += len(p.split())
                    if overlap_words >= self.chunk_overlap:
                        break
                
                overlap_text = "\n\n".join(overlap_paras)
                current_text = overlap_text + "\n\n" + para

            else:
                if current_text:
                    current_text += " " + para
                else:
                    current_text = para

        # Add final chunk
        if current_text:
            chunk = self._create_chunk(
                current_text,
                chunk_id,
                document_name,
                source,
                page_num,
                metadata,
            )
            chunks.append(chunk)

        return chunks

    def _chunk_by_sentence(
        self,
        text: str,
        document_name: str,
        source: DocumentSource,
        metadata: Dict[str, Any] | None = None,
    ) -> List[DocumentChunk]:
        """Chunk by sentence boundaries."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[DocumentChunk] = []
        chunk_id = 0
        
        current_text = ""
        page_num = 1

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence.split()) < 3:
                continue

            if (
                current_text
                and len((current_text + " " + sentence).split())
                > self.chunk_size
            ):
                chunk = self._create_chunk(
                    current_text,
                    chunk_id,
                    document_name,
                    source,
                    page_num,
                    metadata,
                )
                chunks.append(chunk)
                chunk_id += 1

                # Overlap
                # Add overlap (keep recent complete sentences roughly matching overlap size)
                sents = re.split(r"(?<=[.!?])\s+", current_text)
                overlap_words = 0
                overlap_sents = []
                for s in reversed(sents):
                    if not s.strip(): continue
                    overlap_sents.insert(0, s)
                    overlap_words += len(s.split())
                    if overlap_words >= self.chunk_overlap:
                        break
                
                overlap_text = " ".join(overlap_sents)
                current_text = overlap_text + " " + sentence

            else:
                if current_text:
                    current_text += " " + sentence
                else:
                    current_text = sentence

        if current_text:
            chunk = self._create_chunk(
                current_text,
                chunk_id,
                document_name,
                source,
                page_num,
                metadata,
            )
            chunks.append(chunk)

        return chunks

    def _chunk_sliding_window(
        self,
        text: str,
        document_name: str,
        source: DocumentSource,
        metadata: Dict[str, Any] | None = None,
    ) -> List[DocumentChunk]:
        """Chunk using sliding window."""
        words = text.split()
        chunks: List[DocumentChunk] = []
        chunk_id = 0
        page_num = 1

        stride = self.chunk_size - self.chunk_overlap

        for i in range(0, len(words), stride):
            end = min(i + self.chunk_size, len(words))
            chunk_words = words[i:end]
            chunk_text = " ".join(chunk_words)

            chunk = self._create_chunk(
                chunk_text,
                chunk_id,
                document_name,
                source,
                page_num,
                metadata,
            )
            chunks.append(chunk)
            chunk_id += 1

        return chunks

    def _create_chunk(
        self,
        text: str,
        chunk_index: int,
        document_name: str,
        source: DocumentSource,
        page_num: int,
        metadata: Dict[str, Any] | None = None,
    ) -> DocumentChunk:
        """Create a DocumentChunk object."""
        chunk_id = f"{document_name}_{chunk_index}"
        
        return DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            source=source,
            document_name=document_name,
            chunk_index=chunk_index,
            page_number=page_num,
            metadata=metadata or {},
        )

    def get_statistics(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Get statistics about chunks.
        
        Args:
            chunks: List of chunks to analyze
        
        Returns:
            Dict with statistics
        """
        if not chunks:
            return {}

        token_counts = [len(c.text.split()) for c in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(token_counts) / len(token_counts),
            "min_chunk_size": min(token_counts),
            "max_chunk_size": max(token_counts),
            "total_tokens": sum(token_counts),
        }
