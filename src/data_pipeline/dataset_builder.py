"""
File: src/data_pipeline/dataset_builder.py

Purpose:
Orchestrate the complete data pipeline: load PDFs → clean text → chunk → export dataset.
Builds the chunk dataset for retrieval indexing.

Dependencies:
from typing import List, Dict, Any
import json
from pathlib import Path
import logging
from src.core.types import DocumentChunk
from src.data_pipeline.pdf_loader import PDFLoader
from src.data_pipeline.text_cleaner import TextCleaner
from src.data_pipeline.chunker import TextChunker

Implements Interface:
None (orchestration utility for data pipeline)

Notes:
- Orchestrates all data pipeline stages
- Outputs chunks.json dataset
- Includes metadata and statistics
- Error resilient processing
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.core.types import DocumentChunk, DocumentSource
from src.data_pipeline.chunker import TextChunker
from src.data_pipeline.pdf_loader import PDFLoader
from src.data_pipeline.text_cleaner import TextCleaner


logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Orchestrates the complete data pipeline.
    
    Loads PDFs, cleans text, chunks into semantic units,
    and exports to chunk dataset for retrieval.
    """

    def __init__(
        self,
        output_dir: str = "data/chunks",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:
        """
        Initialize dataset builder.
        
        Args:
            output_dir: Directory to save chunks
            chunk_size: Target chunk size in words
            chunk_overlap: Overlap between chunks
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pdf_loader = PDFLoader()
        self.text_cleaner = TextCleaner()
        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy="paragraph",
        )
        self.logger = logging.getLogger(__name__)

    def build_dataset(
        self,
        source_directories: Dict[str, str],
        output_filename: str = "chunks.json",
    ) -> Dict[str, Any]:
        """
        Build complete chunk dataset from PDFs.
        
        Args:
            source_directories: Dict mapping DocumentSource to directory
                Example: {
                    "income_tax": "data/income_tax",
                    "corporate_tax": "data/corporate_tax",
                    "gst": "data/gst"
                }
            output_filename: Output file name
        
        Returns:
            Dict with build statistics and summary
        """
        all_chunks: List[DocumentChunk] = []
        statistics = {
            "by_source": {},
            "total_documents": 0,
            "total_chunks": 0,
            "total_tokens": 0,
        }

        for source_name, source_dir in source_directories.items():
            source_enum = self._get_document_source(source_name)
            
            self.logger.info(
                f"Processing {source_name} from {source_dir}"
            )

            chunks = self._process_directory(source_dir, source_enum)
            all_chunks.extend(chunks)

            # Statistics
            statistics["by_source"][source_name] = {
                "chunk_count": len(chunks),
                "total_tokens": sum(len(c.text.split()) for c in chunks),
            }
            statistics["total_documents"] += 1
            statistics["total_chunks"] += len(chunks)
            statistics["total_tokens"] += sum(
                len(c.text.split()) for c in chunks
            )

            self.logger.info(
                f"Processed {source_name}: {len(chunks)} chunks"
            )

        # Export dataset
        output_path = self.output_dir / output_filename
        self._export_chunks(all_chunks, output_path)

        self.logger.info(f"Dataset exported to {output_path}")
        self.logger.info(f"Statistics: {statistics}")

        return {
            "output_path": str(output_path),
            "total_chunks": len(all_chunks),
            "statistics": statistics,
        }

    def _process_directory(
        self,
        directory: str,
        source: DocumentSource,
    ) -> List[DocumentChunk]:
        """
        Process all PDFs in a directory.
        
        Args:
            directory: Directory containing PDFs
            source: DocumentSource enum
        
        Returns:
            List of chunks from all PDFs
        """
        chunks: List[DocumentChunk] = []

        try:
            pdf_files = self.pdf_loader.load_pdfs_from_directory(directory)
        except Exception as e:
            self.logger.error(f"Failed to load PDFs from {directory}: {e}")
            return chunks

        for pdf_data in pdf_files:
            try:
                # Clean text
                cleaned_text = self.text_cleaner.clean_text(
                    pdf_data["text"]
                )

                # Chunk text
                pdf_chunks = self.text_chunker.chunk_text(
                    cleaned_text,
                    pdf_data["file_name"],
                    source,
                    {"file_path": pdf_data["file_path"]},
                )

                chunks.extend(pdf_chunks)

                self.logger.debug(
                    f"Processed {pdf_data['file_name']}: "
                    f"{len(pdf_chunks)} chunks"
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to process {pdf_data['file_name']}: {e}"
                )

        return chunks

    def _get_document_source(self, source_name: str) -> DocumentSource:
        """
        Map source name to DocumentSource enum.
        
        Args:
            source_name: Name of source
        
        Returns:
            DocumentSource enum value
        """
        mapping = {
            "income_tax": DocumentSource.PERSONAL_TAX,
            "personal_tax": DocumentSource.PERSONAL_TAX,
            "corporate_tax": DocumentSource.CORPORATE_TAX,
            "gst": DocumentSource.GST,
            "investment": DocumentSource.INVESTMENT,
            "regulatory": DocumentSource.REGULATORY,
        }
        return mapping.get(source_name, DocumentSource.REGULATORY)

    def _export_chunks(
        self,
        chunks: List[DocumentChunk],
        output_path: Path,
    ) -> None:
        """
        Export chunks to JSON file.
        
        Args:
            chunks: List of chunks to export
            output_path: Output file path
        """
        chunks_data = []

        for chunk in chunks:
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source.value,
                "document_name": chunk.document_name,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "metadata": chunk.metadata,
            }
            chunks_data.append(chunk_dict)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                chunks_data,
                f,
                indent=2,
                ensure_ascii=False,
            )

    @staticmethod
    def load_chunks_from_file(
        file_path: str,
    ) -> List[DocumentChunk]:
        """
        Load chunks from JSON file.
        
        Args:
            file_path: Path to chunks JSON file
        
        Returns:
            List of DocumentChunk objects
        """
        with open(file_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        chunks: List[DocumentChunk] = []

        for chunk_dict in chunks_data:
            chunk = DocumentChunk(
                chunk_id=chunk_dict["chunk_id"],
                text=chunk_dict["text"],
                source=DocumentSource(chunk_dict["source"]),
                document_name=chunk_dict["document_name"],
                chunk_index=chunk_dict["chunk_index"],
                page_number=chunk_dict.get("page_number"),
                metadata=chunk_dict.get("metadata", {}),
            )
            chunks.append(chunk)

        return chunks

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the built dataset.
        
        Returns:
            Dict with dataset information
        """
        chunks_file = self.output_dir / "chunks.json"

        if not chunks_file.exists():
            return {"status": "no_dataset"}

        chunks = self.load_chunks_from_file(str(chunks_file))
        
        sources_count = {}
        for chunk in chunks:
            source = chunk.source.value
            sources_count[source] = sources_count.get(source, 0) + 1

        return {
            "status": "exists",
            "chunks_file": str(chunks_file),
            "total_chunks": len(chunks),
            "by_source": sources_count,
            "total_tokens": sum(len(c.text.split()) for c in chunks),
        }
