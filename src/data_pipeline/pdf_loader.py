"""
File: src/data_pipeline/pdf_loader.py

Purpose:
Load and extract text from PDF documents.
Handles multi-page PDFs, metadata extraction, and error handling.

Dependencies:
from typing import List, Dict, Any, Optional
from pathlib import Path
from pypdf import PdfReader
import logging

Implements Interface:
None (utility module for data pipeline)

Notes:
- Extracts raw text from PDFs
- Captures page numbers and document metadata
- Handles corrupted PDFs gracefully
- Non-blocking on read errors
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import pdfplumber
import pytesseract
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)


class PDFLoader:
    """
    Loads and extracts text from PDF documents.
    
    Handles metadata extraction, page-level information,
    and error resilience for malformed PDFs.
    """

    def __init__(self, max_pages: Optional[int] = None) -> None:
        """
        Initialize PDF loader.
        
        Args:
            max_pages: Maximum pages to extract per PDF (None = all)
        """
        self.max_pages = max_pages
        self.logger = logging.getLogger(__name__)

    def load_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Load a single PDF and extract all text.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dict with:
                - text: Complete extracted text
                - pages: List of page texts
                - metadata: PDF metadata
                - page_count: Total pages
                - file_name: File name
        
        Raises:
            FileNotFoundError: If PDF not found
            ValueError: If PDF cannot be read
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not path.suffix.lower() == ".pdf":
            raise ValueError(f"Not a PDF file: {pdf_path}")

        try:
            pages_text: List[str] = []
            
            with pdfplumber.open(pdf_path) as pdf:
                metadata = pdf.metadata or {}
                page_count = len(pdf.pages)

                if self.max_pages:
                    page_count = min(page_count, self.max_pages)

                for page_num in range(page_count):
                    page = pdf.pages[page_num]
                    # extract_text(layout=True) keeps tax tables from turning into mush
                    page_text = page.extract_text(layout=True)
                    
                    # OCR Fallback: If page is an image, text will be empty or very short
                    if not page_text or len(page_text.strip()) < 50:
                        self.logger.info(f"Triggering OCR fallback for page {page_num + 1} in {path.name}")
                        try:
                            # Convert that specific page to an image and run Tesseract
                            images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)
                            if images:
                                page_text = pytesseract.image_to_string(images[0])
                        except Exception as ocr_e:
                            self.logger.warning(f"OCR failed for page {page_num + 1}: {ocr_e}")
                            page_text = ""

                    pages_text.append(page_text or "")

            full_text = "\n".join(pages_text)

            return {
                "text": full_text,
                "pages": pages_text,
                "metadata": metadata,
                "page_count": page_count,
                "file_name": path.name,
                "file_path": str(path),
            }

        except Exception as e:
            self.logger.error(f"Failed to load PDF {path.name}: {e}")
            raise ValueError(f"Cannot read PDF {path.name}: {e}") from e

    def load_pdfs_from_directory(
        self,
        directory: str,
        pattern: str = "*.pdf",
    ) -> List[Dict[str, Any]]:
        """
        Load all PDFs from a directory.
        
        Args:
            directory: Directory containing PDFs
            pattern: File pattern to match
        
        Returns:
            List of loaded PDF data dicts
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        pdf_files = list(dir_path.glob(pattern))
        self.logger.info(f"Found {len(pdf_files)} PDFs in {directory}")

        loaded_pdfs: List[Dict[str, Any]] = []
        for pdf_file in pdf_files:
            try:
                pdf_data = self.load_pdf(str(pdf_file))
                loaded_pdfs.append(pdf_data)
                self.logger.info(f"Loaded {pdf_file.name}")
            except Exception as e:
                self.logger.error(f"Skipped {pdf_file.name}: {e}")

        return loaded_pdfs

    def get_page_text(
        self,
        pdf_data: Dict[str, Any],
        page_num: int,
    ) -> str:
        """
        Get text from a specific page.
        
        Args:
            pdf_data: Loaded PDF data dict
            page_num: Page number (0-indexed)
        
        Returns:
            Text from that page
        
        Raises:
            IndexError: If page not found
        """
        pages = pdf_data.get("pages", [])
        if page_num >= len(pages):
            raise IndexError(
                f"Page {page_num} not found in {pdf_data.get('file_name')}"
            )
        return pages[page_num]

    def get_statistics(self, pdf_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Get statistics about a loaded PDF.
        
        Args:
            pdf_data: Loaded PDF data dict
        
        Returns:
            Dict with word count, char count, page count
        """
        text = pdf_data.get("text", "")
        return {
            "word_count": len(text.split()),
            "char_count": len(text),
            "page_count": pdf_data.get("page_count", 0),
        }
