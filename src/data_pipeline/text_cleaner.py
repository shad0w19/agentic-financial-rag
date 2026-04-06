"""
File: src/data_pipeline/text_cleaner.py

Purpose:
Clean and normalize text extracted from PDFs.
Removes artifacts, standardizes formatting, handles special characters.

Dependencies:
from typing import List, Dict, Any
import re
import logging

Implements Interface:
None (utility module for data pipeline)

Notes:
- Removes PDF extraction artifacts
- Normalizes whitespace and line breaks
- Handles special characters and encodings
- Maintains semantic meaning
"""

import logging
import re
from typing import List


logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans and normalizes text extracted from PDFs.
    
    Handles formatting artifacts, whitespace normalization,
    and special character handling while preserving content.
    """

    def __init__(self) -> None:
        """Initialize text cleaner."""
        self.logger = logging.getLogger(__name__)
        pass

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text comprehensively.
        
        Operations:
        1. Decode special characters
        2. Remove extra whitespace
        3. Normalize line breaks
        4. Remove common PDF artifacts
        5. Fix common OCR errors
        
        Args:
            text: Raw text to clean
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove null characters
        text = text.replace("\x00", "")

        # Fix common PDF encoding issues
        text = self._fix_encoding(text)

        # Remove page headers/footers (common patterns)
        text = self._remove_headers_footers(text)

        # Normalize whitespace
        text = self._normalize_whitespace(text)

        # Fix line breaks
        text = self._normalize_line_breaks(text)

        # Remove excessive special characters
        text = self._clean_special_chars(text)

        # Fix common OCR errors
        text = self._fix_ocr_errors(text)

        return text.strip()

    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues from PDF extraction."""
        replacements = {
            "\xe2\x80\x99": "'",  # Right quote
            "\xe2\x80\x98": "'",  # Left quote
            "\xe2\x80\x9c": '"',  # Left double quote
            "\xe2\x80\x9d": '"',  # Right double quote
            "\xe2\x80\x93": "-",  # En dash
            "\xe2\x80\x94": "-",  # Em dash
            "Âº": "°",  # Degree symbol
            "\xc2": "",  # Remove orphaned A with circumflex
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _remove_headers_footers(self, text: str) -> str:
        """Remove common page headers and footers."""
        lines = text.split("\n")
        cleaned_lines = []

        for i, line in enumerate(lines):
            # Skip very short lines that look like page numbers
            if len(line.strip()) < 3 and line.strip().isdigit():
                continue

            # Skip common header patterns
            if any(
                pattern in line.lower()
                for pattern in [
                    "page ",
                    "confidential",
                    "copyright",
                    "© ",
                    "www.",
                ]
            ):
                # Only skip if it's short (header/footer likely)
                if len(line) < 100:
                    continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize and clean whitespace."""
        # Remove multiple spaces but keep single spaces
        text = re.sub(r" {2,}", " ", text)

        # Remove spaces before punctuation
        text = re.sub(r" ([,.!?;:])", r"\1", text)

        # Remove leading/trailing spaces from lines
        lines = text.split("\n")
        lines = [line.strip() for line in lines]
        text = "\n".join(lines)

        return text

    def _normalize_line_breaks(self, text: str) -> str:
        """Normalize line breaks and paragraph spacing."""
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Fix spaces before paragraph breaks
        text = re.sub(r" +\n", "\n", text)

        return text

    def _clean_special_chars(self, text: str) -> str:
        """Remove excessive or problematic special characters."""
        # Remove control characters except newline and tab
        text = "".join(
            char for char in text
            if ord(char) >= 32 or char in "\n\t"
        )

        # Remove excessive repeated punctuation
        text = re.sub(r"[.!?]{2,}", lambda m: m.group(0)[0], text)

        return text

    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR misrecognitions safely without corrupting numbers or financial terms."""
        replacements = {
            r"\bl\b": "I",  # Single lowercase L -> I (in context)
            r"([A-Z])\1{2,}": r"\1",  # Remove repeated capitals
            # REMOVED: dangerous r"rn" -> "m", r"ft" -> "fl", and r"\b0\b" -> "O" replacements
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        return text

    def clean_texts(self, texts: List[str]) -> List[str]:
        """
        Clean multiple texts.
        
        Args:
            texts: List of texts to clean
        
        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]

    def remove_empty_lines(self, text: str) -> str:
        """
        Remove empty lines from text.
        
        Args:
            text: Text to clean
        
        Returns:
            Text with empty lines removed
        """
        lines = text.split("\n")
        non_empty = [line for line in lines if line.strip()]
        return "\n".join(non_empty)

    def extract_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
        
        Returns:
            List of sentences
        """
        # Simple sentence splitting using common delimiters
        sentences = re.split(r"(?<=[.!?])\s+", text)
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences

    def get_statistics(self, text: str) -> dict:
        """
        Get text statistics.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dict with statistics
        """
        sentences = self.extract_sentences(text)
        return {
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": len(text.split("\n")),
            "sentence_count": len(sentences),
            "avg_word_length": (
                sum(len(w) for w in text.split()) / len(text.split())
                if text.split()
                else 0
            ),
        }
