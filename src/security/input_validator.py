"""
File: src/security/input_validator.py

Purpose:
Input validation for queries.
Implements ISecurityValidator interface.

Dependencies:
from typing import Tuple, List, Dict, Any
from src.import_map import ISecurityValidator, Query, SecurityCheckResult

Implements Interface:
ISecurityValidator

Notes:
- Basic validation: empty, length, invalid chars
- Rule-based checks
"""

import logging
from typing import Any, Dict, List, Tuple

from src.import_map import ISecurityValidator, Query, SecurityCheckResult


logger = logging.getLogger(__name__)


class InputValidator(ISecurityValidator):
    """
    Input validation for queries.
    """

    MAX_QUERY_LENGTH = 10000
    MIN_QUERY_LENGTH = 3
    INVALID_CHARS = ["\x00", "\x01", "\x02", "\x03"]
    SHORT_TRIVIAL_ALLOWLIST = {"hi", "hey", "yo", "ok"}

    def __init__(self) -> None:
        """Initialize input validator."""
        self.logger = logging.getLogger(__name__)

    def validate_query(self, query: Query) -> SecurityCheckResult:
        """
        Validate query syntax and structure.
        
        Args:
            query: Query to validate
        
        Returns:
            SecurityCheckResult
        """
        if not query.text:
            return SecurityCheckResult(
                passed=False,
                threat_detected="empty_query",
                confidence=1.0,
                details={"reason": "Query text is empty"},
            )

        query_text = (query.text or "").strip().lower()

        # Allow short trivial greetings to pass security and route normally.
        if query_text in self.SHORT_TRIVIAL_ALLOWLIST:
            return SecurityCheckResult(
                passed=True,
                threat_detected=None,
                confidence=1.0,
                details={"check": "input_validation_short_trivial_allowlist"},
            )

        if len(query.text) < self.MIN_QUERY_LENGTH:
            return SecurityCheckResult(
                passed=False,
                threat_detected="query_too_short",
                confidence=1.0,
                details={"reason": f"Query < {self.MIN_QUERY_LENGTH} chars"},
            )

        if len(query.text) > self.MAX_QUERY_LENGTH:
            return SecurityCheckResult(
                passed=False,
                threat_detected="query_too_long",
                confidence=1.0,
                details={"reason": f"Query > {self.MAX_QUERY_LENGTH} chars"},
            )

        for char in self.INVALID_CHARS:
            if char in query.text:
                return SecurityCheckResult(
                    passed=False,
                    threat_detected="invalid_characters",
                    confidence=1.0,
                    details={"reason": "Invalid control characters"},
                )

        return SecurityCheckResult(
            passed=True,
            threat_detected=None,
            confidence=1.0,
            details={"check": "input_validation"},
        )

    def validate_parameters(
        self,
        parameters: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Validate request parameters.
        
        Args:
            parameters: Parameters dict
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors: List[str] = []

        if not isinstance(parameters, dict):
            errors.append("Parameters must be a dict")
            return False, errors

        if "query" in parameters:
            query_text = parameters.get("query", "")
            if not isinstance(query_text, str):
                errors.append("Query must be a string")
            elif len(query_text) == 0:
                errors.append("Query cannot be empty")
            elif len(query_text) > self.MAX_QUERY_LENGTH:
                errors.append(f"Query exceeds {self.MAX_QUERY_LENGTH} chars")

        if "k" in parameters:
            k = parameters.get("k")
            if not isinstance(k, int) or k <= 0:
                errors.append("k must be positive integer")

        return len(errors) == 0, errors
