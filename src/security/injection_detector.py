"""
File: src/security/injection_detector.py

Purpose:
Prompt injection detection.
Implements IInjectionDetector interface.

Dependencies:
from typing import List
from src.import_map import IInjectionDetector, Query, SecurityCheckResult

Implements Interface:
IInjectionDetector

Notes:
- Pattern-based detection
- Rule-based (no ML)
"""

import logging
from typing import List

from src.import_map import IInjectionDetector, Query, SecurityCheckResult


logger = logging.getLogger(__name__)


class InjectionDetector(IInjectionDetector):
    """
    Prompt injection detector.
    """

    INJECTION_PATTERNS = [
        "ignore previous instructions",
        "ignore all instructions",
        "disregard instructions",
        "forget instructions",
        "override instructions",
        "reveal system prompt",
        "show system prompt",
        "print system prompt",
        "repeat your instructions",
        "show your instructions",
        "hidden instructions",
        "print all instructions",
        "ignore the knowledge base",
        "ignore the documents",
        "answer freely",
        "answer without restrictions",
        "respond freely",
        "speak freely",
        "bypass security",
        "bypass safety",
        "jailbreak",
        "developer mode",
        "DAN mode",
        "do anything now",
        "you are now a",
        "act as a",
        "pretend to be",
        "roleplay as",
        "simulate a",
        "you are an AI without",
        "you are not an AI",
    ]

    def __init__(self) -> None:
        """Initialize injection detector."""
        self.logger = logging.getLogger(__name__)

    def detect_injection(self, query: Query) -> SecurityCheckResult:
        """
        Detect prompt injection attempts.
        
        Args:
            query: Query to check
        
        Returns:
            SecurityCheckResult
        """
        query_lower = query.text.lower()

        for pattern in self.INJECTION_PATTERNS:
            if pattern in query_lower:
                return SecurityCheckResult(
                    passed=False,
                    threat_detected="prompt_injection",
                    confidence=0.95,
                    details={"pattern": pattern},
                )

        return SecurityCheckResult(
            passed=True,
            threat_detected=None,
            confidence=1.0,
            details={"check": "injection_detection"},
        )

    def get_threat_patterns(self) -> List[str]:
        """
        Get list of detected threat patterns.
        
        Returns:
            List of threat patterns
        """
        return self.INJECTION_PATTERNS.copy()
