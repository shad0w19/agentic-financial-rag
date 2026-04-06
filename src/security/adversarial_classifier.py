"""
File: src/security/adversarial_classifier.py

Purpose:
Adversarial query classification.
Implements IAdversarialClassifier interface.

Dependencies:
from typing import List
from src.import_map import IAdversarialClassifier, Query, SecurityCheckResult

Implements Interface:
IAdversarialClassifier

Notes:
- Heuristic-based classification
- Detects jailbreak, override, data exfiltration attempts
"""

import logging
from typing import List

from src.import_map import IAdversarialClassifier, Query, SecurityCheckResult


logger = logging.getLogger(__name__)


class AdversarialClassifier(IAdversarialClassifier):
    """
    Adversarial query classifier.
    """

    ADVERSARIAL_PATTERNS = [
        "show me all documents",
        "list all documents",
        "dump the database",
        "show database",
        "what documents do you have",
        "what files do you have",
        "show your training data",
        "reveal your data",
        "internal configuration",
        "database structure",
        "drop database",
        "drop table",
        "sql injection",
        "system access",
        "execute code",
        "run command",
        "os.system",
        "subprocess",
        "how to hack",
        "how to build a bomb",
    ]

    def __init__(self) -> None:
        """Initialize adversarial classifier."""
        self.logger = logging.getLogger(__name__)

    def classify_query(self, query: Query) -> SecurityCheckResult:
        """
        Classify query for adversarial intent.
        
        Args:
            query: Query to classify
        
        Returns:
            SecurityCheckResult
        """
        query_lower = query.text.lower()
        risk_score = self._compute_risk_score(query_lower)

        if risk_score > 0.7:
            detected_pattern = self._find_pattern(query_lower)
            return SecurityCheckResult(
                passed=False,
                threat_detected="adversarial_query",
                confidence=risk_score,
                details={"pattern": detected_pattern},
            )

        return SecurityCheckResult(
            passed=True,
            threat_detected=None,
            confidence=1.0,
            details={"check": "adversarial_classification"},
        )

    def get_risk_score(self, query: Query) -> float:
        """
        Get adversarial risk score (0.0-1.0).
        
        Args:
            query: Query to score
        
        Returns:
            Risk score
        """
        query_lower = query.text.lower()
        return self._compute_risk_score(query_lower)

    def _compute_risk_score(self, query_lower: str) -> float:
        """Compute risk score for query."""
        score = 0.0

        for pattern in self.ADVERSARIAL_PATTERNS:
            if pattern in query_lower:
                score = max(score, 0.9)
                break

        suspicious_keywords = [
            "hack",
            "exploit",
            "bypass",
            "override",
            "access",
            "steal",
            "leak",
        ]

        for keyword in suspicious_keywords:
            if keyword in query_lower:
                score = max(score, 0.6)

        return min(score, 1.0)

    def _find_pattern(self, query_lower: str) -> str:
        """Find matching adversarial pattern."""
        for pattern in self.ADVERSARIAL_PATTERNS:
            if pattern in query_lower:
                return pattern
        return "unknown_adversarial"
