"""
File: src/security/security_gatekeeper.py

Purpose:
Security gatekeeper orchestrating validation pipeline.
Implements ISecurityGatekeeper interface.

Dependencies:
from typing import Tuple
from src.import_map import ISecurityGatekeeper, Query, SecurityCheckResult

Implements Interface:
ISecurityGatekeeper

Notes:
- Central entry point for security checks
- Orchestrates input validation, injection detection, adversarial classification
- Rule-based implementation (no ML yet)
"""

import logging
from typing import Dict, Any, List, Tuple

from src.import_map import (
    ISecurityGatekeeper,
    Query,
    SecurityCheckResult,
)


logger = logging.getLogger(__name__)


class SecurityGatekeeper(ISecurityGatekeeper):
    """
    Security gatekeeper orchestrating validation pipeline.
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
        "dan mode",
        "do anything now",
        "you are now a",
        "act as a",
        "pretend to be",
        "roleplay as",
        "simulate a",
    ]

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

    def __init__(self, log_path: str = "logs/audit_log.json") -> None:
        """
        Initialize security gatekeeper.
        
        Args:
            log_path: Path to audit log
        """
        self.log_path = log_path
        self.blocked_count = 0
        self.logger = logging.getLogger(__name__)

    def check_query(
        self,
        query: Query,
    ) -> Tuple[bool, SecurityCheckResult]:
        """
        Run complete security check pipeline.
        
        Args:
            query: Query to check
        
        Returns:
            Tuple of (passed: bool, SecurityCheckResult)
        """
        # Input validation
        validation_result = self._validate_input(query)
        if not validation_result.passed:
            self._log_security_event(query, validation_result, "HIGH")
            self.blocked_count += 1
            return False, validation_result

        # Injection detection
        injection_result = self._detect_injection(query)
        if not injection_result.passed:
            self._log_security_event(query, injection_result, "HIGH")
            self.blocked_count += 1
            return False, injection_result

        # Adversarial classification
        adversarial_result = self._classify_adversarial(query)
        if not adversarial_result.passed:
            self._log_security_event(query, adversarial_result, "MEDIUM")
            self.blocked_count += 1
            return False, adversarial_result

        # All checks passed
        passed_result = SecurityCheckResult(
            passed=True,
            threat_detected=None,
            confidence=1.0,
            details={"checks": ["input_validation", "injection_detection", "adversarial_classification"]},
        )
        self._log_security_event(query, passed_result, "LOW")
        return True, passed_result

    def log_security_event(
        self,
        query: Query,
        result: SecurityCheckResult,
        threat_level: str,
    ) -> None:
        """
        Log security event for audit trail.
        
        Args:
            query: Query that triggered event
            result: Security check result
            threat_level: "LOW", "MEDIUM", "HIGH", "CRITICAL"
        """
        self._log_security_event(query, result, threat_level)

    def get_blocked_queries_count(self) -> int:
        """Get count of blocked queries."""
        return self.blocked_count

    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and statistics."""
        return {
            "blocked_queries": self.blocked_count,
            "injection_patterns_count": len(self.INJECTION_PATTERNS),
            "adversarial_patterns_count": len(self.ADVERSARIAL_PATTERNS),
        }

    def _validate_input(self, query: Query) -> SecurityCheckResult:
        """Validate query syntax and structure."""
        if not query.text:
            return SecurityCheckResult(
                passed=False,
                threat_detected="empty_query",
                confidence=1.0,
                details={"reason": "Query text is empty"},
            )

        if len(query.text) > 10000:
            return SecurityCheckResult(
                passed=False,
                threat_detected="query_too_long",
                confidence=1.0,
                details={"reason": "Query exceeds 10000 characters"},
            )

        if len(query.text) < 3:
            return SecurityCheckResult(
                passed=False,
                threat_detected="query_too_short",
                confidence=1.0,
                details={"reason": "Query too short"},
            )

        return SecurityCheckResult(
            passed=True,
            threat_detected=None,
            confidence=1.0,
            details={"check": "input_validation"},
        )

    def _detect_injection(self, query: Query) -> SecurityCheckResult:
        """Detect prompt injection attempts."""
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

    def _classify_adversarial(self, query: Query) -> SecurityCheckResult:
        """Classify query for adversarial intent."""
        query_lower = query.text.lower()

        for pattern in self.ADVERSARIAL_PATTERNS:
            if pattern in query_lower:
                return SecurityCheckResult(
                    passed=False,
                    threat_detected="adversarial_query",
                    confidence=0.90,
                    details={"pattern": pattern},
                )

        return SecurityCheckResult(
            passed=True,
            threat_detected=None,
            confidence=1.0,
            details={"check": "adversarial_classification"},
        )

    def _log_security_event(
        self,
        query: Query,
        result: SecurityCheckResult,
        threat_level: str,
    ) -> None:
        """Log security event."""
        self.logger.info(
            f"Security event - Level: {threat_level}, "
            f"Threat: {result.threat_detected}, "
            f"Query: {query.text[:50]}..."
        )
