"""
File: src/interfaces/security.py

Purpose:
Abstract interface for security operations.
Defines contract for input validation, injection detection, adversarial detection.

Dependencies:
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from src.core.types import Query, SecurityCheckResult, ValidationContext

Implements Interface:
ISecurityValidator, IInjectionDetector, IAdversarialClassifier (abstract base classes)

Notes:
- Security pipeline must run BEFORE retrieval
- All three checks must pass before query proceeds
- Blocked queries never reach agent pipeline
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

from src.core.types import Query, SecurityCheckResult, ValidationContext


class ISecurityValidator(ABC):
    """
    Abstract interface for input validation.
    
    Validates query syntax, length, and basic structure.
    """

    @abstractmethod
    def validate_query(self, query: Query) -> SecurityCheckResult:
        """
        Validate query syntax and structure.
        
        Args:
            query: Query to validate
        
        Returns:
            SecurityCheckResult with validation status
        """
        pass

    @abstractmethod
    def validate_parameters(
        self,
        parameters: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Validate request parameters.
        
        Args:
            parameters: Parameters dict to validate
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        pass


class IInjectionDetector(ABC):
    """
    Abstract interface for prompt injection detection.
    
    Detects attempts to manipulate system behavior through
    prompt injection, role-play manipulation, etc.
    """

    @abstractmethod
    def detect_injection(self, query: Query) -> SecurityCheckResult:
        """
        Detect prompt injection attempts.
        
        Patterns checked:
        - ignore/override instructions
        - system prompt revelation attempts
        - context bypass attempts
        - role manipulation
        
        Args:
            query: Query to check
        
        Returns:
            SecurityCheckResult with detection status
        """
        pass

    @abstractmethod
    def get_threat_patterns(self) -> List[str]:
        """
        Get list of detected threat patterns.
        
        Returns:
            List of threat patterns
        """
        pass


class IAdversarialClassifier(ABC):
    """
    Abstract interface for adversarial query detection.
    
    Detects queries designed to exploit system weaknesses,
    extract sensitive data, or cause harm.
    """

    @abstractmethod
    def classify_query(self, query: Query) -> SecurityCheckResult:
        """
        Classify query for adversarial intent.
        
        Detects:
        - Data exfiltration attempts
        - System abuse attempts
        - Harmful content
        - Retrieval poisoning attempts
        
        Args:
            query: Query to classify
        
        Returns:
            SecurityCheckResult with classification
        """
        pass

    @abstractmethod
    def get_risk_score(self, query: Query) -> float:
        """
        Get adversarial risk score (0.0-1.0).
        
        Args:
            query: Query to score
        
        Returns:
            Risk score where 1.0 is maximum risk
        """
        pass


class ISecurityGatekeeper(ABC):
    """
    Abstract interface for the security gatekeeper.
    
    Orchestrates all security checks and controls access
    to the agent pipeline.
    """

    @abstractmethod
    def check_query(self, query: Query) -> Tuple[bool, SecurityCheckResult]:
        """
        Run complete security check pipeline.
        
        Pipeline:
        1. Input validation
        2. Injection detection
        3. Adversarial classification
        
        All checks must pass for query to proceed.
        
        Args:
            query: Query to check
        
        Returns:
            Tuple of (passed: bool, SecurityCheckResult)
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_blocked_queries_count(self) -> int:
        """
        Get count of blocked queries.
        
        Returns:
            Number of blocked queries
        """
        pass

    @abstractmethod
    def get_security_metrics(self) -> Dict[str, Any]:
        """
        Get security metrics and statistics.
        
        Returns:
            Dict with metrics (blocked rate, threat types, etc.)
        """
        pass
