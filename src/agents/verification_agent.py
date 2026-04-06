"""
File: src/agents/verification_agent.py

Purpose:
Final validation layer for agent outputs using LLM.
Verifies correctness and grounding of answers.

Dependencies:
from typing import List, Dict, Any, Callable
from src.import_map import RetrievalResult, PlanStep

Implements Interface:
None (agent component)

Notes:
- LLM-based verification (uses general_llm)
- Includes rule-based validation as fallback
- Detects unsupported claims and contradictions
"""

import logging
import re
from typing import Any, Callable, Dict, List

from src.import_map import PlanStep, RetrievalResult


logger = logging.getLogger(__name__)


class VerificationAgent:
    """
    Verification agent for validating agent outputs using LLM.
    """

    UNSUPPORTED_CLAIM_KEYWORDS = [
        "always",
        "never",
        "guaranteed",
        "100%",
        "definitely",
        "certainly",
        "absolutely",
    ]

    FINANCIAL_KEYWORDS = [
        "tax",
        "income",
        "deduction",
        "rebate",
        "investment",
        "return",
        "corpus",
        "sip",
    ]

    def __init__(self, llm_generator: Callable[[str], str]) -> None:
        """
        Initialize verification agent with LLM.
        
        Args:
            llm_generator: Callable that takes prompt and returns verification
        """
        self.logger = logging.getLogger(__name__)
        self.llm_generator = llm_generator

    def verify(
        self,
        answer: str,
        retrieved_docs: List[RetrievalResult],
        plan: List[PlanStep],
    ) -> Dict[str, Any]:
        """
        Verify answer correctness and grounding using LLM.
        
        Args:
            answer: Generated answer
            retrieved_docs: Retrieved documents
            plan: Execution plan
        
        Returns:
            Verification result dict
        """
        # Create context from retrieved docs
        context = "\n".join(
            [chunk.text for r in retrieved_docs for chunk in r.chunks]
            if retrieved_docs else []
        )

        # Create verification prompt
        verification_prompt = f"""You are a financial answer verifier. Check if this answer is correct and grounded in the context.

Context (from documents):
{context if context else "NO CONTEXT PROVIDED"}

Answer to verify:
{answer}

Evaluate on:
1. Is the answer grounded in context? (yes/no)
2. Does it have correct numbers? (yes/no/na)
3. Are claims supported? (yes/no/na)
4. Confidence (0-100%)
5. Issues found (list or none)

Respond with JSON:
{{"is_valid": true/false, "grounded": true/false, "has_numbers": true/false, "confidence": 85, "issues": ["issue1"]}}"""

        try:
            # Call LLM for verification
            response = self.llm_generator(verification_prompt)
            
            # Parse response
            try:
                import json
                # Clean JSON response
                json_str = response.strip()
                if "```" in json_str:
                    json_str = json_str.split("```")[1]
                    if json_str.startswith("json"):
                        json_str = json_str[4:]
                    json_str = json_str.strip()
                
                result = json.loads(json_str)
                
                # Ensure required fields
                is_valid = result.get("is_valid", True)
                confidence = float(result.get("confidence", 50)) / 100.0
                issues = result.get("issues", [])
                
                return self._format_result(is_valid, issues, confidence)
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"Could not parse verification JSON: {e}, using fallback")
                return self._rule_based_verify(answer, retrieved_docs)
                
        except Exception as e:
            logger.error(f"LLM verification error: {e}, using fallback")
            return self._rule_based_verify(answer, retrieved_docs)

    def _rule_based_verify(
        self,
        answer: str,
        retrieved_docs: List[RetrievalResult],
    ) -> Dict[str, Any]:
        """Fallback rule-based verification."""
        issues: List[str] = []
        confidence = 1.0

        # Rule 1: If answer is "Information not found", it's always valid
        if answer.strip() == "Information not found.":
            return self._format_result(True, [], 1.0)

        # Rule 2: Check if answer is empty
        if not answer or len(answer.strip()) == 0:
            issues.append("Answer is empty")
            confidence = 0.0
            return self._format_result(False, issues, confidence)

        # Rule 3: Check if docs were retrieved
        if not retrieved_docs or not any(
            r.chunks for r in retrieved_docs
        ):
            issues.append("No retrieved documents to ground answer")
            confidence = 0.5

        # Rule 4: Check for numbers in context but NOT in answer
        context_has_numbers = self._context_has_numbers(retrieved_docs)
        answer_has_numbers = self._answer_has_numbers(answer)
        
        if context_has_numbers and not answer_has_numbers:
            issues.append("Answer lacks numerical extraction")
            confidence *= 0.7

        # Rule 5: Check for contradictions
        if self._answer_contradicts_context(answer, retrieved_docs):
            issues.append("Answer contradicts retrieved context")
            confidence *= 0.5

        # Rule 6: Check for unsupported claims
        unsupported = self._check_unsupported_claims(answer)
        if unsupported:
            issues.extend(unsupported)
            confidence *= 0.8

        # Final confidence calculation
        if issues:
            confidence = 0.5
        else:
            confidence = 0.9

        is_valid = len(issues) == 0 and confidence > 0.5

        return self._format_result(is_valid, issues, confidence)

    def _context_has_numbers(self, retrieved_docs: List[RetrievalResult]) -> bool:
        """Check if context contains any numbers."""
        for result in retrieved_docs:
            for chunk in result.chunks:
                if re.search(r'\d+', chunk.text):
                    return True
        return False

    def _answer_has_numbers(self, answer: str) -> bool:
        """Check if answer contains any numbers."""
        return bool(re.search(r'\d+', answer))

    def _answer_contradicts_context(
        self,
        answer: str,
        retrieved_docs: List[RetrievalResult],
    ) -> bool:
        """Check if answer contradicts context (simple check)."""
        if "no" in answer.lower() or "not" in answer.lower():
            doc_text = " ".join(
                [chunk.text for r in retrieved_docs for chunk in r.chunks]
            )
            # If context is substantial but answer is negative/short
            if len(doc_text) > 500 and len(answer) < 50:
                return True
        return False

    def _check_unsupported_claims(self, answer: str) -> List[str]:
        """Check for unsupported absolute claims."""
        issues = []
        answer_lower = answer.lower()

        for keyword in self.UNSUPPORTED_CLAIM_KEYWORDS:
            if keyword in answer_lower:
                issues.append(
                    f"Answer contains absolute claim: '{keyword}'"
                )

        return issues

    def _check_financial_accuracy(self, answer: str) -> List[str]:
        """Check for basic financial accuracy."""
        issues = []
        answer_lower = answer.lower()

        # Check if financial terms are used appropriately
        has_financial_terms = any(
            term in answer_lower for term in self.FINANCIAL_KEYWORDS
        )

        if not has_financial_terms:
            issues.append("Answer lacks financial terminology")

        # Check for negative percentages
        if "%" in answer:
            percentages = re.findall(r"-?\d+\.?\d*%", answer)
            for pct in percentages:
                try:
                    value = float(pct.rstrip("%"))
                    if value < -100 or value > 100:
                        issues.append(f"Invalid percentage: {pct}")
                except ValueError:
                    pass

        return issues

    def _check_grounding(
        self,
        answer: str,
        retrieved_docs: List[RetrievalResult],
    ) -> float:
        """Check if answer is grounded in retrieved docs."""
        if not retrieved_docs:
            return 0.0

        # Collect all doc text
        doc_text = " ".join(
            [chunk.text for r in retrieved_docs for chunk in r.chunks]
        )

        # Simple word overlap check
        answer_words = set(answer.lower().split())
        doc_words = set(doc_text.lower().split())

        overlap = len(answer_words & doc_words)
        total = len(answer_words)

        if total == 0:
            return 0.0

        return min(overlap / total, 1.0)

    def _format_result(
        self,
        is_valid: bool,
        issues: List[str],
        confidence: float,
    ) -> Dict[str, Any]:
        """Format verification result."""
        return {
            "is_valid": is_valid,
            "issues": issues,
            "confidence": max(0.0, min(confidence, 1.0)),
            "issue_count": len(issues),
        }

    def get_verification_summary(
        self,
        verification_result: Dict[str, Any],
    ) -> str:
        """Get human-readable verification summary."""
        if verification_result["is_valid"]:
            return f"✓ Valid (confidence: {verification_result['confidence']:.2f})"
        else:
            issues_str = "; ".join(verification_result["issues"])
            return f"✗ Invalid - {issues_str}"
