"""
Phase C: Answer Quality Evaluator Module

Purpose:
Evaluate answer quality across three dimensions:
1. Retrieval Quality: How good are the retrieved documents?
2. Reasoning Quality: Is the reasoning chain consistent and sound?
3. Verification Quality: Does the answer verify against source documents?

Replaces:
- Relying on single LLM confidence scores
- No hallucination detection
- No reasoning consistency checks
- No source verification
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class RetrievalQualityMetrics:
    """Metrics for evaluating retrieved documents."""
    num_documents: int
    avg_relevance_score: float  # [0, 1] - BM25/vector similarity
    top_doc_relevance: float    # [0, 1] - Best document relevance
    coverage_score: float       # [0, 1] - How well docs cover query
    recency_score: float        # [0, 1] - Document freshness (for time-sensitive domains)


@dataclass
class ReasoningQualityMetrics:
    """Metrics for evaluating reasoning chain."""
    steps_count: int
    consistency_score: float    # [0, 1] - No contradictions detected
    grounding_score: float      # [0, 1] - Steps grounded in documents
    logic_validity_score: float # [0, 1] - Logical validity of chain
    contradiction_count: int    # Number of contradictions found


@dataclass
class VerificationQualityMetrics:
    """Metrics for verifying answer against sources."""
    verified_claims: int        # Number of claims verified in sources
    unverified_claims: int      # Number of claims without source support
    contradicted_claims: int    # Number of claims contradicting sources
    hallucination_score: float  # [0, 1] - Inverse of hallucination risk


class RetrievalQualityEvaluator:
    """Evaluate quality of retrieved documents."""
    
    MIN_DOCS_FOR_GROUNDING = 2
    GOOD_RELEVANCE_THRESHOLD = 0.70
    EXCELLENT_RELEVANCE_THRESHOLD = 0.85
    
    def __init__(self):
        """Initialize retrieval quality evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def evaluate(
        self,
        query: str,
        retrieved_docs: List[Dict],
        relevance_scores: Optional[List[float]] = None,
    ) -> Tuple[float, RetrievalQualityMetrics]:
        """
        Evaluate quality of retrieved documents.
        
        Args:
            query: Original query
            retrieved_docs: List of retrieved documents
            relevance_scores: Optional relevance scores [0, 1] for each doc
            
        Returns:
            (quality_confidence [0,1], metrics)
        """
        if not retrieved_docs:
            return 0.0, RetrievalQualityMetrics(
                num_documents=0,
                avg_relevance_score=0.0,
                top_doc_relevance=0.0,
                coverage_score=0.0,
                recency_score=0.0,
            )
        
        # Default relevance scores if not provided
        if relevance_scores is None:
            relevance_scores = [0.7] * len(retrieved_docs)  # Assume reasonable retrieval
        
        # Calculate metrics
        num_docs = len(retrieved_docs)
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        top_relevance = max(relevance_scores) if relevance_scores else 0.0
        
        # Coverage: Do documents adequately cover the query terms?
        coverage = self._estimate_coverage(query, retrieved_docs)
        
        # Recency: For time-sensitive domains (tax), prefer recent docs
        recency = self._estimate_recency(retrieved_docs)
        
        metrics = RetrievalQualityMetrics(
            num_documents=num_docs,
            avg_relevance_score=avg_relevance,
            top_doc_relevance=top_relevance,
            coverage_score=coverage,
            recency_score=recency,
        )
        
        # Combine into confidence score
        confidence = self._combine_scores(metrics)
        
        self.logger.debug(
            f"Retrieval quality: {confidence:.2f} | "
            f"Docs: {num_docs}, Relevance: {avg_relevance:.2f}, Coverage: {coverage:.2f}"
        )
        
        return confidence, metrics
    
    def _estimate_coverage(self, query: str, docs: List[Dict]) -> float:
        """Estimate how well documents cover query terms."""
        query_terms = set(query.lower().split())
        query_terms = {t for t in query_terms if len(t) > 2}  # Remove short words
        
        if not query_terms:
            return 0.8  # Default if no good terms
        
        # Check coverage across documents
        covered_terms = set()
        for doc in docs:
            doc_text = (doc.get("text", "") or "").lower()
            for term in query_terms:
                if term in doc_text:
                    covered_terms.add(term)
        
        coverage = len(covered_terms) / len(query_terms) if query_terms else 0.5
        return min(1.0, coverage)
    
    def _estimate_recency(self, docs: List[Dict]) -> float:
        """Estimate document recency (for date-sensitive domains like tax)."""
        # Simple heuristic: if any doc is recent, boost recency score
        # In production, would check actual timestamps
        recency = 0.7  # Default moderate recency
        return min(1.0, recency)
    
    def _combine_scores(self, metrics: RetrievalQualityMetrics) -> float:
        """Combine retrieval metrics into single confidence score."""
        # Weight: relevance (0.5) + coverage (0.3) + num_docs (0.2)
        relevance_boost = metrics.avg_relevance_score * 0.5
        coverage_boost = metrics.coverage_score * 0.3
        doc_count_boost = min(0.2, (metrics.num_documents / 3.0) * 0.2)  # Cap at 0.2 for 3+ docs
        
        confidence = relevance_boost + coverage_boost + doc_count_boost
        return min(1.0, confidence)


class ReasoningQualityEvaluator:
    """Evaluate quality of reasoning chain."""
    
    def __init__(self):
        """Initialize reasoning quality evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def evaluate(
        self,
        reasoning_chain: Optional[str],
        reasoning_steps: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> Tuple[float, ReasoningQualityMetrics]:
        """
        Evaluate reasoning chain consistency and quality.
        
        Args:
            reasoning_chain: Full reasoning narrative
            reasoning_steps: Individual reasoning steps (optional)
            query: Original query for grounding check
            
        Returns:
            (quality_confidence [0,1], metrics)
        """
        if not reasoning_chain or not reasoning_steps:
            # No reasoning available → low confidence
            return 0.5, ReasoningQualityMetrics(
                steps_count=0,
                consistency_score=0.5,
                grounding_score=0.5,
                logic_validity_score=0.5,
                contradiction_count=0,
            )
        
        steps_count = len(reasoning_steps)
        consistency_score = self._check_consistency(reasoning_steps)
        grounding_score = self._estimate_grounding(reasoning_steps)
        logic_validity_score = self._check_logic_validity(reasoning_steps)
        contradictions = self._find_contradictions(reasoning_steps)
        
        metrics = ReasoningQualityMetrics(
            steps_count=steps_count,
            consistency_score=consistency_score,
            grounding_score=grounding_score,
            logic_validity_score=logic_validity_score,
            contradiction_count=len(contradictions),
        )
        
        # Combine into confidence score
        confidence = self._combine_scores(metrics)
        
        self.logger.debug(
            f"Reasoning quality: {confidence:.2f} | "
            f"Steps: {steps_count}, Consistency: {consistency_score:.2f}, "
            f"Contradictions: {len(contradictions)}"
        )
        
        return confidence, metrics
    
    def _check_consistency(self, steps: List[str]) -> float:
        """Check for contradictions within reasoning steps."""
        if not steps:
            return 0.5
        
        # Simple heuristic: check for negation patterns in consecutive steps
        contradictions_found = 0
        for i in range(len(steps) - 1):
            if self._steps_contradict(steps[i], steps[i + 1]):
                contradictions_found += 1
        
        consistency = max(0.0, 1.0 - (contradictions_found * 0.2))
        return consistency
    
    def _steps_contradict(self, step1: str, step2: str) -> bool:
        """Check if two steps logically contradict."""
        # Simple pattern matching: look for opposing words/concepts
        negation_patterns = [
            ("not ", " "),
            ("cannot ", "can "),
            ("deny", "affirm"),
        ]
        
        step1_lower = step1.lower()
        step2_lower = step2.lower()
        
        for neg, pos in negation_patterns:
            if neg in step1_lower and pos in step2_lower:
                return True
        
        return False
    
    def _estimate_grounding(self, steps: List[str]) -> float:
        """Estimate how much reasoning is grounded in premises."""
        # Heuristic: look for source/document references ("according to", "document says", etc.)
        grounding_indicators = ["according to", "document", "source", "says", "states", "mentions"]
        grounded_steps = 0
        
        for step in steps:
            step_lower = step.lower()
            for indicator in grounding_indicators:
                if indicator in step_lower:
                    grounded_steps += 1
                    break
        
        grounding = grounded_steps / len(steps) if steps else 0.5
        return min(1.0, grounding)
    
    def _check_logic_validity(self, steps: List[str]) -> float:
        """Check logical validity of reasoning chain."""
        # Simplified: assume valid unless we find logical fallacies
        fallacy_indicators = ["obviously", "clearly", "everyone knows"]
        
        fallacy_count = 0
        for step in steps:
            step_lower = step.lower()
            for fallacy in fallacy_indicators:
                if fallacy in step_lower:
                    fallacy_count += 1
        
        validity = max(0.0, 1.0 - (fallacy_count * 0.15))
        return validity
    
    def _find_contradictions(self, steps: List[str]) -> List[Tuple[int, int]]:
        """Find contradicting step pairs."""
        contradictions = []
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                if self._steps_contradict(steps[i], steps[j]):
                    contradictions.append((i, j))
        return contradictions
    
    def _combine_scores(self, metrics: ReasoningQualityMetrics) -> float:
        """Combine reasoning metrics into single confidence score."""
        # Weight: consistency (0.4) + validity (0.3) + grounding (0.3)
        consistency_boost = metrics.consistency_score * 0.4
        validity_boost = metrics.logic_validity_score * 0.3
        grounding_boost = metrics.grounding_score * 0.3
        
        # Penalize contradictions
        contradiction_penalty = min(0.3, metrics.contradiction_count * 0.1)
        
        confidence = consistency_boost + validity_boost + grounding_boost - contradiction_penalty
        return max(0.0, min(1.0, confidence))


class VerificationQualityEvaluator:
    """Evaluate answer verification against source documents."""
    
    def __init__(self):
        """Initialize verification quality evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def evaluate(
        self,
        answer: str,
        source_docs: Optional[List[Dict]] = None,
    ) -> Tuple[float, VerificationQualityMetrics]:
        """
        Evaluate answer verification against sources.
        
        Args:
            answer: Answer text to verify
            source_docs: Source documents for verification
            
        Returns:
            (quality_confidence [0,1], metrics)
        """
        if not source_docs or not answer:
            # Can't verify without sources → moderate confidence
            return 0.6, VerificationQualityMetrics(
                verified_claims=0,
                unverified_claims=0,
                contradicted_claims=0,
                hallucination_score=0.6,
            )
        
        # Extract claims from answer
        claims = self._extract_claims(answer)
        
        # Verify each claim against sources
        verified_count = 0
        contradicted_count = 0
        unverified_count = len(claims)
        
        for claim in claims:
            verification_status = self._verify_claim(claim, source_docs)
            if verification_status == "verified":
                verified_count += 1
                unverified_count -= 1
            elif verification_status == "contradicted":
                contradicted_count += 1
                unverified_count -= 1
        
        # Hallucination score: inverse of unverified + contradicted ratio
        total_issues = unverified_count + contradicted_count
        hallucination_risk = total_issues / len(claims) if claims else 0.0
        hallucination_score = max(0.0, 1.0 - hallucination_risk)
        
        metrics = VerificationQualityMetrics(
            verified_claims=verified_count,
            unverified_claims=unverified_count,
            contradicted_claims=contradicted_count,
            hallucination_score=hallucination_score,
        )
        
        # Confidence is hallucination score with penalty if contradictions found
        confidence = hallucination_score - (contradicted_count * 0.2)
        confidence = max(0.0, min(1.0, confidence))
        
        self.logger.debug(
            f"Verification quality: {confidence:.2f} | "
            f"Verified: {verified_count}, Unverified: {unverified_count}, "
            f"Contradicted: {contradicted_count}"
        )
        
        return confidence, metrics
    
    def _extract_claims(self, answer: str) -> List[str]:
        """Extract individual claims from answer text."""
        # Simple heuristic: split by sentences
        sentences = answer.split(".")
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims
    
    def _verify_claim(self, claim: str, source_docs: List[Dict]) -> str:
        """
        Verify a single claim against source documents.
        
        Returns:
            "verified" - claim found in sources
            "contradicted" - contradictory info in sources
            "unverified" - no information in sources
        """
        claim_lower = claim.lower()
        
        for doc in source_docs:
            doc_text = (doc.get("text", "") or "").lower()
            
            # Check for verbatim match or strong similarity
            if claim_lower in doc_text or self._similar_text(claim_lower, doc_text):
                return "verified"
        
        # Check for contradictions
        for doc in source_docs:
            doc_text = (doc.get("text", "") or "").lower()
            if self._contradicts(claim_lower, doc_text):
                return "contradicted"
        
        return "unverified"
    
    def _similar_text(self, text1: str, text2: str) -> bool:
        """Check if text snippets are similar."""
        # Simple: check for 70% word overlap
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2) / len(words1 | words2)
        return overlap > 0.7
    
    def _contradicts(self, claim: str, doc_text: str) -> bool:
        """Check if claim contradicts document text."""
        # Look for negation of claim elements
        if "not" in claim and claim.replace("not ", "") in doc_text:
            return True
        if "no" in claim and "yes" in doc_text:
            return True
        
        return False
