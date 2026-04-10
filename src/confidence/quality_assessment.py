"""
Quality Assessment Pipeline: Compute confidence signals across all three dimensions

Purpose:
Orchestrate the three quality evaluators (retrieval, reasoning, verification)
and combine their outputs into unified confidence signals for the ConfidenceComposer.

Usage:
    assessment_pipeline = QualityAssessmentPipeline()
    result = assessment_pipeline.assess_answer(
        query="...",
        retrieved_docs=[...],
        reasoning_chain="...",
        reasoning_steps=[...],
        answer="...",
        relevance_scores=[...],
    )
    
    retrieval_signal = result['retrieval_signal']  # [0, 1]
    reasoning_signal = result['reasoning_signal']  # [0, 1]
    verification_signal = result['verification_signal']  # [0, 1]
    overall_quality = result['overall_quality']  # [0, 1]
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.confidence.answer_quality_evaluator import (
    RetrievalQualityEvaluator,
    ReasoningQualityEvaluator,
    VerificationQualityEvaluator,
    VerificationQualityMetrics,
)


logger = logging.getLogger(__name__)


@dataclass
class QualityAssessmentResult:
    """Result of quality assessment across all dimensions."""
    retrieval_signal: float        # [0, 1] - Document quality confidence
    reasoning_signal: float        # [0, 1] - Reasoning chain confidence
    verification_signal: float     # [0, 1] - Answer verification confidence
    overall_quality: float         # [0, 1] - Average of all signals
    metrics: Dict[str, Any]        # Detailed metrics from each evaluator
    explanation: str               # Human-readable assessment


class QualityAssessmentPipeline:
    """
    Orchestrate answer quality assessment across three dimensions.
    
    Computes signals that feed into ConfidenceComposer for final confidence
    composition and quality gating decisions.
    """
    
    def __init__(self):
        """Initialize quality assessment pipeline with all evaluators."""
        self.retrieval_evaluator = RetrievalQualityEvaluator()
        self.reasoning_evaluator = ReasoningQualityEvaluator()
        self.verification_evaluator = VerificationQualityEvaluator()
        self.logger = logging.getLogger(__name__)
    
    def assess_answer(
        self,
        query: str,
        retrieved_docs: Optional[List[Dict]] = None,
        reasoning_chain: Optional[str] = None,
        reasoning_steps: Optional[List[str]] = None,
        answer: Optional[str] = None,
        relevance_scores: Optional[List[float]] = None,
        existing_verification: Optional[Dict[str, Any]] = None,
    ) -> QualityAssessmentResult:
        """
        Perform complete quality assessment of an answer.
        
        Args:
            query: Original query text
            retrieved_docs: Documents retrieved for context
            reasoning_chain: Full reasoning narrative
            reasoning_steps: Individual reasoning steps
            answer: Final answer to evaluate
            relevance_scores: Optional relevance scores for documents
            existing_verification: Existing workflow verification result to reuse
            
        Returns:
            QualityAssessmentResult with three signals and overall quality
        """
        # Step 1: Assess retrieval quality
        retrieval_confidence, retrieval_metrics = self.retrieval_evaluator.evaluate(
            query=query,
            retrieved_docs=retrieved_docs or [],
            relevance_scores=relevance_scores,
        )
        
        # Step 2: Assess reasoning quality
        reasoning_confidence, reasoning_metrics = self.reasoning_evaluator.evaluate(
            reasoning_chain=reasoning_chain,
            reasoning_steps=reasoning_steps or [],
            query=query,
        )
        
        # Step 3: Assess verification quality
        if isinstance(existing_verification, dict) and "confidence" in existing_verification:
            verification_confidence = max(0.0, min(1.0, float(existing_verification.get("confidence", 0.0))))
            verification_issues = existing_verification.get("issues", []) or []
            verification_is_valid = bool(existing_verification.get("is_valid", False))
            verification_metrics = VerificationQualityMetrics(
                verified_claims=1 if verification_is_valid else 0,
                unverified_claims=0 if verification_is_valid else len(verification_issues),
                contradicted_claims=0,
                hallucination_score=verification_confidence,
            )
            self.logger.debug(
                "Using workflow verification in Phase C: valid=%s confidence=%.2f",
                verification_is_valid,
                verification_confidence,
            )
        else:
            verification_confidence, verification_metrics = self.verification_evaluator.evaluate(
                answer=answer or "",
                source_docs=retrieved_docs or [],
            )
        
        # Step 4: Combine signals
        overall_quality = (
            retrieval_confidence * 0.33 +
            reasoning_confidence * 0.33 +
            verification_confidence * 0.34
        )
        
        # Step 5: Build explanation
        explanation = self._build_assessment_explanation(
            retrieval_confidence,
            reasoning_confidence,
            verification_confidence,
            retrieval_metrics,
            reasoning_metrics,
            verification_metrics,
        )
        
        # Step 6: Aggregate metrics
        metrics = {
            'retrieval': {
                'confidence': retrieval_confidence,
                'metrics': {
                    'num_documents': retrieval_metrics.num_documents,
                    'avg_relevance_score': retrieval_metrics.avg_relevance_score,
                    'coverage_score': retrieval_metrics.coverage_score,
                }
            },
            'reasoning': {
                'confidence': reasoning_confidence,
                'metrics': {
                    'steps_count': reasoning_metrics.steps_count,
                    'consistency_score': reasoning_metrics.consistency_score,
                    'contradictions': reasoning_metrics.contradiction_count,
                }
            },
            'verification': {
                'confidence': verification_confidence,
                'source': 'workflow' if isinstance(existing_verification, dict) and "confidence" in existing_verification else 'phase_c_recomputed',
                'metrics': {
                    'verified_claims': verification_metrics.verified_claims,
                    'unverified_claims': verification_metrics.unverified_claims,
                    'contradicted_claims': verification_metrics.contradicted_claims,
                }
            },
        }
        
        result = QualityAssessmentResult(
            retrieval_signal=retrieval_confidence,
            reasoning_signal=reasoning_confidence,
            verification_signal=verification_confidence,
            overall_quality=overall_quality,
            metrics=metrics,
            explanation=explanation,
        )
        
        self.logger.info(
            f"Quality assessment complete: "
            f"Retrieval={retrieval_confidence:.2f}, "
            f"Reasoning={reasoning_confidence:.2f}, "
            f"Verification={verification_confidence:.2f}, "
            f"Overall={overall_quality:.2f}"
        )
        
        return result
    
    def _build_assessment_explanation(
        self,
        retrieval_conf: float,
        reasoning_conf: float,
        verification_conf: float,
        retrieval_metrics: Any,
        reasoning_metrics: Any,
        verification_metrics: Any,
    ) -> str:
        """Build human-readable quality assessment explanation."""
        parts = []
        
        # Retrieval explanation
        parts.append(
            f"Retrieval: {retrieval_conf:.0%} "
            f"({retrieval_metrics.num_documents} docs, "
            f"relevance {retrieval_metrics.avg_relevance_score:.0%})"
        )
        
        # Reasoning explanation
        contradiction_note = ""
        if reasoning_metrics.contradiction_count > 0:
            contradiction_note = f", {reasoning_metrics.contradiction_count} contradictions"
        parts.append(
            f"Reasoning: {reasoning_conf:.0%} "
            f"({reasoning_metrics.steps_count} steps, "
            f"consistency {reasoning_metrics.consistency_score:.0%}{contradiction_note})"
        )
        
        # Verification explanation
        parts.append(
            f"Verification: {verification_conf:.0%} "
            f"({verification_metrics.verified_claims} verified, "
            f"{verification_metrics.unverified_claims + verification_metrics.contradicted_claims} unverified/contradicted)"
        )
        
        return " | ".join(parts)
    
    def assess_retrieval_only(
        self,
        query: str,
        retrieved_docs: List[Dict],
        relevance_scores: Optional[List[float]] = None,
    ) -> float:
        """Quick assessment of retrieval quality only."""
        retrieval_confidence, _ = self.retrieval_evaluator.evaluate(
            query=query,
            retrieved_docs=retrieved_docs,
            relevance_scores=relevance_scores,
        )
        return retrieval_confidence
    
    def assess_reasoning_only(
        self,
        reasoning_chain: str,
        reasoning_steps: List[str],
    ) -> float:
        """Quick assessment of reasoning quality only."""
        reasoning_confidence, _ = self.reasoning_evaluator.evaluate(
            reasoning_chain=reasoning_chain,
            reasoning_steps=reasoning_steps,
        )
        return reasoning_confidence
    
    def assess_verification_only(
        self,
        answer: str,
        source_docs: List[Dict],
    ) -> float:
        """Quick assessment of verification quality only."""
        verification_confidence, _ = self.verification_evaluator.evaluate(
            answer=answer,
            source_docs=source_docs,
        )
        return verification_confidence
