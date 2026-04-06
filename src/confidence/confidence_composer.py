"""
Phase C: Confidence Composition Module

Purpose:
Compose confidence scores from multiple sources (retrieval quality, reasoning consistency,
verification) into a single answer confidence. Enable quality gating and answer override.

Architecture:
1. Retrieval Confidence: Quality of retrieved documents and BM25/vector match scores
2. Reasoning Confidence: Internal consistency of reasoning, contradiction detection
3. Verification Confidence: Answer verification against documents, absence of hallucination
4. Composed Confidence: Weighted blend of three signals [0, 1]
5. Quality Gating: Override answer if composed confidence < threshold (hedge, clarify, or admit uncertainty)

Replaces:
- Hard confidence thresholds in individual agents
- Single-source confidence signals
- Inability to detect hallucination/reasoning errors
"""

import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence bands for answer quality."""
    HIGH = "high"              # [0.80, 1.0] - Direct answer
    MEDIUM = "medium"          # [0.60, 0.80) - Answer with caveats
    LOW = "low"                # [0.40, 0.60) - Hedged answer / clarification needed
    VERY_LOW = "very_low"      # [0.0, 0.40) - Admit uncertainty / request clarification


@dataclass
class ConfidenceSignal:
    """Individual confidence signal from a source."""
    source: str                 # "retrieval", "reasoning", "verification"
    confidence: float           # [0, 1]
    details: Dict[str, float]   # Source-specific details


@dataclass
class ComposedConfidence:
    """Final composed confidence with gating decision."""
    overall_confidence: float   # [0, 1] - Final composed score
    confidence_level: ConfidenceLevel
    signals: List[ConfidenceSignal]  # Component signals
    should_override_answer: bool
    override_action: Optional[str]   # "hedge", "clarify", "admit_uncertainty"
    override_replacement: Optional[str]  # Replacement answer if overriding
    explanation: str            # Human-readable explanation


class ConfidenceComposer:
    """
    Compose confidence from multiple sources and apply quality gating.
    
    Weights:
    - Retrieval: 0.35 (document quality is foundation)
    - Reasoning: 0.35 (answer chain must be sound)
    - Verification: 0.30 (must verify against sources)
    
    Thresholds for gating:
    - HIGH (0.80): Direct answer
    - MEDIUM (0.60): Answer with caveats
    - LOW (0.40): Hedged answer + clarification offer
    - VERY_LOW (0.0): Admit uncertainty
    """
    
    # Weights for blending signals
    RETRIEVAL_WEIGHT = 0.35
    REASONING_WEIGHT = 0.35
    VERIFICATION_WEIGHT = 0.30
    
    # Gating thresholds
    OVERRIDE_THRESHOLD_HIGH = 0.80
    OVERRIDE_THRESHOLD_MEDIUM = 0.60
    OVERRIDE_THRESHOLD_LOW = 0.40
    
    def __init__(self):
        """Initialize confidence composer."""
        self.logger = logging.getLogger(__name__)
    
    def compose(
        self,
        retrieval_signal: ConfidenceSignal,
        reasoning_signal: ConfidenceSignal,
        verification_signal: ConfidenceSignal,
        answer_text: str,
    ) -> ComposedConfidence:
        """
        Compose confidence from three signals and determine gating action.
        
        Args:
            retrieval_signal: Document quality and relevance
            reasoning_signal: Consistency of reasoning chain
            verification_signal: Verification against sources
            answer_text: The answer being evaluated
            
        Returns:
            ComposedConfidence with gating decision
        """
        # Validate input signals
        assert 0 <= retrieval_signal.confidence <= 1
        assert 0 <= reasoning_signal.confidence <= 1
        assert 0 <= verification_signal.confidence <= 1
        
        # Step 1: Compute weighted average
        overall_confidence = (
            self.RETRIEVAL_WEIGHT * retrieval_signal.confidence +
            self.REASONING_WEIGHT * reasoning_signal.confidence +
            self.VERIFICATION_WEIGHT * verification_signal.confidence
        )
        
        # Step 2: Determine confidence level
        if overall_confidence >= self.OVERRIDE_THRESHOLD_HIGH:
            level = ConfidenceLevel.HIGH
        elif overall_confidence >= self.OVERRIDE_THRESHOLD_MEDIUM:
            level = ConfidenceLevel.MEDIUM
        elif overall_confidence >= self.OVERRIDE_THRESHOLD_LOW:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW
        
        # Step 3: Determine gating action
        should_override, action, replacement = self._determine_gating_action(
            overall_confidence,
            level,
            answer_text,
            retrieval_signal,
            reasoning_signal,
            verification_signal,
        )
        
        # Step 4: Build explanation
        explanation = self._build_explanation(
            level,
            retrieval_signal.confidence,
            reasoning_signal.confidence,
            verification_signal.confidence,
            should_override,
            action,
        )
        
        self.logger.debug(
            f"Composed confidence: {overall_confidence:.2f} ({level.value}) | "
            f"Signals: R={retrieval_signal.confidence:.2f}, "
            f"Reasoning={reasoning_signal.confidence:.2f}, "
            f"V={verification_signal.confidence:.2f} | "
            f"Override: {should_override} ({action})"
        )
        
        return ComposedConfidence(
            overall_confidence=overall_confidence,
            confidence_level=level,
            signals=[retrieval_signal, reasoning_signal, verification_signal],
            should_override_answer=should_override,
            override_action=action,
            override_replacement=replacement,
            explanation=explanation,
        )
    
    def _determine_gating_action(
        self,
        confidence: float,
        level: ConfidenceLevel,
        answer: str,
        retrieval_sig: ConfidenceSignal,
        reasoning_sig: ConfidenceSignal,
        verification_sig: ConfidenceSignal,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Determine if answer should be gated/overridden and what action to take.
        
        Returns:
            (should_override, action, replacement_answer)
        """
        if level == ConfidenceLevel.HIGH:
            # Direct answer, no override
            return False, None, None
        
        elif level == ConfidenceLevel.MEDIUM:
            # Add caveat to answer
            caveat = " (Note: Based on available information, but please verify for critical decisions.)"
            return False, "caveat", caveat  # Don't override, but could add caveat tag
        
        elif level == ConfidenceLevel.LOW:
            # Offer clarification
            action = "clarify"
            replacement = (
                f"I have partial information about your question, but I'm not fully confident. "
                f"Could you provide more specific details? For example: {self._suggest_clarification(answer)}"
            )
            return True, action, replacement
        
        else:  # VERY_LOW
            # Admit uncertainty
            action = "admit_uncertainty"
            replacement = (
                f"I don't have enough reliable information to answer this question confidently. "
                f"Would you like me to explain what information is available, or would you prefer to "
                f"ask a more specific question?"
            )
            return True, action, replacement
    
    def _suggest_clarification(self, answer: str) -> str:
        """
        Suggest clarification questions based on answer length/content.
        """
        # Simple heuristic: if answer is short, ask for elaboration
        if len(answer) < 100:
            return "What specific aspect would you like to understand better?"
        else:
            return "Which part would you like me to focus on?"
    
    def _build_explanation(
        self,
        level: ConfidenceLevel,
        retrieval_conf: float,
        reasoning_conf: float,
        verification_conf: float,
        should_override: bool,
        action: Optional[str],
    ) -> str:
        """Build human-readable explanation of confidence composition."""
        parts = [
            f"Confidence: {level.value}",
            f"Retrieval quality: {retrieval_conf:.0%}",
            f"Reasoning consistency: {reasoning_conf:.0%}",
            f"Verification: {verification_conf:.0%}",
        ]
        
        if should_override:
            parts.append(f"Action: {action}")
        
        return " | ".join(parts)
    
    def get_confidence_level_description(self, level: ConfidenceLevel) -> str:
        """Get user-friendly description of confidence level."""
        descriptions = {
            ConfidenceLevel.HIGH: "Very confident - can be used as-is",
            ConfidenceLevel.MEDIUM: "Reasonably confident - suggest noting caveats",
            ConfidenceLevel.LOW: "Low confidence - should offer clarification",
            ConfidenceLevel.VERY_LOW: "Very low confidence - should admit uncertainty",
        }
        return descriptions[level]
