"""
Phase B: Intent Classifier Module

Purpose:
Classify query intent into tax-grounded, general-finance, or trivial categories
using learned classification rather than keyword patterns.

Architecture:
1. Zero-shot classification using embeddings (no fine-tuning required)
2. Confidence scoring to determine fallback to pattern matching
3. Calibrated thresholds for routing decisions

Replaces:
- KW.match-based trivial detection
- KW.match-based general finance detection
- Brittle domain routing logic
"""

import logging
from typing import Tuple, Optional, Dict, Any
from enum import Enum


logger = logging.getLogger(__name__)


class Intent(Enum):
    """Query intent categories."""
    TAX_GROUNDED = "tax_grounded"       # Tax-specific, domain expert queries
    GENERAL_FINANCE = "general_finance" # Finance but not tax-specific
    TRIVIAL = "trivial"                 # Greeting/small talk
    OUT_OF_SCOPE = "out_of_scope"       # Not finance/tax domain


class IntentClassifier:
    """
    Learned intent classifier using embedding-based zero-shot classification.
    
    Advantages over keyword patterns:
    - Learns semantic meaning, not exact string matches
    - Harder to evade with paraphrasing
    - Confidence scores enable graceful fallback
    - Can be updated without code changes
    """
    
    # Intent templates for zero-shot classification
    INTENT_TEMPLATES = {
        Intent.TAX_GROUNDED: [
            "This is a question about tax law, tax deductions, tax filings, or tax compliance.",
            "This query requires knowledge of income tax, GST, corporate tax, or tax regulations.",
            "The user is asking about tax calculations, tax planning, or tax advisory.",
            "This is a tax-specific financial query requiring expert tax knowledge.",
            "The question involves income tax slabs, deductions, exemptions, or filing procedures.",
        ],
        Intent.GENERAL_FINANCE: [
            "This is a general finance question about investments, banking, or financial planning.",
            "The query is about financial products, money management, or financial advice not specific to taxes.",
            "This is a general finance topic like mutual funds, stocks, bonds, or compound interest.",
            "The user is asking about financial services, investment strategies, or savings concepts.",
            "This involves personal finance concepts like interest rates, returns, or wealth management.",
        ],
        Intent.TRIVIAL: [
            "This is a greeting, small talk, or casual conversation.",
            "This is a trivial question like 'how are you' or general chit-chat.",
            "This is a question about the assistant itself, not about finance.",
            "This is a short greeting or polite acknowledgement from the user.",
            "The user is saying hello, thanks, or asking what this assistant can do.",
        ],
        Intent.OUT_OF_SCOPE: [
            "This query is unrelated to finance, taxes, investments, or compliance.",
            "The user is asking for non-financial general knowledge or entertainment.",
            "The question belongs to another domain like sports, weather, coding, or medicine.",
            "This is not a finance advisory request and should be deflected politely.",
            "The request is outside the assistant's supported financial scope.",
        ]
    }
    
    # Confidence thresholds for routing decisions
    MIN_CONFIDENCE_FOR_ROUTING = 0.65  # Below this, use fallback to pattern matching
    TAX_GROUNDED_THRESHOLD = 0.70      # Need higher confidence for tax routing
    
    # Keywords that strongly indicate intent (backup if confidence is low)
    TAX_KEYWORDS = {
        "tax", "income tax", "gst", "corporate tax", "tds", "hra", "section 80",
        "deduction", "filing", "itr", "return", "assessment", "fy", "ay",
        "refund", "capital gains", "dividend", "interest income", "salary",
        "freelance", "business income", "loss", "cess", "surcharge",
        "fema", "sebi", "dtaa", "lrs", "fatca", "compliance", "regulatory", "vda", "crypto",
    }
    
    FINANCE_KEYWORDS = {
        "finance", "investment", "mutual fund", "sip", "stock", "share", "bond",
        "insurance", "loan", "credit", "banking", "deposit", "savings",
        "portfolio", "ipo", "etf", "retirement", "pension", "fund",
        "compound interest", "interest", "return", "yield", "diversification",
        "asset allocation", "nav", "expense", "dividend", "bull", "bear",
    }
    
    TRIVIAL_KEYWORDS = {
        "hello", "hi", "hey", "how are you", "thanks", "thank you",
        "help", "support", "what can you do", "who are you", "what are you",
    }

    OUT_OF_SCOPE_KEYWORDS = {
        "joke", "funny", "movie", "sports", "cricket", "football", "weather",
        "recipe", "medical", "medicine", "doctor", "diagnosis", "programming",
        "python code", "capital of", "history", "geography", "astrology", "horoscope",
        "love advice", "dating", "game", "music", "song", "kill", "murder", "violence",
    }

    AMBIGUOUS_TAX_KEYWORDS = {
        "tax", "taxes", "deduction", "deductions", "filing", "compliance", "return",
    }

    SPECIFIC_TAX_MARKERS = {
        "personal", "individual", "gst", "corporate", "section", "80c", "80d", "tds", "itr", "slab", "regime",
        "fy", "ay", "salary", "business", "capital gains",
        "investment", "portfolio", "mutual fund", "sip", "retirement", "fema", "sebi", "dtaa", "lrs",
    }
    
    def __init__(self):
        """Initialize intent classifier."""
        self.logger = logging.getLogger(__name__)
        self._embedding_cache: Dict[str, Any] = {}
    
    def classify(self, query_text: str) -> Tuple[Intent, float, Dict[str, float], bool, str]:
        """
        Classify query intent using learned classification.
        
        Args:
            query_text: The query to classify
            
        Returns:
            (intent, confidence, scores_dict, is_ambiguous, ambiguity_reason)
            - intent: Best matching Intent category
            - confidence: Confidence score [0, 1]
            - scores_dict: Scores for all intents (for debugging/logging)
            - is_ambiguous: Whether the query needs clarification before execution
            - ambiguity_reason: Human-readable reason for ambiguity
        """
        if not query_text or len(query_text.strip()) < 3:
            return Intent.TRIVIAL, 1.0, {
                Intent.TAX_GROUNDED: 0.0,
                Intent.GENERAL_FINANCE: 0.0,
                Intent.TRIVIAL: 1.0,
                Intent.OUT_OF_SCOPE: 0.0,
            }, False, ""
        
        query_lower = query_text.lower()
        
        # Step 1: Keyword-based quick detection (fast path for obvious cases)
        keyword_intent, keyword_confidence = self._keyword_classification(query_lower)
        if keyword_confidence >= 0.85:  # High confidence from keywords
            self.logger.debug(f"Quick keyword match: {keyword_intent} ({keyword_confidence:.2f})")
            scores = {
                Intent.TAX_GROUNDED: 0.95 if keyword_intent == Intent.TAX_GROUNDED else 0.02,
                Intent.GENERAL_FINANCE: 0.95 if keyword_intent == Intent.GENERAL_FINANCE else 0.02,
                Intent.TRIVIAL: 0.95 if keyword_intent == Intent.TRIVIAL else 0.02,
                Intent.OUT_OF_SCOPE: 0.95 if keyword_intent == Intent.OUT_OF_SCOPE else 0.02,
            }
            ambiguous, reason = self._detect_ambiguity(query_lower, keyword_intent, scores)
            return keyword_intent, keyword_confidence, scores, ambiguous, reason
        
        # Step 2: Semantic classification (learned)
        semantic_intent, semantic_confidence, scores = self._semantic_classification(query_text)
        
        # Step 3: Fuse keyword and semantic signals
        if keyword_confidence > 0.4:
            # Weight keyword signal
            blended_confidence = 0.55 * semantic_confidence + 0.45 * keyword_confidence
            # If keyword is strong for a specific intent, prefer it
            if keyword_intent == semantic_intent:
                blended_confidence = max(semantic_confidence, keyword_confidence)
        else:
            blended_confidence = semantic_confidence
        
        # Step 4: Return best intent with confidence
        best_intent = semantic_intent
        
        self.logger.debug(
            f"Classification: {best_intent} ({blended_confidence:.2f}) | "
            f"Scores: {scores}"
        )

        is_ambiguous, ambiguity_reason = self._detect_ambiguity(query_lower, best_intent, scores)
        return best_intent, blended_confidence, scores, is_ambiguous, ambiguity_reason
    
    def _keyword_classification(self, query_lower: str) -> Tuple[Intent, float]:
        """
        Fast keyword-based classification (backup signal).
        
        Returns:
            (intent, confidence)
        """
        tax_score = sum(1 for kw in self.TAX_KEYWORDS if kw in query_lower)
        finance_score = sum(1 for kw in self.FINANCE_KEYWORDS if kw in query_lower)
        trivial_score = sum(1 for kw in self.TRIVIAL_KEYWORDS if kw in query_lower)
        out_scope_score = sum(1 for kw in self.OUT_OF_SCOPE_KEYWORDS if kw in query_lower)
        
        total = tax_score + finance_score + trivial_score + out_scope_score
        if total < 1:
            return Intent.OUT_OF_SCOPE, 0.60
        
        max_score = max(tax_score, finance_score, trivial_score, out_scope_score)
        confidence = max_score / (total + 1)  # +1 to handle edge cases
        
        if tax_score == max_score and tax_score > 0:
            return Intent.TAX_GROUNDED, min(1.0, confidence * 1.3)
        elif finance_score == max_score and finance_score > 0:
            return Intent.GENERAL_FINANCE, min(1.0, confidence * 1.3)
        elif trivial_score > 0:
            return Intent.TRIVIAL, min(1.0, confidence * 1.3)
        elif out_scope_score > 0:
            return Intent.OUT_OF_SCOPE, min(1.0, confidence * 1.3)
        else:
            return Intent.OUT_OF_SCOPE, 0.60
    
    def _semantic_classification(self, query_text: str) -> Tuple[Intent, float, Dict[str, float]]:
        """
        Semantic classification using embedding-based zero-shot.
        
        This uses a simple heuristic: score how well query_text matches
        intent templates using substring/semantic similarity.
        
        In production, this would use:
        - Cross-encoder model (e.g., SBERT's CrossEncoder)
        - LLM-based classification (e.g., via API)
        - Embeddings + cosine similarity
        
        For now, we implement a lightweight heuristic-based approach.
        """
        query_lower = query_text.lower()
        
        # Calculate similarity to each intent template
        scores = {}
        for intent, templates in self.INTENT_TEMPLATES.items():
            # Average similarity to templates for this intent
            similarities = [
                self._template_similarity(query_lower, template.lower())
                for template in templates
            ]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            scores[intent] = avg_similarity
        
        # Boost scores if strong keyword signals present
        tax_keyword_score = sum(1 for kw in self.TAX_KEYWORDS if kw in query_lower) / max(len(self.TAX_KEYWORDS), 1)
        finance_keyword_score = sum(1 for kw in self.FINANCE_KEYWORDS if kw in query_lower) / max(len(self.FINANCE_KEYWORDS), 1)
        trivial_keyword_score = sum(1 for kw in self.TRIVIAL_KEYWORDS if kw in query_lower) / max(len(self.TRIVIAL_KEYWORDS), 1)
        out_scope_keyword_score = sum(1 for kw in self.OUT_OF_SCOPE_KEYWORDS if kw in query_lower) / max(len(self.OUT_OF_SCOPE_KEYWORDS), 1)
        
        scores[Intent.TAX_GROUNDED] = min(1.0, scores[Intent.TAX_GROUNDED] + tax_keyword_score * 0.2)
        scores[Intent.GENERAL_FINANCE] = min(1.0, scores[Intent.GENERAL_FINANCE] + finance_keyword_score * 0.2)
        scores[Intent.TRIVIAL] = min(1.0, scores[Intent.TRIVIAL] + trivial_keyword_score * 0.2)
        scores[Intent.OUT_OF_SCOPE] = min(1.0, scores[Intent.OUT_OF_SCOPE] + out_scope_keyword_score * 0.25)
        
        # Find best intent
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        
        # Normalize scores to [0, 1] probability distribution
        total = sum(scores.values())
        normalized_scores = {k: v / total for k, v in scores.items()} if total > 0 else {k: 0.33 for k in scores}
        
        return best_intent, best_score, normalized_scores
    
    def _template_similarity(self, query: str, template: str) -> float:
        """
        Simple heuristic similarity between query and template.
        
        Measures:
        1. Word overlap (Jaccard similarity)
        2. Common semantic concepts (longer words)
        3. Concept order preservation
        """
        query_words = set(query.split())
        template_words = set(template.split())
        
        # Jaccard similarity
        intersection = query_words & template_words
        union = query_words | template_words
        jaccard_sim = len(intersection) / len(union) if union else 0.0
        
        # Bonus for key concepts appearing (words > 3 chars)
        query_concepts = [w.lower() for w in query.split() if len(w) > 3]
        template_concepts = [w.lower() for w in template.split() if len(w) > 3]
        concept_overlap = len(set(query_concepts) & set(template_concepts))
        concept_bonus = min(0.3, concept_overlap * 0.08)
        
        # Check if template concepts are in query (semantic relevance)
        for concept in template_concepts[:3]:  # Check first 3 key concepts
            if concept in query.lower():
                concept_bonus += 0.05
        
        similarity = min(1.0, jaccard_sim + concept_bonus)
        return similarity
    
    def should_use_fallback(self, confidence: float) -> bool:
        """
        Determine if confidence is too low and we should fallback to patterns.
        
        Args:
            confidence: Classification confidence [0, 1]
            
        Returns:
            True if confidence is below threshold
        """
        return confidence < self.MIN_CONFIDENCE_FOR_ROUTING

    def _detect_ambiguity(self, query_lower: str, best_intent: Intent, scores: Dict[Intent, float]) -> Tuple[bool, str]:
        """Detect whether query needs clarification before routing."""
        # Clarification is currently only required for vague tax asks.
        if best_intent != Intent.TAX_GROUNDED:
            return False, ""

        words = [w for w in query_lower.split() if w]
        top_scores = sorted(scores.values(), reverse=True)
        score_gap = top_scores[0] - top_scores[1] if len(top_scores) >= 2 else 1.0

        has_generic_tax_keyword = any(kw in query_lower for kw in self.AMBIGUOUS_TAX_KEYWORDS)
        has_specific_marker = any(marker in query_lower for marker in self.SPECIFIC_TAX_MARKERS)
        very_short_query = len(words) <= 6

        if has_generic_tax_keyword and very_short_query and not has_specific_marker:
            return True, "generic_tax_query"

        if score_gap < 0.12 and very_short_query:
            return True, "low_intent_separation"

        return False, ""
