"""
Phase B: Domain Classifier Module

Purpose:
Classify query domain into personal_tax, corporate_tax, gst, or multi-domain
using learned classification rather than simple keyword detection.

Architecture:
1. Multi-label capable (query can span multiple domains)
2. Confidence scoring per domain
3. Routes to appropriate retrieval system based on domain

Replaces:
- Simple regex/keyword domain detection
- Hardcoded domain routing rules
"""

import logging
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class Domain(Enum):
    """Query domain categories."""
    PERSONAL_TAX = "personal_tax"       # Individual income tax
    CORPORATE_TAX = "corporate_tax"     # Business/corporate tax
    GST = "gst"                         # GST/indirect tax
    INVESTMENT = "investment"           # Investments and portfolio planning
    REGULATORY = "regulatory"           # Compliance, FEMA, SEBI, DTAA
    MULTI = "multi"                     # Spans multiple domains OR fallback


@dataclass
class DomainClassification:
    """Result of domain classification."""
    primary_domain: Domain
    confidence: float
    domain_scores: Dict[Domain, float]  # Scores for all domains
    is_multi_domain: bool               # Whether query spans multiple domains
    domains_detected: List[Domain]      # All detected domains above threshold


class DomainClassifier:
    """
    Learned domain classifier for routing queries to appropriate retrieval systems.
    
    Advantages over keyword-only:
    - Understands context (e.g., "company dividend" → corporate, "personal dividend" → personal)
    - Confidence scoring enables intelligent fallback
    - Can identify multi-domain queries
    - More robust to paraphrasing
    """
    
    # Domain identification templates
    DOMAIN_INDICATORS = {
        Domain.PERSONAL_TAX: {
            "templates": [
                "Personal income tax for individuals",
                "Salary, wages, and employment income",
                "Personal deductions and exemptions",
                "Individual tax planning and filing",
                "Student, farmer, or individual taxpayer",
            ],
            "keywords": {
                "salary", "wage", "personal income", "individual", "employee",
                "freelance", "hmm", "house rent", "tuition", "dependent",
                "student", "farmer", "ae", "ltp",
            },
            "patterns": ["individual", "personal", "me", "my", "i", "salary", "employee"]
        },
        Domain.CORPORATE_TAX: {
            "templates": [
                "Corporate income tax for businesses",
                "Business, partnership, or company taxation",
                "Depreciation, capital gains, and business expenses",
                "Corporate tax planning and compliance",
                "Profit, loss, and business income",
            ],
            "keywords": {
                "corporate", "business", "company", "partnership", "enterprise",
                "profit", "loss", "depreciation", "capex", "ltd", "pvt",
                "dividend", "bonus", "expense", "deduction", "gdr", "adr",
            },
            "patterns": ["company", "business", "corporate", "profit", "loss", "partnership"]
        },
        Domain.GST: {
            "templates": [
                "Goods and Services Tax (GST)",
                "GST registration, rates, and compliance",
                "Input tax credit and IGST",
                "GST return filing and payment",
                "HSN codes and GST categories",
            ],
            "keywords": {
                "gst", "goods services tax", "tax rate", "hsn", "sac",
                "input credit", "igst", "cgst", "sgst", "cess",
                "invoice", "billing", "supply", "registration",
            },
            "patterns": ["gst", "tax rate", "input credit", "hsn", "sac"]
        },
        Domain.INVESTMENT: {
            "templates": [
                "Investment planning and portfolio strategy",
                "Mutual funds, SIP, stocks, bonds, and ETFs",
                "Retirement corpus and wealth accumulation",
                "Risk profile, diversification, and asset allocation",
                "Returns, compounding, and long-term investing",
            ],
            "keywords": {
                "investment", "mutual fund", "sip", "stp", "swp", "stock", "equity",
                "bond", "etf", "portfolio", "asset allocation", "risk", "return",
                "retirement", "nps", "ppf", "epf", "ulip", "scss", "reit", "invit",
            },
            "patterns": ["investment", "mutual fund", "sip", "portfolio", "retirement", "stock"]
        },
        Domain.REGULATORY: {
            "templates": [
                "Financial regulations and compliance in India",
                "SEBI, FEMA, and DTAA interpretation",
                "Regulatory filing and disclosure obligations",
                "Cross-border remittance and LRS compliance",
                "Circulars, notifications, and legal compliance rules",
            ],
            "keywords": {
                "sebi", "fema", "dtaa", "lrs", "fatca", "schedule fa", "compliance",
                "regulatory", "circular", "notification", "rbi", "cbic", "cbdt",
                "foreign remittance", "nre", "nro", "fcnr", "vda", "crypto", "194s",
            },
            "patterns": ["sebi", "fema", "dtaa", "compliance", "regulatory", "lrs"]
        }
    }
    
    # Thresholds
    MIN_DOMAIN_CONFIDENCE = 0.40       # Minimum to include domain in detection
    MULTI_DOMAIN_THRESHOLD = 2         # Number of domains above threshold → multi
    PRIMARY_DOMAIN_WIN_MARGIN = 0.15   # Gap needed to pick one vs multi
    LOW_CONFIDENCE_FANOUT_THRESHOLD = 0.55
    
    def __init__(self):
        """Initialize domain classifier."""
        self.logger = logging.getLogger(__name__)
    
    def classify(self, query_text: str) -> DomainClassification:
        """
        Classify query domain(s).
        
        Args:
            query_text: The query to classify
            
        Returns:
            DomainClassification with primary domain and confidence
        """
        query_lower = query_text.lower()
        
        # Step 1: Keyword scores
        keyword_scores = self._keyword_scores(query_lower)
        
        # Step 2: Semantic scores
        semantic_scores = self._semantic_scores(query_lower)
        
        # Step 3: Blend signals
        combined_scores = {
            domain: 0.6 * semantic_scores.get(domain, 0.0) + 0.4 * keyword_scores.get(domain, 0.0)
            for domain in Domain
        }
        
        # Step 4: Detect multi-domain
        detected_domains = [
            domain for domain, score in combined_scores.items()
            if score >= self.MIN_DOMAIN_CONFIDENCE
        ]
        
        sorted_domains = sorted(
            combined_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_domain, top_score = sorted_domains[0]
        second_score = sorted_domains[1][1] if len(sorted_domains) > 1 else 0.0
        score_gap = top_score - second_score

        is_multi_domain = (
            len(detected_domains) >= self.MULTI_DOMAIN_THRESHOLD
            and top_score < self.LOW_CONFIDENCE_FANOUT_THRESHOLD
            and score_gap < self.PRIMARY_DOMAIN_WIN_MARGIN
        )
        
        # Step 5: Determine primary domain
        if is_multi_domain:
            primary_domain = Domain.MULTI
            primary_confidence = sum(combined_scores.values()) / len(combined_scores)
        else:
            primary_domain = top_domain
            primary_confidence = top_score
        
        self.logger.debug(
            f"Domain classification: {primary_domain} ({primary_confidence:.2f}) | "
            f"Detected: {detected_domains} | Scores: {combined_scores}"
        )
        
        return DomainClassification(
            primary_domain=primary_domain,
            confidence=primary_confidence,
            domain_scores=combined_scores,
            is_multi_domain=is_multi_domain,
            domains_detected=detected_domains,
        )
    
    def _keyword_scores(self, query_lower: str) -> Dict[Domain, float]:
        """
        Calculate keyword-based scores for each domain.
        """
        scores = {}
        
        for domain, indicators in self.DOMAIN_INDICATORS.items():
            keywords = indicators["keywords"]
            keyword_matches = sum(1 for kw in keywords if kw in query_lower)
            score = keyword_matches / max(len(keywords), 1)  # Normalize by domain keywords
            scores[domain] = min(1.0, score * 1.5)  # Scale up but cap at 1.0
        
        return scores
    
    def _semantic_scores(self, query_lower: str) -> Dict[Domain, float]:
        """
        Calculate semantic (template) scores for each domain.
        """
        scores = {}
        
        for domain, indicators in self.DOMAIN_INDICATORS.items():
            templates = indicators["templates"]
            # Average similarity to templates for this domain
            similarities = [
                self._template_similarity(query_lower, template.lower())
                for template in templates
            ]
            avg_score = sum(similarities) / len(similarities) if similarities else 0.0
            scores[domain] = avg_score
        
        return scores
    
    def _template_similarity(self, query: str, template: str) -> float:
        """
        Calculate similarity between query and template.
        
        Similar to intent classifier but focused on domains.
        """
        query_words = set(query.split())
        template_words = set(template.split())
        
        # Jaccard similarity
        intersection = query_words & template_words
        union = query_words | template_words
        jaccard_sim = len(intersection) / len(union) if union else 0.0
        
        # Concept bonus
        query_concepts = [w for w in query.split() if len(w) > 3]
        template_concepts = [w for w in template.split() if len(w) > 3]
        concept_overlap = set(query_concepts) & set(template_concepts)
        concept_bonus = min(0.15, len(concept_overlap) * 0.04)
        
        similarity = min(1.0, jaccard_sim + concept_bonus)
        return similarity
    
    def get_retrieval_domains(self, classification: DomainClassification) -> List[Domain]:
        """
        Get the domain(s) to use for retrieval based on classification.
        
        Args:
            classification: DomainClassification result
            
        Returns:
            List of domains to query (could be multiple for multi-domain)
        """
        if classification.is_multi_domain:
            # Query all detected domains
            return [d for d in classification.domains_detected if d != Domain.MULTI]
        else:
            # Query single domain
            if classification.primary_domain != Domain.MULTI:
                return [classification.primary_domain]
            else:
                # Fallback: query all domains
                return list(self.DOMAIN_INDICATORS.keys())
    
    def should_use_fallback(self, confidence: float) -> bool:
        """
        Determine if confidence is too low and we should use fallback routing.
        """
        return confidence < 0.5
