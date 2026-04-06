"""
File: src/import_map.py

Purpose:
Central import registry for the entire system.
All modules import core types and interfaces from here to ensure consistency.

Dependencies:
All core types and interface definitions

Implements Interface:
None (registry only)

Notes:
- All modules must use this as their single import source for core/interface definitions
- Prevents circular imports
- Makes it easy to track all dependencies
- Enables clean refactoring without cascading changes
"""

# ============================================================================
# CORE TYPE IMPORTS
# ============================================================================

from src.core.types import (
    # Enums
    QueryType,
    DocumentSource,
    AgentPhase,
    RetrievalStrategy,
    VerificationStatus,
    # Core Domain Types
    Query,
    DocumentChunk,
    RetrievalResult,
    ToolResult,
    PlanStep,
    ProvenanceEvent,
    # Agent State
    AgentState,
    # Validation
    ValidationContext,
    SecurityCheckResult,
    # Tax Types
    IncomeBreakdown,
    DeductionBreakdown,
    TaxCalculationResult,
)

# ============================================================================
# INTERFACE IMPORTS
# ============================================================================

from src.interfaces.retriever import IRetriever
from src.interfaces.calculator import ITaxCalculator, IInvestmentCalculator
from src.interfaces.security import (
    ISecurityValidator,
    IInjectionDetector,
    IAdversarialClassifier,
    ISecurityGatekeeper,
)
from src.interfaces.provenance import IProvenanceTracker, IProvenanceGraph
from src.interfaces.embedding import IEmbeddingModel, IVectorIndex

# ============================================================================
# CLASSIFIER IMPORTS (Phase B: Model-Centric Routing)
# ============================================================================

from src.classifiers.intent_classifier import IntentClassifier, Intent
from src.classifiers.domain_classifier import DomainClassifier, Domain, DomainClassification

# ============================================================================
# CONFIDENCE & QUALITY IMPORTS (Phase C: Confidence Composition & Quality Gating)
# ============================================================================

from src.confidence.confidence_composer import ConfidenceComposer, ConfidenceLevel, ComposedConfidence
from src.confidence.answer_quality_evaluator import (
    RetrievalQualityEvaluator,
    ReasoningQualityEvaluator,
    VerificationQualityEvaluator,
    RetrievalQualityMetrics,
    ReasoningQualityMetrics,
    VerificationQualityMetrics,
)

# ============================================================================
# NOTE: Phase D (Latency Optimization) imports are excluded from central registry
# to avoid circular imports. Import directly from:
#   - src.retrieval.parallel_retriever (ParallelRetriever, etc.)
#   - src.services.response_cache (ResponseCache, etc.)
# ============================================================================

# ============================================================================
# EXPORT ALL SYMBOLS
# ============================================================================

__all__ = [
    # Enums
    "QueryType",
    "DocumentSource",
    "AgentPhase",
    "RetrievalStrategy",
    "VerificationStatus",
    # Core Domain Types
    "Query",
    "DocumentChunk",
    "RetrievalResult",
    "ToolResult",
    "PlanStep",
    "ProvenanceEvent",
    # Agent State
    "AgentState",
    # Validation
    "ValidationContext",
    "SecurityCheckResult",
    # Tax Types
    "IncomeBreakdown",
    "DeductionBreakdown",
    "TaxCalculationResult",
    # Interfaces
    "IRetriever",
    "ITaxCalculator",
    "IInvestmentCalculator",
    "ISecurityValidator",
    "IInjectionDetector",
    "IAdversarialClassifier",
    "ISecurityGatekeeper",
    "IProvenanceTracker",
    "IProvenanceGraph",
    "IEmbeddingModel",
    "IVectorIndex",
    # Classifiers (Phase B)
    "IntentClassifier",
    "Intent",
    "DomainClassifier",
    "Domain",
    "DomainClassification",
    # Confidence & Quality (Phase C)
    "ConfidenceComposer",
    "ConfidenceLevel",
    "ComposedConfidence",
    "RetrievalQualityEvaluator",
    "ReasoningQualityEvaluator",
    "VerificationQualityEvaluator",
    "RetrievalQualityMetrics",
    "ReasoningQualityMetrics",
    "VerificationQualityMetrics",
    # Latency Optimization (Phase D)
    "ParallelRetriever",
    "ParallelRetrievalMetrics",
    "ParallelRetrievalBenchmark",
    "ResponseCache",
    "CacheEntry",
    "CacheStats",
    "CacheWarmer",
]
