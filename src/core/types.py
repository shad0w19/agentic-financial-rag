"""
File: src/core/types.py

Purpose:
Central type definitions and shared schemas for the entire system.
Defines contracts for Query, AgentState, RetrievalResult, DocumentChunk, ToolResult, and PlanStep.

Dependencies:
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
from datetime import datetime

Implements Interface:
None (core type definitions)

Notes:
- All types are frozen dataclasses for immutability
- All fields are required or have explicit defaults
- No circular dependencies
- These types form the contract between all modules
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
from datetime import datetime


# ============================================================================
# ENUMS
# ============================================================================

class QueryType(str, Enum):
    """Types of financial queries the system handles."""
    TAX_PLANNING = "tax_planning"
    INCOME_TAX = "income_tax"
    CORPORATE_TAX = "corporate_tax"
    GST_RELATED = "gst_related"
    INVESTMENT = "investment"
    GENERAL = "general"


class DocumentSource(str, Enum):
    """Source of a document chunk."""
    PERSONAL_TAX = "personal_tax"
    CORPORATE_TAX = "corporate_tax"
    GST = "gst"
    INVESTMENT = "investment"
    REGULATORY = "regulatory"


class AgentPhase(str, Enum):
    """Phases of agent execution."""
    SECURITY_CHECK = "security_check"
    PLANNING = "planning"
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    VERIFICATION = "verification"
    COMPLETE = "complete"


class RetrievalStrategy(str, Enum):
    """Strategy for retrieval."""
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    FEDERATED = "federated"


class VerificationStatus(str, Enum):
    """Status of verification."""
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"


# ============================================================================
# CORE DOMAIN TYPES
# ============================================================================

@dataclass(frozen=True)
class Query:
    """
    User query with metadata.
    
    Attributes:
        text: The actual query text from the user
        query_type: Classified type of query
        user_id: Optional identifier for the user
        session_id: Optional session identifier for tracking
        timestamp: When the query was created
        metadata: Custom metadata key-value pairs
    """
    text: str
    query_type: QueryType = QueryType.GENERAL
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    """
    A chunk of a document with metadata.
    
    Attributes:
        chunk_id: Unique identifier for this chunk
        text: The actual text content
        source: Source of the document
        document_name: Name/path of original document
        chunk_index: Position in the original document
        page_number: Page number if applicable
        metadata: Document-level metadata
        embedding: Optional pre-computed embedding vector
    """
    chunk_id: str
    text: str
    source: DocumentSource
    document_name: str
    chunk_index: int
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass(frozen=True)
class RetrievalResult:
    """
    Result from a retrieval operation.
    
    Attributes:
        chunks: List of retrieved document chunks
        strategy_used: Which retrieval strategy was used
        scores: Relevance scores for each chunk
        query_used: The query that was executed
        timestamp: When retrieval occurred
    """
    chunks: List[DocumentChunk]
    strategy_used: RetrievalStrategy
    scores: List[float]
    query_used: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class ToolResult:
    """
    Result from a tool execution (e.g., tax calculation).
    
    Attributes:
        tool_name: Name of the tool that was executed
        success: Whether the tool execution succeeded
        output: The result data
        error: Error message if execution failed
        execution_time_ms: How long the tool took to execute
        metadata: Additional execution metadata
    """
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanStep:
    """
    A step in an execution plan.
    
    Attributes:
        step_id: Unique identifier for this step
        description: Human-readable description
        action_type: Type of action to perform
        parameters: Parameters for the action
        dependencies: IDs of steps this depends on
        order: Execution order
    """
    step_id: str
    description: str
    action_type: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    order: int = 0


@dataclass(frozen=True)
class ProvenanceEvent:
    """
    An event recorded for audit trail.
    
    Attributes:
        event_id: Unique identifier
        agent_name: Name of agent that generated this event
        action: What action was performed
        metadata: Event metadata
        timestamp: When event occurred
        parent_event_id: ID of parent event (for DAG tracking)
    """
    event_id: str
    agent_name: str
    action: str
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    parent_event_id: Optional[str] = None


# ============================================================================
# AGENT STATE
# ============================================================================

@dataclass
class AgentState:
    """
    Complete state of agent execution.
    
    Mutable state object shared between agents in the workflow.
    
    Attributes:
        query: Original user query
        query_type: Classified query type
        is_valid: Whether query passed security checks
        security_violations: List of security violations found
        plan: List of execution steps
        retrieved_context: Retrieved documents
        reasoning: Agent reasoning trace
        tool_results: Results from executed tools
        verification_status: Status of verification
        verification_notes: Notes from verification
        final_answer: Final response to user
        current_phase: Current execution phase
        error_message: Any error encountered
        execution_timestamp: When execution started
        metadata: Custom metadata
    """
    query: Query
    query_type: QueryType
    is_valid: bool = False
    security_violations: List[str] = field(default_factory=list)
    plan: List[PlanStep] = field(default_factory=list)
    retrieved_context: List[RetrievalResult] = field(default_factory=list)
    reasoning: Dict[str, Any] = field(default_factory=dict)
    tool_results: List[ToolResult] = field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.PENDING
    verification_notes: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None
    current_phase: AgentPhase = AgentPhase.SECURITY_CHECK
    error_message: Optional[str] = None
    execution_timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# VALIDATION CONTEXT
# ============================================================================

@dataclass(frozen=True)
class ValidationContext:
    """
    Context for validation operations.
    
    Attributes:
        query: Query being validated
        threat_patterns: Patterns to check for
        allowed_domains: Allowed domain types
        max_depth: Maximum recursion depth allowed
    """
    query: Query
    threat_patterns: List[str] = field(default_factory=list)
    allowed_domains: List[DocumentSource] = field(default_factory=list)
    max_depth: int = 5


@dataclass(frozen=True)
class SecurityCheckResult:
    """
    Result of a security check.
    
    Attributes:
        passed: Whether security check passed
        threat_detected: Which threat was detected (if any)
        confidence: Confidence level (0.0-1.0)
        details: Details about the check
    """
    passed: bool
    threat_detected: Optional[str] = None
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# TAX CALCULATION TYPES
# ============================================================================

@dataclass(frozen=True)
class IncomeBreakdown:
    """
    Breakdown of different income sources.
    
    Attributes:
        salary: Salary income
        capital_gains: Capital gains
        rental_income: Rental income
        business_income: Business income
        other_income: Other income
    """
    salary: float = 0.0
    capital_gains: float = 0.0
    rental_income: float = 0.0
    business_income: float = 0.0
    other_income: float = 0.0


@dataclass(frozen=True)
class DeductionBreakdown:
    """
    Breakdown of deductions.
    
    Attributes:
        section_80c: Standard deductible items (80C)
        section_80d: Health insurance
        section_80e: Education loan
        section_80tta: Interest on savings
        other_deductions: Other allowed deductions
    """
    section_80c: float = 0.0
    section_80d: float = 0.0
    section_80e: float = 0.0
    section_80tta: float = 0.0
    other_deductions: float = 0.0


@dataclass(frozen=True)
class TaxCalculationResult:
    """
    Result of tax calculation.
    
    Attributes:
        gross_income: Total income
        total_deductions: Total deductible amount
        taxable_income: Income after deductions
        tax_amount: Tax calculated
        effective_tax_rate: Effective tax rate percentage
        applicable_slab: Tax slab applied
        metadata: Additional calculation metadata
    """
    gross_income: float
    total_deductions: float
    taxable_income: float
    tax_amount: float
    effective_tax_rate: float
    applicable_slab: str
    metadata: Dict[str, Any] = field(default_factory=dict)
