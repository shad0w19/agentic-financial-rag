import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from src.agents.planner_agent import PlannerAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.verification_agent import VerificationAgent
from src.orchestration.workflow import AgentWorkflow
from src.security.security_gatekeeper import SecurityGatekeeper
from src.retrieval.federated_router import FederatedRouter
from src.retrieval.embedding_model import EmbeddingModel
from src.retrieval.reranker import Reranker
from src.retrieval.parallel_retriever import ParallelRetriever  # Phase D Tier 2
from src.import_map import DomainClassifier, Domain  # Phase D Tier 2
from src.data_pipeline.dataset_builder import DatasetBuilder
from src.config.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    REASONING_MODEL,
    GENERAL_MODEL,
    DEFAULT_TIMEOUT,
)

logger = logging.getLogger(__name__)
_workflow_instance: AgentWorkflow | None = None
_workflow_warmed = False

# Initialize FastAPI app
app = FastAPI(
    title="Financial Advisor Agent",
    description="Secure federated financial advisory system",
    version="1.0.0",
)


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    confidence: float
    sources: List[str]
    issues: List[str]


# ============ MULTI-LLM FUNCTIONS ============

def _get_reasoning_llm() -> ChatOpenAI:
    """DeepSeek-R1 for financial reasoning (deterministic, temperature=0)"""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")
    
    return ChatOpenAI(
        model=REASONING_MODEL,  # deepseek/deepseek-r1
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0,  # Deterministic reasoning
        timeout=DEFAULT_TIMEOUT,
        max_tokens=1800,  # Allow more detailed reasoning and complete answers
    )


def _get_general_llm() -> ChatOpenAI:
    """Qwen3-30B for general tasks (slight creativity, temperature=0.2)"""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")
    
    return ChatOpenAI(
        model=GENERAL_MODEL,  # qwen/qwen3-30b-a3b
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.2,  # Slight creativity for variety
        timeout=DEFAULT_TIMEOUT,
    )


def _create_reasoning_llm_generator(llm: ChatOpenAI):
    """Wrap LLM for ReasoningAgent with system prompt"""
    system_prompt = """You are an expert Indian Financial and Tax Advisor.
Answer strictly using the provided context.
Always include exact numbers (₹, %, limits).
If the answer is not in the context, output EXACTLY:
'Information not found.'"""
    
    def generator(prompt: str) -> str:
        try:
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])
            return response.content
        except Exception as e:
            logger.error(f"Reasoning LLM error: {e}")
            return "Error generating response"
    
    return generator


def _create_general_llm_generator(llm: ChatOpenAI):
    """Wrap LLM for Planner and Verification"""
    def generator(prompt: str) -> str:
        try:
            response = llm.invoke([{"role": "user", "content": prompt}])
            return response.content
        except Exception as e:
            logger.error(f"General LLM error: {e}")
            return "Error generating response"
    
    return generator


def initialize_workflow() -> AgentWorkflow:
    """Initialize all workflow components with federated retrieval."""
    gatekeeper = SecurityGatekeeper()
    
    # 1. Initialize federated router (replaces old single retriever)
    embedding_model = EmbeddingModel()
    router = FederatedRouter(
        embedding_model=embedding_model,
        index_dir="data/vector_store",
        domain_detection_model="qwen/qwen2.5-7b-instruct",
    )
    
    # Check if indices exist
    available_sources = router.get_available_sources()
    logger.info(f"✅ Available domain indices: {[s.value for s in available_sources]}")
    
    if not available_sources:
        logger.warning("⚠️  No domain indices found. Run: python src/data_pipeline/run_pipeline.py")
    
    # Phase D Tier 2: Initialize ParallelRetriever for multi-domain queries
    domain_classifier = DomainClassifier()
    parallel_retriever = None
    reranker = None

    try:
        reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("✅ Reranker initialized for retrieval merge")
    except Exception as e:
        logger.warning(f"⚠️  Reranker initialization failed: {e}. Continuing without reranker.")
    
    try:
        # For Phase D Tier 2: Use the router itself as domain-specific retrievers
        # (Each domain is accessed through the domain classification in router)
        # This is a simplified approach - can be optimized in Tier 3
        if len(available_sources) >= 2:  # At least 2 domains available
            parallel_retriever = ParallelRetriever(
                domain_classifier=domain_classifier,
                personal_tax_retriever=router,  # Use router for all domains
                corporate_tax_retriever=router,
                gst_retriever=router,
                investment_retriever=router,
                regulatory_retriever=router,
                reranker=reranker,
            )
            logger.info("✅ ParallelRetriever initialized (Phase D Tier 2)")
        else:
            logger.warning("⚠️  Insufficient domain indices for ParallelRetriever (need ≥2 domains)")
    except Exception as e:
        logger.warning(f"⚠️  ParallelRetriever initialization failed: {e}. Will use sequential retrieval.")
        parallel_retriever = None
    
    # 2. Create retrieval agent with federated router AND parallel retriever (Phase D)
    retriever = RetrievalAgent(
        retriever=router,
        parallel_retriever=parallel_retriever,
        domain_classifier=domain_classifier,
    )
    
    # 3. Initialize Multi-LLM for agents
    reasoning_llm = _get_reasoning_llm()
    reasoning_generator = _create_reasoning_llm_generator(reasoning_llm)
    reasoner = ReasoningAgent(llm_generator=reasoning_generator)
    
    general_llm = _get_general_llm()
    general_generator = _create_general_llm_generator(general_llm)
    
    # Wire agents with LLMs
    planner = PlannerAgent(llm_generator=general_generator)
    verifier = VerificationAgent(llm_generator=general_generator)

    workflow = AgentWorkflow(
        gatekeeper=gatekeeper,
        planner=planner,
        retriever=retriever,
        reasoner=reasoner,
        verifier=verifier,
    )
    
    # Attach router and embedding_model for run_demo.py parallelization
    workflow.router = router
    workflow.embedding_model = embedding_model

    return workflow


def _warmup_workflow(workflow: AgentWorkflow) -> None:
    """One-time preload/warmup to reduce first-query latency spikes."""
    global _workflow_warmed
    if _workflow_warmed:
        return

    try:
        router = getattr(workflow, "router", None)
        if router and hasattr(router, "preload_all_retrievers"):
            router.preload_all_retrievers()
        if hasattr(workflow, "retriever") and getattr(workflow.retriever, "domain_classifier", None):
            workflow.retriever.domain_classifier.classify("income tax slab and GST threshold")
        _workflow_warmed = True
        logger.info("✅ Workflow warmup completed")
    except Exception as exc:
        logger.warning(f"Workflow warmup skipped due to error: {exc}")


def get_workflow() -> AgentWorkflow:
    """Lazy workflow initialization to avoid heavy import-time side effects."""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = initialize_workflow()
    _warmup_workflow(_workflow_instance)
    return _workflow_instance


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "financial-advisor-agent"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Process financial advisory query."""
    try:
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if len(request.query) > 10000:
            raise HTTPException(
                status_code=400, detail="Query exceeds maximum length"
            )

        # Run workflow with safe error handling
        try:
            state = get_workflow().run(request.query)
        except Exception as e:
            print("ERROR during workflow execution:", str(e))
            return QueryResponse(
                answer="Service temporarily unavailable. Our AI is currently experiencing high load.",
                confidence=0.0,
                sources=[],
                issues=["llm_execution_error"]
            )

        # Handle blocked queries gracefully - return HTTP 200 with blocked message
        if hasattr(state, 'is_blocked') and state.is_blocked:
            return QueryResponse(
                answer="BLOCKED: Malicious prompt detected",
                confidence=0.0,
                sources=[],
                issues=["security_block"]
            )

        # Extract response data - handle both dataclass and dict cases
        if hasattr(state, 'answer'):
            answer = state.answer or "Unable to generate answer"
        else:
            answer = "Unable to generate answer"
        
        confidence = 0.0
        issues: List[str] = []

        # Handle verification - can be dict or attribute
        if hasattr(state, 'verification') and state.verification:
            verification = state.verification
            if isinstance(verification, dict):
                confidence = verification.get("confidence", 0.0)
                issues = verification.get("issues", [])
            else:
                confidence = 0.0

        # Extract sources
        sources: List[str] = []
        # Handle both attribute access and blocked queries
        if hasattr(state, 'retrieved_docs'):
            retrieved_docs = state.retrieved_docs
        else:
            retrieved_docs = None
        
        if retrieved_docs:
            for result in retrieved_docs:
                if hasattr(result, 'chunks'):
                    for chunk in result.chunks:
                        source = f"{chunk.source.value}:{chunk.document_name}"
                        if source not in sources:
                            sources.append(source)
                elif isinstance(result, dict) and 'chunks' in result:
                    for chunk in result['chunks']:
                        source = f"{chunk.get('source', 'unknown')}:{chunk.get('document_name', 'unknown')}"
                        if source not in sources:
                            sources.append(source)

        return QueryResponse(
            answer=answer,
            confidence=confidence,
            sources=sources,
            issues=issues,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        )


@app.get("/status")
async def status_endpoint() -> dict:
    """Get system status."""
    return {
        "service": "financial-advisor-agent",
        "version": "1.0.0",
        "status": "operational",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)