"""Shared query runtime for Streamlit demos.

This module decouples UI apps from run_demo.py and exposes a stable,
low-latency run_query() method with three-lane routing:
  1. Trivial: Greetings and small talk (instant static response)
  2. General Finance: Finance concepts without RAG (fast LLM direct answer)
  3. Tax RAG: Tax/compliance/grounded questions (full multi-agent pipeline)
"""

from __future__ import annotations

import logging
import time
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, List, Optional, Literal

from langchain_openai import ChatOpenAI

from src.api.server import initialize_workflow
from src.config.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    GENERAL_MODEL,
    DEFAULT_TIMEOUT,
)
from src.security.security_gatekeeper import SecurityGatekeeper
from src.import_map import Query, IntentClassifier, Intent, DomainClassifier, Domain
from src.confidence.quality_assessment import QualityAssessmentPipeline
from src.confidence.confidence_composer import ConfidenceComposer
# Phase D: Latency Optimization (Parallel Retrieval & Caching)
from src.services.response_cache import ResponseCache
from src.retrieval.parallel_retriever import ParallelRetriever


logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """App-facing orchestrator implementing three-lane query routing."""

    # Trivial lane: exact match greetings and small talk
    _TRIVIAL_PROMPTS = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
    }

    # Tax/compliance keywords for heuristic-first routing
    _TAX_COMPLIANCE_KEYWORDS = {
        "tax", "income tax", "itr", "80c", "80d", "80gg", "hra", "deduction",
        "filing", "return", "assessment", "tds", "itc", "gst", "goods",
        "services", "supply", "invoice", "compliance", "threshold",
        "slab", "regime", "corporate tax", "company", "auditor", "audit",
        "form", "schedule", "appendix", "section", "penalty", "interest"
    }

    # General finance keywords for conceptual questions
    _GENERAL_FINANCE_KEYWORDS = {
        "mutual fund", "sip", "etf", "stock", "bond", "investment",
        "portfolio", "diversification", "risk", "return", "inflation",
        "compounding", "asset allocation", "nav", "expense ratio",
        "dividend", "yield", "liquidity", "volatility", "credit", "debit",
        "margin", "leverage", "hedge", "derivative", "option", "future",
    }

    def __init__(self, preload_faiss: bool = False) -> None:
        self.gatekeeper = SecurityGatekeeper()
        # Phase B: Model-centric routing
        self.intent_classifier = IntentClassifier()
        self.domain_classifier = DomainClassifier()
        # Phase C: Quality assessment and confidence composition
        self.quality_assessment = QualityAssessmentPipeline()
        self.confidence_composer = ConfidenceComposer()
        # Phase D: Latency Optimization
        self.response_cache = ResponseCache(cache_size_mb=100, default_ttl_seconds=3600)
        self.workflow = None
        self.router = None
        self._preload_faiss = preload_faiss
        self._general_llm: Optional[ChatOpenAI] = None
        self._timeout = DEFAULT_TIMEOUT or 30

    def run_query(self, query: str, timeout_seconds: int = 45) -> Dict[str, Any]:
        """
        Route query to appropriate lane: trivial, general_finance, or tax_rag.
        
        CRITICAL: All queries are security-checked BEFORE routing (Step 0).
        Returns normalized response payload with route metadata.
        """
        query_text = (query or "").strip()
        if not query_text:
            return self._result_template(
                query="",
                answer="Please enter a valid question.",
                route="error"
            )

        started = time.time()

        # Step 0: GLOBAL SECURITY CHECK (MANDATORY - all queries, all lanes)
        query_obj = Query(text=query_text)
        passed, security_result = self.gatekeeper.check_query(query_obj)
        
        if not passed:
            logger.warning(f"Query blocked by gatekeeper: {security_result.threat_detected}")
            return self._result_template(
                query=query_text,
                answer="Query blocked due to security concerns.",
                confidence=0.0,
                metadata={
                    "sources": [],
                    "security_blocked": True,
                    "threat_detected": security_result.threat_detected,
                    "threat_confidence": security_result.confidence,
                    "gatekeeper_stage": "orchestrator_pre_routing"
                },
                timings={"total": (time.time() - started) * 1000},
                route="blocked"
            )

        # Phase D: Step 0.5 - CHECK RESPONSE CACHE
        cached_response = self.response_cache.get(query_text)
        if cached_response is not None:
            logger.info(f"Cache HIT for query: {query_text[:50]}...")
            cache_ms = (time.time() - started) * 1000
            result = cached_response.copy()
            result["metadata"]["cache_status"] = "hit"
            result["timings"]["cache_check_ms"] = cache_ms
            result["timings"]["total"] = cache_ms
            return result

        # Step 1: Classify query intent
        lane = self._classify_query_intent(query_text)
        logger.info(f"Query routed to lane: {lane}")
        
        # Step 1b: Classify query domain (Phase B) for all tax-related queries
        domain_classification = None
        if lane == "tax_rag":
            domain_classification = self.domain_classifier.classify(query_text)
            logger.debug(
                f"Domain classification: {domain_classification.primary_domain} "
                f"({domain_classification.confidence:.2f})"
            )

        # Step 2: Route to appropriate lane
        try:
            if lane == "trivial":
                result = self._run_trivial_lane(query_text)
            elif lane == "general_finance":
                result = self._run_general_finance_lane(query_text, timeout_seconds)
            elif lane == "tax_rag":
                result = self._run_tax_rag_lane(query_text, timeout_seconds)
            else:
                result = self._result_template(
                    query=query_text,
                    answer="Unexpected routing error.",
                    route="error"
                )
        except Exception as exc:
            logger.exception(f"Query execution failed in lane {lane}: {exc}")
            total_ms = (time.time() - started) * 1000
            result = self._result_template(
                query=query_text,
                answer=f"Error processing query: {exc}",
                metadata={"error": str(exc)},
                timings={"total": total_ms},
                route="error"
            )

        # Add timing and routing metadata
        if "timings" not in result:
            result["timings"] = {}
        result["timings"]["total"] = (time.time() - started) * 1000
        result["route"] = lane
        
        # Mark that this query passed orchestrator-level security precheck
        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"]["gatekeeper_stage"] = "orchestrator_pre_routing_passed"
        result["metadata"]["cache_status"] = "miss"  # Default to miss (will be hit if cache returned early)
        
        # Add domain classification metadata if available (Phase B)
        if domain_classification:
            result["metadata"]["domain"] = domain_classification.primary_domain.value
            result["metadata"]["domain_confidence"] = domain_classification.confidence
            result["metadata"]["is_multi_domain"] = domain_classification.is_multi_domain
            result["metadata"]["domains_detected"] = [d.value for d in domain_classification.domains_detected]

        # Phase D: Cache the result for future queries (only cache successful, non-error responses)
        if result.get("route") not in ("error", "blocked") and result.get("answer"):
            try:
                # Cache with a reasonable TTL (1 hour for general queries, 30 min for date-sensitive)
                ttl = 1800 if "current" in query_text.lower() or "today" in query_text.lower() else 3600
                self.response_cache.put(query_text, result, ttl_seconds=ttl)
                logger.debug(f"Cached response for query: {query_text[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")

        return result

    def _classify_query_intent(self, query: str) -> Literal["trivial", "general_finance", "tax_rag"]:
        """
        Classify query intent using learned classification (Phase B).
        
        Uses IntentClassifier for semantic understanding:
        - tax_grounded: Tax-specific expert queries → tax_rag lane
        - general_finance: Finance conceptual questions → general_finance lane
        - trivial: Small talk/greetings → trivial lane
        
        Fallback:
        - If confidence < threshold, uses keyword patterns as backup
        """
        # Use learned intent classifier
        intent, confidence, scores = self.intent_classifier.classify(query)
        
        logger.debug(
            f"Intent classification: {intent} ({confidence:.2f}) | "
            f"Scores: {scores}"
        )
        
        # Map learned intent to lane
        if intent == Intent.TAX_GROUNDED:
            lane = "tax_rag"
        elif intent == Intent.GENERAL_FINANCE:
            lane = "general_finance"
        else:  # Intent.TRIVIAL
            lane = "trivial"
        
        # Fallback to keyword patterns if confidence is too low
        if self.intent_classifier.should_use_fallback(confidence):
            logger.info(
                f"Intent classification confidence {confidence:.2f} below threshold "
                f"({self.intent_classifier.MIN_CONFIDENCE_FOR_ROUTING}), using keyword fallback"
            )
            lowered = query.lower().strip()
            
            # Fast keyword checks
            if lowered in self._TRIVIAL_PROMPTS or (len(lowered) <= 4 and not any(kw in lowered for kw in self._TAX_COMPLIANCE_KEYWORDS)):
                lane = "trivial"
            elif any(kw in lowered for kw in self._TAX_COMPLIANCE_KEYWORDS):
                lane = "tax_rag"
            elif any(kw in lowered for kw in self._GENERAL_FINANCE_KEYWORDS):
                lane = "general_finance"
        
        logger.info(f"Query routed to lane: {lane} (intent={intent}, confidence={confidence:.2f})")
        return lane

    def _classify_with_fast_llm(self, query: str) -> Literal["trivial", "general_finance", "tax_rag"]:
        """
        Use fast general LLM to classify ambiguous queries.
        
        Deterministic classification without retrieval.
        """
        try:
            if self._general_llm is None:
                self._general_llm = ChatOpenAI(
                    model=GENERAL_MODEL,
                    openai_api_key=OPENROUTER_API_KEY,
                    openai_api_base=OPENROUTER_BASE_URL,
                    temperature=0,
                    timeout=10,
                )

            classifier_prompt = f"""Classify this query into ONE category ONLY:
- tax_rag: Questions about taxes, tax rules, compliance, GST, deductions, ITR, filing, income tax, corporate tax
- general_finance: Questions about finance concepts like SIP, mutual funds, stocks, bonds, investment strategy, portfolio
- trivial: Greetings, small talk, casual conversation

Query: "{query}"

Respond with ONLY the category name, nothing else."""

            response = self._general_llm.invoke([{"role": "user", "content": classifier_prompt}])
            category = response.content.strip().lower()

            if "tax" in category:
                return "tax_rag"
            elif "general" in category or "finance" in category:
                return "general_finance"
            else:
                return "trivial"

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}. Defaulting to general_finance")
            return "general_finance"

    def _run_trivial_lane(self, query: str) -> Dict[str, Any]:
        """Trivial lane: instant static response for greetings."""
        return self._result_template(
            query=query,
            answer="Hi! 👋 I'm your AI financial advisor. Ask me about taxes, GST, investments, or financial concepts, and I'll help.",
            confidence=1.0,
            route="trivial"
        )

    def _run_general_finance_lane(self, query: str, timeout_seconds: int = 10) -> Dict[str, Any]:
        """
        General finance lane: direct LLM call for conceptual finance questions.
        
        Bypasses retrieval, planning, and verification for speed.
        """
        logger.info(f"Executing general finance direct-answer lane (timeout={timeout_seconds}s)")

        started = time.time()

        # Initialize general LLM if needed
        if self._general_llm is None:
            try:
                self._general_llm = ChatOpenAI(
                    model=GENERAL_MODEL,
                    openai_api_key=OPENROUTER_API_KEY,
                    openai_api_base=OPENROUTER_BASE_URL,
                    temperature=0.1,
                    timeout=timeout_seconds,
                )
            except Exception as e:
                logger.error(f"Failed to initialize general LLM: {e}")
                return self._result_template(
                    query=query,
                    answer="Error initializing general finance assistant.",
                    route="general_finance"
                )

        try:
            # Direct LLM call without retrieval
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._call_general_llm, query)
                answer = future.result(timeout=timeout_seconds)

            total_ms = (time.time() - started) * 1000

            return {
                "query": query,
                "blocked": False,
                "answer": answer,
                "confidence": 0.8,  # Reduced confidence (no grounding)
                "retrieved_docs_count": 0,  # No retrieval
                "plan_steps": [],
                "metadata": {
                    "sources": [],
                    "lane": "general_finance",
                    "note": "Direct LLM answer (no document retrieval)"
                },
                "timings": {"total": total_ms},
                "route": "general_finance"
            }

        except TimeoutError:
            total_ms = (time.time() - started) * 1000
            logger.warning(f"General finance lane timed out after {timeout_seconds}s")
            return self._result_template(
                query=query,
                answer=f"General finance query timed out after {timeout_seconds}s. Please try a simpler question.",
                metadata={"timeout": True},
                timings={"total": total_ms},
                route="general_finance"
            )
        except Exception as e:
            logger.exception(f"General finance lane error: {e}")
            total_ms = (time.time() - started) * 1000
            return self._result_template(
                query=query,
                answer=f"Error generating general finance answer: {e}",
                metadata={"error": str(e)},
                timings={"total": total_ms},
                route="general_finance"
            )

    def _call_general_llm(self, query: str) -> str:
        """Call general LLM with finance-specific system prompt."""
        system_prompt = """You are a knowledgeable financial advisor specializing in general finance and investment concepts.

Answer questions about finance, investments, and economic principles based on your knowledge.
- Be concise and clear.
- Provide practical insights.
- If asked about taxes or compliance, clearly state: "This is a tax/compliance question. For authoritative tax advice, please consult the grounded recommendations in the full finance advisor."
- Do NOT fabricate document citations or sources.

Answer directly and confidently without hedging."""

        try:
            response = self._general_llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ])
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise

    def _run_tax_rag_lane(self, query: str, timeout_seconds: int = 45) -> Dict[str, Any]:
        """
        Tax RAG lane: full multi-agent pipeline for tax/compliance questions.
        
        Step 1: Retrieve documents, perform planning, reasoning, and verification.
        Step 2: (Phase C) Assess answer quality across retrieval/reasoning/verification
        Step 3: (Phase C) Compose confidence and apply quality gating
        """
        logger.info(f"Executing tax RAG lane with full pipeline (timeout={timeout_seconds}s)")

        self._ensure_workflow()

        started = time.time()
        try:
            # Run full workflow with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.workflow.run, query)
                state = future.result(timeout=timeout_seconds)

        except TimeoutError:
            total_ms = (time.time() - started) * 1000
            logger.warning(f"Tax RAG workflow timed out after {timeout_seconds}s")
            return self._result_template(
                query=query,
                answer=f"Request timed out after {timeout_seconds}s. Please try a shorter question.",
                metadata={"timeout": True},
                timings={"total": total_ms},
                route="tax_rag"
            )
        except Exception as exc:
            logger.exception(f"Tax RAG workflow failed: {exc}")
            total_ms = (time.time() - started) * 1000
            return self._result_template(
                query=query,
                answer=f"Error processing tax question: {exc}",
                metadata={"error": str(exc)},
                timings={"total": total_ms},
                route="tax_rag"
            )

        # Extract workflow results
        total_ms = (time.time() - started) * 1000
        blocked = state.metadata.get("security_blocked", False)
        answer = state.answer or "No answer generated"
        verification = state.verification if isinstance(state.verification, dict) else {}
        base_confidence = float(verification.get("confidence", 0.0))

        plan_steps = [step.action_type for step in state.plan] if state.plan else []
        retrieved_docs_count = sum(len(r.chunks) for r in (state.retrieved_docs or []))

        metadata = dict(state.metadata or {})
        metadata.setdefault("sources", self._extract_sources(state.retrieved_docs or []))
        metadata["lane"] = "tax_rag"

        # ======================================================================
        # Phase C: Quality Assessment & Confidence Composition
        # ======================================================================
        
        # Extract quality assessment inputs from workflow state
        reasoning_chain = state.reasoning if isinstance(state.reasoning, str) else ""
        reasoning_steps = state.reasoning_steps if isinstance(state.reasoning_steps, list) else []
        retrieved_docs = state.retrieved_docs or []
        
        # Step 5: Assess answer quality across three dimensions
        try:
            quality_result = self.quality_assessment.assess_answer(
                query=query,
                retrieved_docs=[
                    {"text": chunk.content, "source": chunk.source.value}
                    for result in retrieved_docs
                    for chunk in getattr(result, "chunks", [])
                ],
                reasoning_chain=reasoning_chain,
                reasoning_steps=reasoning_steps,
                answer=answer,
            )
            
            logger.debug(f"Quality assessment: {quality_result.explanation}")
            
            # Step 6: Compose confidence from multiple signals
            composed_result = self.confidence_composer.compose(
                retrieval_signal=quality_result.retrieval_signal,
                reasoning_signal=quality_result.reasoning_signal,
                verification_signal=quality_result.verification_signal,
                answer_text=answer,
            )
            
            logger.debug(f"Composed confidence: {composed_result.explanation}")
            
            # Add Phase C metadata
            metadata["quality_assessment"] = {
                "retrieval_confidence": quality_result.retrieval_signal,
                "reasoning_confidence": quality_result.reasoning_signal,
                "verification_confidence": quality_result.verification_signal,
                "overall_quality": quality_result.overall_quality,
                "quality_explanation": quality_result.explanation,
            }
            
            metadata["confidence_composition"] = {
                "composed_confidence": composed_result.overall_confidence,
                "confidence_level": composed_result.level.value,
                "gating_action": composed_result.action,
                "confidence_explanation": composed_result.explanation,
            }
            
            # Apply quality gating if needed
            final_answer = answer
            final_confidence = composed_result.overall_confidence
            
            if composed_result.should_override:
                logger.info(
                    f"Confidence gating activated: action={composed_result.action}, "
                    f"confidence={composed_result.overall_confidence:.2f}"
                )
                
                if composed_result.action == "clarify":
                    # Offer clarification instead of uncertain answer
                    final_answer = composed_result.replacement_answer
                    metadata["gating_applied"] = "clarification"
                    
                elif composed_result.action == "admit_uncertainty":
                    # Admit uncertainty instead of potentially wrong answer
                    final_answer = composed_result.replacement_answer
                    metadata["gating_applied"] = "uncertainty_admission"
                
                else:
                    metadata["gating_applied"] = composed_result.action
            
            elif composed_result.action == "caveat":
                # Add caveat without replacing answer
                caveat_prefix = f"⚠️ {composed_result.replacement_answer}\n\n"
                final_answer = caveat_prefix + answer
                metadata["gating_applied"] = "caveat"
            
            return {
                "query": query,
                "blocked": blocked,
                "answer": final_answer,
                "confidence": final_confidence,
                "retrieved_docs_count": retrieved_docs_count,
                "plan_steps": plan_steps,
                "metadata": metadata,
                "timings": {"total": total_ms},
                "route": "tax_rag"
            }
        
        except Exception as e:
            logger.exception(f"Phase C quality assessment failed: {e}. Returning base answer.")
            
            # Fallback: return base answer if quality assessment fails
            return {
                "query": query,
                "blocked": blocked,
                "answer": answer,
                "confidence": base_confidence,
                "retrieved_docs_count": retrieved_docs_count,
                "plan_steps": plan_steps,
                "metadata": {
                    **metadata,
                    "phase_c_error": str(e),
                },
                "timings": {"total": total_ms},
                "route": "tax_rag"
            }

    def _ensure_workflow(self) -> None:
        """Lazy initialize workflow for tax RAG lane only."""
        if self.workflow is not None:
            return

        self.workflow = initialize_workflow()
        self.router = getattr(self.workflow, "router", None)

        if self._preload_faiss and self.router and hasattr(self.router, "preload_all_retrievers"):
            preload_start = time.time()
            self.router.preload_all_retrievers()
            logger.info("FAISS preload completed in %.2fs", time.time() - preload_start)

    def _extract_sources(self, retrieved_docs: List[Any]) -> List[str]:
        """Extract source list from retrieved documents."""
        sources: List[str] = []
        seen = set()

        for result in retrieved_docs:
            for chunk in getattr(result, "chunks", []):
                source = f"{chunk.source.value}:{chunk.document_name}"
                if source not in seen:
                    seen.add(source)
                    sources.append(source)

        return sources

    def _result_template(
        self,
        query: str,
        answer: str,
        confidence: float = 0.0,
        metadata: Dict[str, Any] | None = None,
        timings: Dict[str, float] | None = None,
        route: str = "unknown"
    ) -> Dict[str, Any]:
        """Generate normalized response payload."""
        return {
            "query": query,
            "blocked": False,
            "answer": answer,
            "confidence": confidence,
            "retrieved_docs_count": 0,
            "plan_steps": [],
            "metadata": metadata or {"sources": []},
            "timings": timings or {"total": 0.0},
            "route": route
        }
