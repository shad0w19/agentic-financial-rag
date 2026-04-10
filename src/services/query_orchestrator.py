"""Shared query runtime for Streamlit demos.

This module decouples UI apps from run_demo.py and exposes a stable,
low-latency run_query() method with domain-aware routing:
    1. Trivial: Greetings and small talk (instant static response)
    2. Out-of-Scope: Polite domain deflection
    3. Ambiguous In-Domain: Clarification prompt before heavy execution
    4. General Finance: Finance concepts without RAG (fast LLM direct answer)
    5. Tax RAG: Tax/compliance/grounded questions (full multi-agent pipeline)
"""

from __future__ import annotations

import logging
import time
import re
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, List, Optional, Literal
try:
    from openai import APITimeoutError
except ImportError:
    APITimeoutError = TimeoutError  # type: ignore[assignment,misc]

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
from src.confidence.confidence_composer import ConfidenceComposer, ConfidenceSignal
# Phase D: Latency Optimization (Parallel Retrieval & Caching)
from src.services.response_cache import ResponseCache
from src.retrieval.parallel_retriever import ParallelRetriever


logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """App-facing orchestrator implementing domain-aware query routing."""

    TAX_RAG_STAGE_BUDGETS = {
        "planner": 20.0,
        "retrieval": 20.0,
        "reasoning": 60.0,
        "verification": 20.0,
    }
    TAX_RAG_TOP_K = 6
    FAST_LANE_TOP_K = 3
    _TAX_GUARD_TERMS = {
        "80c", "80d", "80e", "80gg", "hra", "tds", "gst", "itr",
        "deduction", "regime", "threshold", "limit", "slab", "income tax",
        "corporate tax", "section", "rebate", "exemption", "standard deduction",
        "capital gains", "business loss", "interest", "loan", "assessment year", "financial year",
    }

    _INDIA_TAX_MARKERS = {
        "india", "indian", "gst", "itr", "section 80", "old regime", "new regime", "tds",
    }

    _NON_INDIA_TAX_MARKERS = {
        "irs", "internal revenue service", "u.s.", "united states", "federal tax",
        "state tax", "itemized deduction", "mortgage interest deduction", "1040",
    }

    # Trivial lane: exact match greetings and small talk
    _TRIVIAL_PROMPTS = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good evening",
        "how are you",
    }

    # Tax/compliance keywords for heuristic-first routing
    _TAX_COMPLIANCE_KEYWORDS = {
        "tax", "income tax", "itr", "80c", "80d", "80gg", "hra", "deduction",
        "filing", "return", "assessment", "tds", "itc", "gst", "goods",
        "services", "supply", "invoice", "compliance", "threshold",
        "slab", "regime", "corporate tax", "company", "auditor", "audit",
        "form", "schedule", "appendix", "section", "penalty", "interest",
        "capital gains", "exemption", "business loss", "business losses", "home loan",
    }

    # General finance keywords for conceptual questions
    _GENERAL_FINANCE_KEYWORDS = {
        "mutual fund", "sip", "etf", "stock", "bond", "investment",
        "portfolio", "diversification", "risk", "return", "inflation",
        "compounding", "asset allocation", "nav", "expense ratio",
        "dividend", "yield", "liquidity", "volatility", "credit", "debit",
        "margin", "leverage", "hedge", "derivative", "option", "future",
    }

    _TRACE_PREVIEW_CHUNK_COUNT = 5
    _TRACE_PREVIEW_TEXT_CHARS = 220

    _STRUCTURED_QUERY_MARKERS = {
        "compare", "comparison", "table", "list", "breakdown", "rate", "rates", "slab", "slabs",
        "regime", "difference", "show all", "show me all",
        "calculate", "calculation",
    }

    _GENERIC_ANSWER_MARKERS = {
        "the documents don't cover this specifically",
        "please consult a tax professional",
        "information not found",
        "i need a bit more detail",
        "i have partial information",
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
        self._awaiting_tax_clarification = False
        self._last_ambiguous_query = ""

    def _build_retrieval_trace(
        self,
        retrieved_docs: List[Any],
        max_chunks: int | None = None,
        max_chars: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Create a compact preview of retrieved chunks for debugging."""
        chunk_limit = max_chunks or self._TRACE_PREVIEW_CHUNK_COUNT
        char_limit = max_chars or self._TRACE_PREVIEW_TEXT_CHARS
        previews: List[Dict[str, Any]] = []

        for result in retrieved_docs:
            scores = list(getattr(result, "scores", []) or [])
            for index, chunk in enumerate(getattr(result, "chunks", []) or []):
                preview_text = (getattr(chunk, "text", "") or "").strip().replace("\n", " ")
                previews.append(
                    {
                        "source": getattr(getattr(chunk, "source", None), "value", str(getattr(chunk, "source", "unknown"))),
                        "document_name": getattr(chunk, "document_name", ""),
                        "chunk_index": getattr(chunk, "chunk_index", None),
                        "score": scores[index] if index < len(scores) else None,
                        "text_preview": preview_text[:char_limit],
                    }
                )
                if len(previews) >= chunk_limit:
                    return previews

        return previews

    def _build_verification_trace(self, verification: Any) -> Dict[str, Any]:
        """Normalize verification output into a compact debug payload."""
        if not isinstance(verification, dict):
            return {}

        return {
            "is_valid": verification.get("is_valid"),
            "confidence": verification.get("confidence"),
            "issues": verification.get("issues", []),
        }

    def _extract_numeric_tokens(self, text: str) -> List[str]:
        """Extract compact numeric tokens from text for grounding checks."""
        if not text:
            return []
        return re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text)

    def _has_extended_finance_indices(self) -> bool:
        """Check whether retrieval indices exist for extended finance domains."""
        root = Path(__file__).resolve().parents[2]
        index_paths = [
            root / "data" / "vector_store" / "investment" / "index.faiss",
            root / "data" / "vector_store" / "regulatory" / "index.faiss",
        ]
        return any(path.exists() for path in index_paths)

    def _extract_tax_terms(self, text: str) -> List[str]:
        """Extract tax-specific terms present in text."""
        lowered = (text or "").lower()
        terms: List[str] = []
        for term in self._TAX_GUARD_TERMS:
            if term in lowered:
                terms.append(term)
        return sorted(set(terms))

    def _is_structured_output_query(self, query: str) -> bool:
        """Detect whether user likely expects tabular/list output."""
        lowered = (query or "").lower()
        return any(marker in lowered for marker in self._STRUCTURED_QUERY_MARKERS)

    def _looks_generic_answer(self, answer: str) -> bool:
        """Detect weak/generic fallback answers."""
        lowered = (answer or "").strip().lower()
        return any(marker in lowered for marker in self._GENERIC_ANSWER_MARKERS)

    def _rank_fast_lane_chunks(self, query: str, chunks: List[Any]) -> List[Any]:
        """Lightweight lexical ranking for fast-lane chunks."""
        query_terms = {
            token
            for token in re.findall(r"[a-zA-Z0-9]+", (query or "").lower())
            if len(token) > 2 and token not in {"what", "how", "the", "for", "and", "tax"}
        }
        if not query_terms:
            return chunks

        def score_chunk(chunk: Any) -> float:
            text = (getattr(chunk, "text", "") or "").lower()
            overlap = sum(1 for term in query_terms if term in text)
            has_number = bool(re.search(r"\d", text))
            return float(overlap) + (0.25 if overlap > 0 and has_number else 0.0)

        return sorted(chunks, key=score_chunk, reverse=True)

    def _coerce_structured_output(self, answer: str, query: str) -> str:
        """Coerce output into a simple markdown table for structure-seeking queries."""
        if not self._is_structured_output_query(query):
            return answer
        if not answer or "|" in answer:
            return answer

        if self._looks_generic_answer(answer):
            safe_answer = answer.replace("|", " ")
            return "\n".join([
                "| Status | Details |",
                "|---|---|",
                f"| Insufficient grounded detail | {safe_answer} |",
            ])

        sentences = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?])\s+", answer)
            if segment.strip()
        ]
        if not sentences:
            return answer

        rows = ["| Item | Details |", "|---|---|"]
        for idx, sentence in enumerate(sentences[:5], start=1):
            safe_sentence = sentence.replace("|", " ")
            rows.append(f"| Point {idx} | {safe_sentence} |")
        return "\n".join(rows)

    def _build_fast_evidence_summary(self, query: str, chunks: List[Any], structured_expected: bool) -> str:
        """Build concise evidence-backed fallback when model answer is generic."""
        query_terms = {
            token
            for token in re.findall(r"[a-zA-Z0-9]+", (query or "").lower())
            if len(token) > 2
        }
        evidence_rows: List[tuple[str, str]] = []

        for chunk in chunks:
            text = re.sub(r"\s+", " ", (getattr(chunk, "text", "") or "").strip())
            if not text:
                continue
            lowered = text.lower()
            has_overlap = any(term in lowered for term in query_terms)
            has_number = bool(re.search(r"\d", lowered))
            if not (has_overlap and has_number):
                continue
            snippet = text[:200].replace("|", " ")
            source = getattr(getattr(chunk, "source", None), "value", "unknown")
            evidence_rows.append((str(source), snippet))
            if len(evidence_rows) >= 3:
                break

        if not evidence_rows:
            return ""

        if structured_expected:
            lines = ["| Source | Evidence |", "|---|---|"]
            for source, snippet in evidence_rows:
                lines.append(f"| {source} | {snippet} |")
            return "\n".join(lines)

        lines = ["I found relevant evidence in retrieved tax documents:"]
        for source, snippet in evidence_rows:
            lines.append(f"- Source: {source} | {snippet}")
        lines.append("Please share one more specific detail so I can give an exact final answer.")
        return "\n".join(lines)

    def _compute_fast_lane_confidence(self, answer: str, chunks: List[Any], context: str) -> float:
        """Compute confidence for fast-lane answer quality."""
        lowered = (answer or "").strip().lower()
        if not lowered or lowered.startswith("error generating") or "timed out" in lowered:
            return 0.1

        score = 0.30
        if len(chunks) >= 2:
            score += 0.20
        if len(chunks) >= 3:
            score += 0.10
        if len(context) >= 600:
            score += 0.10

        has_numbers = bool(self._extract_numeric_tokens(answer))
        has_tax_terms = bool(self._extract_tax_terms(answer))
        if has_numbers:
            score += 0.12
        if has_tax_terms:
            score += 0.12

        if self._looks_generic_answer(answer):
            score -= 0.30

        return max(0.05, min(score, 0.9))

    def _is_generic_tax_answer(self, answer: str) -> bool:
        """Detect generic responses that likely missed grounded extraction."""
        lowered = (answer or "").lower()
        generic_markers = (
            "information not found",
            "could you be more specific",
            "i can help with indian tax",
            "i am focused on finance",
            "please share your context",
            "no answer generated",
        )
        return any(marker in lowered for marker in generic_markers)

    def _is_probably_tax_query(self, query: str) -> bool:
        """Heuristic safety net to keep valid tax queries in the tax lane."""
        lowered = (query or "").lower().strip()
        if not lowered:
            return False

        if any(keyword in lowered for keyword in self._TAX_COMPLIANCE_KEYWORDS):
            return True

        section_pattern = r"\bsection\s*\d+[a-z]*\b|\b\d{2,3}[a-z]{0,2}\b"
        return bool(re.search(section_pattern, lowered))

    def _detect_tax_domain_drift(self, query: str, answer: str, evidence_text: str) -> bool:
        """Detect when answer drifts to non-Indian tax domain despite Indian evidence."""
        answer_lower = (answer or "").lower()
        evidence_lower = (evidence_text or "").lower()

        answer_non_india = any(marker in answer_lower for marker in self._NON_INDIA_TAX_MARKERS)
        evidence_india = any(marker in evidence_lower for marker in self._INDIA_TAX_MARKERS)
        query_tax_like = self._is_probably_tax_query(query)
        return answer_non_india and (evidence_india or query_tax_like)

    def _collect_evidence_snippets(self, retrieved_docs: List[Any], max_snippets: int = 4) -> List[str]:
        """Collect short evidence snippets containing numbers and tax terms."""
        snippets: List[str] = []
        for result in retrieved_docs:
            for chunk in getattr(result, "chunks", []) or []:
                chunk_text = (getattr(chunk, "text", "") or "").strip()
                if not chunk_text:
                    continue
                has_number = bool(self._extract_numeric_tokens(chunk_text))
                has_term = bool(self._extract_tax_terms(chunk_text))
                if not (has_number and has_term):
                    continue
                compact = re.sub(r"\s+", " ", chunk_text)
                snippets.append(compact[:240])
                if len(snippets) >= max_snippets:
                    return snippets
        return snippets

    def _build_constrained_correction_answer(self, query: str, evidence_snippets: List[str]) -> str:
        """Create a deterministic correction answer from evidence snippets only."""
        if not evidence_snippets:
            return ""

        lines = [
            "I found concrete evidence in the retrieved tax sources:",
        ]
        for snippet in evidence_snippets[:3]:
            lines.append(f"- {snippet}")
        lines.append("Please tell me which specific sub-part you want applied to your case.")
        return "\n".join(lines)

    def _apply_tax_grounding_guard(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Any],
    ) -> Dict[str, Any]:
        """Guard against extraction misses when evidence has concrete tax values."""
        answer_numbers = sorted(set(self._extract_numeric_tokens(answer or "")))
        answer_terms = sorted(set(self._extract_tax_terms(answer or "")))

        evidence_text = "\n".join(
            (getattr(chunk, "text", "") or "")
            for result in retrieved_docs
            for chunk in getattr(result, "chunks", []) or []
        )
        evidence_numbers = sorted(set(self._extract_numeric_tokens(evidence_text)))
        evidence_terms = sorted(set(self._extract_tax_terms(evidence_text)))

        has_strong_evidence = len(evidence_numbers) >= 1 and len(evidence_terms) >= 1
        answer_missing_concrete = len(answer_numbers) == 0 or len(set(answer_terms) & set(evidence_terms)) == 0
        is_generic = self._is_generic_tax_answer(answer)
        has_domain_drift = self._detect_tax_domain_drift(query, answer, evidence_text)
        should_correct = has_strong_evidence and (answer_missing_concrete or has_domain_drift) and (is_generic or has_domain_drift)

        corrected_answer = answer
        evidence_snippets: List[str] = []
        if should_correct:
            evidence_snippets = self._collect_evidence_snippets(retrieved_docs)
            correction = self._build_constrained_correction_answer(query, evidence_snippets)
            if correction:
                corrected_answer = correction

        return {
            "answer": corrected_answer,
            "debug": {
                "enabled": True,
                "activated": bool(should_correct and corrected_answer != answer),
                "evidence_numbers_count": len(evidence_numbers),
                "answer_numbers_count": len(answer_numbers),
                "evidence_terms": evidence_terms[:8],
                "answer_terms": answer_terms[:8],
                "is_generic_answer": is_generic,
                "domain_drift_detected": has_domain_drift,
                "evidence_snippets_used": len(evidence_snippets[:3]),
            },
        }

    def _prepend_chat_context(
        self,
        query_text: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        max_messages: int = 3,
    ) -> str:
        """Prepend recent conversation turns to improve follow-up context."""
        if not chat_history:
            return query_text

        recent_messages = chat_history[-max_messages:]
        context_lines: List[str] = []

        for idx, message in enumerate(recent_messages):
            if not isinstance(message, dict):
                continue

            role_raw = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if not content:
                continue

            # Avoid duplicating the current prompt if UI already appended it.
            is_last = idx == len(recent_messages) - 1
            if is_last and role_raw.lower() == "user" and content == query_text:
                continue

            if role_raw.lower() == "user":
                role_label = "User"
            elif role_raw.lower() == "assistant":
                role_label = "Assistant"
            else:
                role_label = role_raw.title() if role_raw else "Unknown"

            context_lines.append(f"{role_label}: {content}")

        if not context_lines:
            return query_text

        prefix = "Previous conversation:\n" + "\n".join(context_lines)
        return f"{prefix}\n\nCurrent question: {query_text}"

    def _normalize_mode(self, mode: str) -> str:
        """Normalize user-selected mode into supported values."""
        return "deep" if str(mode).strip().lower() == "deep" else "fast"

    def _cache_key(self, execution_query: str, requested_mode: str) -> str:
        """Build cache key that preserves mode-specific behavior."""
        return f"{requested_mode}:{execution_query}"

    def _effective_mode_for_lane(self, lane: str, requested_mode: str) -> str:
        """Return the mode actually used by a routed lane."""
        if lane == "tax_rag" and requested_mode == "deep":
            return "deep"
        return "fast"

    def _normalize_response_schema(
        self,
        result: Dict[str, Any],
        lane: str,
        requested_mode: str,
        effective_mode: str,
    ) -> Dict[str, Any]:
        """Ensure a stable payload contract across all lanes for UI rendering."""
        normalized: Dict[str, Any] = dict(result or {})
        metadata = dict(normalized.get("metadata") or {})
        timings = dict(normalized.get("timings") or {})

        plan_steps = list(normalized.get("plan_steps") or [])
        sources = list(metadata.get("sources") or [])

        normalized["metadata"] = metadata
        normalized["timings"] = timings
        normalized["plan_steps"] = plan_steps
        normalized["retrieved_docs_count"] = int(normalized.get("retrieved_docs_count") or 0)
        normalized["blocked"] = bool(normalized.get("blocked", False))
        if lane == "blocked":
            normalized["blocked"] = True
        normalized["route"] = str(normalized.get("route") or lane)

        metadata["sources"] = sources
        metadata["requested_mode"] = requested_mode
        metadata["mode_used"] = effective_mode
        metadata["mode_honored"] = requested_mode == effective_mode
        metadata["hops"] = len(plan_steps)

        return normalized

    def run_query(
        self,
        query: str,
        timeout_seconds: int = 120,
        mode: str = "fast",
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Route query to appropriate lane: trivial, general_finance, or tax_rag.

        mode="fast"  → tax queries: direct retrieve → LLM (~10-15 s, no planner/verification)
        mode="deep"  → full multi-agent pipeline (planner, reasoning, verification, ~40-45 s)

        CRITICAL: All queries are security-checked BEFORE routing (Step 0).
        Returns normalized response payload with route metadata.
        """
        query_text = (query or "").strip()
        requested_mode = self._normalize_mode(mode)
        if not query_text:
            empty_result = self._result_template(
                query="",
                answer="Please enter a valid question.",
                route="error"
            )
            return self._normalize_response_schema(
                empty_result,
                lane="error",
                requested_mode=requested_mode,
                effective_mode="fast",
            )

        execution_query = query_text
        forced_lane = None
        query_id = str(uuid.uuid4())

        # If previous turn asked for tax clarification, allow short follow-up answers
        # (for example: "personal tax", "gst", "corporate") to jump directly into tax_rag.
        if self._awaiting_tax_clarification:
            expanded = self._expand_tax_clarification_query(query_text)
            if expanded:
                execution_query = expanded
                forced_lane = "tax_rag"

        execution_query = self._prepend_chat_context(
            execution_query,
            chat_history=chat_history,
        )
        cache_key = self._cache_key(execution_query, requested_mode)

        started = time.time()

        # Step 0: GLOBAL SECURITY CHECK (MANDATORY - all queries, all lanes)
        query_obj = Query(text=query_text)
        passed, security_result = self.gatekeeper.check_query(query_obj)
        
        if not passed:
            logger.warning(f"Query blocked by gatekeeper: {security_result.threat_detected}")
            blocked_result = self._result_template(
                query=query_text,
                answer="Query blocked due to security concerns.",
                confidence=0.0,
                metadata={
                    "sources": [],
                    "security_blocked": True,
                    "query_id": query_id,
                    "threat_detected": security_result.threat_detected,
                    "threat_confidence": security_result.confidence,
                    "gatekeeper_stage": "orchestrator_pre_routing"
                },
                timings={"total": (time.time() - started) * 1000},
                route="blocked"
            )
            return self._normalize_response_schema(
                blocked_result,
                lane="blocked",
                requested_mode=requested_mode,
                effective_mode="fast",
            )

        # Phase D: Step 0.5 - CHECK RESPONSE CACHE
        cached_response = self.response_cache.get(cache_key)
        if cached_response is not None:
            logger.info(f"Cache HIT for query: {query_text[:50]}...")
            cache_ms = (time.time() - started) * 1000
            cached_route = str(cached_response.get("route") or "unknown")
            result = self._normalize_response_schema(
                cached_response,
                lane=cached_route,
                requested_mode=requested_mode,
                effective_mode=self._effective_mode_for_lane(cached_route, requested_mode),
            )
            result["metadata"]["cache_status"] = "hit"
            result["metadata"]["conversation_context_used"] = bool(chat_history)
            result["timings"]["cache_check_ms"] = cache_ms
            result["timings"]["total"] = cache_ms
            result["query"] = query_text
            result["metadata"]["query_id"] = query_id
            self._emit_query_timing_log(result)
            return result

        # Step 1: Classify query intent
        if forced_lane:
            lane = forced_lane
            route_details = {
                "intent": Intent.TAX_GROUNDED.value,
                "intent_confidence": 1.0,
                "ambiguity_reason": "clarification_followup",
            }
        else:
            lowered = execution_query.lower()
            if lowered in self._TRIVIAL_PROMPTS:
                lane = "trivial"
                route_details = {"intent": "trivial", "intent_confidence": 1.0}
                logger.debug("Heuristic routing hit: trivial")
            elif any(k in lowered for k in self._TAX_COMPLIANCE_KEYWORDS):
                lane = "tax_rag"
                route_details = {"intent": "tax_grounded", "intent_confidence": 0.9}
                logger.debug("Heuristic routing hit: tax_rag")
            elif any(k in lowered for k in self._GENERAL_FINANCE_KEYWORDS):
                if self._has_extended_finance_indices():
                    lane = "tax_rag"
                    route_details = {
                        "intent": "general_finance",
                        "intent_confidence": 0.9,
                        "lane_upgrade": "general_finance_to_tax_rag",
                    }
                    logger.debug("Heuristic routing upgrade: general_finance -> tax_rag")
                else:
                    lane = "general_finance"
                    route_details = {"intent": "general_finance", "intent_confidence": 0.9}
                    logger.debug("Heuristic routing hit: general_finance")
            else:
                lane, route_details = self._classify_query_intent(execution_query)

        # Phase 1.1: Rescue likely tax questions from non-tax lanes.
        if lane in {"out_of_scope", "ambiguous_in_domain"} and self._is_probably_tax_query(execution_query):
            route_details["lane_rescue"] = f"{lane}_to_tax_rag"
            route_details["ambiguity_reason"] = "tax_query_rescue"
            lane = "tax_rag"
        logger.info(f"Query routed to lane: {lane}")
        
        # Step 1b: Classify query domain (Phase B) for all tax-related queries
        domain_classification = None
        if lane == "tax_rag":
            domain_classification = self.domain_classifier.classify(execution_query)
            logger.debug(
                f"Domain classification: {domain_classification.primary_domain} "
                f"({domain_classification.confidence:.2f})"
            )

        # Step 2: Route to appropriate lane
        try:
            if lane == "trivial":
                result = self._run_trivial_lane(execution_query)
            elif lane == "out_of_scope":
                result = self._run_out_of_scope_lane(execution_query)
            elif lane == "ambiguous_in_domain":
                result = self._run_ambiguous_in_domain_lane(execution_query)
            elif lane == "general_finance":
                result = self._run_general_finance_lane(execution_query, timeout_seconds)
            elif lane == "tax_rag":
                # Extract domain hint from already-computed classification (skips duplicate LLM call in retriever)
                domain_hint = (
                    domain_classification.primary_domain.value
                    if domain_classification and domain_classification.primary_domain.value != "multi"
                    else None
                )
                result = self._run_tax_rag_lane(execution_query, query_id, timeout_seconds, requested_mode, domain_hint)
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

        effective_mode = self._effective_mode_for_lane(lane, requested_mode)
        result = self._normalize_response_schema(
            result,
            lane=lane,
            requested_mode=requested_mode,
            effective_mode=effective_mode,
        )

        # Add timing and routing metadata
        result["timings"]["total"] = (time.time() - started) * 1000
        result["route"] = lane
        result["query"] = query_text
        
        # Mark that this query passed orchestrator-level security precheck
        result["metadata"]["gatekeeper_stage"] = "orchestrator_pre_routing_passed"
        result["metadata"]["cache_status"] = "miss"  # Default to miss (will be hit if cache returned early)
        result["metadata"]["conversation_context_used"] = bool(chat_history)
        result["metadata"]["query_id"] = query_id
        result["metadata"]["intent"] = route_details.get("intent")
        result["metadata"]["intent_confidence"] = route_details.get("intent_confidence")
        if route_details.get("ambiguity_reason"):
            result["metadata"]["ambiguity_reason"] = route_details.get("ambiguity_reason")

        # Track whether we are waiting for tax clarification on next user message.
        if lane == "ambiguous_in_domain":
            self._awaiting_tax_clarification = True
            self._last_ambiguous_query = query_text
        else:
            self._awaiting_tax_clarification = False
            self._last_ambiguous_query = ""
        
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
                self.response_cache.put(cache_key, result, ttl_seconds=ttl)
                logger.debug(f"Cached response for query: {query_text[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")

        self._emit_query_timing_log(result)

        return result

    def _emit_query_timing_log(self, result: Dict[str, Any]) -> None:
        """Emit structured timing payload for each query."""
        metadata = result.get("metadata", {})
        timings = result.get("timings", {})
        stage_timings = metadata.get("stage_timings_ms", {})
        payload = {
            "query_id": metadata.get("query_id"),
            "route": result.get("route"),
            "planner_time_ms": stage_timings.get("planner", 0.0),
            "retrieval_time_ms": stage_timings.get("retrieval", 0.0),
            "reasoning_time_ms": stage_timings.get("reasoning", 0.0),
            "verification_time_ms": stage_timings.get("verification", 0.0),
            "total_time_ms": timings.get("total", 0.0),
            "timeout_stage": metadata.get("timeout_stage"),
            "degraded_flags": metadata.get("degraded_flags", []),
        }
        logger.info("TIMING_PAYLOAD %s", payload)

    def _expand_tax_clarification_query(self, user_reply: str) -> Optional[str]:
        """Map short clarification replies to an actionable tax query."""
        reply = user_reply.lower().strip()
        if not reply or len(reply) > 80:
            return None

        personal_markers = ("personal", "individual", "salary", "salaried", "self-employed", "freelancer")
        corporate_markers = ("corporate", "company", "business", "firm", "startup")
        gst_markers = ("gst", "indirect tax", "input tax credit", "itc")

        if any(marker in reply for marker in personal_markers):
            return f"{self._last_ambiguous_query}. Focus on Indian personal income tax rules for individuals."
        if any(marker in reply for marker in corporate_markers):
            return f"{self._last_ambiguous_query}. Focus on Indian corporate tax compliance for businesses."
        if any(marker in reply for marker in gst_markers):
            return f"{self._last_ambiguous_query}. Focus on Indian GST rules and compliance."

        return None

    def _classify_query_intent(
        self,
        query: str,
    ) -> tuple[Literal["trivial", "out_of_scope", "ambiguous_in_domain", "general_finance", "tax_rag"], Dict[str, Any]]:
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
        intent, confidence, scores, is_ambiguous, ambiguity_reason = self.intent_classifier.classify(query)

        route_details: Dict[str, Any] = {
            "intent": intent.value,
            "intent_confidence": confidence,
            "ambiguity_reason": ambiguity_reason,
        }
        
        logger.debug(
            f"Intent classification: {intent} ({confidence:.2f}) | "
            f"Scores: {scores}"
        )
        
        # Map learned intent to lane
        if intent == Intent.OUT_OF_SCOPE:
            lane = "out_of_scope"
        elif is_ambiguous and intent == Intent.TAX_GROUNDED:
            lane = "ambiguous_in_domain"
        elif intent == Intent.TAX_GROUNDED:
            lane = "tax_rag"
        elif intent == Intent.GENERAL_FINANCE:
            lane = "tax_rag" if self._has_extended_finance_indices() else "general_finance"
            if lane == "tax_rag":
                route_details["lane_upgrade"] = "general_finance_to_tax_rag"
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
                # Keep generic tax asks in clarification path.
                if is_ambiguous:
                    lane = "ambiguous_in_domain"
                else:
                    lane = "tax_rag"
            elif any(kw in lowered for kw in self._GENERAL_FINANCE_KEYWORDS):
                lane = "tax_rag" if self._has_extended_finance_indices() else "general_finance"
                if lane == "tax_rag":
                    route_details["lane_upgrade"] = "general_finance_to_tax_rag"
            else:
                lane = "out_of_scope"
                route_details["intent"] = Intent.OUT_OF_SCOPE.value
        
        logger.info(f"Query routed to lane: {lane} (intent={intent}, confidence={confidence:.2f})")
        return lane, route_details

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
                    request_timeout=10,
                    max_tokens=1200,
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
            answer="Hi! I'm your finance and tax advisor. Ask me about income tax, GST, investments, compliance, or financial planning.",
            confidence=1.0,
            route="trivial"
        )

    def _run_out_of_scope_lane(self, query: str) -> Dict[str, Any]:
        """Out-of-scope lane: politely deflect to finance/tax domain."""
        return self._result_template(
            query=query,
            answer=(
                "I am focused on finance, taxation, and investments. "
                "I cannot help with unrelated topics, but I can help with Indian income tax, GST, deductions, filings, and investment concepts."
            ),
            confidence=1.0,
            metadata={"sources": [], "scope": "out_of_domain"},
            route="out_of_scope",
        )

    def _run_ambiguous_in_domain_lane(self, query: str) -> Dict[str, Any]:
        """Ambiguous in-domain lane: ask a clarifying question before execution."""
        return self._result_template(
            query=query,
            answer=(
                "I can help with Indian tax. Could you be more specific: personal income tax, GST, or corporate tax? "
                "If possible, also share your context (salaried, self-employed, or business owner)."
            ),
            confidence=0.45,
            metadata={"sources": [], "needs_clarification": True},
            route="ambiguous_in_domain",
        )

    def _run_general_finance_lane(self, query: str, timeout_seconds: int = 20) -> Dict[str, Any]:
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
                    temperature=0.2,
                    timeout=timeout_seconds,
                    request_timeout=timeout_seconds,
                    max_tokens=1200,
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
            answer = self._call_general_llm(query)
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

        except APITimeoutError:
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
            if not response or not getattr(response, "content", None):
                raise ValueError("Empty LLM response")
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise

    def _run_fast_tax_lane(
        self,
        query: str,
        query_id: str,
        domain_hint: Optional[str],
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        """
        Fast tax lane: direct retrieval → single LLM call → return.

        No GraphState, no planner, no verification, no multi-agent overhead.
        Latency target: 8-15 s.
        Falls back to deep lane if retrieval fails or router is unavailable.
        """
        logger.info("Executing FAST tax lane (retrieve → LLM, no planner/verification)")
        self._ensure_workflow()

        if self.router is None:
            logger.warning("Router unavailable on fast lane, falling back to deep lane")
            return self._run_tax_rag_lane(query, query_id, timeout_seconds, mode="deep", domain_hint=None)

        started = time.time()

        # ── Step 1: Direct retrieval ──────────────────────────────────────────
        retrieval_ms = 0.0
        try:
            retrieval_started = time.time()
            retrieval_result = self.router.search(
                query, k=self.FAST_LANE_TOP_K, domain_hint=domain_hint
            )
            retrieval_ms = (time.time() - retrieval_started) * 1000
            if retrieval_ms > 4000:
                logger.warning("Fast lane retrieval exceeded 4s: %.2f ms", retrieval_ms)
                return self._result_template(
                    query=query,
                    answer="The document database is currently under heavy load. Please try again in a moment or try Deep Mode.",
                    metadata={"degraded": "retrieval_timeout"},
                    route="tax_rag",
                )
            chunks = retrieval_result.chunks if retrieval_result else []
            chunks = self._rank_fast_lane_chunks(query, chunks)
        except Exception as exc:
            logger.error("Fast lane retrieval failed: %s — falling back to deep lane", exc)
            return self._run_tax_rag_lane(query, query_id, timeout_seconds, mode="deep", domain_hint=None)

        # ── Step 2: Build context (2 000-char cap) ───────────────────────────
        context_parts: List[str] = []
        char_count = 0
        for chunk in chunks:
            text = (getattr(chunk, "text", "") or "").strip()
            if not text:
                continue
            remaining = max(0, 1200 - char_count)
            if remaining == 0:
                break
            context_parts.append(text[:remaining])
            char_count += min(len(text), remaining)

        context = "\n\n".join(context_parts)
        sources = list({
            f"{chunk.source.value}:{getattr(chunk, 'document_name', '')}"
            for chunk in chunks
        })

        if not context:
            return self._result_template(
                query=query,
                answer=(
                    "I couldn't find specific information on this. "
                    "Could you clarify: personal income tax, GST, or corporate tax?"
                ),
                confidence=0.0,
                metadata={
                    "sources": [],
                    "lane": "tax_rag_fast",
                    "query_id": query_id,
                    "fast_lane_no_context": True,
                },
                timings={"total": (time.time() - started) * 1000},
                route="tax_rag",
            )

        # ── Step 3: Single LLM call ───────────────────────────────────────────
        if self._general_llm is None:
            self._general_llm = ChatOpenAI(
                model=GENERAL_MODEL,
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base=OPENROUTER_BASE_URL,
                temperature=0.2,
                timeout=timeout_seconds,
                request_timeout=timeout_seconds,
                max_tokens=1200,
            )

        structured_expected = self._is_structured_output_query(query)

        system_prompt = (
            "You are an expert Indian Tax and Finance Advisor.\n"
            "Answer the user's question using ONLY the context below from official Indian tax documents.\n"
            "Include exact figures (\u20b9 amounts, %, limits, section numbers) when present in the context.\n"
            "If the context does not contain the answer, say: "
            "\"The documents don't cover this specifically. Please consult a tax professional.\"\n"
            "If the user asks for comparisons, lists, rates, or slabs, return markdown table output.\n"
            "Be concise and direct."
        )
        user_message = f"Context from tax documents:\n{context}\n\nQuestion: {query}"

        llm_ms = 0.0
        answer = ""
        llm_started = time.time()
        try:
            response = self._general_llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
            )
            if not response or not getattr(response, "content", None):
                raise ValueError("Empty LLM response")
            llm_ms = (time.time() - llm_started) * 1000
            answer = response.content.strip()
            if self._looks_generic_answer(answer):
                evidence_answer = self._build_fast_evidence_summary(query, chunks, structured_expected)
                if evidence_answer:
                    answer = evidence_answer
            if structured_expected:
                answer = self._coerce_structured_output(answer, query)
        except APITimeoutError:
            logger.warning("Fast lane LLM timed out")
            answer = (
                "Request timed out. For complex multi-step tax questions, "
                "please try deep mode."
            )
            llm_ms = min(25, timeout_seconds) * 1000
        except Exception as exc:
            logger.error("Fast lane LLM failed: %s", exc)
            answer = f"Error generating answer: {exc}"

        total_ms = (time.time() - started) * 1000

        fast_confidence = self._compute_fast_lane_confidence(answer, chunks, context)

        return {
            "query": query,
            "blocked": False,
            "answer": answer,
            "confidence": fast_confidence,
            "retrieved_docs_count": len(chunks),
            "plan_steps": [],
            "metadata": {
                "sources": sources,
                "lane": "tax_rag_fast",
                "domain_hint": domain_hint,
                "fast_lane_chunks": len(chunks),
                "fast_lane_context_chars": len(context),
                "cache_status": "miss",
                "query_id": query_id,
            },
            "timings": {
                "retrieval_time_ms": retrieval_ms,
                "llm_time_ms": llm_ms,
                "total": total_ms,
            },
            "route": "tax_rag",
        }

    def _run_tax_rag_lane(
        self,
        query: str,
        query_id: str,
        timeout_seconds: int = 120,
        mode: str = "fast",
        domain_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Tax RAG lane dispatcher.

        fast  → _run_fast_tax_lane: direct retrieve → LLM (no planner/verification)
        deep  → full multi-agent pipeline (planner, retrieval, reasoning, verification)
        """
        if mode == "fast":
            return self._run_fast_tax_lane(query, query_id, domain_hint, timeout_seconds)

        # ── Deep mode: full multi-agent pipeline ───────────────────────────────
        logger.info(f"Executing DEEP tax RAG lane with full pipeline (timeout={timeout_seconds}s)")

        self._ensure_workflow()

        started = time.time()
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            self.workflow.run,
            query,
            {
                "query_id": query_id,
                "gatekeeper_stage": "orchestrator_pre_routing_passed",
            },
            self.TAX_RAG_STAGE_BUDGETS,
            self.TAX_RAG_TOP_K,
        )
        try:
            # Run full workflow with timeout
            state = future.result(timeout=timeout_seconds)
            executor.shutdown(wait=False)

        except TimeoutError:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
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
            executor.shutdown(wait=False, cancel_futures=True)
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
        metadata["debug_trace"] = {}

        # ======================================================================
        # Phase C: Quality Assessment & Confidence Composition
        # ======================================================================
        
        # Extract quality assessment inputs from workflow state
        state_reasoning = getattr(state, "reasoning", None)
        state_reasoning_steps = getattr(state, "reasoning_steps", None)
        reasoning_chain = state_reasoning if isinstance(state_reasoning, str) else ""
        reasoning_steps = state_reasoning_steps if isinstance(state_reasoning_steps, list) else []
        retrieved_docs = state.retrieved_docs or []

        # Phase 1: Deterministic grounding guard for tax lane only.
        guard_result = self._apply_tax_grounding_guard(query, answer, retrieved_docs)
        answer = guard_result.get("answer", answer)
        metadata["grounding_guard"] = guard_result.get("debug", {})
        
        # Step 5: Assess answer quality across three dimensions
        try:
            quality_result = self.quality_assessment.assess_answer(
                query=query,
                retrieved_docs=[
                    {"text": chunk.text, "source": chunk.source.value}
                    for result in retrieved_docs
                    for chunk in getattr(result, "chunks", [])
                ],
                reasoning_chain=reasoning_chain,
                reasoning_steps=reasoning_steps,
                answer=answer,
                existing_verification=verification,
            )
            
            logger.debug(f"Quality assessment: {quality_result.explanation}")
            
            # Step 6: Compose confidence from multiple signals
            # Wrap float signals into ConfidenceSignal objects as required by compose()
            retrieval_cs = ConfidenceSignal(
                source="retrieval",
                confidence=float(quality_result.retrieval_signal),
                details={},
            )
            reasoning_cs = ConfidenceSignal(
                source="reasoning",
                confidence=float(quality_result.reasoning_signal),
                details={},
            )
            verification_cs = ConfidenceSignal(
                source="verification",
                confidence=float(quality_result.verification_signal),
                details={},
            )
            composed_result = self.confidence_composer.compose(
                retrieval_signal=retrieval_cs,
                reasoning_signal=reasoning_cs,
                verification_signal=verification_cs,
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
                "confidence_level": composed_result.confidence_level.value,
                "gating_action": composed_result.override_action,
                "confidence_explanation": composed_result.explanation,
            }
            metadata["debug_trace"]["gating_decision"] = {
                "should_override_answer": composed_result.should_override_answer,
                "override_action": composed_result.override_action,
                "override_replacement": composed_result.override_replacement,
                "final_confidence": composed_result.overall_confidence,
                "confidence_level": composed_result.confidence_level.value,
            }
            
            # Apply quality gating if needed
            final_answer = answer
            final_confidence = composed_result.overall_confidence
            
            if composed_result.should_override_answer:
                logger.info(
                    f"Confidence gating activated: action={composed_result.override_action}, "
                    f"confidence={composed_result.overall_confidence:.2f}"
                )
                
                if composed_result.override_action == "clarify":
                    # Offer clarification instead of uncertain answer
                    final_answer = composed_result.override_replacement
                    metadata["gating_applied"] = "clarification"
                    
                elif composed_result.override_action == "admit_uncertainty":
                    # Admit uncertainty instead of potentially wrong answer
                    final_answer = composed_result.override_replacement
                    metadata["gating_applied"] = "uncertainty_admission"
                
                else:
                    metadata["gating_applied"] = composed_result.override_action
            
            elif composed_result.override_action == "caveat":
                # Add caveat without replacing answer
                caveat_prefix = f"⚠️ {composed_result.override_replacement}\n\n"
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
                "timings": {
                    "planner_time_ms": metadata.get("stage_timings_ms", {}).get("planner", 0.0),
                    "retrieval_time_ms": metadata.get("stage_timings_ms", {}).get("retrieval", 0.0),
                    "reasoning_time_ms": metadata.get("stage_timings_ms", {}).get("reasoning", 0.0),
                    "verification_time_ms": metadata.get("stage_timings_ms", {}).get("verification", 0.0),
                    "total": total_ms,
                },
                "route": "tax_rag"
            }
        
        except Exception as e:
            logger.exception(f"Phase C quality assessment failed: {e}. Returning base answer.")
            metadata.setdefault("debug_trace", {})["gating_decision"] = {
                "should_override_answer": False,
                "override_action": None,
                "override_replacement": None,
                "phase_c_error": str(e),
            }
            
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
                "timings": {
                    "planner_time_ms": metadata.get("stage_timings_ms", {}).get("planner", 0.0),
                    "retrieval_time_ms": metadata.get("stage_timings_ms", {}).get("retrieval", 0.0),
                    "reasoning_time_ms": metadata.get("stage_timings_ms", {}).get("reasoning", 0.0),
                    "verification_time_ms": metadata.get("stage_timings_ms", {}).get("verification", 0.0),
                    "total": total_ms,
                },
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
