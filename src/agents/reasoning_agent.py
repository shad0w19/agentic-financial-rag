"""
File: src/agents/reasoning_agent.py

Purpose:
LLM-based reasoning agent for generating final answers.
Synthesizes retrieved context into grounded responses.

Dependencies:
from typing import List, Dict, Any, Optional, Callable
from src.import_map import RetrievalResult, PlanStep

Implements Interface:
None (agent component)

Notes:
- Injectable LLM (no hardcoded provider)
- Context-grounded generation
- Strict grounding with no hallucination
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from src.import_map import PlanStep, RetrievalResult


logger = logging.getLogger(__name__)


# Strict grounding system prompt for financial context
SYSTEM_PROMPT = """You are an expert Indian Financial and Tax Advisor.

You MUST follow these rules strictly:

1. Answer ONLY using the provided context. Do NOT use prior knowledge.
2. Always extract and use exact numerical values (tax rates, slabs, limits, percentages).
3. If the query requires calculation (e.g., 'calculate', 'tax', 'how much', 'total'), perform exact arithmetic step-by-step using numbers from the context.
4. Do NOT guess or infer missing values.
5. If the answer is not present in the context, output EXACTLY:
Information not found.

Output format:

Reasoning:
<Step-by-step explanation and exact calculations using context>

Final Answer:
<Clear, concise final answer>

Sources Used:
<Quote exact lines or key facts from context>"""


class ReasoningAgent:
    """
    Reasoning agent for generating final answers.
    """

    _QUERY_STOPWORDS = {
        "what", "how", "does", "the", "for", "and", "with", "can", "claim", "rate",
        "tax", "income", "about", "under", "into", "from", "that", "this", "your",
    }
    _MAX_CONTEXT_CHUNKS = 4
    _MAX_CONTEXT_CHARS = 3200

    def __init__(
        self,
        llm_generator: Callable[[str], str],
    ) -> None:
        """
        Initialize reasoning agent.
        
        Args:
            llm_generator: Callable that takes prompt and returns answer
        """
        self.llm_generator = llm_generator
        self.logger = logging.getLogger(__name__)

    def reason(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult],
        plan: List[PlanStep],
        calculations: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate final answer using retrieved context.
        
        Args:
            query: Original user query
            retrieved_docs: List of RetrievalResult
            plan: List of PlanStep
            calculations: Optional calculation results
        
        Returns:
            Final answer string
        """
        # SHORT-CIRCUIT: If no context, return immediately (saves API calls)
        if not retrieved_docs or len(retrieved_docs) == 0:
            return "Information not found."
        
        # Check if any chunks exist
        has_chunks = any(r.chunks for r in retrieved_docs)
        if not has_chunks:
            return "Information not found."

        # If retrieved chunks are too weak/short, ask for a more specific question.
        has_substantive_chunks = any(
            chunk.text and len(chunk.text.strip()) >= 80
            for result in retrieved_docs
            for chunk in result.chunks
        )
        if not has_substantive_chunks:
            return (
                "I need a bit more detail to give an accurate finance answer. "
                "Please specify your context and exact topic, for example: "
                "'personal income tax deductions under 80C', 'GST registration threshold', or 'corporate tax rate for domestic companies'."
            )

        # Build focused context from the most query-relevant retrieved docs.
        context = self._build_context(retrieved_docs, query=query)

        # Build prompt with strict formatting
        prompt = self._build_prompt(
            query=query,
            context=context,
            calculations=calculations,
        )

        # Generate answer
        try:
            answer = self.llm_generator(prompt)
            
            # Strip reasoning leak (DeepSeek/Qwen output format cleanup)
            if "Final answer:" in answer.lower():
                answer = answer.split("Final answer:")[-1].strip()
            elif "Final Answer:" in answer:
                answer = answer.split("Final Answer:")[-1].strip()
            elif "Reasoning:" in answer and "\n\n" in answer:
                # If starts with "Reasoning:", extract content after first double newline
                parts = answer.split("\n\n", 1)
                if len(parts) > 1:
                    answer = parts[-1].strip()

            if self._looks_like_not_found(answer):
                fallback_answer = self._build_evidence_backed_fallback(query, retrieved_docs)
                if fallback_answer:
                    return fallback_answer
            
            return answer
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            fallback_answer = self._build_evidence_backed_fallback(query, retrieved_docs)
            return fallback_answer or "Information not found."

    def reason_with_calculations(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult],
        calculations: Dict[str, Any],
    ) -> str:
        """
        Generate answer combining retrieval and calculations.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            calculations: Calculation results
        
        Returns:
            Final answer
        """
        context = self._build_context(retrieved_docs, query=query)

        prompt = self._build_calculation_prompt(
            query=query,
            context=context,
            calculations=calculations,
        )

        try:
            answer = self.llm_generator(prompt)
            
            # Strip reasoning leak (DeepSeek/Qwen output format cleanup)
            if "Final answer:" in answer.lower():
                answer = answer.split("Final answer:")[-1].strip()
            elif "Final Answer:" in answer:
                answer = answer.split("Final Answer:")[-1].strip()
            elif "Reasoning:" in answer and "\n\n" in answer:
                parts = answer.split("\n\n", 1)
                if len(parts) > 1:
                    answer = parts[-1].strip()

            if self._looks_like_not_found(answer):
                fallback_answer = self._build_evidence_backed_fallback(query, retrieved_docs)
                if fallback_answer:
                    return fallback_answer
            
            return answer
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            fallback_answer = self._build_evidence_backed_fallback(query, retrieved_docs)
            return fallback_answer or "Information not found."

    def _build_context(
        self,
        retrieved_docs: List[RetrievalResult],
        query: Optional[str] = None,
    ) -> str:
        """Build context string from retrieved documents."""
        if not retrieved_docs:
            return "No relevant documents found."

        selected_chunks = self._select_relevant_chunks(retrieved_docs, query=query)
        context_parts = []

        total_chars = 0
        for chunk in selected_chunks:
            chunk_text = chunk.text or ""
            entry = f"Source: {chunk.source.value}\n{chunk_text}"
            if total_chars + len(entry) > self._MAX_CONTEXT_CHARS and context_parts:
                break
            context_parts.append(entry)
            total_chars += len(entry)

        return "\n\n---\n\n".join(context_parts)

    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract meaningful lexical terms from the user query."""
        tokens = re.findall(r"[a-zA-Z0-9]+", (query or "").lower())
        return [token for token in tokens if len(token) > 2 and token not in self._QUERY_STOPWORDS]

    def _score_chunk_for_query(self, chunk_text: str, query_terms: List[str]) -> float:
        """Score chunk relevance using lexical overlap and presence of concrete values."""
        text = (chunk_text or "").lower()
        if not text:
            return 0.0

        overlap = sum(1 for term in query_terms if term in text)
        has_numbers = bool(re.search(r"\d", text))
        return float(overlap) + (0.35 if overlap and has_numbers else 0.0)

    def _select_relevant_chunks(
        self,
        retrieved_docs: List[RetrievalResult],
        query: Optional[str] = None,
    ) -> List[Any]:
        """Pick the most relevant retrieved chunks for the current query."""
        all_chunks = [chunk for result in retrieved_docs for chunk in result.chunks]
        if not all_chunks:
            return []

        query_terms = self._extract_query_terms(query or "")
        if not query_terms:
            return all_chunks[: self._MAX_CONTEXT_CHUNKS]

        scored_chunks = [
            (chunk, self._score_chunk_for_query(chunk.text or "", query_terms))
            for chunk in all_chunks
        ]
        positive_scored = [(chunk, score) for chunk, score in scored_chunks if score > 0]
        ranked_pairs = sorted(
            positive_scored or scored_chunks,
            key=lambda item: item[1],
            reverse=True,
        )
        ranked = [chunk for chunk, _ in ranked_pairs]
        return ranked[: self._MAX_CONTEXT_CHUNKS]

    def _looks_like_not_found(self, answer: str) -> bool:
        """Detect model answers that amount to a not-found response."""
        lowered = (answer or "").strip().lower()
        return lowered.startswith("information not found") or "information not found" in lowered[:120]

    def _build_evidence_backed_fallback(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult],
    ) -> str:
        """Build a constrained answer from relevant evidence when the model misses it."""
        relevant_chunks = self._select_relevant_chunks(retrieved_docs, query=query)
        query_terms = self._extract_query_terms(query)
        evidence_lines: List[str] = []

        for chunk in relevant_chunks:
            text = re.sub(r"\s+", " ", (chunk.text or "").strip())
            if not text:
                continue
            lowered = text.lower()
            overlap = sum(1 for term in query_terms if term in lowered)
            if overlap <= 0:
                continue
            snippet = self._extract_local_snippet(text, query_terms)
            evidence_lines.append(f"- Source: {chunk.source.value} | {snippet}")
            if len(evidence_lines) >= 3:
                break

        if not evidence_lines:
            return ""

        return "\n".join([
            "I found relevant information in the retrieved documents:",
            *evidence_lines,
            "The retrieved material is relevant, but it does not cleanly provide a single direct answer for the exact wording of the question.",
        ])

    def _extract_local_snippet(self, text: str, query_terms: List[str]) -> str:
        """Extract a compact local snippet around the first matched query term."""
        lowered = text.lower()
        for term in query_terms:
            idx = lowered.find(term)
            if idx >= 0:
                start = max(0, idx - 80)
                end = min(len(text), idx + 160)
                return text[start:end].strip()
        return text[:220]

    def _build_prompt(
        self,
        query: str,
        context: str,
        calculations: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build prompt for LLM with strict context formatting.
        Includes calculation keyword detection.
        """
        # Detect calculation keywords in query
        calculation_keywords = ["calculate", "tax", "how much", "total", "amount"]
        has_calculation = any(kw in query.lower() for kw in calculation_keywords)
        
        # Build strict formatted prompt
        prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}
"""
        
        # Add calculation emphasis if detected
        if has_calculation:
            prompt += "\nIMPORTANT: Perform exact step-by-step calculations using only the numbers in the context."
        
        if calculations:
            prompt += f"\n\nCalculation Results:\n{self._format_calculations(calculations)}"
        
        return prompt

    def _build_calculation_prompt(
        self,
        query: str,
        context: str,
        calculations: Dict[str, Any],
    ) -> str:
        """Build prompt for calculation-based reasoning."""
        prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

Calculation Results:
{self._format_calculations(calculations)}

IMPORTANT: Perform exact step-by-step calculations using only the numbers in the context.
"""
        return prompt

    def _format_calculations(
        self,
        calculations: Dict[str, Any],
    ) -> str:
        """Format calculation results for prompt."""
        lines = []
        for key, value in calculations.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _fallback_answer(self, query: str, context: str) -> str:
        """Fallback answer when LLM fails."""
        return "Information not found."

    def _fallback_calculation_answer(
        self,
        query: str,
        context: str,
        calculations: Dict[str, Any],
    ) -> str:
        """Fallback answer for calculation-based queries."""
        return "Information not found."
