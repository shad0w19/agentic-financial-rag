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

        # Build context from retrieved docs
        context = self._build_context(retrieved_docs)

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
            
            return answer
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return "Information not found."

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
        context = self._build_context(retrieved_docs)

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
            
            return answer
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return "Information not found."

    def _build_context(
        self,
        retrieved_docs: List[RetrievalResult],
    ) -> str:
        """Build context string from retrieved documents."""
        if not retrieved_docs:
            return "No relevant documents found."

        context_parts = []

        for result in retrieved_docs:
            for chunk in result.chunks:
                context_parts.append(
                    f"Source: {chunk.source.value}\n{chunk.text}"
                )

        return "\n\n---\n\n".join(context_parts)

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
