"""
File: src/agents/planner_agent.py

Purpose:
Planner agent for query decomposition into structured plan.
Converts user query into List[PlanStep] using LLM.

Dependencies:
from typing import List, Dict, Any, Callable
from src.import_map import PlanStep, Query

Implements Interface:
None (agent component)

Notes:
- LLM-based planning (uses general_llm for query decomposition)
- Detects query type and generates steps
- Includes error handling for LLM calls
"""

import json
import logging
from typing import Any, Callable, Dict, List

from src.import_map import PlanStep, Query


logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    Planner agent for query decomposition using LLM.
    """

    TAX_KEYWORDS = [
        "tax",
        "income tax",
        "gst",
        "deduction",
        "rebate",
        "slab",
        "filing",
        "return",
        "assessment",
    ]

    CALCULATION_KEYWORDS = [
        "calculate",
        "compute",
        "how much",
        "what is",
        "amount",
        "corpus",
        "sip",
        "returns",
    ]

    INVESTMENT_KEYWORDS = [
        "investment",
        "mutual fund",
        "ppf",
        "fd",
        "elss",
        "retirement",
        "savings",
        "portfolio",
    ]

    def __init__(self, llm_generator: Callable[[str], str]) -> None:
        """
        Initialize planner agent with LLM.
        
        Args:
            llm_generator: Callable that takes prompt and returns plan
        """
        self.logger = logging.getLogger(__name__)
        self.llm_generator = llm_generator
        self.step_counter = 0

    def plan(self, query: Query | str) -> List[PlanStep]:
        """
        Generate plan from query using LLM.
        
        Args:
            query: Query object or string
        
        Returns:
            List of PlanStep
        """
        query_text = query.text if isinstance(query, Query) else query
        query_lower = query_text.lower()

        # Create LLM prompt for planning
        planning_prompt = f"""You are a financial query planner. Analyze this query and create a structured plan.

Query: {query_text}

⚠️  CRITICAL: For ALL financial/tax queries, ALWAYS include retrieval step!
Specific financial advice MUST be grounded in official documents.
Never skip retrieval - it is MANDATORY.

Generate a JSON plan with these steps:
1. security - Validate query safety (ALWAYS include)
2. retrieval - Retrieve official documents (ALWAYS include for financial queries)
3. calculation - Calculate values only if needed
4. reasoning - Generate answer grounded in retrieved documents
5. verification - Verify answer correctness

Return JSON array with required steps.
Each step must have:
- step_id: "step_1", "step_2", etc.
- action_type: "security", "retrieval", "calculation", "reasoning", or "verification"
- description: What this step does
- dependencies: ["step_id"] of prior steps
- parameters: {{}}

MANDATORY SEQUENCE: security → retrieval → reasoning
Modify only if needed: Add calculation | Add verification

Return ONLY valid JSON array, no other text."""

        try:
            # Call LLM to generate plan
            response = self.llm_generator(planning_prompt)
            logger.debug(f"[PLANNER] LLM Raw Response (first 500 chars): {response[:500]}")
            
            # Parse JSON response
            try:
                # Clean JSON response
                json_str = response.strip()
                if json_str.startswith("```"):
                    # Remove markdown code blocks if present
                    json_str = json_str.split("```")[1]
                    if json_str.startswith("json"):
                        json_str = json_str[4:]
                    json_str = json_str.strip()
                
                logger.debug(f"[PLANNER] Cleaned JSON (first 300 chars): {json_str[:300]}")
                steps_data = json.loads(json_str)
                
                # Convert to PlanStep objects
                plan = []
                for step_data in steps_data:
                    try:
                        step = PlanStep(
                            step_id=step_data.get("step_id", f"step_{len(plan)+1}"),
                            action_type=step_data.get("action_type", "unknown"),
                            description=step_data.get("description", ""),
                            dependencies=step_data.get("dependencies", []),
                            parameters=step_data.get("parameters", {})
                        )
                        plan.append(step)
                        logger.debug(f"[PLANNER] ✓ Added step: {step.action_type}")
                    except Exception as e:
                        logger.debug(f"[PLANNER] ✗ Parse error: {e}, skipping step")
                        continue
                
                if plan:
                    step_types = [s.action_type for s in plan]
                    
                    # SAFETY CHECK: Ensure retrieval is included
                    has_retrieval = any(s.action_type == "retrieval" for s in plan)
                    if not has_retrieval:
                        logger.warning("[PLANNER] ⚠️  LLM didn't include retrieval step, adding manually")
                        retrieval_step = PlanStep(
                            step_id="step_retrieval",
                            action_type="retrieval",
                            description="Retrieve official financial documents for accuracy",
                            dependencies=[plan[0].step_id] if plan else [],
                            parameters={"k": 5}
                        )
                        # Insert after security step (index 1 if security exists, else 0)
                        security_idx = next((i for i, s in enumerate(plan) if s.action_type == "security"), -1)
                        insert_idx = security_idx + 1 if security_idx >= 0 else 1
                        plan.insert(insert_idx, retrieval_step)
                        step_types = [s.action_type for s in plan]
                        logger.info(f"[PLANNER] ✅ FORCED retrieval - Plan now: {len(plan)} steps → {step_types}")
                    else:
                        logger.info(f"[PLANNER] ✅ LLM Plan: {len(plan)} steps → {step_types}")
                    
                    return plan
                else:
                    logger.warning("[PLANNER] ⚠️  No valid steps from LLM, using fallback")
                    fallback = self._generate_fallback_plan(query_text)
                    fallback_types = [s.action_type for s in fallback]
                    logger.info(f"[PLANNER] ✅ Fallback Plan: {len(fallback)} steps → {fallback_types}")
                    return fallback
                    
            except json.JSONDecodeError as e:
                logger.error(f"[PLANNER] ❌ JSON parse error: {e}")
                logger.error(f"[PLANNER] ⚠️  Raw response: {response[:1000]}")
                fallback = self._generate_fallback_plan(query_text)
                fallback_types = [s.action_type for s in fallback]
                logger.info(f"[PLANNER] ✅ Fallback Plan: {len(fallback)} steps → {fallback_types}")
                return fallback
                
        except Exception as e:
            logger.error(f"LLM planning error: {e}")
            return self._generate_fallback_plan(query_text)

    def _generate_fallback_plan(self, query_text: str) -> List[PlanStep]:
        """Generate fallback rule-based plan."""
        query_lower = query_text.lower()

        plan: List[PlanStep] = []

        # Detect query type
        is_tax_query = any(kw in query_lower for kw in self.TAX_KEYWORDS)
        is_calculation = any(kw in query_lower for kw in self.CALCULATION_KEYWORDS)
        is_investment = any(kw in query_lower for kw in self.INVESTMENT_KEYWORDS)

        # Security check step (always first)
        security_step = self._create_step(
            action_type="security",
            description="Validate query for security threats",
            dependencies=[],
        )
        plan.append(security_step)

        # Retrieval step - ALWAYS include for tax/investment (mandatory)
        # Only skip if it's a pure calculation query with no tax/investment context
        if is_tax_query or is_investment or not is_calculation:
            retrieval_step = self._create_step(
                action_type="retrieval",
                description="Retrieve relevant financial documents for accuracy",
                dependencies=[security_step.step_id],
            )
            plan.append(retrieval_step)
            last_step_id = retrieval_step.step_id
        else:
            # Even for pure calculations, retrieve docs for context
            logger.debug("[PLANNER] Pure calculation detected, but still including retrieval for context")
            retrieval_step = self._create_step(
                action_type="retrieval",
                description="Retrieve relevant documents for calculation context",
                dependencies=[security_step.step_id],
            )
            plan.append(retrieval_step)
            last_step_id = retrieval_step.step_id

        # Calculation step
        if is_calculation or is_tax_query or is_investment:
            calc_type = self._detect_calculation_type(query_lower)
            calculation_step = self._create_step(
                action_type="calculation",
                description=f"Perform {calc_type} calculation",
                dependencies=[last_step_id],
                parameters={"type": calc_type},
            )
            plan.append(calculation_step)
            last_step_id = calculation_step.step_id

        # Reasoning step
        reasoning_step = self._create_step(
            action_type="reasoning",
            description="Synthesize context and generate advice",
            dependencies=[last_step_id],
            parameters={"query": query_text},
        )
        plan.append(reasoning_step)

        # Verification step
        verification_step = self._create_step(
            action_type="verification",
            description="Verify citations and consistency",
            dependencies=[reasoning_step.step_id],
        )
        plan.append(verification_step)

        return plan

    def _create_step(
        self,
        action_type: str,
        description: str,
        dependencies: List[str],
        parameters: Dict[str, Any] | None = None,
    ) -> PlanStep:
        """Create a plan step."""
        self.step_counter += 1
        step_id = f"step_{self.step_counter}"

        return PlanStep(
            step_id=step_id,
            action_type=action_type,
            description=description,
            dependencies=dependencies,
            parameters=parameters or {},
        )

    def _detect_calculation_type(self, query_lower: str) -> str:
        """Detect type of calculation needed."""
        if "tax" in query_lower:
            return "tax"
        elif "investment" in query_lower or "sip" in query_lower:
            return "investment"
        elif "gst" in query_lower:
            return "gst"
        else:
            return "general"

    def get_plan_summary(self, plan: List[PlanStep]) -> Dict[str, Any]:
        """
        Get summary of plan.
        
        Args:
            plan: List of PlanStep
        
        Returns:
            Plan summary dict
        """
        return {
            "total_steps": len(plan),
            "steps": [
                {
                    "step_id": step.step_id,
                    "action": step.action_type,
                    "description": step.description,
                }
                for step in plan
            ],
        }
