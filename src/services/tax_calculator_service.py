"""
File: src/services/tax_calculator_service.py

Purpose:
Deterministic tax calculation service.
Implements ITaxCalculator interface.

Dependencies:
from typing import Dict, List, Any
from src.import_map import ITaxCalculator, IncomeBreakdown, DeductionBreakdown, TaxCalculationResult

Implements Interface:
ITaxCalculator

Notes:
- Pure calculation logic (no LLM)
- Indian tax slabs (FY 2024-25)
- Deterministic output
"""

import logging
from typing import Any, Dict, List

from src.import_map import (
    DeductionBreakdown,
    IncomeBreakdown,
    ITaxCalculator,
    TaxCalculationResult,
)


logger = logging.getLogger(__name__)


class TaxCalculatorService(ITaxCalculator):
    """
    Tax calculation service for Indian income tax.
    """

    TAX_SLABS_2024_25 = [
        (250000, 0.0),
        (500000, 0.05),
        (1000000, 0.20),
        (1500000, 0.30),
        (float("inf"), 0.30),
    ]

    DEDUCTION_LIMITS = {
        "80C": 150000,
        "80D": 100000,
        "80E": 50000,
        "80TTA": 10000,
    }

    def __init__(self) -> None:
        """Initialize tax calculator."""
        self.logger = logging.getLogger(__name__)

    def calculate_income_tax(
        self,
        income_breakdown: IncomeBreakdown,
        deduction_breakdown: DeductionBreakdown,
        financial_year: str | None = None,
    ) -> TaxCalculationResult:
        """
        Calculate income tax.
        
        Args:
            income_breakdown: Income sources
            deduction_breakdown: Deductible items
            financial_year: FY (unused, uses 2024-25)
        
        Returns:
            TaxCalculationResult
        """
        gross_income = (
            income_breakdown.salary
            + income_breakdown.capital_gains
            + income_breakdown.rental_income
            + income_breakdown.business_income
            + income_breakdown.other_income
        )

        total_deductions = min(
            deduction_breakdown.section_80c, self.DEDUCTION_LIMITS["80C"]
        )
        total_deductions += min(
            deduction_breakdown.section_80d, self.DEDUCTION_LIMITS["80D"]
        )
        total_deductions += min(
            deduction_breakdown.section_80e, self.DEDUCTION_LIMITS["80E"]
        )
        total_deductions += min(
            deduction_breakdown.section_80tta, self.DEDUCTION_LIMITS["80TTA"]
        )
        total_deductions += deduction_breakdown.other_deductions

        taxable_income = max(0, gross_income - total_deductions)

        tax_amount = self._calculate_tax_from_slabs(taxable_income)
        effective_rate = (
            (tax_amount / gross_income * 100) if gross_income > 0 else 0
        )

        applicable_slab = self._get_applicable_slab(taxable_income)

        return TaxCalculationResult(
            gross_income=gross_income,
            total_deductions=total_deductions,
            taxable_income=taxable_income,
            tax_amount=tax_amount,
            effective_tax_rate=effective_rate,
            applicable_slab=applicable_slab,
        )

    def calculate_gst(
        self,
        amount: float,
        gst_category: str,
    ) -> Dict[str, float]:
        """
        Calculate GST.
        
        Args:
            amount: Amount subject to GST
            gst_category: GST rate ("5%", "12%", "18%", "28%")
        
        Returns:
            Dict with base_amount, gst, total
        """
        rate_map = {
            "5%": 0.05,
            "12%": 0.12,
            "18%": 0.18,
            "28%": 0.28,
        }

        rate = rate_map.get(gst_category, 0.18)
        gst_amount = amount * rate

        return {
            "base_amount": amount,
            "gst": gst_amount,
            "total": amount + gst_amount,
        }

    def get_eligible_rebates(
        self,
        income: float,
        financial_year: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Get eligible rebates.
        
        Args:
            income: Annual income
            financial_year: FY
        
        Returns:
            List of eligible rebates
        """
        rebates = []

        if income <= 500000:
            rebates.append(
                {
                    "name": "Section 87A Rebate",
                    "amount": min(income * 0.05, 12500),
                    "condition": "Income <= 5 lakhs",
                }
            )

        return rebates

    def get_deduction_limits(
        self,
        section: str,
        financial_year: str | None = None,
    ) -> Dict[str, Any]:
        """
        Get deduction limits.
        
        Args:
            section: Tax section
            financial_year: FY
        
        Returns:
            Dict with limit and conditions
        """
        limits = {
            "80C": {
                "limit": 150000,
                "items": ["LIC", "PPF", "ELSS", "NSC"],
            },
            "80D": {
                "limit": 100000,
                "items": ["Health Insurance"],
            },
            "80E": {
                "limit": 50000,
                "items": ["Education Loan Interest"],
            },
            "80TTA": {
                "limit": 10000,
                "items": ["Savings Account Interest"],
            },
        }

        return limits.get(section, {})

    def calculate_tax_saving_options(
        self,
        income: float,
        current_tax: float,
        financial_year: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Get tax saving recommendations.
        
        Args:
            income: Annual income
            current_tax: Current tax liability
            financial_year: FY
        
        Returns:
            List of tax saving options
        """
        options = []

        if income > 250000:
            options.append(
                {
                    "option": "Section 80C Investment",
                    "potential_saving": 150000 * 0.20,
                    "description": "Invest in LIC, PPF, ELSS",
                }
            )

        if income > 500000:
            options.append(
                {
                    "option": "Health Insurance (80D)",
                    "potential_saving": 100000 * 0.30,
                    "description": "Health insurance premium",
                }
            )

        return options

    def _calculate_tax_from_slabs(self, taxable_income: float) -> float:
        """Calculate tax using slabs."""
        tax = 0.0
        previous_limit = 0

        for limit, rate in self.TAX_SLABS_2024_25:
            if taxable_income <= previous_limit:
                break

            taxable_in_slab = min(taxable_income, limit) - previous_limit
            tax += taxable_in_slab * rate
            previous_limit = limit

        return tax

    def _get_applicable_slab(self, taxable_income: float) -> str:
        """Get applicable tax slab."""
        for limit, rate in self.TAX_SLABS_2024_25:
            if taxable_income <= limit:
                return f"{int(rate * 100)}%"
        return "30%"
