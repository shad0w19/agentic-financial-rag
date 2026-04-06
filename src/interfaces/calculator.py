"""
File: src/interfaces/calculator.py

Purpose:
Abstract interface for financial calculations.
Defines contract for tax and investment calculation services.
LLM must never compute taxes directly - only through this interface.

Dependencies:
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from src.core.types import IncomeBreakdown, DeductionBreakdown, TaxCalculationResult

Implements Interface:
ITaxCalculator (abstract base class)

Notes:
- All tax/investment calculations must be deterministic
- LLM can only call these tools, never compute directly
- Results must be traceable and auditable
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

from src.core.types import (
    IncomeBreakdown,
    DeductionBreakdown,
    TaxCalculationResult,
)


class ITaxCalculator(ABC):
    """
    Abstract interface for deterministic tax calculations.
    
    LLM agents must use this interface for all calculations.
    Never compute taxes in the LLM reasoning chain directly.
    """

    @abstractmethod
    def calculate_income_tax(
        self,
        income_breakdown: IncomeBreakdown,
        deduction_breakdown: DeductionBreakdown,
        financial_year: str | None = None,
    ) -> TaxCalculationResult:
        """
        Calculate income tax based on income and deductions.
        
        Args:
            income_breakdown: Breakdown of income sources
            deduction_breakdown: Breakdown of deductible items
            financial_year: Financial year (e.g., "2024-25")
        
        Returns:
            TaxCalculationResult with tax amount and effective rate
        
        Raises:
            ValueError: If inputs are invalid
        """
        pass

    @abstractmethod
    def calculate_gst(
        self,
        amount: float,
        gst_category: str,
    ) -> Dict[str, float]:
        """
        Calculate GST and components.
        
        Args:
            amount: Amount subject to GST
            gst_category: GST category (e.g., "5%", "18%", "28%")
        
        Returns:
            Dict with base_amount, gst, total
        
        Raises:
            ValueError: If category or amount invalid
        """
        pass

    @abstractmethod
    def get_eligible_rebates(
        self,
        income: float,
        financial_year: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Get eligible tax rebates for given income level.
        
        Args:
            income: Annual income
            financial_year: Financial year
        
        Returns:
            List of eligible rebates with details
        """
        pass

    @abstractmethod
    def get_deduction_limits(
        self,
        section: str,
        financial_year: str | None = None,
    ) -> Dict[str, Any]:
        """
        Get deduction limits for a specific section.
        
        Args:
            section: Tax section (e.g., "80C", "80D")
            financial_year: Financial year
        
        Returns:
            Dict with limit and applicable conditions
        """
        pass

    @abstractmethod
    def calculate_tax_saving_options(
        self,
        income: float,
        current_tax: float,
        financial_year: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate tax saving recommendations.
        
        Args:
            income: Annual income
            current_tax: Current tax liability
            financial_year: Financial year
        
        Returns:
            List of tax saving options with potential savings
        """
        pass


class IInvestmentCalculator(ABC):
    """
    Abstract interface for investment calculations.
    """

    @abstractmethod
    def calculate_returns(
        self,
        principal: float,
        rate: float,
        years: int,
        compound_frequency: str = "annual",
    ) -> Dict[str, float]:
        """
        Calculate investment returns.
        
        Args:
            principal: Initial investment
            rate: Interest rate (percentage)
            years: Investment period
            compound_frequency: Compounding frequency
        
        Returns:
            Dict with interest earned and final amount
        """
        pass

    @abstractmethod
    def get_investment_options(
        self,
        amount: float,
        timeline_months: int,
        risk_profile: str = "moderate",
    ) -> List[Dict[str, Any]]:
        """
        Get investment options recommendations.
        
        Args:
            amount: Amount to invest
            timeline_months: Investment horizon
            risk_profile: Risk tolerance level
        
        Returns:
            List of recommended investment options
        """
        pass
