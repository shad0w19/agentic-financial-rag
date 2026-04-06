"""
File: src/services/investment_service.py

Purpose:
Investment calculation service.
Implements IInvestmentCalculator interface.

Dependencies:
from typing import Dict, List, Any
from src.import_map import IInvestmentCalculator

Implements Interface:
IInvestmentCalculator

Notes:
- Pure calculation logic (no LLM)
- Deterministic output
- No external APIs
"""

import logging
import math
from typing import Any, Dict, List

from src.import_map import IInvestmentCalculator


logger = logging.getLogger(__name__)


class InvestmentService(IInvestmentCalculator):
    """
    Investment calculation service.
    """

    INVESTMENT_OPTIONS = {
        "FD": {
            "name": "Fixed Deposit",
            "rate_range": (4.0, 7.0),
            "liquidity": "Low",
            "risk": "Very Low",
        },
        "PPF": {
            "name": "Public Provident Fund",
            "rate_range": (7.0, 8.0),
            "liquidity": "Medium",
            "risk": "Very Low",
        },
        "ELSS": {
            "name": "Equity Linked Savings Scheme",
            "rate_range": (10.0, 15.0),
            "liquidity": "Medium",
            "risk": "Medium",
        },
        "MF": {
            "name": "Mutual Fund",
            "rate_range": (8.0, 12.0),
            "liquidity": "High",
            "risk": "Medium",
        },
        "STOCKS": {
            "name": "Stocks",
            "rate_range": (12.0, 20.0),
            "liquidity": "High",
            "risk": "High",
        },
    }

    def __init__(self) -> None:
        """Initialize investment service."""
        self.logger = logging.getLogger(__name__)

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
        if principal <= 0 or rate < 0 or years <= 0:
            raise ValueError("Invalid input parameters")

        frequency_map = {
            "annual": 1,
            "semi-annual": 2,
            "quarterly": 4,
            "monthly": 12,
        }

        n = frequency_map.get(compound_frequency, 1)
        r = rate / 100

        final_amount = principal * (1 + r / n) ** (n * years)
        interest_earned = final_amount - principal

        return {
            "principal": principal,
            "interest_earned": interest_earned,
            "final_amount": final_amount,
            "rate": rate,
            "years": years,
        }

    def get_investment_options(
        self,
        amount: float,
        timeline_months: int,
        risk_profile: str = "moderate",
    ) -> List[Dict[str, Any]]:
        """
        Get investment recommendations.
        
        Args:
            amount: Amount to invest
            timeline_months: Investment horizon
            risk_profile: Risk tolerance
        
        Returns:
            List of recommended options
        """
        if amount <= 0 or timeline_months <= 0:
            raise ValueError("Invalid input parameters")

        recommendations = []

        risk_mapping = {
            "conservative": ["FD", "PPF"],
            "moderate": ["PPF", "ELSS", "MF"],
            "aggressive": ["ELSS", "MF", "STOCKS"],
        }

        selected_options = risk_mapping.get(risk_profile, ["PPF", "MF"])

        for option_key in selected_options:
            if option_key not in self.INVESTMENT_OPTIONS:
                continue

            option = self.INVESTMENT_OPTIONS[option_key]
            min_rate, max_rate = option["rate_range"]
            avg_rate = (min_rate + max_rate) / 2

            years = timeline_months / 12
            returns = self.calculate_returns(amount, avg_rate, years)

            recommendations.append(
                {
                    "option": option_key,
                    "name": option["name"],
                    "expected_rate": avg_rate,
                    "expected_return": returns["interest_earned"],
                    "final_amount": returns["final_amount"],
                    "liquidity": option["liquidity"],
                    "risk": option["risk"],
                    "timeline_months": timeline_months,
                }
            )

        return sorted(
            recommendations,
            key=lambda x: x["expected_return"],
            reverse=True,
        )

    def calculate_sip(
        self,
        monthly_amount: float,
        rate: float,
        months: int,
    ) -> Dict[str, float]:
        """
        Calculate SIP (Systematic Investment Plan) returns.
        
        Args:
            monthly_amount: Monthly investment
            rate: Annual interest rate (percentage)
            months: Investment period
        
        Returns:
            Dict with total invested, interest, final amount
        """
        if monthly_amount <= 0 or rate < 0 or months <= 0:
            raise ValueError("Invalid input parameters")

        r = rate / 100 / 12
        total_invested = monthly_amount * months

        if r == 0:
            final_amount = total_invested
        else:
            final_amount = monthly_amount * (((1 + r) ** months - 1) / r) * (
                1 + r
            )

        interest_earned = final_amount - total_invested

        return {
            "monthly_amount": monthly_amount,
            "total_invested": total_invested,
            "interest_earned": interest_earned,
            "final_amount": final_amount,
            "months": months,
        }

    def calculate_retirement_corpus(
        self,
        current_age: int,
        retirement_age: int,
        monthly_expense: float,
        inflation_rate: float = 5.0,
        return_rate: float = 8.0,
    ) -> Dict[str, float]:
        """
        Calculate retirement corpus needed.
        
        Args:
            current_age: Current age
            retirement_age: Retirement age
            monthly_expense: Current monthly expense
            inflation_rate: Expected inflation
            return_rate: Expected return rate
        
        Returns:
            Dict with corpus needed and monthly SIP
        """
        years_to_retirement = retirement_age - current_age
        years_in_retirement = 25

        inflation_factor = (1 + inflation_rate / 100) ** years_to_retirement
        future_monthly_expense = monthly_expense * inflation_factor
        future_annual_expense = future_monthly_expense * 12

        real_return = (
            (1 + return_rate / 100) / (1 + inflation_rate / 100) - 1
        ) * 100

        corpus_needed = future_annual_expense * (
            (1 - (1 + real_return / 100) ** (-years_in_retirement))
            / (real_return / 100)
        )

        monthly_sip = self._calculate_required_sip(
            corpus_needed, return_rate, years_to_retirement * 12
        )

        return {
            "corpus_needed": corpus_needed,
            "monthly_sip_required": monthly_sip,
            "years_to_retirement": years_to_retirement,
            "future_monthly_expense": future_monthly_expense,
        }

    def _calculate_required_sip(
        self,
        target_amount: float,
        annual_rate: float,
        months: int,
    ) -> float:
        """Calculate required monthly SIP."""
        r = annual_rate / 100 / 12

        if r == 0:
            return target_amount / months

        monthly_sip = target_amount / (
            (((1 + r) ** months - 1) / r) * (1 + r)
        )
        return monthly_sip
