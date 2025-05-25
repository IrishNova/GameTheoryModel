"""
Payoff calculations using real trade weights and FX data
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.parameters import PAYOFF_MATRIX, TRADE_WEIGHTS
from core.country import Country
from typing import Tuple


def get_trade_weight(country1: Country, country2: Country) -> float:
    """
    Get normalized trade weight between two countries from YOUR data
    Returns value between 0.5 and 2.0
    """
    # Try both orderings since trade is bidirectional
    key1 = f"{country1.currency}-{country2.currency}"
    key2 = f"{country2.currency}-{country1.currency}"

    weight = TRADE_WEIGHTS.get(key1) or TRADE_WEIGHTS.get(key2)

    if weight is None:
        # Default weight if not found
        print(f"Warning: No trade weight found for {key1}, using default 1.0")
        return 1.0

    return weight


def calculate_base_payoffs(action1: int, action2: int) -> Tuple[float, float]:
    """Get base payoffs from game theory matrix"""
    return PAYOFF_MATRIX[(action1, action2)]


def apply_fx_effects(
        base_payoff1: float,
        base_payoff2: float,
        country1: Country,
        country2: Country,
        exchange_rate: float
) -> Tuple[float, float]:
    """
    Apply exchange rate effects based on currency regimes
    Exchange rate is country1_currency / country2_currency
    """
    fx_impact1 = 1.0
    fx_impact2 = 1.0

    # Fixed currencies are less affected by FX changes
    if country1.currency_regime == 'fixed' and country2.currency_regime == 'fixed':
        # Both fixed - minimal FX impact
        return base_payoff1, base_payoff2

    # Calculate FX impacts based on rate deviation from 1.0
    rate_deviation = abs(exchange_rate - 1.0)

    if country1.currency_regime != 'fixed' and country2.currency_regime != 'fixed':
        # Both floating/managed - normal FX impact
        if exchange_rate > 1.0:
            # Country1's currency is stronger
            fx_impact1 = 1 - 0.1 * rate_deviation  # Slight disadvantage
            fx_impact2 = 1 + 0.05 * rate_deviation  # Slight advantage
        else:
            # Country2's currency is stronger
            fx_impact1 = 1 + 0.05 * rate_deviation
            fx_impact2 = 1 - 0.1 * rate_deviation

    elif country1.currency_regime == 'fixed':
        # Country1 fixed, country2 floating/managed
        fx_impact1 = 1.0  # No impact on fixed currency
        fx_impact2 = 1 + 0.1 * rate_deviation if exchange_rate > 1 else 1 - 0.1 * rate_deviation

    elif country2.currency_regime == 'fixed':
        # Country2 fixed, country1 floating/managed
        fx_impact1 = 1 - 0.1 * rate_deviation if exchange_rate > 1 else 1 + 0.1 * rate_deviation
        fx_impact2 = 1.0

    return base_payoff1 * fx_impact1, base_payoff2 * fx_impact2


def apply_reserve_benefits(
        payoff: float,
        country: Country,
        global_uncertainty: float = 0.0
) -> float:
    """
    Apply reserve currency benefits
    Uses actual reserve status from YOUR data
    """
    # Base funding advantage (lower borrowing costs)
    funding_benefit = 1 + 0.05 * country.reserve_status

    # Transaction cost reduction (more important during uncertainty)
    transaction_benefit = 1 + 0.03 * country.reserve_status * (1 + global_uncertainty)

    # Global demand effect (people want to hold reserve currencies)
    demand_benefit = 1 + 0.08 * country.reserve_status * country.currency_confidence

    return payoff * funding_benefit * transaction_benefit * demand_benefit


def get_payoffs(
        country1: Country,
        country2: Country,
        action1: int,
        action2: int,
        exchange_rate: float,
        global_uncertainty: float = 0.0
) -> Tuple[float, float]:
    """
    Calculate final payoffs incorporating all effects
    Uses YOUR trade weights and reserve status data
    """
    # 1. Get base payoffs from game theory
    base1, base2 = calculate_base_payoffs(action1, action2)

    # 2. Apply trade weight from YOUR data
    trade_weight1 = get_trade_weight(country1, country2)
    trade_weight2 = get_trade_weight(country2, country1)

    weighted1 = base1 * trade_weight1
    weighted2 = base2 * trade_weight2

    # 3. Apply FX effects
    fx_adjusted1, fx_adjusted2 = apply_fx_effects(
        weighted1, weighted2, country1, country2, exchange_rate
    )

    # 4. Apply reserve currency benefits using YOUR data
    final1 = apply_reserve_benefits(fx_adjusted1, country1, global_uncertainty)
    final2 = apply_reserve_benefits(fx_adjusted2, country2, global_uncertainty)

    return final1, final2


# Test the system
if __name__ == "__main__":
    from config.countries_data import create_countries

    countries = create_countries()
    us = next(c for c in countries if c.name == "US")
    china = next(c for c in countries if c.name == "China")

    print("=== Testing Payoff System with Your Data ===\n")

    # Test different scenarios
    scenarios = [
        ("Both cooperate", 0, 0, 1.0),
        ("US defects", 1, 0, 1.0),
        ("China defects", 0, 1, 1.0),
        ("Both defect", 1, 1, 1.0),
        ("Both cooperate, USD strong", 0, 0, 1.2),
    ]

    for scenario, act1, act2, fx_rate in scenarios:
        pay1, pay2 = get_payoffs(us, china, act1, act2, fx_rate)
        print(f"{scenario}:")
        print(f"  US payoff: {pay1:.3f}")
        print(f"  China payoff: {pay2:.3f}")
        print(f"  Trade weight used: {get_trade_weight(us, china):.2f}")
        print()