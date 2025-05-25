"""
Country configurations using researched data
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.country import Country
from config.parameters import RESERVE_SHARES


def create_countries():
    """
    Create all countries with parameters based on research
    Reserve status comes from YOUR IMF data
    """
    countries = [
        Country(
            name="US",
            currency="USD",
            strategy="tit_for_tat",
            cooperation_tendency=0.6,
            currency_regime="floating",
            reserve_status=RESERVE_SHARES.get('USD', 0.59),  # From your data: 59%
            currency_confidence=1.0,
            usd_exposure=0.0  # US has no USD exposure (it IS the USD)
        ),

        Country(
            name="Eurozone",
            currency="EUR",
            strategy="tit_for_tat",
            cooperation_tendency=0.7,
            currency_regime="floating",
            reserve_status=RESERVE_SHARES.get('EUR', 0.205),  # From your data: 20.5%
            currency_confidence=1.0,
            usd_exposure=0.4
        ),

        Country(
            name="China",
            currency="CNY",
            strategy="aggressive",
            cooperation_tendency=0.3,  # Low cooperation
            currency_regime="managed",  # China manages the yuan
            reserve_status=RESERVE_SHARES.get('CNY', 0.028),  # From your data: 2.8%
            currency_confidence=1.0,
            usd_exposure=0.7  # High USD reserves
        ),

        Country(
            name="Japan",
            currency="JPY",
            strategy="tit_for_tat",
            cooperation_tendency=0.6,
            currency_regime="floating",
            reserve_status=RESERVE_SHARES.get('JPY', 0.055),  # From your data: 5.5%
            currency_confidence=1.0,
            usd_exposure=0.6
        ),

        Country(
            name="Canada",
            currency="CAD",
            strategy="generous_tit_for_tat",
            cooperation_tendency=0.8,  # High cooperation
            currency_regime="floating",
            reserve_status=RESERVE_SHARES.get('CAD', 0.021),  # From your data: 2.1%
            currency_confidence=1.0,
            usd_exposure=0.8  # Very integrated with US
        ),

        Country(
            name="Mexico",
            currency="MXN",
            strategy="tit_for_tat",
            cooperation_tendency=0.7,
            currency_regime="floating",
            reserve_status=0.01,  # Not a reserve currency
            currency_confidence=1.0,
            usd_exposure=0.9  # Extremely high USD dependence
        ),

        Country(
            name="UK",
            currency="GBP",
            strategy="tit_for_tat",
            cooperation_tendency=0.65,
            currency_regime="floating",
            reserve_status=RESERVE_SHARES.get('GBP', 0.049),  # From your data: 4.9%
            currency_confidence=1.0,
            usd_exposure=0.5
        ),

        Country(
            name="Singapore",
            currency="SGD",
            strategy="cooperative",
            cooperation_tendency=0.7,
            currency_regime="managed",  # Singapore manages against basket
            reserve_status=0.02,  # Minor reserve currency
            currency_confidence=1.0,
            usd_exposure=0.7
        )
    ]

    return countries


def get_country_by_name(countries, name):
    """Helper to find country by name"""
    for country in countries:
        if country.name == name:
            return country
    return None


def get_country_by_currency(countries, currency):
    """Helper to find country by currency"""
    for country in countries:
        if country.currency == currency:
            return country
    return None


# Test the setup
if __name__ == "__main__":
    countries = create_countries()

    print("=== Countries Created with Your Researched Data ===\n")

    for country in countries:
        print(f"{country.name}:")
        print(f"  Currency: {country.currency}")
        print(f"  Strategy: {country.strategy}")
        print(f"  Reserve Status: {country.reserve_status:.1%}")
        print(f"  Currency Regime: {country.currency_regime}")
        print(f"  Cooperation Level: {country.cooperation_tendency}")
        print()

    # Verify reserve shares match your data
    print("=== Reserve Currency Total ===")
    total_reserve = sum(c.reserve_status for c in countries
                        if c.currency in ['USD', 'EUR', 'JPY', 'GBP', 'CNY', 'CAD'])
    print(f"Total tracked reserve share: {total_reserve:.1%}")