"""
Core simulation engine for trade game
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List, Dict, Tuple
from core.country import Country
from dynamics.payoffs import get_payoffs  # Fixed import path
from config.parameters import DEFAULT_ROUNDS, CURRENT_FX_RATES, VOLATILITIES
import pandas as pd


class TradeSimulation:
    """Main simulation engine"""

    def __init__(self, countries: List[Country], rounds: int = DEFAULT_ROUNDS):
        self.countries = countries
        self.rounds = rounds
        self.current_round = 0

        # Initialize exchange rates from YOUR IBKR data
        self.exchange_rates = self._initialize_exchange_rates()

        # Track metrics
        self.round_data = []
        self.global_uncertainty = 0.0

    def _initialize_exchange_rates(self) -> Dict[str, float]:
        """Initialize FX rates from your IBKR data"""
        rates = {}

        # Use your actual current rates
        for pair, rate in CURRENT_FX_RATES.items():
            rates[pair] = rate

            # Also add CNY mapping for CNH pairs
            if 'CNH' in pair:
                cny_pair = pair.replace('CNH', 'CNY')
                rates[cny_pair] = rate

        # Add any missing pairs by calculation
        self._fill_cross_rates(rates)

        return rates

    def _fill_cross_rates(self, rates: Dict[str, float]):
        """Calculate missing cross rates"""
        currencies = set()
        for pair in rates.keys():
            currencies.add(pair[:3])
            currencies.add(pair[3:])

        # Fill in missing cross rates
        for curr1 in currencies:
            for curr2 in currencies:
                if curr1 != curr2:
                    pair = f"{curr1}{curr2}"
                    if pair not in rates:
                        # Try to calculate via USD
                        try:
                            if f"{curr1}USD" in rates and f"{curr2}USD" in rates:
                                rates[pair] = rates[f"{curr1}USD"] / rates[f"{curr2}USD"]
                            elif f"USD{curr1}" in rates and f"USD{curr2}" in rates:
                                rates[pair] = rates[f"USD{curr2}"] / rates[f"USD{curr1}"]
                        except:
                            pass

    def get_exchange_rate(self, country1: Country, country2: Country) -> float:
        """Get exchange rate between two countries"""
        # Handle CNY/CNH mapping - IBKR uses CNH for Chinese Yuan
        curr1 = 'CNH' if country1.currency == 'CNY' else country1.currency
        curr2 = 'CNH' if country2.currency == 'CNY' else country2.currency

        pair = f"{curr1}{curr2}"
        if pair in self.exchange_rates:
            return self.exchange_rates[pair]

        # Try reverse pair
        reverse_pair = f"{curr2}{curr1}"
        if reverse_pair in self.exchange_rates:
            return 1.0 / self.exchange_rates[reverse_pair]

        # Try to calculate via USD
        try:
            if curr1 == 'USD':
                usd_to_curr2 = f"USD{curr2}" if f"USD{curr2}" in self.exchange_rates else f"{curr2}USD"
                if usd_to_curr2 in self.exchange_rates:
                    return self.exchange_rates[usd_to_curr2] if usd_to_curr2.startswith('USD') else 1/self.exchange_rates[usd_to_curr2]
            elif curr2 == 'USD':
                curr1_to_usd = f"{curr1}USD" if f"{curr1}USD" in self.exchange_rates else f"USD{curr1}"
                if curr1_to_usd in self.exchange_rates:
                    return self.exchange_rates[curr1_to_usd] if curr1_to_usd.endswith('USD') else 1/self.exchange_rates[curr1_to_usd]
            else:
                # Cross rate via USD
                curr1_usd = f"{curr1}USD" if f"{curr1}USD" in self.exchange_rates else None
                curr2_usd = f"{curr2}USD" if f"{curr2}USD" in self.exchange_rates else None
                if curr1_usd and curr2_usd:
                    return self.exchange_rates[curr1_usd] / self.exchange_rates[curr2_usd]
        except:
            pass

        # Default to 1.0 if not found
        print(f"Warning: No FX rate for {pair}, using 1.0")
        return 1.0

    def update_exchange_rates(self):
        """Update exchange rates using YOUR volatility data"""
        for pair, rate in self.exchange_rates.items():
            # Get volatility from your IBKR data
            # Handle CNY/CNH mapping
            lookup_pair = pair.replace('CNY', 'CNH') if 'CNY' in pair else pair

            vol_data = VOLATILITIES.get(lookup_pair, {})
            if isinstance(vol_data, dict):
                volatility = vol_data.get('average', 0.03)
            else:
                volatility = 0.03

            # Daily volatility (annualized / sqrt(252))
            daily_vol = volatility / np.sqrt(252)

            # Random walk with your actual volatility
            change = np.random.normal(0, daily_vol)
            self.exchange_rates[pair] = rate * (1 + change)

    def simulate_round(self):
        """Simulate one round of the game"""
        round_results = {
            'round': self.current_round,
            'actions': {},
            'payoffs': {},
            'fx_rates': self.exchange_rates.copy()
        }

        # Each pair of countries interacts
        for i, country1 in enumerate(self.countries):
            for j, country2 in enumerate(self.countries):
                if i >= j:  # Skip self and avoid double-counting
                    continue

                # Get actions
                action1 = country1.choose_action(country2.name)
                action2 = country2.choose_action(country1.name)

                # Get exchange rate
                fx_rate = self.get_exchange_rate(country1, country2)

                # Calculate payoffs using YOUR data
                payoff1, payoff2 = get_payoffs(
                    country1, country2, action1, action2,
                    fx_rate, self.global_uncertainty
                )

                # Update histories
                country1.update_history(country2.name, action2, payoff1)
                country2.update_history(country1.name, action1, payoff2)

                # Store results
                pair_key = f"{country1.name}-{country2.name}"
                round_results['actions'][pair_key] = (action1, action2)
                round_results['payoffs'][pair_key] = (payoff1, payoff2)

        # Update exchange rates for next round
        self.update_exchange_rates()

        # Update global uncertainty based on defection rate
        self._update_global_conditions(round_results)

        self.round_data.append(round_results)
        self.current_round += 1

    def _update_global_conditions(self, round_results):
        """Update global uncertainty based on cooperation levels"""
        total_defections = 0
        total_actions = 0

        for actions in round_results['actions'].values():
            total_actions += 2
            total_defections += sum(actions)

        defection_rate = total_defections / total_actions if total_actions > 0 else 0

        # Uncertainty increases with defection rate
        self.global_uncertainty = 0.7 * self.global_uncertainty + 0.3 * defection_rate

    def run(self):
        """Run the complete simulation"""
        print(f"Starting simulation with {len(self.countries)} countries for {self.rounds} rounds")

        for round_num in range(self.rounds):
            self.simulate_round()

            if round_num % 10 == 0:
                print(f"  Round {round_num}: Global uncertainty = {self.global_uncertainty:.3f}")

        print("Simulation complete!")
        return self.get_results()

    def get_results(self) -> Dict:
        """Compile simulation results"""
        results = {
            'countries': [c.name for c in self.countries],
            'rounds': self.rounds,
            'round_data': self.round_data,
            'final_payoffs': {},
            'cooperation_rates': {},
            'average_payoffs': {}
        }

        # Calculate final statistics
        for country in self.countries:
            total_payoff = sum(sum(payoffs) for payoffs in country.payoff_history.values())
            total_interactions = sum(len(payoffs) for payoffs in country.payoff_history.values())

            results['final_payoffs'][country.name] = total_payoff
            results['average_payoffs'][country.name] = total_payoff / total_interactions if total_interactions > 0 else 0

            # Cooperation rate
            total_cooperations = 0
            total_actions = 0
            for opponent, history in country.history.items():
                total_actions += len(history)
                total_cooperations += len([a for a in history if a == 0])

            results['cooperation_rates'][country.name] = total_cooperations / total_actions if total_actions > 0 else 0

        return results


# Test the simulation
if __name__ == "__main__":
    from config.countries_data import create_countries

    # Create countries with your data
    countries = create_countries()

    # Run a short test simulation
    sim = TradeSimulation(countries, rounds=10)
    results = sim.run()

    print("\n=== Simulation Results ===")
    print(f"Average Payoffs (using YOUR trade weights & FX data):")
    for country, payoff in results['average_payoffs'].items():
        print(f"  {country}: {payoff:.3f}")

    print(f"\nCooperation Rates:")
    for country, rate in results['cooperation_rates'].items():
        print(f"  {country}: {rate:.1%}")