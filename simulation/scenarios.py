"""
Scenario testing for trade simulation
Test trade wars, currency crises, and strategy changes
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from simulation.engine import TradeSimulation
from simulation.monte_carlo import MonteCarloSimulation
from config.countries_data import create_countries
import json
from datetime import datetime


class ScenarioTester:
    """Test different scenarios and compare outcomes"""

    def __init__(self):
        self.results = {}

    def run_baseline(self, iterations: int = 20) -> Dict:
        """Run baseline scenario with normal parameters"""
        print("\n=== BASELINE SCENARIO ===")
        print("All countries use default strategies")

        mc = MonteCarloSimulation(n_iterations=iterations, n_rounds=40)
        analysis = mc.run(save_results=False)
        mc.print_summary(analysis)

        self.results['baseline'] = analysis
        return analysis

    def run_trade_war(self, warring_countries: List[str], iterations: int = 20) -> Dict:
        """Run trade war scenario where specific countries turn aggressive"""
        print(f"\n=== TRADE WAR SCENARIO ===")
        print(f"Countries turning aggressive: {', '.join(warring_countries)}")

        war_results = []

        for i in range(iterations):
            # Create countries
            countries = create_countries()

            # Make warring countries aggressive
            for country in countries:
                if country.name in warring_countries:
                    country.strategy = 'aggressive'
                    country.cooperation_tendency *= 0.5  # Halve cooperation

            # Run simulation
            sim = TradeSimulation(countries, rounds=40)
            results = sim.run()
            war_results.append(results)

        # Analyze
        analysis = self._analyze_results(war_results)
        self.results[f'trade_war_{"-".join(warring_countries)}'] = analysis

        # Print summary
        self._print_scenario_summary(analysis)

        return analysis

    def run_currency_crisis(self, crisis_country: str, crisis_magnitude: float = 0.3,
                            crisis_round: int = 20, iterations: int = 20) -> Dict:
        """Run currency crisis scenario"""
        print(f"\n=== CURRENCY CRISIS SCENARIO ===")
        print(f"{crisis_country} currency devalues by {crisis_magnitude:.0%} at round {crisis_round}")

        crisis_results = []

        for i in range(iterations):
            countries = create_countries()
            sim = TradeSimulation(countries, rounds=40)

            # Inject crisis
            sim.crisis_settings = {
                'country': crisis_country,
                'magnitude': crisis_magnitude,
                'round': crisis_round
            }

            # Override simulate_round to inject crisis
            original_simulate = sim.simulate_round

            def simulate_with_crisis():
                original_simulate()

                # Check if it's crisis time
                if sim.current_round == crisis_round + 1:  # After the round increments
                    # Devalue currency
                    crisis_curr = next(c.currency for c in countries if c.name == crisis_country)
                    for pair, rate in sim.exchange_rates.items():
                        if crisis_curr in pair:
                            if pair.startswith(crisis_curr):
                                sim.exchange_rates[pair] *= (1 - crisis_magnitude)
                            elif pair.endswith(crisis_curr):
                                sim.exchange_rates[pair] *= (1 + crisis_magnitude)

                    # Reduce confidence
                    for country in countries:
                        if country.name == crisis_country:
                            country.currency_confidence *= 0.7

                    print(f"  💥 Currency crisis triggered for {crisis_country}!")

            sim.simulate_round = simulate_with_crisis
            results = sim.run()
            crisis_results.append(results)

        # Analyze
        analysis = self._analyze_results(crisis_results)
        self.results[f'crisis_{crisis_country}'] = analysis

        self._print_scenario_summary(analysis)

        return analysis

    def run_cooperation_test(self, country_name: str, cooperation_levels: List[float],
                             iterations: int = 10) -> Dict:
        """Test different cooperation levels for a specific country"""
        print(f"\n=== COOPERATION SENSITIVITY TEST ===")
        print(f"Testing {country_name} with cooperation levels: {cooperation_levels}")

        coop_results = {}

        for coop_level in cooperation_levels:
            level_results = []

            for i in range(iterations):
                countries = create_countries()

                # Adjust cooperation level
                for country in countries:
                    if country.name == country_name:
                        country.cooperation_tendency = coop_level

                sim = TradeSimulation(countries, rounds=40)
                results = sim.run()
                level_results.append(results)

            analysis = self._analyze_results(level_results)
            coop_results[coop_level] = analysis

            print(f"\nCooperation {coop_level:.1f}:")
            print(f"  {country_name} payoff: {analysis['payoffs'][country_name]['mean']:.2f}")
            print(f"  Global cooperation: {analysis['global_metrics']['mean_cooperation']:.1%}")

        self.results[f'cooperation_test_{country_name}'] = coop_results
        return coop_results

    def compare_scenarios(self):
        """Compare all run scenarios"""
        if len(self.results) < 2:
            print("Run at least 2 scenarios to compare")
            return

        print("\n=== SCENARIO COMPARISON ===")

        # Create comparison dataframe
        comparison_data = []

        for scenario_name, analysis in self.results.items():
            if isinstance(analysis, dict) and 'payoffs' in analysis:
                for country, stats in analysis['payoffs'].items():
                    comparison_data.append({
                        'Scenario': scenario_name,
                        'Country': country,
                        'Payoff': stats['mean'],
                        'Cooperation': analysis['cooperation'][country]['mean']
                    })

        df = pd.DataFrame(comparison_data)

        # Print payoff changes
        print("\nPayoff Changes vs Baseline:")
        if 'baseline' in self.results:
            baseline_payoffs = {c: s['mean'] for c, s in self.results['baseline']['payoffs'].items()}

            for scenario in self.results:
                if scenario != 'baseline':
                    print(f"\n{scenario}:")
                    scenario_df = df[df['Scenario'] == scenario]
                    for _, row in scenario_df.iterrows():
                        baseline = baseline_payoffs.get(row['Country'], 0)
                        change = row['Payoff'] - baseline
                        pct_change = (change / abs(baseline) * 100) if baseline != 0 else 0
                        print(f"  {row['Country']:12} {change:+6.2f} ({pct_change:+5.1f}%)")

    def save_results(self):
        """Save all scenario results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f"/Users/ryanmoloney/Desktop/DePaul 24/GameTheoryModel/data/historical/scenarios_{timestamp}.json"

        # Convert results to serializable format
        save_data = {}
        for scenario, data in self.results.items():
            if isinstance(data, dict):
                save_data[scenario] = data

        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"\nScenario results saved to: {path}")

    def _analyze_results(self, results_list: List[Dict]) -> Dict:
        """Analyze a list of simulation results"""
        # Similar to MonteCarloSimulation.analyze_results but simpler
        all_payoffs = {country: [] for country in results_list[0]['countries']}
        all_cooperation = {country: [] for country in results_list[0]['countries']}

        for result in results_list:
            for country, payoff in result['average_payoffs'].items():
                all_payoffs[country].append(payoff)
            for country, rate in result['cooperation_rates'].items():
                all_cooperation[country].append(rate)

        analysis = {
            'payoffs': {},
            'cooperation': {},
            'global_metrics': {}
        }

        for country in all_payoffs:
            analysis['payoffs'][country] = {
                'mean': np.mean(all_payoffs[country]),
                'std': np.std(all_payoffs[country])
            }
            analysis['cooperation'][country] = {
                'mean': np.mean(all_cooperation[country]),
                'std': np.std(all_cooperation[country])
            }

        global_coop = [np.mean(list(r['cooperation_rates'].values())) for r in results_list]
        analysis['global_metrics']['mean_cooperation'] = np.mean(global_coop)

        return analysis

    def _print_scenario_summary(self, analysis: Dict):
        """Print summary for a scenario"""
        print("\nResults:")

        # Sort by payoff
        payoffs = [(c, s['mean']) for c, s in analysis['payoffs'].items()]
        payoffs.sort(key=lambda x: x[1], reverse=True)

        print("Average Payoffs:")
        for country, payoff in payoffs:
            coop = analysis['cooperation'][country]['mean']
            print(f"  {country:12} {payoff:6.2f} (cooperation: {coop:.1%})")

        print(f"\nGlobal Cooperation: {analysis['global_metrics']['mean_cooperation']:.1%}")


# Test scenarios
if __name__ == "__main__":
    tester = ScenarioTester()

    # 1. Baseline
    tester.run_baseline(iterations=10)

    # 2. US-China Trade War
    tester.run_trade_war(['US', 'China'], iterations=10)

    # 3. Mexican Peso Crisis
    tester.run_currency_crisis('Mexico', crisis_magnitude=0.25, crisis_round=20, iterations=10)

    # 4. China cooperation sensitivity
    tester.run_cooperation_test('China', [0.1, 0.3, 0.5, 0.7], iterations=5)

    # 5. Compare all scenarios
    tester.compare_scenarios()

    # Save results
    tester.save_results()