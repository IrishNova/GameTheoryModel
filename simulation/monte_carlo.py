"""
Monte Carlo simulation runner
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import List, Dict
import json
from datetime import datetime
from simulation.engine import TradeSimulation
from config.countries_data import create_countries
from config.parameters import DEFAULT_ROUNDS, DEFAULT_ITERATIONS
import time


class MonteCarloSimulation:
    """Run multiple simulations and aggregate results"""

    def __init__(self, n_iterations: int = DEFAULT_ITERATIONS, n_rounds: int = DEFAULT_ROUNDS):
        self.n_iterations = n_iterations
        self.n_rounds = n_rounds
        self.results = []

    def run(self, save_results: bool = True):
        """Run Monte Carlo simulation"""
        print(f"=== Monte Carlo Simulation ===")
        print(f"Running {self.n_iterations} iterations of {self.n_rounds} rounds each")
        print(f"Using YOUR researched FX and trade data\n")

        start_time = time.time()

        for i in range(self.n_iterations):
            # Create fresh countries for each iteration
            countries = create_countries()

            # Run simulation
            sim = TradeSimulation(countries, self.n_rounds)
            results = sim.run()

            # Store results
            self.results.append(results)

            # Progress update
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (self.n_iterations - i - 1) / rate
                print(f"Progress: {i + 1}/{self.n_iterations} iterations "
                      f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")

        total_time = time.time() - start_time
        print(f"\nCompleted in {total_time:.1f} seconds")

        # Analyze results
        analysis = self.analyze_results()

        # Save if requested
        if save_results:
            self.save_results(analysis)

        return analysis

    def analyze_results(self) -> Dict:
        """Analyze Monte Carlo results"""
        print("\n=== Analyzing Results ===")

        # Aggregate metrics across all iterations
        all_payoffs = {country: [] for country in self.results[0]['countries']}
        all_cooperation = {country: [] for country in self.results[0]['countries']}
        global_cooperation = []

        for result in self.results:
            # Collect average payoffs
            for country, payoff in result['average_payoffs'].items():
                all_payoffs[country].append(payoff)

            # Collect cooperation rates
            for country, rate in result['cooperation_rates'].items():
                all_cooperation[country].append(rate)

            # Global cooperation
            global_rate = np.mean(list(result['cooperation_rates'].values()))
            global_cooperation.append(global_rate)

        # Calculate statistics
        analysis = {
            'metadata': {
                'iterations': self.n_iterations,
                'rounds_per_iteration': self.n_rounds,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'payoffs': {},
            'cooperation': {},
            'global_metrics': {}
        }

        # Country-level statistics
        for country in all_payoffs:
            analysis['payoffs'][country] = {
                'mean': np.mean(all_payoffs[country]),
                'std': np.std(all_payoffs[country]),
                'min': np.min(all_payoffs[country]),
                'max': np.max(all_payoffs[country]),
                'median': np.median(all_payoffs[country])
            }

            analysis['cooperation'][country] = {
                'mean': np.mean(all_cooperation[country]),
                'std': np.std(all_cooperation[country]),
                'min': np.min(all_cooperation[country]),
                'max': np.max(all_cooperation[country])
            }

        # Global metrics
        analysis['global_metrics'] = {
            'mean_cooperation': np.mean(global_cooperation),
            'std_cooperation': np.std(global_cooperation),
            'payoff_variance': np.mean([np.std(all_payoffs[c]) for c in all_payoffs])
        }

        return analysis

    def print_summary(self, analysis: Dict):
        """Print summary of results"""
        print("\n=== MONTE CARLO RESULTS SUMMARY ===")
        print(f"Based on {self.n_iterations} simulations of {self.n_rounds} rounds each")

        print("\nAverage Payoffs (Mean ± Std):")
        payoff_data = []
        for country, stats in analysis['payoffs'].items():
            payoff_data.append({
                'Country': country,
                'Mean': stats['mean'],
                'Std': stats['std']
            })

        # Sort by mean payoff
        payoff_df = pd.DataFrame(payoff_data).sort_values('Mean', ascending=False)
        for _, row in payoff_df.iterrows():
            print(f"  {row['Country']:12} {row['Mean']:6.2f} ± {row['Std']:5.2f}")

        print("\nCooperation Rates:")
        coop_data = []
        for country, stats in analysis['cooperation'].items():
            coop_data.append({
                'Country': country,
                'Mean': stats['mean'],
                'Std': stats['std']
            })

        coop_df = pd.DataFrame(coop_data).sort_values('Mean', ascending=False)
        for _, row in coop_df.iterrows():
            print(f"  {row['Country']:12} {row['Mean']:5.1%} ± {row['Std']:5.1%}")

        print(f"\nGlobal Cooperation Rate: {analysis['global_metrics']['mean_cooperation']:.1%}")

    def save_results(self, analysis: Dict):
        """Save results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save summary
        summary_path = f"/Users/ryanmoloney/Desktop/DePaul 24/GameTheoryModel/data/historical/monte_carlo_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"\nResults saved to: {summary_path}")

        # Also save raw results for detailed analysis
        raw_path = f"/Users/ryanmoloney/Desktop/DePaul 24/GameTheoryModel/data/historical/monte_carlo_raw_{timestamp}.json"

        # Convert results to serializable format
        raw_data = []
        for i, result in enumerate(self.results):
            raw_data.append({
                'iteration': i,
                'average_payoffs': result['average_payoffs'],
                'cooperation_rates': result['cooperation_rates'],
                'final_payoffs': result['final_payoffs']
            })

        with open(raw_path, 'w') as f:
            json.dump(raw_data, f, indent=2)

        print(f"Raw data saved to: {raw_path}")


# Run Monte Carlo simulation
if __name__ == "__main__":
    # Quick test with fewer iterations
    mc = MonteCarloSimulation(n_iterations=20, n_rounds=40)
    analysis = mc.run()
    mc.print_summary(analysis)