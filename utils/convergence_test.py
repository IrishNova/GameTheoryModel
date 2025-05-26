"""
Convergence test to determine optimal number of simulations
Tests how many iterations are needed for statistically stable results
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time
from simulation.engine import TradeSimulation
from config.countries_data import create_countries


class ConvergenceAnalyzer:
    """Analyze how many iterations needed for stable results"""

    def __init__(self, rounds_per_sim: int = 40):
        self.rounds_per_sim = rounds_per_sim
        self.results = {}

    def run_convergence_test(self, iteration_counts: List[int],
                             scenario: str = 'baseline',
                             us_profile: str = 'moderate',
                             n_bootstrap: int = 10) -> Dict:
        """
        Run simulations with increasing iteration counts

        Args:
            iteration_counts: List of iteration counts to test [50, 100, 500, etc]
            scenario: Test scenario name
            us_profile: US leadership profile
            n_bootstrap: Number of times to repeat each iteration count
        """
        print(f"\n=== Convergence Test: {scenario} ===")
        print(f"Testing iteration counts: {iteration_counts}")
        print(f"Bootstrap samples per count: {n_bootstrap}")

        convergence_data = {
            'iteration_counts': iteration_counts,
            'payoff_means': {},
            'payoff_stds': {},
            'payoff_ci': {},
            'cooperation_means': {},
            'cooperation_stds': {},
            'time_taken': []
        }

        # For each iteration count
        for n_iter in iteration_counts:
            print(f"\nTesting {n_iter} iterations...")

            # Bootstrap to get confidence intervals
            bootstrap_payoffs = {country: [] for country in ['US', 'China', 'Japan', 'Eurozone']}
            bootstrap_cooperation = {country: [] for country in ['US', 'China', 'Japan', 'Eurozone']}

            start_time = time.time()

            for bootstrap in range(n_bootstrap):
                # Run mini Monte Carlo
                payoffs, cooperation = self._run_mini_monte_carlo(n_iter, us_profile)

                for country in bootstrap_payoffs:
                    bootstrap_payoffs[country].append(payoffs[country])
                    bootstrap_cooperation[country].append(cooperation[country])

                print(f"  Bootstrap {bootstrap + 1}/{n_bootstrap} complete")

            elapsed = time.time() - start_time
            convergence_data['time_taken'].append(elapsed)

            # Calculate statistics
            for country in bootstrap_payoffs:
                # Payoff statistics
                payoff_samples = bootstrap_payoffs[country]
                convergence_data['payoff_means'].setdefault(country, []).append(np.mean(payoff_samples))
                convergence_data['payoff_stds'].setdefault(country, []).append(np.std(payoff_samples))

                # 95% confidence interval
                ci_low, ci_high = np.percentile(payoff_samples, [2.5, 97.5])
                convergence_data['payoff_ci'].setdefault(country, []).append((ci_low, ci_high))

                # Cooperation statistics
                coop_samples = bootstrap_cooperation[country]
                convergence_data['cooperation_means'].setdefault(country, []).append(np.mean(coop_samples))
                convergence_data['cooperation_stds'].setdefault(country, []).append(np.std(coop_samples))

        self.results[scenario] = convergence_data
        return convergence_data

    def _run_mini_monte_carlo(self, n_iterations: int, us_profile: str) -> Tuple[Dict, Dict]:
        """Run a small Monte Carlo and return average payoffs and cooperation rates"""
        all_payoffs = []
        all_cooperation = []

        for i in range(n_iterations):
            countries = create_countries()
            sim = TradeSimulation(countries, rounds=self.rounds_per_sim,
                                  us_leadership_profile=us_profile,
                                  enable_leadership_dynamics=True)
            results = sim.run()

            all_payoffs.append(results['average_payoffs'])
            all_cooperation.append(results['cooperation_rates'])

        # Calculate averages
        avg_payoffs = {}
        avg_cooperation = {}

        for country in all_payoffs[0].keys():
            avg_payoffs[country] = np.mean([p[country] for p in all_payoffs])
            avg_cooperation[country] = np.mean([c[country] for c in all_cooperation])

        return avg_payoffs, avg_cooperation

    def analyze_convergence(self, scenario: str = 'baseline') -> Dict:
        """Analyze when results converge to stable values"""
        data = self.results[scenario]
        analysis = {}

        print(f"\n=== Convergence Analysis: {scenario} ===")

        # For each country, find when payoffs stabilize
        for country in data['payoff_means']:
            means = data['payoff_means'][country]
            stds = data['payoff_stds'][country]

            # Calculate coefficient of variation (CV) for each iteration count
            cvs = [std / abs(mean) if mean != 0 else float('inf')
                   for mean, std in zip(means, stds)]

            # Find first iteration count where CV < 0.05 (5% variation)
            convergence_idx = None
            for i, cv in enumerate(cvs):
                if cv < 0.05:
                    convergence_idx = i
                    break

            convergence_n = data['iteration_counts'][convergence_idx] if convergence_idx is not None else None

            analysis[country] = {
                'convergence_n': convergence_n,
                'final_mean': means[-1],
                'final_std': stds[-1],
                'final_cv': cvs[-1]
            }

            print(f"\n{country}:")
            print(f"  Converges at: {convergence_n} iterations" if convergence_n else "  Does not converge")
            print(f"  Final mean: {means[-1]:.3f} ± {stds[-1]:.3f}")
            print(f"  Final CV: {cvs[-1]:.3%}")

        # Overall recommendation
        all_convergence_n = [a['convergence_n'] for a in analysis.values() if a['convergence_n']]
        recommended_n = max(all_convergence_n) if all_convergence_n else data['iteration_counts'][-1]

        print(f"\n=== RECOMMENDATION ===")
        print(f"Minimum iterations for convergence: {recommended_n}")
        print(f"Conservative recommendation: {int(recommended_n * 1.5)}")

        analysis['recommended_iterations'] = recommended_n
        analysis['conservative_iterations'] = int(recommended_n * 1.5)

        return analysis

    def plot_convergence(self):
        """Visualize convergence patterns"""
        n_scenarios = len(self.results)
        fig, axes = plt.subplots(2, n_scenarios, figsize=(6 * n_scenarios, 10))

        if n_scenarios == 1:
            axes = axes.reshape(-1, 1)

        for idx, (scenario, data) in enumerate(self.results.items()):
            # Payoff convergence
            ax1 = axes[0, idx]
            for country in ['US', 'China', 'Japan', 'Eurozone']:
                means = data['payoff_means'][country]
                stds = data['payoff_stds'][country]
                iterations = data['iteration_counts']

                # Plot with error bars
                ax1.errorbar(iterations, means, yerr=stds,
                             label=country, marker='o', capsize=5)

            ax1.set_xlabel('Number of Iterations')
            ax1.set_ylabel('Average Payoff')
            ax1.set_title(f'{scenario}: Payoff Convergence')
            ax1.set_xscale('log')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Cooperation convergence
            ax2 = axes[1, idx]
            for country in ['US', 'China', 'Japan', 'Eurozone']:
                means = data['cooperation_means'][country]
                stds = data['cooperation_stds'][country]

                ax2.errorbar(iterations, means, yerr=stds,
                             label=country, marker='s', capsize=5)

            ax2.set_xlabel('Number of Iterations')
            ax2.set_ylabel('Cooperation Rate')
            ax2.set_title(f'{scenario}: Cooperation Convergence')
            ax2.set_xscale('log')
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confidence_intervals(self):
        """Plot confidence intervals for different iteration counts"""
        fig, axes = plt.subplots(1, len(self.results), figsize=(8 * len(self.results), 6))

        if len(self.results) == 1:
            axes = [axes]

        for idx, (scenario, data) in enumerate(self.results.items()):
            ax = axes[idx]

            countries = ['US', 'China', 'Japan', 'Eurozone']
            x_positions = np.arange(len(countries))
            width = 0.15

            # Plot bars for different iteration counts
            for i, (n_iter, color) in enumerate(zip(data['iteration_counts'],
                                                    ['lightblue', 'blue', 'darkblue', 'black'])):
                means = [data['payoff_means'][c][i] for c in countries]
                ci_ranges = [data['payoff_ci'][c][i] for c in countries]
                errors = [[m - ci[0] for m, ci in zip(means, ci_ranges)],
                          [ci[1] - m for m, ci in zip(means, ci_ranges)]]

                ax.bar(x_positions + i * width, means, width,
                       yerr=errors, capsize=3, label=f'{n_iter} iter',
                       color=color, alpha=0.8)

            ax.set_xlabel('Country')
            ax.set_ylabel('Average Payoff')
            ax.set_title(f'{scenario}: Confidence Intervals by Iteration Count')
            ax.set_xticks(x_positions + width * 1.5)
            ax.set_xticklabels(countries)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('confidence_intervals.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_scenario_comparison(self):
        """Compare convergence between different scenarios"""
        # Test moderate US
        self.run_convergence_test(
            iteration_counts=[50, 100, 250, 500, 1000],
            scenario='moderate_us',
            us_profile='moderate',
            n_bootstrap=5
        )

        # Test volatile US
        self.run_convergence_test(
            iteration_counts=[50, 100, 250, 500, 1000],
            scenario='volatile_us',
            us_profile='volatile_populist',
            n_bootstrap=5
        )

        # Analyze both
        moderate_analysis = self.analyze_convergence('moderate_us')
        volatile_analysis = self.analyze_convergence('volatile_us')

        # Compare requirements
        print("\n=== SCENARIO COMPARISON ===")
        print(f"Moderate US needs: {moderate_analysis['recommended_iterations']} iterations")
        print(f"Volatile US needs: {volatile_analysis['recommended_iterations']} iterations")
        print(
            f"\nOverall recommendation: {max(moderate_analysis['recommended_iterations'], volatile_analysis['recommended_iterations'])}")


# Run the convergence analysis
if __name__ == "__main__":
    analyzer = ConvergenceAnalyzer(rounds_per_sim=40)

    # Quick test with fewer bootstraps for speed
    print("Running convergence analysis...")
    print("This will take a few minutes...\n")

    # Test single scenario first
    analyzer.run_convergence_test(
        iteration_counts=[25, 50, 100, 200, 400],
        scenario='baseline',
        us_profile='moderate',
        n_bootstrap=5  # Reduced for testing
    )

    # Analyze results
    analysis = analyzer.analyze_convergence('baseline')

    # Create plots
    analyzer.plot_convergence()
    analyzer.plot_confidence_intervals()

    # Optional: Run full scenario comparison (takes longer)
    # analyzer.run_scenario_comparison()