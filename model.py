#!/usr/bin/env python3

"""

@rìan

Game Theory Model for ECO 525 final paper

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any


class Country:
    def __init__(self, name: str, currency: str, strategy: str = 'tit_for_tat',
                 cooperation_tendency: float = 0.5, currency_regime: str = 'floating'):
        self.name = name
        self.currency = currency
        self.strategy = strategy
        self.history = {}  # Dict to store history against each opponent
        self.payoff_history = {}  # Dict to store payoffs against each opponent
        self.cooperation_tendency = cooperation_tendency  # 0.0 (uncooperative) to 1.0 (cooperative)
        self.currency_regime = currency_regime  # 'floating', 'fixed', or 'managed'

    def choose_action(self, opponent_name: str) -> int:
        """Choose tariff level (0 = low, 1 = high) based on strategy and cooperation tendency."""
        if opponent_name not in self.history or not self.history[opponent_name]:
            # First round decision influenced by cooperation tendency
            return 0 if np.random.random() < self.cooperation_tendency else 1

        # Base decision on strategy
        base_action = self._get_strategy_action(opponent_name)

        # Apply cooperation tendency modifier
        # Higher tendency increases chance of cooperation (low tariffs)
        if base_action == 1:  # If strategy suggests defection
            # Chance to override and cooperate instead, based on cooperation tendency
            if np.random.random() < self.cooperation_tendency:
                return 0  # Choose cooperation despite strategy
        else:  # If strategy suggests cooperation
            # Chance to override and defect instead, based on non-cooperation tendency
            if np.random.random() < (1 - self.cooperation_tendency):
                return 1  # Choose defection despite strategy

        return base_action

    def _get_strategy_action(self, opponent_name: str) -> int:
        """Get the action suggested by the basic strategy."""
        if self.strategy == 'tit_for_tat':
            return self.history[opponent_name][-1]  # Copy opponent's last move
        elif self.strategy == 'generous_tit_for_tat':
            if self.history[opponent_name][-1] == 1 and np.random.random() < 0.1:
                return 0  # 10% chance to forgive
            return self.history[opponent_name][-1]
        elif self.strategy == 'random':
            return np.random.choice([0, 1])
        elif self.strategy == 'aggressive':
            return 1  # Always defect (high tariffs)
        elif self.strategy == 'cooperative':
            return 0  # Always cooperate (low tariffs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


def generate_exchange_rates(countries: List[Country], rounds: int,
                            correlations: np.ndarray = None, volatilities: Dict[str, float] = None,
                            initial_rates: Dict[str, float] = None,
                            fixed_band_width: Dict[str, float] = None) -> Dict[str, List[float]]:
    """
    Generate exchange rate paths accounting for different currency regimes.

    Parameters:
    - countries: List of country objects
    - rounds: Number of time periods to simulate
    - correlations: Correlation matrix between currencies
    - volatilities: Dict mapping currency to its volatility
    - initial_rates: Dict mapping currency to its initial exchange rate vs USD
    - fixed_band_width: Dict mapping currency to allowed deviation from peg (for managed currencies)

    Returns:
    - Dict mapping currency pairs to their exchange rate paths
    """
    currencies = [country.currency for country in countries]
    currency_regimes = {country.currency: country.currency_regime for country in countries}

    # Default values if not provided
    if volatilities is None:
        volatilities = {curr: 0.03 for curr in currencies}
        # Adjust volatilities based on currency regime
        for curr, regime in currency_regimes.items():
            if regime == 'fixed':
                volatilities[curr] = 0.001  # Very low volatility for fixed currencies
            elif regime == 'managed':
                volatilities[curr] = 0.015  # Lower volatility for managed currencies

    if initial_rates is None:
        initial_rates = {curr: 1.0 for curr in currencies}

    if fixed_band_width is None:
        fixed_band_width = {curr: 0.02 for curr in currencies}  # Default 2% band

    # Create correlation matrix if not provided (simplified for this example)
    if correlations is None:
        n_currencies = len(currencies)
        correlations = np.eye(n_currencies)  # Start with identity matrix

        # Add realistic correlations based on economic relationships
        currency_idx = {curr: i for i, curr in enumerate(currencies)}

        # Example correlations - would be data-driven in a real model
        # EUR and GBP tend to move together
        if 'EUR' in currency_idx and 'GBP' in currency_idx:
            i, j = currency_idx['EUR'], currency_idx['GBP']
            correlations[i, j] = correlations[j, i] = 0.7

        # CAD and MXN correlated with USD (but differently)
        if 'USD' in currency_idx:
            if 'CAD' in currency_idx:
                i, j = currency_idx['USD'], currency_idx['CAD']
                correlations[i, j] = correlations[j, i] = 0.6
            if 'MXN' in currency_idx:
                i, j = currency_idx['USD'], currency_idx['MXN']
                correlations[i, j] = correlations[j, i] = 0.5

    # Initialize exchange rates dictionary
    exchange_rates = {}

    # Generate paths for each currency pair (vs USD)
    for i, base_curr in enumerate(currencies):
        if base_curr == "USD":
            continue  # Skip USD vs USD

        path = [initial_rates[base_curr]]
        regime = currency_regimes[base_curr]

        if regime == 'fixed':
            # Fixed exchange rate with tiny random noise for numerical stability
            path = [initial_rates[base_curr] * (1 + np.random.normal(0, 0.0005)) for _ in range(rounds)]

        elif regime == 'managed':
            # Managed float: allow movement within a band, with intervention
            band_width = fixed_band_width[base_curr]
            for _ in range(rounds - 1):
                # Generate candidate new rate
                curr_vol = volatilities[base_curr]
                shock = np.random.normal(0, curr_vol)
                candidate_rate = path[-1] * np.exp(shock)

                # Check if outside allowed band
                upper_limit = initial_rates[base_curr] * (1 + band_width)
                lower_limit = initial_rates[base_curr] * (1 - band_width)

                if candidate_rate > upper_limit:
                    # Central bank intervention - push back toward center
                    new_rate = upper_limit * (0.9 + 0.1 * np.random.random())
                elif candidate_rate < lower_limit:
                    # Central bank intervention - push back toward center
                    new_rate = lower_limit * (1.0 + 0.1 * np.random.random())
                else:
                    new_rate = candidate_rate

                path.append(new_rate)

        else:  # Floating exchange rate
            # Create Cholesky decomposition for correlated random walks
            n_currencies = len(currencies)
            cholesky_matrix = np.linalg.cholesky(correlations)

            for _ in range(rounds - 1):
                # Generate correlated random shocks
                random_shocks = np.random.normal(0, 1, n_currencies)
                correlated_shocks = cholesky_matrix @ random_shocks
                curr_shock = correlated_shocks[i]

                # Apply shock to generate next rate (log-normal random walk)
                curr_vol = volatilities[base_curr]
                new_rate = path[-1] * np.exp(curr_vol * curr_shock)
                path.append(new_rate)

        exchange_rates[f"USD/{base_curr}"] = path

    # Generate cross-rates for all currency pairs
    for i, base_curr in enumerate(currencies):
        for j, quote_curr in enumerate(currencies):
            if base_curr == quote_curr or (base_curr == "USD" and quote_curr == "USD"):
                continue

            if base_curr == "USD":
                # Direct USD/quote rate
                exchange_rates[f"{base_curr}/{quote_curr}"] = exchange_rates[f"USD/{quote_curr}"]
            elif quote_curr == "USD":
                # Inverse of USD/base for base/USD
                exchange_rates[f"{base_curr}/{quote_curr}"] = [1 / rate for rate in exchange_rates[f"USD/{base_curr}"]]
            else:
                # Cross rate: base/quote = (base/USD) * (USD/quote)
                usd_base = [1 / rate for rate in exchange_rates[f"USD/{base_curr}"]]
                usd_quote = exchange_rates[f"USD/{quote_curr}"]
                exchange_rates[f"{base_curr}/{quote_curr}"] = [b * q for b, q in zip(usd_base, usd_quote)]

    return exchange_rates


def get_payoffs(country1: Country, country2: Country, action1: int, action2: int,
                exchange_rate: float, trade_volumes: Dict[Tuple[str, str], float]) -> Tuple[float, float]:
    """
    Calculate payoffs for both countries based on their actions and exchange rate.

    - Prisoner's Dilemma payoff structure modified by:
      1. Exchange rate effects (different by currency regime)
      2. Relative trade importance (trade volume)
    """
    # Base payoff matrix (row player, column player)
    payoff_matrix = np.array([
        [[3, 3], [0, 5]],  # Row player cooperates
        [[5, 0], [1, 1]]  # Row player defects
    ])

    base_payoff1 = payoff_matrix[action1, action2, 0]
    base_payoff2 = payoff_matrix[action1, action2, 1]

    # Trade volume adjustment
    trade_weight1 = trade_volumes.get((country1.name, country2.name), 1.0)
    trade_weight2 = trade_volumes.get((country2.name, country1.name), 1.0)

    # Exchange rate effect varies based on currency regime
    fx_impact1 = 1.0
    fx_impact2 = 1.0

    # For fixed currencies, exchange rate has minimal impact
    if country1.currency_regime != 'fixed' and country2.currency_regime != 'fixed':
        # Both floating or managed currencies - normal exchange rate impact
        if country1.currency == "USD" or country2.currency == "USD":
            # USD case
            if country1.currency == "USD":
                # Higher exchange rate means stronger USD
                if exchange_rate > 1:  # Stronger USD
                    fx_impact1 = 1 - 0.1 * (exchange_rate - 1)
                    fx_impact2 = 1 + 0.05 * (exchange_rate - 1)
                else:  # Weaker USD
                    fx_impact1 = 1 + 0.05 * (1 - exchange_rate)
                    fx_impact2 = 1 - 0.1 * (1 - exchange_rate)
            else:  # country2.currency == "USD"
                # Lower exchange rate means stronger USD
                if exchange_rate < 1:  # Stronger USD
                    fx_impact1 = 1 + 0.05 * (1 - exchange_rate)
                    fx_impact2 = 1 - 0.1 * (1 - exchange_rate)
                else:  # Weaker USD
                    fx_impact1 = 1 - 0.1 * (exchange_rate - 1)
                    fx_impact2 = 1 + 0.05 * (exchange_rate - 1)
        else:
            # Non-USD currency pair
            if exchange_rate > 1:  # country1's currency is stronger
                fx_impact1 = 1 - 0.1 * (exchange_rate - 1)
                fx_impact2 = 1 + 0.05 * (exchange_rate - 1)
            else:  # country2's currency is stronger
                fx_impact1 = 1 + 0.05 * (1 - exchange_rate)
                fx_impact2 = 1 - 0.1 * (1 - exchange_rate)
    elif country1.currency_regime == 'fixed' and country2.currency_regime != 'fixed':
        # Country1 has fixed currency - less affected by exchange rate
        fx_impact1 = 1.0
        if exchange_rate > 1:
            fx_impact2 = 1 + 0.03 * (exchange_rate - 1)  # Reduced impact
        else:
            fx_impact2 = 1 - 0.06 * (1 - exchange_rate)  # Increased impact (fixed advantage)
    elif country1.currency_regime != 'fixed' and country2.currency_regime == 'fixed':
        # Country2 has fixed currency - less affected by exchange rate
        if exchange_rate > 1:
            fx_impact1 = 1 - 0.06 * (exchange_rate - 1)  # Increased impact (disadvantage)
        else:
            fx_impact1 = 1 + 0.03 * (1 - exchange_rate)  # Reduced impact
        fx_impact2 = 1.0
    else:
        # Both fixed - minimal exchange rate impact
        fx_impact1 = 1.0
        fx_impact2 = 1.0

    # Calculate final payoffs with all modifiers
    payoff1 = base_payoff1 * fx_impact1 * trade_weight1
    payoff2 = base_payoff2 * fx_impact2 * trade_weight2

    return payoff1, payoff2


def run_multi_country_simulation(countries: List[Country], rounds: int,
                                 exchange_rates: Dict[str, List[float]],
                                 trade_volumes: Dict[Tuple[str, str], float]) -> Dict[str, Any]:
    """Run a single simulation with multiple countries."""
    # Initialize history and payoffs
    for country in countries:
        country.history = {other.name: [] for other in countries if other.name != country.name}
        country.payoff_history = {other.name: [] for other in countries if other.name != country.name}

    # Track global stats
    global_stats = {
        "free_trade_ratio": [],  # % of country pairs with mutual cooperation
        "trade_war_ratio": [],  # % of country pairs with mutual defection
        "mixed_strategy_ratio": []  # % of country pairs with mixed strategies
    }

    # Run simulation for each round
    for round_idx in range(rounds):
        # Track stats for this round
        cooperation_count = 0
        defection_count = 0
        mixed_count = 0
        total_pairs = 0

        # For each pair of countries
        for i, country1 in enumerate(countries):
            for j, country2 in enumerate(countries):
                if i >= j:  # Avoid duplicate pairs and self-interaction
                    continue

                total_pairs += 1

                # Get exchange rate for this currency pair
                exchange_rate_key = f"{country1.currency}/{country2.currency}"
                if exchange_rate_key not in exchange_rates:
                    exchange_rate_key = f"{country2.currency}/{country1.currency}"
                    exchange_rate = 1 / exchange_rates[exchange_rate_key][round_idx]
                else:
                    exchange_rate = exchange_rates[exchange_rate_key][round_idx]

                # Countries choose actions based on opponent's history and cooperation tendency
                action1 = country1.choose_action(country2.name)
                action2 = country2.choose_action(country1.name)

                # Update cooperation/defection counts
                if action1 == 0 and action2 == 0:
                    cooperation_count += 1
                elif action1 == 1 and action2 == 1:
                    defection_count += 1
                else:
                    mixed_count += 1

                # Calculate payoffs
                payoff1, payoff2 = get_payoffs(country1, country2, action1, action2,
                                               exchange_rate, trade_volumes)

                # Update histories
                country1.history[country2.name].append(action2)
                country2.history[country1.name].append(action1)
                country1.payoff_history[country2.name].append(payoff1)
                country2.payoff_history[country1.name].append(payoff2)

        # Update global stats
        global_stats["free_trade_ratio"].append(cooperation_count / total_pairs)
        global_stats["trade_war_ratio"].append(defection_count / total_pairs)
        global_stats["mixed_strategy_ratio"].append(mixed_count / total_pairs)

    # Calculate summary statistics
    results = {
        "country_stats": {},
        "global_free_trade_ratio": np.mean(global_stats["free_trade_ratio"]),
        "global_trade_war_ratio": np.mean(global_stats["trade_war_ratio"]),
        "global_mixed_strategy_ratio": np.mean(global_stats["mixed_strategy_ratio"]),
        # Fix the key error - use the actual keys from global_stats
        "free_trade_trend": global_stats["free_trade_ratio"],
        "trade_war_trend": global_stats["trade_war_ratio"]  # This was the error
    }

    # Compute stats for each country
    for country in countries:
        total_payoff = 0
        cooperation_rates = {}
        bilateral_payoffs = {}
        round_payoffs = [0] * rounds  # Track payoff by round for time series analysis

        for opponent in country.payoff_history:
            payoffs = country.payoff_history[opponent]
            total_payoff += sum(payoffs)

            # Add to round payoffs for time series analysis
            for r, p in enumerate(payoffs):
                round_payoffs[r] += p

            # Calculate cooperation rate against this opponent
            opponent_actions = country.history[opponent]
            cooperation_rates[opponent] = 1 - sum(opponent_actions) / len(opponent_actions)

            # Store bilateral payoffs
            bilateral_payoffs[opponent] = sum(payoffs)

        results["country_stats"][country.name] = {
            "total_payoff": total_payoff,
            "average_payoff": total_payoff / ((len(countries) - 1) * rounds),
            "cooperation_rates": cooperation_rates,
            "bilateral_payoffs": bilateral_payoffs,
            "round_payoffs": round_payoffs  # Add round-by-round payoffs for time series
        }

    return results


def run_monte_carlo_multi_country(iterations: int, rounds: int, countries: List[Country],
                                  trade_volumes: Dict[Tuple[str, str], float] = None) -> pd.DataFrame:
    """Run multiple Monte Carlo simulations with multiple countries."""
    all_results = []

    # Default trade volumes if not provided
    if trade_volumes is None:
        trade_volumes = {}
        for i, country1 in enumerate(countries):
            for j, country2 in enumerate(countries):
                if i != j:
                    # Random trade volume between 0.5 and 2.0
                    trade_volumes[(country1.name, country2.name)] = np.random.uniform(0.5, 2.0)

    for iteration in range(iterations):
        # Generate exchange rate paths for this iteration
        exchange_rates = generate_exchange_rates(countries, rounds)

        # Run simulation
        sim_result = run_multi_country_simulation(countries, rounds, exchange_rates, trade_volumes)
        sim_result["iteration"] = iteration

        # Flatten the results for easier analysis
        flat_result = {
            "iteration": iteration,
            "global_free_trade_ratio": sim_result["global_free_trade_ratio"],
            "global_trade_war_ratio": sim_result["global_trade_war_ratio"],
            "global_mixed_strategy_ratio": sim_result["global_mixed_strategy_ratio"]
        }

        # Add country-specific stats
        for country in countries:
            country_stats = sim_result["country_stats"][country.name]
            flat_result[f"{country.name}_total_payoff"] = country_stats["total_payoff"]
            flat_result[f"{country.name}_avg_payoff"] = country_stats["average_payoff"]

            # Add bilateral stats
            for opponent in country_stats["cooperation_rates"]:
                flat_result[f"{country.name}_coop_with_{opponent}"] = country_stats["cooperation_rates"][opponent]
                flat_result[f"{country.name}_payoff_vs_{opponent}"] = country_stats["bilateral_payoffs"][opponent]

        all_results.append(flat_result)

    return pd.DataFrame(all_results)


def analyze_multi_country_results(results_df: pd.DataFrame, countries: List[Country]):
    """Analyze and visualize simulation results for multiple countries."""
    # Summary statistics
    print(f"Monte Carlo Results Summary (n={len(results_df)} iterations):")
    print("\nGlobal Statistics:")
    print(f"Average Free Trade Ratio: {results_df['global_free_trade_ratio'].mean():.2%}")
    print(f"Average Trade War Ratio: {results_df['global_trade_war_ratio'].mean():.2%}")
    print(f"Average Mixed Strategy Ratio: {results_df['global_mixed_strategy_ratio'].mean():.2%}")

    print("\nCountry Performance:")
    for country in countries:
        print(
            f"\n{country.name} ({country.strategy} strategy, cooperation tendency: {country.cooperation_tendency:.2f}):")
        print(f"  Currency: {country.currency} ({country.currency_regime})")
        print(f"  Average Payoff: {results_df[f'{country.name}_avg_payoff'].mean():.2f}")

        # Bilateral stats
        for opponent in [c.name for c in countries if c.name != country.name]:
            coop_col = f"{country.name}_coop_with_{opponent}"
            payoff_col = f"{country.name}_payoff_vs_{opponent}"
            if coop_col in results_df.columns and payoff_col in results_df.columns:
                print(f"  vs {opponent}: Avg Cooperation {results_df[coop_col].mean():.2%}, "
                      f"Avg Payoff {results_df[payoff_col].mean():.2f}")

    # Correlation analysis of key metrics
    payoff_cols = [f"{country.name}_avg_payoff" for country in countries]
    corr_cols = payoff_cols + ['global_free_trade_ratio', 'global_trade_war_ratio']

    print("\nCorrelation Matrix of Key Metrics:")
    correlations = results_df[corr_cols].corr()
    print(correlations)

    # Visualizations
    plt.figure(figsize=(16, 12))

    # 1. Payoff distribution by country
    plt.subplot(2, 2, 1)
    payoff_data = [results_df[f"{country.name}_avg_payoff"] for country in countries]
    plt.boxplot(payoff_data, labels=[f"{c.name}\n{c.currency}\n({c.strategy[:3]})" for c in countries])
    plt.title('Payoff Distribution by Country and Strategy')
    plt.ylabel('Average Payoff')
    plt.xticks(rotation=45)

    # 2. Correlation heatmap
    plt.subplot(2, 2, 2)
    sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Key Metrics')

    # 3. Cooperation tendency vs payoff
    plt.subplot(2, 2, 3)
    x = [country.cooperation_tendency for country in countries]
    y = [results_df[f"{country.name}_avg_payoff"].mean() for country in countries]
    plt.scatter(x, y)
    for i, country in enumerate(countries):
        plt.annotate(country.name, (x[i], y[i]))
    plt.xlabel('Cooperation Tendency')
    plt.ylabel('Average Payoff')
    plt.title('Cooperation Tendency vs Average Payoff')

    # 4. US bilateral cooperation rates
    plt.subplot(2, 2, 4)
    coop_cols = [col for col in results_df.columns if col.startswith('US_coop_with_')]
    if coop_cols:
        coop_data = results_df[coop_cols].mean().sort_values()
        coop_data.plot(kind='bar')
        plt.title('US Average Cooperation Rates with Partners')
        plt.ylabel('Cooperation Rate')
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    return correlations


# Example usage
if __name__ == "__main__":
    # Define countries with their currencies, strategies, and cooperation tendencies
    countries = [
        Country("US", "USD", "tit_for_tat", cooperation_tendency=0.6, currency_regime='floating'),
        Country("Eurozone", "EUR", "tit_for_tat", cooperation_tendency=0.7, currency_regime='floating'),
        Country("China", "CNY", "aggressive", cooperation_tendency=0.3, currency_regime='managed'),
        # Low cooperation, managed currency
        Country("Japan", "JPY", "tit_for_tat", cooperation_tendency=0.6, currency_regime='floating'),
        Country("Canada", "CAD", "generous_tit_for_tat", cooperation_tendency=0.8, currency_regime='floating'),
        # High cooperation
        Country("Mexico", "MXN", "tit_for_tat", cooperation_tendency=0.7, currency_regime='floating'),
        Country("UK", "GBP", "tit_for_tat", cooperation_tendency=0.65, currency_regime='floating'),
        Country("Singapore", "SGD", "cooperative", cooperation_tendency=0.7, currency_regime='managed')
    ]

    # Define trade volumes (relative importance of bilateral trade)
    trade_volumes = {
        ("US", "China"): 2.0,  # High volume
        ("China", "US"): 2.0,
        ("US", "Canada"): 1.8,  # Very integrated economies
        ("Canada", "US"): 1.8,
        ("US", "Mexico"): 1.7,  # USMCA partners
        ("Mexico", "US"): 1.7,
        ("US", "Eurozone"): 1.5,  # Important transatlantic trade
        ("Eurozone", "US"): 1.5,
        ("US", "Japan"): 1.2,
        ("Japan", "US"): 1.2,
        ("US", "UK"): 1.1,
        ("UK", "US"): 1.1,
        ("US", "Singapore"): 0.8,
        ("Singapore", "US"): 0.8,
        ("China", "Eurozone"): 1.3,
        ("Eurozone", "China"): 1.3,
    }

    # Fill in missing trade volumes with default value
    for i, country1 in enumerate(countries):
        for j, country2 in enumerate(countries):
            if i != j and (country1.name, country2.name) not in trade_volumes:
                trade_volumes[(country1.name, country2.name)] = 1.0

    # Set simulation parameters
    iterations = 1000  # Number of Monte Carlo iterations
    rounds = 40  # Rounds in each game

    # Run Monte Carlo simulation
    results = run_monte_carlo_multi_country(iterations, rounds, countries, trade_volumes)

    # Analyze results
    analyze_multi_country_results(results, countries)

    # Test different US strategies against fixed partner strategies
    print("\nTesting US strategies against partners with fixed behaviors")
    us_strategies = ["tit_for_tat", "generous_tit_for_tat", "cooperative", "aggressive", "random"]

    # Store results for comparison
    strategy_results = []

    for strategy in us_strategies:
        print(f"\n---- US using {strategy} strategy ----")
        # Update US strategy
        countries[0].strategy = strategy

        # Run simulation with fewer iterations for quicker results
        results = run_monte_carlo_multi_country(100, rounds, countries, trade_volumes)

        # Store key metric
        us_payoff = results['US_avg_payoff'].mean()
        global_free_trade = results['global_free_trade_ratio'].mean()
        strategy_results.append((strategy, us_payoff, global_free_trade))

        # Analyze key metrics only
        print(f"US Average Payoff: {us_payoff:.2f}")
        print(f"Global Free Trade Ratio: {global_free_trade:.2%}")
        print(f"Global Trade War Ratio: {results['global_trade_war_ratio'].mean():.2%}")

    # Compare US strategies
    print("\nUS Strategy Comparison:")
    for strategy, payoff, free_trade in sorted(strategy_results, key=lambda x: x[1], reverse=True):
        print(f"{strategy:20} - Payoff: {payoff:.2f}, Free Trade: {free_trade:.2%}")

    # Additional scenario: Testing different cooperation tendencies for China
    print("\nTesting different China cooperation tendencies")

    tendency_results = []
    original_tendency = countries[2].cooperation_tendency

    for tendency in [0.1, 0.3, 0.5, 0.7, 0.9]:
        countries[2].cooperation_tendency = tendency
        countries[0].strategy = "tit_for_tat"  # Reset US strategy

        print(f"\n---- China cooperation tendency: {tendency:.1f} ----")
        results = run_monte_carlo_multi_country(100, rounds, countries, trade_volumes)

        us_payoff = results['US_avg_payoff'].mean()
        china_payoff = results['China_avg_payoff'].mean()
        global_free_trade = results['global_free_trade_ratio'].mean()

        tendency_results.append((tendency, us_payoff, china_payoff, global_free_trade))

        print(f"US Average Payoff: {us_payoff:.2f}")
        print(f"China Average Payoff: {china_payoff:.2f}")
        print(f"Global Free Trade Ratio: {global_free_trade:.2%}")

    # Reset China's cooperation tendency
    countries[2].cooperation_tendency = original_tendency

    # Plot the impact of China's cooperation tendency
    plt.figure(figsize=(12, 8))
    tendencies = [r[0] for r in tendency_results]
    us_payoffs = [r[1] for r in tendency_results]
    china_payoffs = [r[2] for r in tendency_results]
    free_trade = [r[3] for r in tendency_results]

    plt.subplot(2, 1, 1)
    plt.plot(tendencies, us_payoffs, 'b-o', label='US Payoff')
    plt.plot(tendencies, china_payoffs, 'r-s', label='China Payoff')
    plt.xlabel('China Cooperation Tendency')
    plt.ylabel('Average Payoff')
    plt.title('Impact of China\'s Cooperation Tendency on Payoffs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(2, 1, 2)
    plt.plot(tendencies, free_trade, 'g-^')
    plt.xlabel('China Cooperation Tendency')
    plt.ylabel('Global Free Trade Ratio')
    plt.title('Impact of China\'s Cooperation Tendency on Global Free Trade')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Scenario: Currency crisis simulation
    print("\nSimulating Currency Crisis Scenario")


    # Function to simulate currency crisis
    def run_currency_crisis_simulation(countries, rounds, trade_volumes, crisis_currency,
                                       crisis_round, crisis_magnitude):
        """
        Simulate a currency crisis scenario.

        Parameters:
        - countries: List of country objects
        - rounds: Number of rounds
        - trade_volumes: Dict of trade volumes
        - crisis_currency: Currency that experiences the crisis (e.g., "MXN")
        - crisis_round: Round when the crisis occurs
        - crisis_magnitude: Magnitude of devaluation (e.g., 0.3 for 30% devaluation)
        """
        # Generate normal exchange rates
        exchange_rates = generate_exchange_rates(countries, rounds)

        # Modify exchange rates to simulate crisis
        for pair, rates in exchange_rates.items():
            if crisis_currency in pair:
                base, quote = pair.split('/')
                if base == crisis_currency:
                    # Currency is base: reduce value (e.g., MXN/USD decreases)
                    for i in range(crisis_round, rounds):
                        rates[i] = rates[i] * (1 - crisis_magnitude)
                elif quote == crisis_currency:
                    # Currency is quote: increase value (e.g., USD/MXN increases)
                    for i in range(crisis_round, rounds):
                        rates[i] = rates[i] * (1 + crisis_magnitude)

        # Run simulation with crisis exchange rates
        sim_result = run_multi_country_simulation(countries, rounds, exchange_rates, trade_volumes)

        return sim_result


    # Reset country strategies
    for country in countries:
        country.strategy = "tit_for_tat"

    # Run baseline simulation (no crisis)
    baseline_result = run_multi_country_simulation(
        countries, rounds,
        generate_exchange_rates(countries, rounds),
        trade_volumes
    )

    # Run MXN crisis simulation
    mxn_crisis_result = run_currency_crisis_simulation(
        countries, rounds, trade_volumes,
        "MXN", crisis_round=15, crisis_magnitude=0.25
    )

    # Compare results
    print("\nMexican Peso Crisis Impact:")
    print("                    Baseline    Crisis    Change")
    print("-----------------------------------------------")
    for country in countries:
        baseline_payoff = baseline_result["country_stats"][country.name]["average_payoff"]
        crisis_payoff = mxn_crisis_result["country_stats"][country.name]["average_payoff"]
        change = crisis_payoff - baseline_payoff
        change_pct = change / baseline_payoff * 100 if baseline_payoff > 0 else 0

        print(f"{country.name:12} {baseline_payoff:8.2f}  {crisis_payoff:8.2f}  {change:+6.2f} ({change_pct:+5.1f}%)")

    print("\nGlobal Free Trade Ratio:")
    print(f"  Baseline: {baseline_result['global_free_trade_ratio']:.2%}")
    print(f"  Crisis:   {mxn_crisis_result['global_free_trade_ratio']:.2%}")

    # Scenario: Trade war simulation
    print("\nSimulating US-China Trade War Scenario")


    def run_trade_war_simulation(countries, rounds, trade_volumes,
                                 warring_parties, war_start_round):
        """Simulate a trade war between specific countries."""
        # Create deep copies of countries to avoid modifying originals
        countries_copy = []
        for country in countries:
            # Create new country object with same attributes
            new_country = Country(
                country.name, country.currency, country.strategy,
                country.cooperation_tendency, country.currency_regime
            )
            countries_copy.append(new_country)

            # If country is part of the trade war, change strategy to aggressive
            if country.name in warring_parties and war_start_round > 0:
                new_country.original_strategy = country.strategy
                new_country.war_start_round = war_start_round

        # Generate exchange rates
        exchange_rates = generate_exchange_rates(countries_copy, rounds)

        # Custom simulation that can change strategies mid-simulation
        # Initialize history and payoffs
        for country in countries_copy:
            country.history = {other.name: [] for other in countries_copy if other.name != country.name}
            country.payoff_history = {other.name: [] for other in countries_copy if other.name != country.name}

        # Track global stats
        global_stats = {
            "free_trade_ratio": [],
            "trade_war_ratio": [],
            "mixed_strategy_ratio": []
        }

        # Run simulation for each round
        for round_idx in range(rounds):
            # Change strategies if trade war starts
            if round_idx == war_start_round:
                for country in countries_copy:
                    if country.name in warring_parties:
                        country.strategy = "aggressive"  # Switch to aggressive during trade war

            # Track stats for this round
            cooperation_count = 0
            defection_count = 0
            mixed_count = 0
            total_pairs = 0

            # For each pair of countries
            for i, country1 in enumerate(countries_copy):
                for j, country2 in enumerate(countries_copy):
                    if i >= j:  # Avoid duplicate pairs and self-interaction
                        continue

                    total_pairs += 1

                    # Get exchange rate for this currency pair
                    exchange_rate_key = f"{country1.currency}/{country2.currency}"
                    if exchange_rate_key not in exchange_rates:
                        exchange_rate_key = f"{country2.currency}/{country1.currency}"
                        exchange_rate = 1 / exchange_rates[exchange_rate_key][round_idx]
                    else:
                        exchange_rate = exchange_rates[exchange_rate_key][round_idx]

                    # If both countries are in trade war with each other, force defection
                    if country1.name in warring_parties and country2.name in warring_parties and round_idx >= war_start_round:
                        action1 = 1  # Defect
                        action2 = 1  # Defect
                    else:
                        # Normal action selection
                        action1 = country1.choose_action(country2.name)
                        action2 = country2.choose_action(country1.name)

                    # Update cooperation/defection counts
                    if action1 == 0 and action2 == 0:
                        cooperation_count += 1
                    elif action1 == 1 and action2 == 1:
                        defection_count += 1
                    else:
                        mixed_count += 1

                    # Calculate payoffs
                    payoff1, payoff2 = get_payoffs(country1, country2, action1, action2,
                                                   exchange_rate, trade_volumes)

                    # Update histories
                    country1.history[country2.name].append(action2)
                    country2.history[country1.name].append(action1)
                    country1.payoff_history[country2.name].append(payoff1)
                    country2.payoff_history[country1.name].append(payoff2)

            # Update global stats
            global_stats["free_trade_ratio"].append(cooperation_count / total_pairs)
            global_stats["trade_war_ratio"].append(defection_count / total_pairs)
            global_stats["mixed_strategy_ratio"].append(mixed_count / total_pairs)

        # Calculate summary statistics (same as in run_multi_country_simulation)
        results = {
            "country_stats": {},
            "global_free_trade_ratio": np.mean(global_stats["free_trade_ratio"]),
            "global_trade_war_ratio": np.mean(global_stats["trade_war_ratio"]),
            "global_mixed_strategy_ratio": np.mean(global_stats["mixed_strategy_ratio"]),
            "free_trade_trend": global_stats["free_trade_ratio"],
            "trade_war_trend": global_stats["trade_war_ratio"],
            "round_data": {
                "free_trade_ratio": global_stats["free_trade_ratio"],
                "trade_war_ratio": global_stats["trade_war_ratio"]
            }
        }

        # Compute stats for each country
        for country in countries_copy:
            total_payoff = 0
            cooperation_rates = {}
            bilateral_payoffs = {}
            round_payoffs = [0] * rounds  # Track payoff by round

            for opponent in country.payoff_history:
                payoffs = country.payoff_history[opponent]
                total_payoff += sum(payoffs)

                # Add to round payoffs
                for r, p in enumerate(payoffs):
                    round_payoffs[r] += p

                # Calculate cooperation rate against this opponent
                opponent_actions = country.history[opponent]
                cooperation_rates[opponent] = 1 - sum(opponent_actions) / len(opponent_actions)

                # Store bilateral payoffs
                bilateral_payoffs[opponent] = sum(payoffs)

            results["country_stats"][country.name] = {
                "total_payoff": total_payoff,
                "average_payoff": total_payoff / ((len(countries_copy) - 1) * rounds),
                "cooperation_rates": cooperation_rates,
                "bilateral_payoffs": bilateral_payoffs,
                "round_payoffs": round_payoffs
            }

        return results


    # Run baseline (no trade war)
    baseline = run_trade_war_simulation(countries, rounds, trade_volumes, [], -1)

    # Run US-China trade war starting at round 10
    trade_war = run_trade_war_simulation(countries, rounds, trade_volumes, ["US", "China"], 10)

    # Plotting the effects of the trade war over time
    plt.figure(figsize=(14, 10))

    # Plot global free trade ratio trend
    plt.subplot(2, 2, 1)
    plt.plot(baseline["round_data"]["free_trade_ratio"], 'g-', label='Baseline')
    plt.plot(trade_war["round_data"]["free_trade_ratio"], 'r-', label='Trade War')
    plt.axvline(x=10, color='k', linestyle='--', label='War Start')
    plt.xlabel('Round')
    plt.ylabel('Global Free Trade Ratio')
    plt.title('Impact of US-China Trade War on Global Free Trade')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot US and China payoffs over time
    plt.subplot(2, 2, 2)
    plt.plot(baseline["country_stats"]["US"]["round_payoffs"], 'b-', label='US Baseline')
    plt.plot(trade_war["country_stats"]["US"]["round_payoffs"], 'b--', label='US Trade War')
    plt.plot(baseline["country_stats"]["China"]["round_payoffs"], 'r-', label='China Baseline')
    plt.plot(trade_war["country_stats"]["China"]["round_payoffs"], 'r--', label='China Trade War')
    plt.axvline(x=10, color='k', linestyle='--', label='War Start')
    plt.xlabel('Round')
    plt.ylabel('Round Payoff')
    plt.title('US and China Payoffs During Trade War')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot third-party country payoffs
    plt.subplot(2, 2, 3)
    for country in [c for c in countries if c.name not in ["US", "China"]]:
        baseline_payoffs = baseline["country_stats"][country.name]["round_payoffs"]
        war_payoffs = trade_war["country_stats"][country.name]["round_payoffs"]
        plt.plot(war_payoffs, label=country.name)

    plt.axvline(x=10, color='k', linestyle='--', label='War Start')
    plt.xlabel('Round')
    plt.ylabel('Round Payoff')
    plt.title('Third-Party Country Payoffs During US-China Trade War')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot net effect of trade war by country
    plt.subplot(2, 2, 4)
    countries_names = [country.name for country in countries]
    baseline_avg = [baseline["country_stats"][c]["average_payoff"] for c in countries_names]
    war_avg = [trade_war["country_stats"][c]["average_payoff"] for c in countries_names]
    diff = [w - b for w, b in zip(war_avg, baseline_avg)]

    plt.bar(countries_names, diff)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.xlabel('Country')
    plt.ylabel('Change in Average Payoff')
    plt.title('Net Effect of US-China Trade War by Country')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    plt.tight_layout()
    plt.show()

    # Scenario: Testing the effect of currency regime on trade outcomes
    print("\nSimulating Impact of Currency Regime on Trade Outcomes")

    # Create a copy of countries with different currency regimes
    floating_countries = []
    fixed_countries = []

    for country in countries:
        # Create floating version
        float_country = Country(
            f"{country.name}", country.currency, country.strategy,
            country.cooperation_tendency, "floating"
        )
        floating_countries.append(float_country)

        # Create fixed version
        fixed_country = Country(
            f"{country.name}", country.currency, country.strategy,
            country.cooperation_tendency, "fixed"
        )
        fixed_countries.append(fixed_country)

    # Run simulations with each set
    float_results = run_monte_carlo_multi_country(100, rounds, floating_countries, trade_volumes)
    fixed_results = run_monte_carlo_multi_country(100, rounds, fixed_countries, trade_volumes)

    # Compare results
    print("\nCurrency Regime Impact on Payoffs and Cooperation:")
    print("\nFloating vs Fixed Currency Regimes")
    print("                   Floating     Fixed     Difference")
    print("------------------------------------------------------")

    for country in countries:
        float_payoff = float_results[f"{country.name}_avg_payoff"].mean()
        fixed_payoff = fixed_results[f"{country.name}_avg_payoff"].mean()
        diff = fixed_payoff - float_payoff

        print(f"{country.name:12} {float_payoff:10.2f} {fixed_payoff:10.2f} {diff:+10.2f}")

    print("\nGlobal Trade Statistics:")
    print(f"Free Trade Ratio (Floating): {float_results['global_free_trade_ratio'].mean():.2%}")
    print(f"Free Trade Ratio (Fixed):    {fixed_results['global_free_trade_ratio'].mean():.2%}")
    print(f"Trade War Ratio (Floating):  {float_results['global_trade_war_ratio'].mean():.2%}")
    print(f"Trade War Ratio (Fixed):     {fixed_results['global_trade_war_ratio'].mean():.2%}")

    # Visualization of currency regime impact
    plt.figure(figsize=(12, 6))

    # Plot payoffs by currency regime
    countries_names = [country.name for country in countries]
    float_payoffs = [float_results[f"{c}_avg_payoff"].mean() for c in countries_names]
    fixed_payoffs = [fixed_results[f"{c}_avg_payoff"].mean() for c in countries_names]

    x = np.arange(len(countries_names))
    width = 0.35

    plt.bar(x - width / 2, float_payoffs, width, label='Floating')
    plt.bar(x + width / 2, fixed_payoffs, width, label='Fixed')

    plt.xlabel('Country')
    plt.ylabel('Average Payoff')
    plt.title('Impact of Currency Regime on Country Payoffs')
    plt.xticks(x, countries_names, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    plt.tight_layout()
    plt.show()