"""
Country class for trade simulation
"""
import numpy as np
from typing import Dict, List, Optional


class Country:
    """Represents a country in the trade simulation"""

    def __init__(
            self,
            name: str,
            currency: str,
            strategy: str = 'tit_for_tat',
            cooperation_tendency: float = 0.5,
            currency_regime: str = 'floating',
            reserve_status: float = 0.0,
            currency_confidence: float = 1.0,
            usd_exposure: float = 0.5
    ):
        """
        Initialize a country

        Args:
            name: Country name (e.g., 'US', 'China')
            currency: Currency code (e.g., 'USD', 'CNY')
            strategy: Trading strategy ('tit_for_tat', 'aggressive', etc.)
            cooperation_tendency: 0.0 (never cooperate) to 1.0 (always cooperate)
            currency_regime: 'floating', 'managed', or 'fixed'
            reserve_status: 0.0 to 1.0 (from your RESERVE_SHARES data)
            currency_confidence: Current confidence level (1.0 = normal)
            usd_exposure: Exposure to USD (for currency crisis modeling)
        """
        self.name = name
        self.currency = currency
        self.strategy = strategy
        self.cooperation_tendency = cooperation_tendency
        self.currency_regime = currency_regime
        self.reserve_status = reserve_status
        self.currency_confidence = currency_confidence
        self.usd_exposure = usd_exposure

        # History tracking
        self.history = {}  # {opponent_name: [actions]}
        self.payoff_history = {}  # {opponent_name: [payoffs]}

    def choose_action(self, opponent_name: str) -> int:
        """
        Choose action against opponent
        0 = cooperate (low tariffs)
        1 = defect (high tariffs)
        """
        # First round - use cooperation tendency
        if opponent_name not in self.history or not self.history[opponent_name]:
            return 0 if np.random.random() < self.cooperation_tendency else 1

        # Get base action from strategy
        base_action = self._get_strategy_action(opponent_name)

        # Apply cooperation tendency modifier
        if base_action == 1:  # Strategy suggests defection
            if np.random.random() < self.cooperation_tendency * 0.3:  # 30% max override
                return 0  # Cooperate anyway
        else:  # Strategy suggests cooperation
            if np.random.random() < (1 - self.cooperation_tendency) * 0.3:  # 30% max override
                return 1  # Defect anyway

        return base_action

    def _get_strategy_action(self, opponent_name: str) -> int:
        """Get action based on strategy"""
        last_opponent_action = self.history[opponent_name][-1]

        if self.strategy == 'tit_for_tat':
            return last_opponent_action  # Copy opponent's last move

        elif self.strategy == 'aggressive':
            return 1  # Always defect

        elif self.strategy == 'cooperative':
            return 0  # Always cooperate

        elif self.strategy == 'generous_tit_for_tat':
            if last_opponent_action == 1 and np.random.random() < 0.1:
                return 0  # 10% chance to forgive
            return last_opponent_action

        elif self.strategy == 'random':
            return np.random.choice([0, 1])

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def update_history(self, opponent_name: str, opponent_action: int, my_payoff: float):
        """Update history after a round"""
        if opponent_name not in self.history:
            self.history[opponent_name] = []
            self.payoff_history[opponent_name] = []

        self.history[opponent_name].append(opponent_action)
        self.payoff_history[opponent_name].append(my_payoff)

    def get_average_payoff(self, opponent_name: Optional[str] = None) -> float:
        """Get average payoff against specific opponent or overall"""
        if opponent_name:
            if opponent_name in self.payoff_history:
                return np.mean(self.payoff_history[opponent_name])
            return 0.0
        else:
            # Overall average
            all_payoffs = []
            for payoffs in self.payoff_history.values():
                all_payoffs.extend(payoffs)
            return np.mean(all_payoffs) if all_payoffs else 0.0

    def __repr__(self):
        return f"Country({self.name}, {self.currency}, {self.strategy})"


# Test the class
if __name__ == "__main__":
    # Create test countries
    us = Country("US", "USD", strategy="tit_for_tat",
                 cooperation_tendency=0.6, reserve_status=0.59)
    china = Country("China", "CNY", strategy="aggressive",
                    cooperation_tendency=0.3, currency_regime="managed")

    print(f"Created: {us}")
    print(f"Created: {china}")

    # Test action selection
    print(f"\nUS first action vs China: {us.choose_action('China')}")
    print(f"China first action vs US: {china.choose_action('US')}")