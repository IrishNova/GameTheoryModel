"""
Country class for trade simulation
Updated to support volatile leadership and strategy overrides
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
            strategy: Trading strategy ('tit_for_tat', 'aggressive', 'reactive_volatile', etc.)
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

        # Volatile leadership support
        self.volatile_overrides = {}  # {opponent_name: {'strategy': str, 'cooperation': float}}

    def choose_action(self, opponent_name: str) -> int:
        """
        Choose action against opponent
        0 = cooperate (low tariffs)
        1 = defect (high tariffs)
        """
        # Check for volatile override for this specific opponent
        if opponent_name in self.volatile_overrides:
            override = self.volatile_overrides[opponent_name]
            temp_strategy = override.get('strategy', self.strategy)
            temp_cooperation = override.get('cooperation', self.cooperation_tendency)

            # Use override for this decision
            if opponent_name not in self.history or not self.history[opponent_name]:
                # First round with override
                return 0 if np.random.random() < temp_cooperation else 1

            # Get action using override strategy
            base_action = self._get_strategy_action(opponent_name, temp_strategy)

            # Apply override cooperation tendency
            if base_action == 1:  # Strategy suggests defection
                if np.random.random() < temp_cooperation * 0.3:  # 30% max override
                    return 0  # Cooperate anyway
            else:  # Strategy suggests cooperation
                if np.random.random() < (1 - temp_cooperation) * 0.3:  # 30% max override
                    return 1  # Defect anyway

            return base_action

        # Normal behavior (no override)
        # First round - use cooperation tendency
        if opponent_name not in self.history or not self.history[opponent_name]:
            return 0 if np.random.random() < self.cooperation_tendency else 1

        # Get base action from strategy
        base_action = self._get_strategy_action(opponent_name, self.strategy)

        # Apply cooperation tendency modifier
        if base_action == 1:  # Strategy suggests defection
            if np.random.random() < self.cooperation_tendency * 0.3:  # 30% max override
                return 0  # Cooperate anyway
        else:  # Strategy suggests cooperation
            if np.random.random() < (1 - self.cooperation_tendency) * 0.3:  # 30% max override
                return 1  # Defect anyway

        return base_action

    def _get_strategy_action(self, opponent_name: str, strategy: str = None) -> int:
        """Get action based on strategy"""
        # Use provided strategy or default strategy
        strat = strategy or self.strategy

        # Get opponent's last action
        last_opponent_action = self.history[opponent_name][-1]

        if strat == 'tit_for_tat':
            return last_opponent_action  # Copy opponent's last move

        elif strat == 'aggressive':
            return 1  # Always defect

        elif strat == 'cooperative':
            return 0  # Always cooperate

        elif strat == 'generous_tit_for_tat':
            if last_opponent_action == 1 and np.random.random() < 0.1:
                return 0  # 10% chance to forgive
            return last_opponent_action

        elif strat == 'random':
            return np.random.choice([0, 1])

        elif strat == 'reactive_volatile':
            # Base volatile behavior (when no specific override)
            # More unpredictable than regular tit-for-tat
            if last_opponent_action == 1:
                # Opponent defected - high chance of retaliation
                return 1 if np.random.random() < 0.85 else 0
            else:
                # Opponent cooperated - but still somewhat unpredictable
                return 0 if np.random.random() < 0.7 else 1

        else:
            raise ValueError(f"Unknown strategy: {strat}")

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

    def get_recent_payoffs(self, opponent_name: str, n_periods: int = 3) -> List[float]:
        """Get recent payoffs against specific opponent"""
        if opponent_name in self.payoff_history:
            return self.payoff_history[opponent_name][-n_periods:]
        return []

    def get_cooperation_rate(self, opponent_name: Optional[str] = None) -> float:
        """Get cooperation rate with specific opponent or overall"""
        if opponent_name:
            if opponent_name in self.history:
                actions = self.history[opponent_name]
                return 1 - (sum(actions) / len(actions))  # 0 = cooperate, 1 = defect
            return 0.5  # No history
        else:
            # Overall cooperation rate
            total_cooperations = 0
            total_actions = 0
            for actions in self.history.values():
                total_actions += len(actions)
                total_cooperations += len([a for a in actions if a == 0])
            return total_cooperations / total_actions if total_actions > 0 else 0.5

    def __repr__(self):
        return f"Country({self.name}, {self.currency}, {self.strategy})"


# Test the updated class
if __name__ == "__main__":
    # Create test countries
    us = Country("US", "USD", strategy="reactive_volatile",
                 cooperation_tendency=0.3, reserve_status=0.59)
    china = Country("China", "CNY", strategy="aggressive",
                    cooperation_tendency=0.3, currency_regime="managed")
    canada = Country("Canada", "CAD", strategy="generous_tit_for_tat",
                     cooperation_tendency=0.8)

    print(f"Created: {us}")
    print(f"Created: {china}")
    print(f"Created: {canada}")

    # Test normal behavior
    print(f"\nUS first action vs China: {us.choose_action('China')}")
    print(f"China first action vs US: {china.choose_action('US')}")

    # Test with volatile override
    print("\nAdding grudge against China...")
    us.volatile_overrides['China'] = {
        'strategy': 'aggressive',
        'cooperation': 0.1
    }

    # Simulate some history
    us.history['China'] = [1, 1, 0]  # China defected twice, then cooperated
    china.history['US'] = [0, 1, 1]  # US cooperated, then defected twice

    print(f"US action vs China (with grudge): {us.choose_action('China')}")
    print(f"US action vs Canada (no grudge): {us.choose_action('Canada')}")

    # Test negotiation override
    print("\nAdding negotiation with China...")
    us.volatile_overrides['China'] = {
        'strategy': 'generous_tit_for_tat',
        'cooperation': 0.8
    }

    print(f"US action vs China (negotiating): {us.choose_action('China')}")