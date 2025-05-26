"""
Leadership dynamics - Model US strategy changes including volatile leadership
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, List, Optional, Tuple
from core.country import Country


class LeadershipProfile:
    """Represents a leadership style/administration"""

    def __init__(self, name: str, strategy: str, cooperation_tendency: float,
                 trade_philosophy: str, duration_months: int = 48,
                 is_volatile: bool = False, volatility_params: Dict = None):
        """
        Args:
            name: Administration name (e.g., "Internationalist", "Protectionist")
            strategy: Base strategy type
            cooperation_tendency: Base cooperation level
            trade_philosophy: "free_trade", "fair_trade", "protectionist", "aggressive"
            duration_months: How long this leadership typically lasts (default 48 = 4 years)
            is_volatile: Whether this leadership has volatile reactive behavior
            volatility_params: Parameters for volatile behavior
        """
        self.name = name
        self.strategy = strategy
        self.cooperation_tendency = cooperation_tendency
        self.trade_philosophy = trade_philosophy
        self.duration_months = duration_months
        self.is_volatile = is_volatile
        self.volatility_params = volatility_params or {}


# Define US leadership profiles based on different approaches
US_LEADERSHIP_PROFILES = {
    'internationalist': LeadershipProfile(
        name="Internationalist",
        strategy="generous_tit_for_tat",
        cooperation_tendency=0.8,
        trade_philosophy="free_trade",
        duration_months=48
    ),
    'moderate': LeadershipProfile(
        name="Moderate",
        strategy="tit_for_tat",
        cooperation_tendency=0.6,
        trade_philosophy="fair_trade",
        duration_months=48
    ),
    'protectionist': LeadershipProfile(
        name="Protectionist",
        strategy="tit_for_tat",
        cooperation_tendency=0.4,
        trade_philosophy="protectionist",
        duration_months=48
    ),
    'aggressive': LeadershipProfile(
        name="Aggressive",
        strategy="aggressive",
        cooperation_tendency=0.2,
        trade_philosophy="aggressive",
        duration_months=48
    ),
    'volatile_populist': LeadershipProfile(
        name="Volatile Populist",
        strategy="reactive_volatile",
        cooperation_tendency=0.3,  # Base level, but highly variable
        trade_philosophy="aggressive",
        duration_months=48,
        is_volatile=True,
        volatility_params={
            'overreaction_multiplier': 3.0,  # How much to escalate responses
            'grudge_min_duration': 1,  # Minimum grudge length (months)
            'grudge_max_duration': 6,  # Maximum grudge length (months)
            'negotiation_probability': 0.12,  # 12% chance per month
            'negotiation_cooperation_boost': 0.7,  # Cooperation during deals
            'slight_threshold': -2.0,  # Payoff loss to trigger reaction
            'major_slight_threshold': -5.0,  # Payoff loss for major reaction
            'reaction_decay': 0.8,  # How quickly reactions fade
            'blame_threshold': 0.6  # Global defection rate to blame others
        }
    )
}


class USLeadershipDynamics:
    """Manages US strategy changes over time including volatile reactions"""

    def __init__(self,
                 initial_profile: str = 'moderate',
                 change_probability: float = 0.25,
                 election_cycles: List[int] = None):
        """
        Args:
            initial_profile: Starting leadership profile
            change_probability: Probability of leadership change at election
            election_cycles: Months when elections occur (default every 48 months)
        """
        self.current_profile = US_LEADERSHIP_PROFILES[initial_profile]
        self.change_probability = change_probability
        self.election_cycles = election_cycles or [48, 96, 144, 192]
        self.history = [(0, initial_profile)]

        # Volatile leadership state
        self.volatile_state = {
            'grudges': {},  # {country_name: expiration_month}
            'recent_payoffs': {},  # {country_name: [recent payoffs]}
            'negotiation_active': {},  # {country_name: expiration_month}
            'current_mood': 'normal',  # 'aggressive', 'cooperative', 'normal'
            'mood_duration': 0
        }

        # Transition probabilities between profiles
        self.transition_matrix = {
            'internationalist': {
                'internationalist': 0.4,
                'moderate': 0.4,
                'protectionist': 0.15,
                'aggressive': 0.04,
                'volatile_populist': 0.01
            },
            'moderate': {
                'internationalist': 0.25,
                'moderate': 0.45,
                'protectionist': 0.20,
                'aggressive': 0.05,
                'volatile_populist': 0.05
            },
            'protectionist': {
                'internationalist': 0.10,
                'moderate': 0.25,
                'protectionist': 0.40,
                'aggressive': 0.15,
                'volatile_populist': 0.10
            },
            'aggressive': {
                'internationalist': 0.05,
                'moderate': 0.15,
                'protectionist': 0.30,
                'aggressive': 0.35,
                'volatile_populist': 0.15
            },
            'volatile_populist': {
                'internationalist': 0.10,
                'moderate': 0.30,
                'protectionist': 0.25,
                'aggressive': 0.20,
                'volatile_populist': 0.15  # Can be voted out!
            }
        }

    def process_month(self, month_num: int, us_country: Country,
                      round_results: Dict, global_conditions: Dict = None) -> Dict:
        """
        Process all leadership dynamics for the month

        Returns:
            Dict of any events that occurred
        """
        events = {
            'leadership_change': False,
            'negotiation_triggered': [],
            'grudges_added': [],
            'grudges_expired': [],
            'mood_change': None
        }

        # Check for leadership change (elections)
        if self.check_leadership_change(month_num, us_country, global_conditions):
            events['leadership_change'] = True

        # If volatile leadership, process reactions
        if self.current_profile.is_volatile:
            volatile_events = self.process_volatile_reactions(
                month_num, us_country, round_results, global_conditions
            )
            events.update(volatile_events)

        return events

    def process_volatile_reactions(self, month_num: int, us_country: Country,
                                   round_results: Dict, global_conditions: Dict) -> Dict:
        """Process volatile leader's reactions to events"""
        events = {
            'negotiation_triggered': [],
            'grudges_added': [],
            'grudges_expired': [],
            'mood_change': None
        }
        params = self.current_profile.volatility_params

        # Update recent payoffs tracking
        self._update_recent_payoffs(us_country, round_results)

        # Check for expired grudges
        expired = []
        for country, expiry in list(self.volatile_state['grudges'].items()):
            if month_num >= expiry:
                expired.append(country)
                del self.volatile_state['grudges'][country]
        events['grudges_expired'] = expired

        # Check for expired negotiations
        for country, expiry in list(self.volatile_state['negotiation_active'].items()):
            if month_num >= expiry:
                del self.volatile_state['negotiation_active'][country]

        # Analyze interactions and determine reactions
        for opponent_name, payoffs in self.volatile_state['recent_payoffs'].items():
            if not payoffs:
                continue

            recent_avg = np.mean(payoffs[-3:])  # Last 3 months

            # Check for slights
            if recent_avg < params['major_slight_threshold']:
                # Major slight - longer grudge
                duration = np.random.randint(
                    params['grudge_min_duration'] + 2,
                    params['grudge_max_duration'] + 1
                )
                self._add_grudge(opponent_name, month_num + duration)
                events['grudges_added'].append((opponent_name, duration))
                print(f"US takes major offense at {opponent_name}! "
                      f"Grudge for {duration} months")

            elif recent_avg < params['slight_threshold']:
                # Minor slight - shorter grudge
                if opponent_name not in self.volatile_state['grudges']:
                    duration = np.random.randint(
                        params['grudge_min_duration'],
                        params['grudge_min_duration'] + 3
                    )
                    self._add_grudge(opponent_name, month_num + duration)
                    events['grudges_added'].append((opponent_name, duration))
                    print(f"US upset with {opponent_name}. "
                          f"Grudge for {duration} months")

        # Check for random negotiation events
        negotiation_events = self._check_negotiation_opportunities(month_num, us_country)
        events['negotiation_triggered'] = negotiation_events

        # Update mood based on overall performance
        mood_change = self._update_volatile_mood(us_country, global_conditions)
        if mood_change:
            events['mood_change'] = mood_change

        # Apply volatile state to country
        self._apply_volatile_state(us_country)

        return events

    def _update_recent_payoffs(self, us_country: Country, round_results: Dict):
        """Track recent payoffs against each opponent"""
        for pair_key, payoffs in round_results.get('payoffs', {}).items():
            if 'US-' in pair_key:
                opponent = pair_key.split('-')[1]
                us_payoff = payoffs[0]
            elif '-US' in pair_key:
                opponent = pair_key.split('-')[0]
                us_payoff = payoffs[1]
            else:
                continue

            if opponent not in self.volatile_state['recent_payoffs']:
                self.volatile_state['recent_payoffs'][opponent] = []

            self.volatile_state['recent_payoffs'][opponent].append(us_payoff)
            # Keep only last 6 months
            if len(self.volatile_state['recent_payoffs'][opponent]) > 6:
                self.volatile_state['recent_payoffs'][opponent].pop(0)

    def _add_grudge(self, country_name: str, expiration_month: int):
        """Add or extend a grudge"""
        if country_name in self.volatile_state['grudges']:
            # Extend existing grudge
            self.volatile_state['grudges'][country_name] = max(
                self.volatile_state['grudges'][country_name],
                expiration_month
            )
        else:
            self.volatile_state['grudges'][country_name] = expiration_month

    def _check_negotiation_opportunities(self, month_num: int,
                                         us_country: Country) -> List[Tuple[str, int]]:
        """Check for random negotiation breakthroughs"""
        events = []
        params = self.current_profile.volatility_params

        # Can't negotiate if already in too many negotiations
        if len(self.volatile_state['negotiation_active']) >= 2:
            return events

        # Random chance for each country relationship
        for country_name in self.volatile_state['recent_payoffs'].keys():
            if country_name in self.volatile_state['negotiation_active']:
                continue

            if np.random.random() < params['negotiation_probability']:
                # Negotiation breakthrough!
                duration = np.random.randint(1, 4)  # 1-3 months
                self.volatile_state['negotiation_active'][country_name] = month_num + duration

                # Clear any grudge
                if country_name in self.volatile_state['grudges']:
                    del self.volatile_state['grudges'][country_name]

                events.append((country_name, duration))
                print(f" 'Beautiful deal' with {country_name}! "
                      f"Cooperation for {duration} months")

        return events

    def _update_volatile_mood(self, us_country: Country,
                              global_conditions: Dict) -> Optional[str]:
        """Update overall mood based on performance"""
        avg_payoff = us_country.get_average_payoff()
        old_mood = self.volatile_state['current_mood']

        # Mood can shift based on performance
        if avg_payoff < -5:
            new_mood = 'aggressive'
            duration = np.random.randint(2, 5)
        elif avg_payoff > 5:
            new_mood = 'cooperative'
            duration = np.random.randint(1, 3)
        else:
            new_mood = 'normal'
            duration = 1

        if new_mood != old_mood:
            self.volatile_state['current_mood'] = new_mood
            self.volatile_state['mood_duration'] = duration
            return f"{old_mood} -> {new_mood}"

        return None

    def _apply_volatile_state(self, us_country: Country):
        """Apply current volatile state to US behavior"""
        base_coop = self.current_profile.cooperation_tendency
        params = self.current_profile.volatility_params

        # Start with base cooperation
        effective_coop = base_coop

        # Mood effects
        if self.volatile_state['current_mood'] == 'aggressive':
            effective_coop *= 0.5
        elif self.volatile_state['current_mood'] == 'cooperative':
            effective_coop *= 1.5

        # Grudge effects (overrides mood for specific countries)
        us_country.volatile_overrides = {}

        for country_name in self.volatile_state['grudges']:
            us_country.volatile_overrides[country_name] = {
                'strategy': 'aggressive',
                'cooperation': 0.1
            }

        # Negotiation effects (overrides everything)
        for country_name in self.volatile_state['negotiation_active']:
            us_country.volatile_overrides[country_name] = {
                'strategy': 'generous_tit_for_tat',
                'cooperation': params['negotiation_cooperation_boost']
            }

        # Set overall cooperation level
        us_country.cooperation_tendency = max(0.1, min(0.9, effective_coop))

    def check_leadership_change(self, month_num: int, us_country: Country,
                                global_conditions: Dict = None) -> bool:
        """Check if leadership changes this month (unchanged from before)"""
        # Check if it's an election cycle
        if month_num not in self.election_cycles:
            # Check for crisis-triggered change (rare)
            if global_conditions and global_conditions.get('crisis', False):
                if np.random.random() < 0.1:  # 10% chance during crisis
                    return self._change_leadership(month_num, us_country, crisis=True)
            return False

        # Election cycle - determine if change occurs
        change_factors = self._calculate_change_factors(us_country, global_conditions)

        if np.random.random() < change_factors['total_probability']:
            return self._change_leadership(month_num, us_country)

        return False

    def _calculate_change_factors(self, us_country: Country,
                                  global_conditions: Dict = None) -> Dict:
        """Calculate probability of leadership change based on conditions"""
        base_probability = self.change_probability

        # Economic performance factor
        avg_payoff = us_country.get_average_payoff()
        if avg_payoff < 0:
            # Poor performance increases change probability
            performance_modifier = min(0.3, abs(avg_payoff) / 10)
        else:
            # Good performance decreases change probability
            performance_modifier = -min(0.2, avg_payoff / 20)

        # Global uncertainty factor
        uncertainty_modifier = 0.0
        if global_conditions:
            uncertainty = global_conditions.get('global_uncertainty', 0)
            if uncertainty > 0.5:
                uncertainty_modifier = 0.2  # High uncertainty increases change

        # Current profile duration factor
        duration_modifier = 0.0
        current_key = self._get_profile_key(self.current_profile)
        if current_key in ['aggressive', 'protectionist', 'volatile_populist']:
            # Extreme positions are less stable
            duration_modifier = 0.1

        total_probability = min(0.8, max(0.1,
                                         base_probability + performance_modifier +
                                         uncertainty_modifier + duration_modifier
                                         ))

        return {
            'base': base_probability,
            'performance': performance_modifier,
            'uncertainty': uncertainty_modifier,
            'duration': duration_modifier,
            'total_probability': total_probability
        }

    def _change_leadership(self, month_num: int, us_country: Country,
                           crisis: bool = False) -> bool:
        """Execute leadership change"""
        current_key = self._get_profile_key(self.current_profile)

        if crisis:
            # Crisis tends to push toward more extreme positions
            if current_key in ['internationalist', 'moderate']:
                new_profile_key = np.random.choice(
                    ['protectionist', 'aggressive', 'volatile_populist'],
                    p=[0.5, 0.3, 0.2]
                )
            else:
                new_profile_key = 'aggressive'
        else:
            # Normal transition based on matrix
            transitions = self.transition_matrix[current_key]
            profiles = list(transitions.keys())
            probabilities = list(transitions.values())
            new_profile_key = np.random.choice(profiles, p=probabilities)

        # Update profile
        self.current_profile = US_LEADERSHIP_PROFILES[new_profile_key]
        self.history.append((month_num, new_profile_key))

        # Update US country parameters
        us_country.strategy = self.current_profile.strategy
        us_country.cooperation_tendency = self.current_profile.cooperation_tendency

        # Reset volatile state if changing leadership
        self.volatile_state = {
            'grudges': {},
            'recent_payoffs': {},
            'negotiation_active': {},
            'current_mood': 'normal',
            'mood_duration': 0
        }

        print(f"  🗳️ US Leadership Change at month {month_num}: "
              f"{current_key} → {new_profile_key}")

        return True

    def _get_profile_key(self, profile: LeadershipProfile) -> str:
        """Get the key for a profile"""
        for key, p in US_LEADERSHIP_PROFILES.items():
            if p.name == profile.name:
                return key
        return 'moderate'

    def get_current_strategy_override(self, opponent_name: str) -> Optional[Dict]:
        """Get strategy override for specific opponent (volatile leaders only)"""
        if not self.current_profile.is_volatile:
            return None

        # This would be called by the Country class to check for overrides
        # The actual overrides are set in _apply_volatile_state
        return None  # Overrides are handled directly in Country object


# Test the leadership system
if __name__ == "__main__":
    from config.countries_data import create_countries

    # Create countries
    countries = create_countries()
    us = next(c for c in countries if c.name == "US")

    # Create leadership dynamics starting with volatile populist
    dynamics = USLeadershipDynamics(initial_profile='volatile_populist')

    print("=== Volatile US Leadership Test ===")
    print(f"Initial profile: {dynamics.current_profile.name}")
    print(f"Strategy: {us.strategy}")
    print(f"Cooperation: {us.cooperation_tendency}")

    # Simulate some interactions
    fake_results = {
        'payoffs': {
            'US-China': (-3.0, 5.0),  # US loses to China
            'US-Mexico': (2.0, 2.0),  # Fair trade with Mexico
            'US-Canada': (3.0, 3.0),  # Good cooperation with Canada
        }
    }

    print("\nSimulating volatile reactions over 12 months:")
    for month in range(12):
        print(f"\nMonth {month}:")
        events = dynamics.process_month(month, us, fake_results)

        # Show current state
        if dynamics.volatile_state['grudges']:
            print(f"  Active grudges: {dynamics.volatile_state['grudges']}")
        if dynamics.volatile_state['negotiation_active']:
            print(f"  Active negotiations: {dynamics.volatile_state['negotiation_active']}")

        # Vary the results to trigger different reactions
        if month == 3:
            fake_results['payoffs']['US-China'] = (-6.0, 8.0)  # Major loss
        elif month == 6:
            fake_results['payoffs']['US-Mexico'] = (-2.0, 4.0)  # Mexico defects