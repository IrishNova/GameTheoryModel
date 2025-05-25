"""
Constants and data loading for trade simulation model
"""
import json
import os
import pandas as pd

# Path to data directory
DATA_DIR = "/Users/ryanmoloney/Desktop/DePaul 24/GameTheoryModel/data/historical"

# Load FX parameters
fx_params_path = os.path.join(DATA_DIR, "fx_model_parameters.json")
with open(fx_params_path, 'r') as f:
    FX_DATA = json.load(f)

# Load economic parameters
econ_params_path = os.path.join(DATA_DIR, "economic_model_parameters.json")
with open(econ_params_path, 'r') as f:
    ECON_DATA = json.load(f)

# Game theory payoff matrix
# (my_action, their_action) -> (my_payoff, their_payoff)
PAYOFF_MATRIX = {
    (0, 0): (3, 3),  # Both cooperate
    (0, 1): (0, 5),  # I cooperate, they defect
    (1, 0): (5, 0),  # I defect, they cooperate
    (1, 1): (1, 1),  # Both defect
}

# Simulation parameters
DEFAULT_ROUNDS = 40
DEFAULT_ITERATIONS = 100

# Currency regime volatility adjustments
VOLATILITY_ADJUSTMENTS = {
    'floating': 1.0,
    'managed': 0.5,
    'fixed': 0.033
}

# Quick test to make sure data loads correctly
if __name__ == "__main__":
    print("Data loaded successfully!")
    print(f"FX pairs available: {list(FX_DATA['current_rates'].keys())}")
    print(f"Trade weights available: {len(ECON_DATA['trade_weights'])} pairs")
    print(f"Reserve currencies: {list(ECON_DATA['reserve_currency_shares'].keys())}")