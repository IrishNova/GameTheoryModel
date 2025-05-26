"""
Parameters for trade simulation model
Loads REAL data from IBKR and economic research
"""
import json
import os

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PACKAGE_DIR, "data", "historical")

# Load YOUR IBKR FX data
print("Loading researched FX data from IBKR...")
fx_params_path = os.path.join(DATA_DIR, "fx_model_parameters.json")
with open(fx_params_path, 'r') as f:
    FX_DATA = json.load(f)

# Load YOUR economic data
print("Loading researched economic data...")
econ_params_path = os.path.join(DATA_DIR, "economic_model_parameters.json")
with open(econ_params_path, 'r') as f:
    ECON_DATA = json.load(f)

# Extract key parameters from YOUR data
VOLATILITIES = FX_DATA['volatilities']  # Real volatilities from IBKR
CORRELATIONS = FX_DATA['correlations']  # Real correlations from IBKR
CURRENT_FX_RATES = FX_DATA['current_rates']  # Real current rates from IBKR

RESERVE_SHARES = ECON_DATA['reserve_currency_shares']  # From economic research
TRADE_WEIGHTS = ECON_DATA['trade_weights']  # From economic research

# Game theory parameters
PAYOFF_MATRIX = {
    (0, 0): (3, 3),  # Both cooperate
    (0, 1): (0, 5),  # I cooperate, they defect
    (1, 0): (5, 0),  # I defect, they cooperate
    (1, 1): (1, 1),  # Both defect
}

# Simulation parameters
DEFAULT_ROUNDS = 40
DEFAULT_ITERATIONS = 100

# Currency regime parameters
REGIME_SETTINGS = {
    'floating': {
        'volatility_multiplier': 1.0,
        'intervention_threshold': None
    },
    'managed': {
        'volatility_multiplier': 0.5,
        'intervention_threshold': 0.02  # 2% band
    },
    'fixed': {
        'volatility_multiplier': 0.033,
        'intervention_threshold': 0.002  # 0.2% band
    }
}

# Test data loading
if __name__ == "__main__":
    print("\n=== Your Researched Data Loaded Successfully! ===")
    print(f"\nFX Data from IBKR:")
    print(f"  Currency pairs: {list(CURRENT_FX_RATES.keys())}")
    print(f"  Example rate - EURUSD: {CURRENT_FX_RATES.get('EURUSD', 'N/A'):.4f}")
    print(f"  Example volatility - EURUSD: {VOLATILITIES.get('EURUSD', {}).get('average', 'N/A'):.4f}")

    print(f"\nEconomic Data:")
    print(f"  Reserve shares: {RESERVE_SHARES}")
    print(f"  Number of trade relationships: {len(TRADE_WEIGHTS)}")
    print(f"  Example trade weight (USD-EUR): {TRADE_WEIGHTS.get('USD-EUR', 'N/A'):.2f}")