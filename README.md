# Multi-Country Trade Simulation with Currency Dynamics

A sophisticated game theory model simulating international trade relationships, incorporating real-world foreign exchange data, reserve currency dynamics, and volatile leadership behavior.

## Project Overview

This simulation models trade interactions between 8 major economies using:
- **Game Theory**: Iterated prisoner's dilemma with multiple strategies
- **Real FX Data**: Actual currency volatilities and correlations from Interactive Brokers
- **Reserve Currency Effects**: Modeling the advantages of reserve currency status
- **Leadership Dynamics**: Including volatile/unpredictable leadership patterns
- **Monthly Timeline**: 40-month simulations capturing medium-term dynamics

### Key Research Questions
1. Does reserve currency status provide meaningful protection in trade conflicts?
2. How does leadership volatility affect global trade cooperation?
3. Which strategies optimize outcomes in uncertain environments?
4. Do managed currency regimes provide trade advantages?

## Data Sources

### Real Data Collected
- **Interactive Brokers (IBKR)**: 
  - 2 years of daily FX rates for 6 major currency pairs
  - Calculated volatilities and correlation matrices
  - Current exchange rates

### Approximated Data
- **Trade Volumes**: Bilateral trade weights based on major trading relationships
- **Reserve Currency Shares**: Based on IMF COFER reports
- **Economic Parameters**: Cooperation tendencies and strategic profiles

## Project Structure

```
GameTheoryModel/
├── analysis/
│   ├── convergence_test.py      # Statistical convergence analysis
│   ├── results_compiler.py      # Systematic experiment runner
│   └── visualization_suite.py   # Publication-ready figures
├── config/
│   ├── countries_data.py        # Country configurations
│   └── parameters.py            # Load researched data
├── core/
│   └── country.py               # Country class with strategies
├── data/
│   └── historical/              # IBKR FX data and parameters
├── dynamics/
│   ├── leadership.py            # US leadership dynamics
│   └── payoffs.py               # Payoff calculations
├── simulation/
│   ├── engine.py                # Core simulation engine
│   ├── monte_carlo.py           # Monte Carlo wrapper
│   └── scenarios.py             # Special scenario testing
└── results/                     # Experiment outputs
```

## Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scipy
pip install ib_insync  # For IBKR data collection
```

### Running a Basic Simulation
```python
from config.countries_data import create_countries
from simulation.engine import TradeSimulation

# Create countries with real data
countries = create_countries()

# Run simulation
sim = TradeSimulation(countries, rounds=40, us_leadership_profile='moderate')
results = sim.run()

# View results
for country, payoff in results['average_payoffs'].items():
    print(f"{country}: {payoff:.2f}")
```

### Running Full Analysis
```python
from analysis.results_compiler import ResultsCompiler

# Run all experiments (takes ~20-30 minutes)
compiler = ResultsCompiler(base_iterations=200)
compiler.run_all_experiments()

# Results saved to /results/paper_results_[timestamp]/
```

## 🔬 Key Features

### 1. **Country Strategies**
- **Tit-for-Tat**: Reciprocate opponent's last action
- **Aggressive**: Always impose tariffs
- **Cooperative**: Always cooperate
- **Generous Tit-for-Tat**: Forgive defections occasionally
- **Reactive Volatile**: Unpredictable responses with grudges

### 2. **Leadership Dynamics**
- **Moderate**: Balanced approach
- **Internationalist**: Pro-free trade
- **Protectionist**: Tariff-friendly
- **Volatile Populist**: Reactive with grudges and surprise negotiations

### 3. **Currency Regimes**
- **Floating**: Market-determined rates (US, EU, Japan, UK, Canada, Mexico)
- **Managed**: Controlled fluctuation (China, Singapore)
- **Reserve Status**: Advantages in funding costs and transaction fees

### 4. **Volatile Leadership Mechanics**
- **Grudges**: 1-6 month retaliation periods
- **Negotiations**: Random "beautiful deals"
- **Mood Swings**: Overall cooperation changes
- **Reactive**: Responds to payoff losses

## Experiments

### Experiment 1: Leadership Comparison
Compares moderate vs volatile US leadership impacts on global trade.

### Experiment 2: Reserve Currency Analysis
Tests effects of changing reserve currency status (US decline, China rise).

### Experiment 3: Strategy Tournament
Evaluates which US strategies perform best.

### Experiment 4: Crisis Scenarios
Models currency crisis impacts (peso crisis, pound crisis).

### Experiment 5: Cooperation Sensitivity
Finds optimal cooperation levels under volatile leadership.

## Key Findings

1. **Volatile Leadership**: Creates persistent conflicts (especially with Japan) but can improve US outcomes
2. **Reserve Currency**: Provides advantages but not decisive in trade conflicts
3. **Japan Success**: Balanced tit-for-tat strategy consistently wins
4. **China Strategy**: Aggressive approach pays off despite low cooperation
5. **Convergence**: Model stabilizes quickly (25-50 iterations sufficient)

## Statistical Validation

- **Convergence Test**: Shows stable results after 25 iterations
- **Coefficient of Variation**: < 0.5% for all major metrics
- **Bootstrap Analysis**: Confidence intervals included
- **Monte Carlo**: 200+ iterations for publication quality

## Visualizations

The project generates 6 publication-ready figures:
1. Leadership comparison (payoffs & cooperation)
2. Reserve currency impact analysis
3. Strategy tournament results
4. Cooperation sensitivity curves
5. Trade network visualization
6. Volatile leadership timeline

## 🔧 Configuration

### Modifying Countries
Edit `config/countries_data.py`:
```python
Country(
    name="US",
    currency="USD",
    strategy="tit_for_tat",
    cooperation_tendency=0.6,
    currency_regime="floating",
    reserve_status=0.59,  # From IMF data
    usd_exposure=0.0
)
```

### Changing Parameters
Edit `config/parameters.py` to adjust:
- Game theory payoff matrix
- Simulation rounds
- Currency regime settings

## Academic Use

### Citation
If using this model in academic work, please cite:
```
[Your Name]. (2024). Multi-Country Trade Simulation with Currency Dynamics 
and Volatile Leadership. [University/Institution].
```

### Paper Structure Suggestions
1. **Introduction**: Reserve currency advantage in trade wars
2. **Model**: Game theory + FX dynamics + leadership
3. **Data**: IBKR real FX data + economic parameters
4. **Results**: 5 experiments with statistical validation
5. **Discussion**: Policy implications

## Contributing

This is an academic project. For questions or collaboration:
- Raise an issue for bugs
- Submit PRs for enhancements
- Contact for research collaboration

## License

This project is for academic research. Please contact before commercial use.

## Acknowledgments

- Interactive Brokers for FX data access
- Game theory framework inspired by Axelrod's tournaments
- Currency dynamics based on international finance literature

---

**Note**: This model uses real FX data but simplified trade relationships. Results should be interpreted as theoretical insights rather than specific predictions.