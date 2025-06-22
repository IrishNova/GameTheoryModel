**# Contributing to Multi-Country Trade Simulation

Thank you for your interest in contributing to this game theory model of international trade relationships! This project simulates trade interactions between major economies using real FX data and leadership dynamics.

## Project Overview

This is an academic research project that models:
- **Game Theory**: Iterated prisoner's dilemma with multiple strategies
- **Real FX Data**: Actual currency volatilities from Interactive Brokers
- **Leadership Dynamics**: Including volatile/unpredictable behavior patterns
- **Reserve Currency Effects**: Modeling advantages of reserve currency status

## Getting Started

### Prerequisites

```bash
# Required Python packages
pip install pandas numpy matplotlib seaborn scipy networkx
pip install ib_insync  # For IBKR data collection (optional)
```

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/[your-username]/GameTheoryModel.git
   cd GameTheoryModel
   ```

2. **Test Installation**
   ```bash
   python simulation/engine.py  # Should run basic test
   ```

3. **Verify Data Access**
   ```bash
   python config/parameters.py  # Check IBKR data loads correctly
   ```

## Project Structure

Understanding the codebase architecture:

```
GameTheoryModel/
├── analysis/          # Results compilation and visualization
├── config/           # Country configurations and parameters
├── core/             # Country class and core game logic
├── data/historical/  # IBKR FX data and results storage
├── dynamics/         # Leadership and payoff calculation logic
├── simulation/       # Main simulation engines
└── results/          # Experimental outputs
```

## Contributing Guidelines

### Types of Contributions

**🔬 Research Contributions**
- New country strategies or behaviors
- Enhanced leadership dynamics models
- Additional economic parameters or data sources
- Validation studies or sensitivity analyses

**💻 Code Contributions**
- Performance optimizations
- New visualization capabilities
- Extended scenario testing
- Bug fixes and code quality improvements

**📊 Data Contributions**
- Additional FX data sources
- Trade volume data
- Economic indicators
- Crisis event parameters

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/new-strategy-model
   ```

2. **Code Standards**
   - Follow existing naming conventions (`snake_case` for variables/functions)
   - Add docstrings to all new classes and methods
   - Include type hints where applicable
   - Keep functions focused and modular

3. **Testing Requirements**
   - Test new strategies with `simulation/engine.py`
   - Run Monte Carlo validation with `simulation/monte_carlo.py`
   - Verify results with `analysis/results_compiler.py` (subset)

4. **Documentation**
   - Update docstrings for any modified functions
   - Add comments for complex game theory logic
   - Update README.md if adding new features

### Code Style Guidelines

**Strategy Implementation**
```python
def choose_action(self, opponent_name: str) -> int:
    """
    Choose cooperation (0) or defection (1) based on strategy.
    
    Args:
        opponent_name: Name of the opposing country
        
    Returns:
        0 for cooperation, 1 for defection
    """
    # Strategy logic here
    pass
```

**Simulation Extensions**
```python
class CustomLeadershipDynamics:
    """Custom leadership behavior model"""
    
    def __init__(self, profile_params: Dict):
        """Initialize with research-backed parameters"""
        pass
        
    def process_month(self, month: int, country: Country, 
                     conditions: Dict) -> Dict:
        """Process monthly leadership decisions"""
        pass
```

## Adding New Features

### New Country Strategies

1. **Implement in `core/country.py`**
   ```python
   elif self.strategy == 'your_new_strategy':
       return self._your_strategy_logic(opponent_name)
   ```

2. **Add strategy method**
   ```python
   def _your_strategy_logic(self, opponent_name: str) -> int:
       """Implement your strategy logic with game theory rationale"""
       pass
   ```

3. **Test extensively**
   - Run against all existing strategies
   - Validate with tournament play
   - Check edge cases and convergence

### New Economic Data

1. **Add to `config/parameters.py`**
   ```python
   NEW_ECONOMIC_DATA = {
       'parameter_name': value,
       'source': 'Research Source',
       'date_collected': '2024-XX-XX'
   }
   ```

2. **Update country configurations**
   - Modify `config/countries_data.py` as needed
   - Ensure backward compatibility

3. **Document data sources**
   - Add citation in docstring
   - Include data collection methodology

### New Leadership Dynamics

1. **Extend `dynamics/leadership.py`**
2. **Create new profile class**
3. **Test with existing countries**
4. **Validate historical plausibility**

## Experimental Standards

### Statistical Rigor

- **Minimum iterations**: 50 for preliminary testing, 200 for publication
- **Convergence testing**: Use `analysis/convergence_test.py`
- **Confidence intervals**: Include error bars in all results
- **Reproducibility**: Set random seeds for validation

### Scenario Testing

When adding new scenarios:

1. **Baseline comparison**: Always compare to existing baseline
2. **Multiple iterations**: Use Monte Carlo approach
3. **Statistical significance**: Test for meaningful differences
4. **Real-world plausibility**: Ensure scenarios are realistic

## Submission Process

### Pull Request Checklist

- [ ] Code follows existing style conventions
- [ ] New strategies tested against tournament
- [ ] Monte Carlo validation completed (≥50 iterations)
- [ ] Documentation updated
- [ ] No breaking changes to existing experiments
- [ ] Results are reproducible with set seed

### PR Description Template

```markdown
## Description
Brief description of changes and motivation

## Type of Change
- [ ] New country strategy
- [ ] Leadership dynamics enhancement
- [ ] Data addition/update
- [ ] Bug fix
- [ ] Performance improvement

## Testing
- Baseline comparison: [results]
- New feature validation: [methodology]
- Iterations tested: [number]

## Academic Justification
[Game theory or economics rationale for changes]
```

## Research Collaboration

### Academic Standards

- **Citations required**: All economic data and game theory models
- **Methodology transparency**: Document all assumptions
- **Reproducible results**: Include random seeds and parameters
- **Peer review welcome**: Academic contributors encouraged

### Publication Considerations

If using this model for academic work:

1. **Fork for your research**: Create academic branch
2. **Document modifications**: Track all parameter changes  
3. **Share methodology**: Include in appendix
4. **Cite original work**: Reference this repository

## Getting Help

### Discussion Topics

- **Game Theory**: Strategy design and equilibrium analysis
- **Economics**: FX modeling and trade relationship parameters
- **Technical**: Python optimization and simulation architecture
- **Academic**: Research methodology and validation approaches

### Code Review Focus

- **Economic plausibility**: Do parameters reflect real-world conditions?
- **Game theory validity**: Are strategies theoretically sound?
- **Statistical rigor**: Are results statistically meaningful?
- **Code quality**: Is implementation efficient and maintainable?

## Data Usage and Ethics

### Real Data Handling

- **IBKR data**: Used under academic research provisions
- **Respect ToS**: Don't redistribute raw financial data
- **Academic use**: Research and educational purposes only
- **Attribution**: Credit data sources appropriately

### Model Limitations

Remember this model:
- Uses simplified trade relationships
- Approximates complex economic dynamics
- Is for theoretical insights, not predictions
- Should be validated against empirical research

## Recognition

Contributors will be acknowledged in:
- Repository contributor list
- Academic papers using the model (with permission)
- Conference presentations
- Research documentation

---

**Questions?** Open an issue for research methodology discussions or code-related questions. For academic collaboration, contact the repository maintainer directly.

**Academic Use?** Please cite this repository and contact for research collaboration opportunities.

This is a research project - we welcome contributions that advance the academic understanding of game theory in international trade!**