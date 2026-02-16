# Decentralized Building Simulator (db-sims)

A simulator for studying location choice incentives in distributed and decentralized block building regimes.

## Overview

This simulator models block builders (agents) that:
- Choose regions (geographic locations) and information sources
- Learn optimal strategies through reinforcement learning (EMA-Softmax or UCB)
- Receive rewards based on distance-ranked sharing with concentration penalties
- Start from an even distribution and migrate over time

The simulator is designed to study how different block building regimes (e.g., multiple concurrent proposers, distributed building networks, decentralized ordering protocols) affect the geographic distribution of builders and the resulting centralization dynamics.

## Key Features

- **Distance-ranked sharing**: Closer regions to sources get higher rank weights (w_j = 2^-j)
- **Concentration penalty**: Crowded regions split rewards, incentivizing diversity
- **Persistent proposer distribution**: Track how proposers migrate between regions over time
- **Multiple learning policies**: EMA-Softmax and UCB
- **Comprehensive metrics**: Gini, entropy, rewards tracked over time
- **Easy experimentation**: Centralized configuration and comparison tools

## Quick Start

### Run the default experiments:
```bash
python my_experiments.py
```

This will:
- Run 4 different experiment configurations
- Compare EMA vs UCB policies, exploration levels, and migration costs
- Generate plots showing how metrics evolve over time
- Save results to `experiment_results/` directory

### Customize your experiments:

Edit `my_experiments.py` and modify the `define_experiments()` function:

```python
def define_experiments():
    experiments = []

    experiments.append(ExperimentConfig(
        name="my_experiment",

        # Geography
        n_regions=5,
        region_names=["West", "CentralWest", "Central", "CentralEast", "East"],

        # Information sources: (name, value, home_region)
        sources_config=[
            ("LowValue", 8.0, 0),      # At West (distance 0)
            ("MedValue", 12.0, 2),     # At Central (distance 0)
            ("HighValue", 18.0, 4)     # At East (distance 0)
        ],

        # Policy
        policy_type="EMA",  # or "UCB"
        eta=0.12,           # Learning rate
        beta_reg=1.5,       # Region selection temperature (lower = more exploration)
        beta_src=2.5,       # Source selection temperature
        cost_c=0.2,         # Migration cost

        # Simulation
        n_proposers=80,     # Total number of proposers
        K=8,                # Concurrent proposers per slot
        n_slots=10000,      # Simulation length
        seed=42
    ))

    return experiments
```

## Core Concepts

### Information Sources
- Positioned **at** regions (distance = 0 to home region)
- Each source has a constant value V_I
- Proposers choose which source to query

### Distance Matrix
- D[r, I] = distance from region r to source I
- For a source at region r_home: D[r, I] = |r - r_home|
- Distance = 0 when proposer is at source's home region

### Omega (Ω) Mechanism
Reward sharing based on distance ranking:
1. Regions ranked by distance to source (closest = rank 0)
2. Rank weights: $w_j = 2^{-j}$
3. Per-proposer share: $\Omega_r = w_j / Σ(w_k × x_k)$
   - where $x_k$ = number of proposers at rank k
4. Individual reward: $R = \Omega_r × V_I$

**Key properties:**
- Proximity advantage: Closer regions get higher rewards
- Concentration penalty: More proposers → lower per-proposer share
- Trade-off: Best region may not be closest, depends on crowding

### Proposer Distribution Tracking

**Initial state**: Proposers start evenly distributed across regions
- 100 proposers, 5 regions → 20 per region initially

**Evolution**: When selected proposers choose regions, they migrate
- `current_region` attribute updated

**Metrics tracked:**
- **Proposer Distribution Gini**: Inequality in distribution (0 = equal, 1 = concentrated)
- **Proposer Distribution Entropy**: Diversity (0 = concentrated, 1 = uniform)
- **Region Counts**: Per-slot choices of K selected proposers
- **Distinction**: Distribution tracks ALL proposers' locations; counts track per-slot choices

## File Structure

### Core Library
- **`mcp_simulator.py`** - Core simulator with Omega mechanism, policies, and tracking (retains original name for compatibility)
- **`experiment_runner.py`** - Experiment framework with configuration, comparison, and visualization

### Experiments & Demos
- **`my_experiments.py`** - Main experiment file (edit this for your experiments)
- **`omega_mechanism_demo.py`** - Educational demonstrations of Omega mechanism properties

### Documentation
- **`README.md`** - This file

## Available Metrics

### For Comparison Plots
Compare multiple experiments on the same plot:
```python
compare_experiments(
    results,
    metrics=[
        'proposer_dist_gini',      # Proposer distribution inequality
        'proposer_dist_entropy',   # Proposer distribution diversity
        'region_gini',             # Per-slot region choice inequality
        'region_entropy',          # Per-slot region choice diversity
        'source_gini',             # Source selection inequality
        'source_entropy',          # Source selection diversity
        'reward'                   # Average reward per proposer
    ]
)
```

### For Individual Experiment Details
Plot detailed time-series for a single experiment:
```python
plot_experiment_details(result, save_plots=True)
```

This creates a comprehensive 6-panel plot showing:
- **Region Selection**: Stacked area showing which regions proposers choose per slot
- **Source Selection**: Stacked area showing which sources proposers choose per slot
- **Proposer Distribution**: Stacked area showing where ALL proposers reside over time
- **Gini Over Time**: Inequality metrics for proposer distribution, region selection, and source selection
- **Entropy Over Time**: Diversity metrics for proposer distribution, region selection, and source selection
- **Reward Over Time**: Average reward per proposer evolution

### Statistics Available
```python
result = run_experiment(config)

# Final averages
result.stats['avg_reward']
result.stats['avg_proposer_distribution']  # Where proposers reside
result.stats['avg_region_counts']          # Per-slot choices
result.stats['proposer_dist_gini']
result.stats['proposer_dist_entropy']

# Time series
result.proposer_distribution              # Shape: (n_slots, n_regions)
result.proposer_dist_gini_over_time       # List of Gini values
result.proposer_dist_entropy_over_time    # List of entropy values
result.region_counts                      # Per-slot region choices
result.rewards                            # Average reward per slot
```

## Policy Parameters

### EMA-Softmax (`policy_type="EMA"`)
- **`eta`** (0.05-0.2): Learning rate. Higher = faster adaptation
- **`beta_reg`** (0.5-3.0): Region selection temperature. Higher = more exploitation
- **`beta_src`** (1.0-4.0): Source selection temperature. Higher = more exploitation
- **`cost_c`** (0.0-2.0): Migration cost. Higher = more sticky to current region

### UCB (`policy_type="UCB"`)
- **`alpha`** (0.5-3.0): Exploration parameter. Higher = more exploration

## Example Research Questions

1. **Convergence**: How quickly does builder distribution reach equilibrium?
2. **Policy comparison**: Does EMA or UCB achieve better rewards/diversity?
3. **Migration cost effects**: How does `cost_c` affect final distribution?
4. **Source placement**: How does information source location influence equilibrium?
5. **Exploration vs exploitation**: What's the optimal balance for different building regimes?
6. **Fairness**: Is the final distribution equitable across regions?
7. **Decentralization trade-offs**: How do different block building protocols affect geographic centralization?
8. **Network effects**: How does the number of concurrent builders affect location choices?

## Advanced Usage

### Load Saved Results
```python
from experiment_runner import ExperimentResult

result = ExperimentResult.load("experiment_results/my_experiment_results.npz")
print(result.stats)
print(result.proposer_dist_gini_over_time)
```

### Use Preset Configurations
```python
from experiment_runner import get_preset_config, run_experiment

config = get_preset_config("large_diverse")
config.n_slots = 15000  # Modify if needed
result = run_experiment(config)
```

Available presets:
- `"small_uniform"` - 3 regions, 3 uniform-value sources
- `"large_diverse"` - 5 regions, 3 sources with different values
- `"ucb_exploration"` - UCB policy with high exploration
- `"high_migration_cost"` - EMA with high migration cost

### Custom Distance Matrix
```python
import numpy as np
from mcp_simulator import Region, Source, Proposer, MCPSimulator, EMASoftmaxPolicy

# Create custom scenario
regions = [Region(0, "A"), Region(1, "B"), Region(2, "C")]
sources = [Source(0, "S1", value=10.0, home_region=0)]

# Custom distances (not just linear)
distance_matrix = np.array([
    [0.0],  # Region A: at source
    [2.0],  # Region B: medium distance
    [5.0]   # Region C: far distance
])

# Create proposers
proposers = []
for i in range(50):
    policy = EMASoftmaxPolicy(3, 1, eta=0.1, beta_reg=2.0, beta_src=2.0)
    proposers.append(Proposer(i, policy))

# Run simulation
sim = MCPSimulator(regions, sources, proposers, distance_matrix, K=5, seed=42)
sim.run(1000)
stats = sim.get_statistics()
```

## Understanding the Output

### Comparison Table
```
Experiment           Policy   Reward     PropDist   PropDist   Region    Region    Source    Source
                                         Gini       Entropy    Gini      Entropy   Gini      Entropy
-----------------------------------------------------------------------------------------------------------
baseline_ema         EMA      12.3456    0.2345     0.8765     0.1234    0.9012    0.0987    0.9543
high_exploration     EMA      11.8765    0.1543     0.9234     0.0876    0.9456    0.0654    0.9678
```

- **Lower Gini** = more equal/diverse
- **Higher Entropy** = more diverse/uniform
- **Proposer Dist vs Region**: Distribution = where ALL proposers reside; Region = per-slot choices

### Plots
- **Proposer Distribution Gini**: Shows how inequality evolves (starts at ~0 for even distribution)
- **Proposer Distribution Entropy**: Shows diversity evolution (starts at ~1.0 for even distribution)
- **Reward**: Shows learning progress and convergence

## Educational Demo

Run the Omega mechanism demonstration:
```bash
python omega_mechanism_demo.py
```

This shows:
- Proximity advantage: Closer regions get higher rewards
- Concentration penalty: Crowding dilutes per-proposer rewards
- Equilibrium analysis: Finding balanced distributions
- Rank weight effects: How w_j impacts incentives

## Tips

- Start with 5000-10000 slots to see full evolution
- Use K = 10-20% of proposers for reasonable sampling
- Compare proposer_dist_gini vs region_gini to understand migration patterns
- Lower beta_reg = more exploration, higher convergence time
- Higher cost_c = slower migration, more persistent distribution
- Sources at distance 0 from their home region by design

## Dependencies

```bash
pip install numpy matplotlib
```

- Python 3.7+
- NumPy (required)
- Matplotlib (optional, for visualizations)

## References

This simulator was inspired by research on geographic decentralization in blockchain systems:
- GeoDec paper: [arXiv:2509.21475v2](https://arxiv.org/pdf/2509.21475v2)
- Original GeoDec repo: [geographical-decentralization-simulation](https://github.com/syang-ng/geographical-decentralization-simulation)

The simulator has been generalized to study location choice incentives across various distributed and decentralized block building regimes.

## License

Open source - feel free to modify and extend for your research.
