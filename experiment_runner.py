#!/usr/bin/env python3
"""
Experiment Runner: Configurable simulator for studying location choice in decentralized building.

This script provides:
1. Centralized configuration for all experiment parameters
2. Easy-to-run experiments with different settings
3. Comparison tools to analyze metrics over time across experiments

Supports various distributed/decentralized block building regimes.
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import json
from pathlib import Path
from mcp_simulator import (
    Region, Source, Proposer, MCPSimulator,
    EMASoftmaxPolicy, UCBPolicy
)

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not found. Install with 'pip install matplotlib' for visualizations.")


# ============================================================================
# CONFIGURATION SECTION - Edit parameters here
# ============================================================================

@dataclass
class ExperimentConfig:
    """All experiment parameters in one place."""

    # Experiment identification
    name: str = "default_experiment"

    # Regions configuration
    n_regions: int = 5
    region_names: List[str] = None  # If None, will auto-generate

    # Sources configuration
    # Each source: (name, value, home_region)
    sources_config: List[tuple] = None

    # Policy configuration
    policy_type: str = "EMA"  # "EMA" or "UCB"

    # EMA policy parameters
    eta: float = 0.12
    beta_reg: float = 1.5
    beta_src: float = 2.5
    cost_c: float = 0.2

    # UCB policy parameters
    alpha: float = 2.0

    # Simulation parameters
    n_proposers: int = 80
    K: int = 8  # Concurrent proposers per slot
    n_slots: int = 10000
    seed: int = 42

    # Output configuration
    save_results: bool = True
    results_dir: str = "experiment_results"

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.region_names is None:
            self.region_names = [f"Region_{i}" for i in range(self.n_regions)]

        if self.sources_config is None:
            # Default: 3 sources spread across regions
            self.sources_config = [
                ("SourceA", 8.0, 0),
                ("SourceB", 12.0, self.n_regions // 2),
                ("SourceC", 18.0, self.n_regions - 1)
            ]


# ============================================================================
# EXPERIMENT PRESETS - Quick configurations for common scenarios
# ============================================================================

def get_preset_config(preset_name: str) -> ExperimentConfig:
    """Get a predefined experiment configuration."""

    presets = {
        "small_uniform": ExperimentConfig(
            name="small_uniform",
            n_regions=3,
            region_names=["West", "Central", "East"],
            sources_config=[
                ("Oracle1", 8.0, 0),
                ("Oracle2", 8.0, 1),
                ("Oracle3", 8.0, 2)
            ],
            policy_type="EMA",
            n_proposers=60,
            K=6,
            n_slots=5000
        ),

        "large_diverse": ExperimentConfig(
            name="large_diverse",
            n_regions=5,
            region_names=["West", "CentralWest", "Central", "CentralEast", "East"],
            sources_config=[
                ("FastOracle", 8.0, 0),
                ("BalancedOracle", 12.0, 2),
                ("PremiumOracle", 18.0, 4)
            ],
            policy_type="EMA",
            eta=0.12,
            beta_reg=1.5,
            beta_src=2.5,
            n_proposers=80,
            K=8,
            n_slots=10000
        ),

        "ucb_exploration": ExperimentConfig(
            name="ucb_exploration",
            n_regions=5,
            sources_config=[
                ("Source1", 10.0, 0),
                ("Source2", 15.0, 2),
                ("Source3", 20.0, 4)
            ],
            policy_type="UCB",
            alpha=2.0,
            n_proposers=80,
            K=8,
            n_slots=10000
        ),

        "high_migration_cost": ExperimentConfig(
            name="high_migration_cost",
            n_regions=5,
            sources_config=[
                ("Source1", 10.0, 0),
                ("Source2", 15.0, 4)
            ],
            policy_type="EMA",
            eta=0.1,
            beta_reg=2.0,
            beta_src=2.0,
            cost_c=1.0,  # High migration cost
            n_proposers=80,
            K=8,
            n_slots=10000
        )
    }

    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

    return presets[preset_name]


# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

class ExperimentResult:
    """Store results from a single experiment."""

    def __init__(self, config: ExperimentConfig, simulator: MCPSimulator):
        self.config = config
        self.simulator = simulator
        self.stats = simulator.get_statistics()

        # Time series data
        self.region_counts = np.array(simulator.region_counts_history)
        self.source_counts = np.array(simulator.source_counts_history)
        self.proposer_distribution = np.array(simulator.proposer_distribution_history)
        self.rewards = [np.mean(r) if r else 0 for r in simulator.reward_history]

        # Compute time-series metrics
        self._compute_time_series_metrics()

    def _compute_time_series_metrics(self):
        """Compute Gini, entropy, etc. over time."""
        def gini(x):
            sorted_x = np.sort(x)
            n = len(x)
            if np.sum(x) == 0:
                return 0.0
            cumsum = np.cumsum(sorted_x)
            return (2 * np.sum((np.arange(1, n+1)) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n

        def entropy(counts):
            total = np.sum(counts)
            if total == 0:
                return 0.0
            probs = counts / total
            probs = probs[probs > 0]
            return -np.sum(probs * np.log(probs)) / np.log(len(counts)) if len(probs) > 0 else 0.0

        # Compute for each time step
        self.region_gini_over_time = []
        self.region_entropy_over_time = []
        self.source_gini_over_time = []
        self.source_entropy_over_time = []
        self.proposer_dist_gini_over_time = []
        self.proposer_dist_entropy_over_time = []

        for t in range(len(self.region_counts)):
            self.region_gini_over_time.append(gini(self.region_counts[t]))
            self.region_entropy_over_time.append(entropy(self.region_counts[t]))
            self.source_gini_over_time.append(gini(self.source_counts[t]))
            self.source_entropy_over_time.append(entropy(self.source_counts[t]))
            self.proposer_dist_gini_over_time.append(gini(self.proposer_distribution[t]))
            self.proposer_dist_entropy_over_time.append(entropy(self.proposer_distribution[t]))

    def save(self, filepath: Optional[str] = None):
        """Save results to disk."""
        if filepath is None:
            results_dir = Path(self.config.results_dir)
            results_dir.mkdir(exist_ok=True)
            filepath = results_dir / f"{self.config.name}_results.npz"

        # Save numpy arrays and metadata
        np.savez(
            filepath,
            region_counts=self.region_counts,
            source_counts=self.source_counts,
            proposer_distribution=self.proposer_distribution,
            rewards=np.array(self.rewards),
            region_gini_over_time=np.array(self.region_gini_over_time),
            region_entropy_over_time=np.array(self.region_entropy_over_time),
            source_gini_over_time=np.array(self.source_gini_over_time),
            source_entropy_over_time=np.array(self.source_entropy_over_time),
            proposer_dist_gini_over_time=np.array(self.proposer_dist_gini_over_time),
            proposer_dist_entropy_over_time=np.array(self.proposer_dist_entropy_over_time),
            config=np.array([asdict(self.config)], dtype=object),
            stats=np.array([self.stats], dtype=object)
        )

        print(f"Results saved to: {filepath}")
        return filepath

    @staticmethod
    def load(filepath: str) -> 'ExperimentResult':
        """Load results from disk."""
        data = np.load(filepath, allow_pickle=True)

        # Reconstruct config
        config_dict = data['config'].item()
        config = ExperimentConfig(**config_dict)

        # Create a minimal result object
        result = object.__new__(ExperimentResult)
        result.config = config
        result.stats = data['stats'].item()
        result.region_counts = data['region_counts']
        result.source_counts = data['source_counts']
        result.proposer_distribution = data['proposer_distribution']
        result.rewards = list(data['rewards'])
        result.region_gini_over_time = list(data['region_gini_over_time'])
        result.region_entropy_over_time = list(data['region_entropy_over_time'])
        result.source_gini_over_time = list(data['source_gini_over_time'])
        result.source_entropy_over_time = list(data['source_entropy_over_time'])
        result.proposer_dist_gini_over_time = list(data['proposer_dist_gini_over_time'])
        result.proposer_dist_entropy_over_time = list(data['proposer_dist_entropy_over_time'])

        return result


def create_scenario_from_config(config: ExperimentConfig):
    """Create regions, sources, and distance matrix from config."""
    # Create regions
    regions = [Region(i, config.region_names[i]) for i in range(config.n_regions)]

    # Create sources
    sources = []
    for i, (name, value, home_region) in enumerate(config.sources_config):
        sources.append(Source(i, name, value, home_region))

    # Create distance matrix
    distance_matrix = np.zeros((config.n_regions, len(sources)))
    for r in range(config.n_regions):
        for i, source in enumerate(sources):
            distance_matrix[r, i] = abs(r - source.home_region)

    return regions, sources, distance_matrix


def run_experiment(config: ExperimentConfig, verbose: bool = True) -> ExperimentResult:
    """Run a single experiment with given configuration."""

    if verbose:
        print(f"\n{'='*70}")
        print(f"Running Experiment: {config.name}")
        print(f"{'='*70}")
        print(f"Policy: {config.policy_type}")
        print(f"Regions: {config.n_regions}, Sources: {len(config.sources_config)}")
        print(f"Proposers: {config.n_proposers}, K: {config.K}, Slots: {config.n_slots}")

    # Create scenario
    regions, sources, distance_matrix = create_scenario_from_config(config)

    if verbose:
        print(f"\nSources: {[(s.name, f'V={s.value}', f'home={s.home_region}') for s in sources]}")
        print(f"\nDistance Matrix:")
        print(distance_matrix)

    # Create proposers
    proposers = []
    for i in range(config.n_proposers):
        if config.policy_type == "EMA":
            policy = EMASoftmaxPolicy(
                config.n_regions, len(sources),
                eta=config.eta,
                beta_reg=config.beta_reg,
                beta_src=config.beta_src,
                cost_c=config.cost_c
            )
        elif config.policy_type == "UCB":
            policy = UCBPolicy(config.n_regions, len(sources), alpha=config.alpha)
        else:
            raise ValueError(f"Unknown policy: {config.policy_type}")

        proposers.append(Proposer(i, policy))

    # Create and run simulator
    sim = MCPSimulator(regions, sources, proposers, distance_matrix, K=config.K, seed=config.seed)

    if verbose:
        print(f"\nRunning simulation...")

    sim.run(config.n_slots)

    # Create result object
    result = ExperimentResult(config, sim)

    if verbose:
        print_results(result, regions, sources)

    # Save if requested
    if config.save_results:
        result.save()

    return result


def print_results(result: ExperimentResult, regions: List[Region], sources: List[Source]):
    """Print experiment results."""
    stats = result.stats

    print(f"\n{'='*70}")
    print("Results")
    print(f"{'='*70}")
    print(f"Average reward per proposer per slot: {stats['avg_reward']:.4f}")

    print(f"\nProposer distribution across regions (avg over time):")
    for i, count in enumerate(stats['avg_proposer_distribution']):
        print(f"  {regions[i].name:15s}: {count:6.2f} proposers")

    print(f"\nRegion selection per slot (avg proposers per slot):")
    for i, count in enumerate(stats['avg_region_counts']):
        print(f"  {regions[i].name:15s}: {count:6.2f}")

    print(f"\nSource selection (avg selections per slot):")
    for i, count in enumerate(stats['avg_source_counts']):
        print(f"  {sources[i].name:15s} (V={sources[i].value:4.1f}): {count:6.2f}")

    print(f"\nDiversity metrics:")
    print(f"  Proposer Dist Gini:    {stats['proposer_dist_gini']:.4f} (lower = more equal)")
    print(f"  Proposer Dist Entropy: {stats['proposer_dist_entropy']:.4f} (higher = more equal)")
    print(f"  Region Gini:           {stats['region_gini']:.4f}")
    print(f"  Region Entropy:        {stats['region_entropy']:.4f}")
    print(f"  Source Gini:           {stats['source_gini']:.4f}")
    print(f"  Source Entropy:        {stats['source_entropy']:.4f}")


# ============================================================================
# COMPARISON TOOLS
# ============================================================================

def compare_experiments(results: List[ExperimentResult],
                       metrics: List[str] = None,
                       save_plots: bool = True):
    """
    Compare multiple experiments by plotting metrics over time.

    Args:
        results: List of ExperimentResult objects to compare
        metrics: List of metrics to plot. Options:
                 - 'region_gini'
                 - 'region_entropy'
                 - 'source_gini'
                 - 'source_entropy'
                 - 'proposer_dist_gini'
                 - 'proposer_dist_entropy'
                 - 'reward'
                 If None, plots all metrics
        save_plots: Whether to save plots to disk
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Cannot create comparison plots.")
        return

    if metrics is None:
        metrics = ['proposer_dist_gini', 'proposer_dist_entropy', 'region_gini', 'region_entropy', 'reward']

    print(f"\n[DEBUG] compare_experiments called with {len(results)} results:")
    for i, r in enumerate(results):
        print(f"  [{i}] {r.config.name} ({r.config.policy_type})")

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4*n_metrics))

    if n_metrics == 1:
        axes = [axes]

    # Create descriptive title with experiment names
    exp_names = ", ".join([r.config.name for r in results])
    if len(exp_names) > 80:
        exp_names = ", ".join([r.config.name for r in results[:3]])
        if len(results) > 3:
            exp_names += f", +{len(results)-3} more"
    fig.suptitle(f'Experiment Comparison: {exp_names}', fontsize=14, fontweight='bold')

    # Color scheme and line styles for better distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    line_styles = ['-', '--', '-.', ':']  # Solid, dashed, dash-dot, dotted
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']  # Different markers

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]

        for result_idx, result in enumerate(results):
            # Get data for this metric
            if metric == 'region_gini':
                data = result.region_gini_over_time
                ylabel = 'Region Gini (lower = more diverse)'
            elif metric == 'region_entropy':
                data = result.region_entropy_over_time
                ylabel = 'Region Entropy (higher = more diverse)'
            elif metric == 'source_gini':
                data = result.source_gini_over_time
                ylabel = 'Source Gini (lower = more diverse)'
            elif metric == 'source_entropy':
                data = result.source_entropy_over_time
                ylabel = 'Source Entropy (higher = more diverse)'
            elif metric == 'proposer_dist_gini':
                data = result.proposer_dist_gini_over_time
                ylabel = 'Proposer Distribution Gini (lower = more equal)'
            elif metric == 'proposer_dist_entropy':
                data = result.proposer_dist_entropy_over_time
                ylabel = 'Proposer Distribution Entropy (higher = more equal)'
            elif metric == 'reward':
                data = result.rewards
                ylabel = 'Average Reward per Proposer'
            else:
                print(f"Unknown metric: {metric}")
                continue

            # Smooth with moving average
            window = min(100, len(data) // 10)
            if window > 1:
                smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
                slots = np.arange(window-1, len(data))
            else:
                smoothed = data
                slots = np.arange(len(data))

            # Plot with distinct line style
            label = f"{result.config.name} ({result.config.policy_type})"
            linestyle = line_styles[result_idx % len(line_styles)]
            print(f"    Plotting {result.config.name}: {len(smoothed)} points, color idx {result_idx}, style {linestyle}")
            ax.plot(slots, smoothed, label=label, linewidth=2.5,
                   color=colors[result_idx], linestyle=linestyle, alpha=0.9)

        ax.set_xlabel('Slot', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()} Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='best', framealpha=0.95, fontsize=10, edgecolor='black', fancybox=True)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        print(f"  [DEBUG] Plotted {len(results)} lines for metric '{metric}'")

    plt.tight_layout()

    if save_plots:
        # Create descriptive filename with experiment names
        exp_names = "_vs_".join([r.config.name[:15] for r in results[:3]])  # Limit to first 3 names
        if len(results) > 3:
            exp_names += f"_and_{len(results)-3}_more"
        filename = f'comparison_{exp_names}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to: {filename}")

    plt.show()


def plot_experiment_details(result: ExperimentResult, save_plots: bool = True):
    """
    Plot detailed time-series for a single experiment.

    Creates separate plots for:
    - Region selection over time (stacked area or line plot)
    - Source selection over time
    - Proposer distribution over time
    - Diversity metrics (Gini, Entropy)
    - Rewards

    Args:
        result: ExperimentResult object
        save_plots: Whether to save plots to disk
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Cannot create plots.")
        return

    print(f"\n[DEBUG] Plotting details for experiment: {result.config.name}")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # Define grid: 3 rows, 2 columns
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    slots = np.arange(len(result.region_counts))

    # Get dimensions
    n_regions = result.config.n_regions
    n_sources = len(result.config.sources_config)

    # 1. Region Selection Over Time (stacked area)
    ax1 = fig.add_subplot(gs[0, 0])
    region_counts_T = result.region_counts.T  # Transpose for stacking
    colors_regions = plt.cm.tab10(np.linspace(0, 1, n_regions))

    ax1.stackplot(slots, *region_counts_T,
                  labels=[result.config.region_names[i] for i in range(n_regions)],
                  colors=colors_regions, alpha=0.7)
    ax1.set_xlabel('Slot', fontsize=10)
    ax1.set_ylabel('Number of Proposers', fontsize=10)
    ax1.set_title('Region Selection Per Slot', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # 2. Source Selection Over Time (stacked area)
    ax2 = fig.add_subplot(gs[0, 1])
    source_counts_T = result.source_counts.T

    # Get source names from config
    source_names = [f"{src[0]} (V={src[1]})" for src in result.config.sources_config]
    colors_sources = plt.cm.Set2(np.linspace(0, 1, n_sources))

    ax2.stackplot(slots, *source_counts_T,
                  labels=source_names,
                  colors=colors_sources, alpha=0.7)
    ax2.set_xlabel('Slot', fontsize=10)
    ax2.set_ylabel('Number of Proposers', fontsize=10)
    ax2.set_title('Source Selection Per Slot', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # 3. Proposer Distribution Over Time (stacked area)
    ax3 = fig.add_subplot(gs[1, 0])
    proposer_dist_T = result.proposer_distribution.T

    ax3.stackplot(slots, *proposer_dist_T,
                  labels=[result.config.region_names[i] for i in range(n_regions)],
                  colors=colors_regions, alpha=0.7)
    ax3.set_xlabel('Slot', fontsize=10)
    ax3.set_ylabel('Number of Proposers', fontsize=10)
    ax3.set_title('Proposer Distribution (All Proposers)', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    # 4. Diversity Metrics - Gini
    ax4 = fig.add_subplot(gs[1, 1])

    # Smooth data
    window = min(100, len(result.proposer_dist_gini_over_time) // 10)
    if window > 1:
        smooth_prop_gini = np.convolve(result.proposer_dist_gini_over_time,
                                       np.ones(window)/window, mode='valid')
        smooth_region_gini = np.convolve(result.region_gini_over_time,
                                         np.ones(window)/window, mode='valid')
        smooth_source_gini = np.convolve(result.source_gini_over_time,
                                         np.ones(window)/window, mode='valid')
        slots_smooth = np.arange(window-1, len(slots))
    else:
        smooth_prop_gini = result.proposer_dist_gini_over_time
        smooth_region_gini = result.region_gini_over_time
        smooth_source_gini = result.source_gini_over_time
        slots_smooth = slots

    ax4.plot(slots_smooth, smooth_prop_gini, label='Proposer Distribution', linewidth=2, alpha=0.8)
    ax4.plot(slots_smooth, smooth_region_gini, label='Region Selection', linewidth=2, alpha=0.8)
    ax4.plot(slots_smooth, smooth_source_gini, label='Source Selection', linewidth=2, alpha=0.8)
    ax4.set_xlabel('Slot', fontsize=10)
    ax4.set_ylabel('Gini Coefficient', fontsize=10)
    ax4.set_title('Gini (Inequality) Over Time', fontsize=11, fontweight='bold')
    ax4.legend(loc='best', fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3)

    # 5. Diversity Metrics - Entropy
    ax5 = fig.add_subplot(gs[2, 0])

    if window > 1:
        smooth_prop_ent = np.convolve(result.proposer_dist_entropy_over_time,
                                      np.ones(window)/window, mode='valid')
        smooth_region_ent = np.convolve(result.region_entropy_over_time,
                                        np.ones(window)/window, mode='valid')
        smooth_source_ent = np.convolve(result.source_entropy_over_time,
                                        np.ones(window)/window, mode='valid')
    else:
        smooth_prop_ent = result.proposer_dist_entropy_over_time
        smooth_region_ent = result.region_entropy_over_time
        smooth_source_ent = result.source_entropy_over_time

    ax5.plot(slots_smooth, smooth_prop_ent, label='Proposer Distribution', linewidth=2, alpha=0.8)
    ax5.plot(slots_smooth, smooth_region_ent, label='Region Selection', linewidth=2, alpha=0.8)
    ax5.plot(slots_smooth, smooth_source_ent, label='Source Selection', linewidth=2, alpha=0.8)
    ax5.set_xlabel('Slot', fontsize=10)
    ax5.set_ylabel('Normalized Entropy', fontsize=10)
    ax5.set_title('Entropy (Diversity) Over Time', fontsize=11, fontweight='bold')
    ax5.legend(loc='best', fontsize=9, framealpha=0.9)
    ax5.grid(True, alpha=0.3)

    # 6. Average Reward Over Time
    ax6 = fig.add_subplot(gs[2, 1])

    if window > 1:
        smooth_rewards = np.convolve(result.rewards, np.ones(window)/window, mode='valid')
    else:
        smooth_rewards = result.rewards

    ax6.plot(slots_smooth, smooth_rewards, linewidth=2, color='darkgreen', alpha=0.8)
    ax6.set_xlabel('Slot', fontsize=10)
    ax6.set_ylabel('Average Reward', fontsize=10)
    ax6.set_title('Average Reward Per Proposer', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Main title
    policy_info = result.config.policy_type
    if result.config.policy_type == "EMA":
        policy_info += f" (η={result.config.eta}, β_reg={result.config.beta_reg}, β_src={result.config.beta_src}, c={result.config.cost_c})"
    else:
        policy_info += f" (α={result.config.alpha})"

    fig.suptitle(f'Experiment: {result.config.name} | {policy_info}',
                 fontsize=13, fontweight='bold')

    if save_plots:
        results_dir = Path(result.config.results_dir)
        results_dir.mkdir(exist_ok=True)
        filename = results_dir / f"{result.config.name}_details.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Detail plot saved to: {filename}")

    plt.show()


def print_comparison_table(results: List[ExperimentResult]):
    """Print a comparison table of final metrics."""
    print(f"\n{'='*120}")
    print("EXPERIMENT COMPARISON TABLE")
    print(f"{'='*120}")

    # Header
    print(f"\n{'Experiment':<20} {'Policy':<8} {'Reward':<10} {'PropDist':<10} {'PropDist':<10} "
          f"{'Region':<10} {'Region':<10} {'Source':<10} {'Source':<10}")
    print(f"{'':20} {'':8} {'':10} {'Gini':<10} {'Entropy':<10} "
          f"{'Gini':<10} {'Entropy':<10} {'Gini':<10} {'Entropy':<10}")
    print("-" * 120)

    # Data rows
    for result in results:
        stats = result.stats
        print(f"{result.config.name:<20} {result.config.policy_type:<8} "
              f"{stats['avg_reward']:<10.4f} {stats['proposer_dist_gini']:<10.4f} "
              f"{stats['proposer_dist_entropy']:<10.4f} {stats['region_gini']:<10.4f} "
              f"{stats['region_entropy']:<10.4f} {stats['source_gini']:<10.4f} "
              f"{stats['source_entropy']:<10.4f}")

    print("="*120)


# Example usage: See my_experiments.py for how to use this framework
