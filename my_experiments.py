#!/usr/bin/env python3
"""
My Experiments - Simple template for running custom experiments.

Edit the configurations below and run this script to perform your experiments.
"""

from experiment_runner import (
    ExperimentConfig,
    run_experiment,
    compare_experiments,
    print_comparison_table
)


# ============================================================================
# CONFIGURE YOUR EXPERIMENTS HERE
# ============================================================================

def define_experiments():
    """Define all experiments you want to run."""

    experiments = []

    # Experiment 1: Baseline EMA
    experiments.append(ExperimentConfig(
        name="baseline_ema",

        # Regions
        n_regions=5,
        region_names=["West", "CentralWest", "Central", "CentralEast", "East"],

        # Sources: (name, value, home_region)
        sources_config=[
            ("LowValue", 8.0, 0),
            ("MedValue", 12.0, 2),
            ("HighValue", 18.0, 4)
        ],

        # Policy
        policy_type="EMA",
        eta=0.12,
        beta_reg=1.5,
        beta_src=2.5,
        cost_c=0.2,

        # Simulation
        n_proposers=80,
        K=8,
        n_slots=10000,
        seed=42
    ))

    # Experiment 2: High exploration EMA
    experiments.append(ExperimentConfig(
        name="high_exploration_ema",
        n_regions=5,
        region_names=["West", "CentralWest", "Central", "CentralEast", "East"],
        sources_config=[
            ("LowValue", 8.0, 0),
            ("MedValue", 12.0, 2),
            ("HighValue", 18.0, 4)
        ],
        policy_type="EMA",
        eta=0.12,
        beta_reg=1.0,  # Lower beta = more exploration
        beta_src=1.5,
        cost_c=0.2,
        n_proposers=80,
        K=8,
        n_slots=10000,
        seed=42
    ))

    # Experiment 3: UCB policy
    experiments.append(ExperimentConfig(
        name="ucb_baseline",
        n_regions=5,
        region_names=["West", "CentralWest", "Central", "CentralEast", "East"],
        sources_config=[
            ("LowValue", 8.0, 0),
            ("MedValue", 12.0, 2),
            ("HighValue", 18.0, 4)
        ],
        policy_type="UCB",
        alpha=2.0,
        n_proposers=80,
        K=8,
        n_slots=10000,
        seed=42
    ))

    # Experiment 4: High migration cost
    experiments.append(ExperimentConfig(
        name="high_migration_cost",
        n_regions=5,
        region_names=["West", "CentralWest", "Central", "CentralEast", "East"],
        sources_config=[
            ("LowValue", 8.0, 0),
            ("MedValue", 12.0, 2),
            ("HighValue", 18.0, 4)
        ],
        policy_type="EMA",
        eta=0.12,
        beta_reg=1.5,
        beta_src=2.5,
        cost_c=1.0,  # Higher migration cost
        n_proposers=80,
        K=8,
        n_slots=10000,
        seed=42
    ))

    return experiments


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all experiments and compare results."""

    print("="*70)
    print("Running Custom Experiments")
    print("="*70)

    # Define experiments
    configs = define_experiments()

    # Run all experiments
    results = []
    for config in configs:
        result = run_experiment(config, verbose=True)
        results.append(result)
        print(f"✓ Completed: {config.name}")

    # Print comparison table
    print("\n\n")
    print(f"Total experiments completed: {len(results)}")
    print(f"Experiment names: {[r.config.name for r in results]}")
    print_comparison_table(results)

    # Plot comparisons
    print("\n\nGenerating comparison plots...")
    print(f"Plotting {len(results)} experiments: {[r.config.name for r in results]}")

    # Compare proposer distribution and reward metrics
    compare_experiments(
        results,
        metrics=['proposer_dist_gini', 'proposer_dist_entropy', 'reward'],
        save_plots=True
    )

    print("\n" + "="*70)
    print("All experiments complete!")
    print(f"Results saved to: experiment_results/")
    print("="*70)


if __name__ == "__main__":
    main()
