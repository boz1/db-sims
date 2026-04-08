#!/usr/bin/env python3
"""
Experiment A: Symmetric Baseline
"""
import copy
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import load_config
from analysis.experiment_runner import run_experiment
from analysis.poa import compute_poa_stats

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = [
    ROOT/"configs/ABR/sym_flat_equal.yaml",
    ROOT/"configs/ABR/sym_flat_crowded.yaml",
    ROOT/"configs/ABR/sym_linear_equal.yaml",
    ROOT/"configs/ABR/sym_linear_crowded.yaml",
]

PLACEMENTS = ["dispersed", "concentrated", "random_1", "random_2", "random_3"]
RANDOM_PLACEMENT_SEEDS = {"random_1": 1, "random_2": 2, "random_3": 3}

RESULTS_DIR = ROOT / "results/symmetric_baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def convergence_slot(volatility, threshold: float = 1e-6) -> int:
    """First slot where builder distribution stops changing."""
    v = np.array(volatility)
    for i in range(len(v)):
        if v[i] <= threshold and np.all(v[i:] <= threshold):
            return i
    return len(v)


def run_all():
    table_rows = []

    for config_path in CONFIGS:
        base_config = load_config(config_path)
        placement_results = {}  # placement -> list of results across runs

        for placement in PLACEMENTS:
            config = copy.deepcopy(base_config)
            config.initial_placement = "random" if placement in RANDOM_PLACEMENT_SEEDS else placement
            config.placement_seed = RANDOM_PLACEMENT_SEEDS.get(placement, 0)
            config.save_results = False

            print(f"\n{'='*60}")
            print(f"{config.name} | placement={placement}")
            print(f"{'='*60}")

            result = run_experiment(config, verbose=False, compute_poa=False)

            # Compute PoA on the worst-welfare result
            poa_stats = compute_poa_stats(result, method='greedy')

            conv = convergence_slot(result.builder_dist_volatility_over_time)
            final_dist = result.region_counts[-1].astype(int).tolist()
            location_hhi = result.location_hhi_over_time[-1]
            utility_hhi = result.stats['utility_hhi']

            print(f"PoA:          {poa_stats['poa']:.4f}")
            print(f"Location HHI: {location_hhi:.4f}")
            print(f"Utility HHI:  {utility_hhi:.4f}")
            print(f"Converged:    slot {conv}")
            print(f"Final dist:   {final_dist}")

            placement_results[placement] = result
            table_rows.append({
                "config": config.name,
                "placement": placement,
                "poa": poa_stats['poa'],
                "w_star": poa_stats['w_star'],
                "w_converged": poa_stats['w_converged'],
                "location_hhi": location_hhi,
                "utility_hhi": utility_hhi,
                "convergence_slot": conv,
                "final_dist": final_dist,
            })

        _plot_config(base_config, placement_results)

    _print_table(table_rows)


def _plot_config(config, placement_results: dict):
    """One figure per config: Gini + HHI + region occupancy per builder placement."""
    n_placements = len(PLACEMENTS)
    fig = plt.figure(figsize=(18, 4 * (n_placements + 2)))
    fig.suptitle(f"{config.name} — Initialisation Comparison", fontsize=13, fontweight='bold')

    colors = {"dispersed": "#2196F3", "concentrated": "#F44336",
              "random_1": "#FF9800", "random_2": "#4CAF50", "random_3": "#9C27B0"}

    # Top panel: Gini
    ax_gini = fig.add_subplot(n_placements + 2, 1, 1)
    for placement, result in placement_results.items():
        ax_gini.plot(result.location_gini_over_time,
                     label=f"{placement} (location)", color=colors[placement], linewidth=2, alpha=0.85)
        ax_gini.plot(result.utility_gini_over_time,
                     color=colors[placement], linewidth=1.5, alpha=0.6, linestyle='--')
    ax_gini.set_title("Gini — Location (solid) vs Utility (dashed)", fontsize=11)
    ax_gini.set_xlabel("Slot")
    ax_gini.set_ylabel("Gini")
    ax_gini.legend()
    ax_gini.grid(True, alpha=0.3)

    # Second panel: HHI
    ax_hhi = fig.add_subplot(n_placements + 2, 1, 2)
    for placement, result in placement_results.items():
        ax_hhi.plot(result.location_hhi_over_time,
                    label=f"{placement} (location)", color=colors[placement], linewidth=2, alpha=0.85)
        ax_hhi.plot(result.utility_hhi_over_time,
                    color=colors[placement], linewidth=1.5, alpha=0.6, linestyle='--')
    ax_hhi.set_title("HHI — Location (solid) vs Utility (dashed)", fontsize=11)
    ax_hhi.set_xlabel("Slot")
    ax_hhi.set_ylabel("HHI")
    ax_hhi.legend()
    ax_hhi.grid(True, alpha=0.3)

    # One row per placement: region occupancy heatmap
    for row_idx, placement in enumerate(PLACEMENTS):
        result = placement_results[placement]
        ax = fig.add_subplot(n_placements + 2, 1, row_idx + 3)
        region_counts_matrix = np.array(result.region_counts)  # (slots, regions)
        im = ax.imshow(region_counts_matrix.T, aspect='auto', interpolation='nearest',
                       cmap='YlOrRd', vmin=0)
        ax.set_title(f"Region occupancy — {placement}", fontsize=10)
        ax.set_xlabel("Slot")
        ax.set_ylabel("Region")
        ax.set_yticks(range(config.n_regions))
        ax.set_yticklabels(config.region_names, fontsize=7)
        plt.colorbar(im, ax=ax, label='Builders', shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = RESULTS_DIR / f"{config.name}_init_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {path}")


def _print_table(rows):
    print(f"\n\n{'='*160}")
    print("EXPERIMENT A — SYMMETRIC BASELINE SUMMARY")
    print(f"{'='*160}")
    header = (f"{'Config':<25} {'Placement':<14} {'PoA':>8} {'Loc.HHI':>10} {'Util.HHI':>10} "
              f"{'Conv.slot':>10} {'W*':>14} {'W_converged':>14}  Final distribution")
    print(header)
    print("-" * 160)
    for r in rows:
        print(f"{r['config']:<25} {r['placement']:<14} {r['poa']:>8.4f} "
              f"{r['location_hhi']:>10.4f} {r['utility_hhi']:>10.4f} "
              f"{r['convergence_slot']:>10} {r['w_star']:>14.6f} {r['w_converged']:>14.6f}  "
              f"{r['final_dist']}")
    print("=" * 160)


if __name__ == "__main__":
    run_all()
