# scripts/poa_experiment_helpers.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import numpy as np

from sim.datasets import load_gcp, subregion
from sim.simulator import (
    Source, Builder, Region, FixedLatencyPropagationModel,
    StochasticTransactionGenerator, EqualSplitSharingRule,
    LocationGamesSimulator, FixedPolicy,
)
from analysis.poa import optimal_welfare_brute_force, optimal_welfare_greedy, _compute_welfare_analytical


REGIONS_DEFAULT = [
    # US/EU high-value cluster
    "us-east1", "us-east4", "us-central1", "us-west1",
    "europe-west1", "europe-west3", "europe-west4", "europe-north1",
    # Asia/Pacific/SA/Africa distant cluster
    "asia-northeast1", "asia-southeast1", "asia-south1",
    "australia-southeast1", "australia-southeast2",
    "southamerica-east1", "southamerica-west1",
    "africa-south1",
    # Intermediate
    "me-central1", "me-west1", "europe-southwest1", "europe-central2",
]

HIGH_VALUE_SOURCE_REGIONS_DEFAULT = [
    "us-east1", "us-east4", "europe-west1", "europe-west3", "europe-west4",
]
DISTANT_SOURCE_REGIONS_DEFAULT = [
    "asia-northeast1", "asia-southeast1", "australia-southeast1",
    "southamerica-east1", "africa-south1",
]


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_propagation_model(regions=REGIONS_DEFAULT, latency_std_fraction=0.15):
    """Returns (regions, propagation_model, region_index_map)."""
    all_regions, full_mean, full_std = load_gcp(latency_std_fraction)
    _, latency_mean, _ = subregion(all_regions, full_mean, full_std, regions)
    prop = FixedLatencyPropagationModel(latency_mean)
    region_index_map = {r: i for i, r in enumerate(regions)}
    return regions, prop, region_index_map


def build_two_cluster_sources(alpha, total_value, region_index_map,
                              high_value_regions=HIGH_VALUE_SOURCE_REGIONS_DEFAULT,
                              distant_regions=DISTANT_SOURCE_REGIONS_DEFAULT,
                              lambda_rate=1.0, sigma_val=0.0):
    """
    Build sources split between a high-value cluster and a distant cluster.

    alpha: fraction of total_value in the high-value cluster.
    Each cluster's value is divided equally among its sources.
    """
    n_high = len(high_value_regions)
    n_distant = len(distant_regions)
    value_per_high = total_value * alpha / n_high
    value_per_distant = total_value * (1 - alpha) / n_distant

    sources = []
    sid = 0
    for r in high_value_regions:
        sources.append(Source(
            id=sid, name=f"src_{r}", region=region_index_map[r],
            lambda_rate=lambda_rate, mu_val=np.log(value_per_high),
            sigma_val=sigma_val,
        ))
        sid += 1
    for r in distant_regions:
        sources.append(Source(
            id=sid, name=f"src_{r}", region=region_index_map[r],
            lambda_rate=lambda_rate, mu_val=np.log(value_per_distant),
            sigma_val=sigma_val,
        ))
        sid += 1
    return sources


def compute_optimal(K, n_regions, sources, prop, delta, n_t=200,
                    method="auto", max_brute=100_000):
    """Auto-select brute-force or greedy depending on problem size."""
    if method == "greedy":
        return optimal_welfare_greedy(K, n_regions, sources, prop, delta, n_t)
    if method == "brute":
        return optimal_welfare_brute_force(K, n_regions, sources, prop, delta, n_t)

    from math import comb
    n_profiles = comb(n_regions + K - 1, K)
    if n_profiles <= max_brute:
        return optimal_welfare_brute_force(K, n_regions, sources, prop, delta, n_t)
    return optimal_welfare_greedy(K, n_regions, sources, prop, delta, n_t)

def run_abr(K, sources, prop, regions, delta, init_placement,
            n_slots=200, n_t=200, seed=42):
    """Run ABR to convergence; returns (final_profile, converged_welfare)."""
    n_regions = len(regions)
    regions_list = [Region(id=i, name=regions[i]) for i in range(n_regions)]
    builders = [Builder(id=i, policy=FixedPolicy(n_regions)) for i in range(K)]
    sim = LocationGamesSimulator(
        regions=regions_list, sources=sources, builders=builders,
        tx_generator=StochasticTransactionGenerator(),
        propagation_model=prop, sharing_rule=EqualSplitSharingRule(),
        delta=delta, seed=seed, initial_placement=init_placement,
    )
    sim.run_abr(n_slots=n_slots, n_t=n_t)
    final_profile = [b.current_region for b in sim.builders]
    w_converged = _compute_welfare_analytical(
        final_profile, sources, prop, delta, n_t,
    )
    return final_profile, w_converged


def evaluate_instance(K, sources, prop, regions, delta, n_slots=200, n_t=200,
                      opt_method="auto"):
    """Run ABR from both inits, compute optimum, return PoA for each."""
    n_regions = len(regions)
    w_opt, opt_profile = compute_optimal(
        K, n_regions, sources, prop, delta, n_t, method=opt_method,
    )
    profile_c, w_c = run_abr(K, sources, prop, regions, delta,
                             init_placement="concentrated",
                             n_slots=n_slots, n_t=n_t)
    profile_d, w_d = run_abr(K, sources, prop, regions, delta,
                             init_placement="dispersed",
                             n_slots=n_slots, n_t=n_t)
    return {
        "w_opt": w_opt, "opt_profile": opt_profile,
        "w_concentrated": w_c, "profile_concentrated": profile_c,
        "w_dispersed": w_d, "profile_dispersed": profile_d,
        "poa_concentrated": w_opt / w_c if w_c > 1e-12 else float("inf"),
        "poa_dispersed": w_opt / w_d if w_d > 1e-12 else float("inf"),
    }


def plot_poa_curve(x, poa_concentrated, poa_dispersed, K, x_label, title,
                   filename, log_x=False):
    """Standard two-line PoA plot with theoretical bound annotation."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, poa_concentrated, "o-", color="#d62728",
            label="ABR from concentrated init")
    ax.plot(x, poa_dispersed, "s-", color="#1f77b4",
            label="ABR from dispersed init")
    ax.axhline(2 - 1/K, ls="--", color="gray",
               label=f"Theoretical bound: $2 - 1/K = {2 - 1/K:.2f}$")
    ax.axhline(1.0, ls=":", color="black", alpha=0.5)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Empirical PoA")
    ax.set_title(title)
    ax.set_ylim(0.95, 2 - 1/K + 0.1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    print(f"Saved {FIGURES_DIR / filename}")
