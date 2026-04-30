# scripts/exp_poa_vs_n_dominant_geo.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
Path("figures").mkdir(exist_ok=True)

"""
Experiment: PoA vs number of dominant sources, varying geographic distribution.

For each n_dom in {1, 2, ..., M}, we test three different choices of which
sources are dominant:
- "clustered": dominant sources are all in one continental cluster (eg US).
- "spread": dominant sources are spread globally (one per continent).
- "random": dominant sources are randomly chosen, averaged across seeds.
"""
import numpy as np
import matplotlib.pyplot as plt

from sim.datasets import load_gcp, subregion
from sim.simulator import Source, FixedLatencyPropagationModel
from analysis.poa import optimal_welfare_greedy
from scripts.poa_experiment_helpers import run_abr


# 10 source regions, organised so we can pick clustered vs spread subsets
SOURCE_REGIONS = [
    "us-east1", "us-west1", "us-central1",  # 0, 1, 2: US cluster
    "europe-west1", "europe-north1", # 3, 4: EU cluster
    "asia-northeast1", "asia-southeast1", "asia-south1", # 5, 6, 7: Asia cluster
    "australia-southeast1", # 8: Australia
    "southamerica-east1", # 9: South America
]

# Geographic "slot" ordering for spread selection - take one per area
SPREAD_ORDER = [0, 3, 5, 8, 9, 1, 4, 6, 2, 7]  # US, EU, Asia, AUS, SA, then second-tier

# "Clustered" picks: nearby regions first
CLUSTERED_ORDER = [0, 2, 1, 3, 4, 5, 6, 7, 8, 9]  # 3 US, 2 EU, 3 Asia, AUS, SA

EXTRA_REGIONS = [
    "us-east4", "europe-west3", "europe-west4", "europe-southwest1",
    "europe-central2", "australia-southeast2", "africa-south1",
    "me-central1", "me-west1", "asia-east1",
]

REGIONS = list(dict.fromkeys(SOURCE_REGIONS + EXTRA_REGIONS))
M = len(SOURCE_REGIONS)
N_REGIONS = len(REGIONS)

K = M
DELTA_SLOT = 0.1
DELTA_MU = 0.001

LAMBDA_RATE = 4.28
MU_VAL_PERIPHERAL = -11.29
SIGMA_VAL = 1.86

N_RANDOM_SEEDS = 5

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def build_sources_with_dominant_set(M, dominant_indices, delta_mu,
                                     mu_peripheral=MU_VAL_PERIPHERAL,
                                     sigma_val=SIGMA_VAL,
                                     lambda_rate=LAMBDA_RATE):
    """Build M sources where the given indices are dominant."""
    n_dominant = len(dominant_indices)
    if n_dominant == 0:
        # all peripheral
        mus = [mu_peripheral] * M
    else:
        dominant_multiplier = (M / n_dominant) * (1 + delta_mu)
        mu_dominant = mu_peripheral + np.log(dominant_multiplier)
        dominant_set = set(dominant_indices)
        mus = [mu_dominant if i in dominant_set else mu_peripheral
               for i in range(M)]

    return [
        Source(id=i, name=f"src{i}", region=i,
               lambda_rate=lambda_rate, mu_val=mus[i], sigma_val=sigma_val)
        for i in range(M)
    ]


def evaluate_poa(K, sources, prop, delta_slot, init_placement="concentrated"):
    w_opt, _ = optimal_welfare_greedy(K, N_REGIONS, sources, prop, delta_slot)
    profile, w_abr = run_abr(K, sources, prop, REGIONS, delta_slot, init_placement)
    return w_opt / w_abr if w_abr > 1e-12 else float("inf"), profile


def main():
    all_regions, full_mean, _ = load_gcp(latency_std_fraction=0.15)
    _, latency_mean, _ = subregion(all_regions, full_mean, full_mean*0.15, REGIONS)
    prop = FixedLatencyPropagationModel(latency_mean)

    print(f"M = {M}, K = {K}, |R| = {N_REGIONS}, Delta = {DELTA_SLOT}s")
    print()

    n_dominant_grid = list(range(1, M + 1))

    poa_clustered = []
    poa_spread = []
    poa_random_mean = []
    poa_random_std = []

    for n_dom in n_dominant_grid:
        # Clustered: pick n_dom regions in CLUSTERED_ORDER
        clustered_set = CLUSTERED_ORDER[:n_dom]
        sources = build_sources_with_dominant_set(M, clustered_set, DELTA_MU)
        poa_c, _ = evaluate_poa(K, sources, prop, DELTA_SLOT)
        poa_clustered.append(poa_c)

        # Spread: pick n_dom regions in SPREAD_ORDER (one per area first)
        spread_set = SPREAD_ORDER[:n_dom]
        sources = build_sources_with_dominant_set(M, spread_set, DELTA_MU)
        poa_s, _ = evaluate_poa(K, sources, prop, DELTA_SLOT)
        poa_spread.append(poa_s)

        # Random: average over multiple random selections
        rng = np.random.default_rng(42)
        poa_random_seeds = []
        for _ in range(N_RANDOM_SEEDS):
            random_set = list(rng.choice(M, size=n_dom, replace=False))
            sources = build_sources_with_dominant_set(M, random_set, DELTA_MU)
            poa_r, _ = evaluate_poa(K, sources, prop, DELTA_SLOT)
            poa_random_seeds.append(poa_r)
        poa_random_mean.append(np.mean(poa_random_seeds))
        poa_random_std.append(np.std(poa_random_seeds))

        print(f"n_dom={n_dom:2d}  clustered={poa_clustered[-1]:.3f}  "
              f"spread={poa_spread[-1]:.3f}  "
              f"random={poa_random_mean[-1]:.3f} ± {poa_random_std[-1]:.3f}")
        print(f"     clustered set: {[SOURCE_REGIONS[i] for i in clustered_set]}")
        print(f"     spread set:    {[SOURCE_REGIONS[i] for i in spread_set]}")
        print()

    fig, ax = plt.subplots(figsize=(8, 5))

    poa_random_mean = np.array(poa_random_mean)
    poa_random_std = np.array(poa_random_std)

    ax.plot(n_dominant_grid, poa_clustered, "-", linewidth=2, color="#27ae60",
            label="Clustered (US/EU first)")
    ax.plot(n_dominant_grid, poa_spread, "-", linewidth=2, color="#c0392b",
            label="Spread (one per continent first)")
    ax.plot(n_dominant_grid, poa_random_mean, "-", linewidth=2, color="#2c3e50",
            label=f"Random (mean of {N_RANDOM_SEEDS} seeds)")
    ax.fill_between(
        n_dominant_grid,
        poa_random_mean - poa_random_std,
        poa_random_mean + poa_random_std,
        alpha=0.2, color="#2c3e50",
    )

    ax.axhline(1.0, ls=":", color="black", alpha=0.5, linewidth=1)

    ax.set_xlabel("Number of dominant sources $n_{\\rm dom}$")
    ax.set_ylabel("Empirical PoA")
    ax.set_title(rf"PoA vs. $n_{{\rm dom}}$ across geographic distributions "
                 rf"($K=M={M}$, $\Delta={DELTA_SLOT}$s, fixed total dominant value)")
    ax.set_xticks(n_dominant_grid)
    ax.set_ylim(0.95, 2.05)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/exp_poa_vs_n_dominant_geo.pdf", bbox_inches="tight")
    fig.savefig("figures/exp_poa_vs_n_dominant_geo.png", dpi=200, bbox_inches="tight")
    print("Saved figures/exp_poa_vs_n_dominant_geo.pdf and .png")


if __name__ == "__main__":
    main()
