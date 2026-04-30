# scripts/exp_poa_vs_K_fixed_M.py
import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

"""
Experiment: PoA vs number of builders K, with fixed source count

Setup: M = 5 sources at 5 well-separated GCP regions plus 15 non-source
regions (20 total). One dominant source with Lambda_0 = M*(1+delta_mu),
peripherals with Lambda = 1.

Two learning algorithms (both from dispersed init):
  - ABR: single run per K.
  - EXP3: N_SEEDS runs per K, time-averaged welfare with burn-in, plotted with mean +/- 1 std band.

"""
import numpy as np
import matplotlib.pyplot as plt

from sim.datasets import load_gcp, subregion
from sim.simulator import (
    Source, Builder, Region, FixedLatencyPropagationModel,
    StochasticTransactionGenerator, EqualSplitSharingRule,
    LocationGamesSimulator, EXP3Policy,
)
from scripts.poa_experiment_helpers import run_abr
from analysis.poa import (
    _compute_welfare_analytical,
    optimal_welfare_brute_force,
    optimal_welfare_greedy,
)


SOURCE_REGIONS = [
    "us-east1", "us-west1", "europe-west1", "asia-northeast1", "australia-southeast1",
]

EXTRA_REGIONS = [
    "us-east4", "us-central1", "europe-west3", "europe-west4", "europe-north1",
    "europe-southwest1", "europe-central2",
    "asia-southeast1", "asia-south1", "australia-southeast2",
    "southamerica-east1", "africa-south1",
    "me-central1", "me-west1",
    "northamerica-northeast1",
]


def build_region_pool():
    seen = set()
    pool = []
    for r in SOURCE_REGIONS + EXTRA_REGIONS:
        if r not in seen:
            seen.add(r)
            pool.append(r)
    return pool


REGIONS = build_region_pool()
M = len(SOURCE_REGIONS)
N_REGIONS = len(REGIONS)

DELTA_LATENCY_RATIO = 0.99
DELTA_MU = 0.001

LAMBDA_RATE = 4.28
MU_VAL_PERIPHERAL = -11.29
SIGMA_VAL = 1.86

K_MIN = 2
K_MAX = 12
GREEDY_THRESHOLD = 10

ETA = 0.05
GAMMA = 0.5
GAMMA_MIN = 0.002
GAMMA_DECAY = 0.0005
GAMMA_SCHEDULE = "exponential"
NORM_ALPHA = 0.01
N_SLOTS = 15000
N_T_EXP3 = 100
BURN_IN_FRACTION = 0.05
N_SEEDS = 5

INIT_PLACEMENT = "dispersed"

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / f"exp_poa_vs_K_fixed_M"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def min_pairwise_latency(latency_mean):
    n = latency_mean.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.min(latency_mean[mask]))


def build_sources(M, delta_mu, mu_peripheral=MU_VAL_PERIPHERAL,
                  sigma_val=SIGMA_VAL, lambda_rate=LAMBDA_RATE):
    mu_0 = mu_peripheral + np.log(M * (1 + delta_mu))
    mus = [mu_0] + [mu_peripheral] * (M - 1)
    return [
        Source(id=i, name=f"src{i}", region=i,
               lambda_rate=lambda_rate, mu_val=mus[i], sigma_val=sigma_val)
        for i in range(M)
    ]


def initial_belief_estimate(sources, delta):
    return sum(
        s.lambda_rate * delta * np.exp(s.mu_val + s.sigma_val ** 2 / 2)
        for s in sources
    )


def compute_optimum(K, n_regions, sources, prop, delta, n_t=200):
    if K >= GREEDY_THRESHOLD:
        return optimal_welfare_greedy(K, n_regions, sources, prop, delta, n_t)
    return optimal_welfare_brute_force(K, n_regions, sources, prop, delta, n_t)



def run_exp3_instance(K, n_regions, sources, prop, delta_slot, init_placement,
                      n_slots=N_SLOTS, n_t=N_T_EXP3, seed=42):
    regions = [Region(id=i, name=REGIONS[i]) for i in range(n_regions)]
    initial_belief = initial_belief_estimate(sources, delta_slot)

    builders = [
        Builder(id=i, policy=EXP3Policy(
            n_regions=n_regions,
            eta=ETA,
            gamma=GAMMA,
            gamma_min=GAMMA_MIN,
            gamma_decay=GAMMA_DECAY,
            gamma_schedule=GAMMA_SCHEDULE,
            norm_alpha=NORM_ALPHA,
            initial_belief=initial_belief,
        ))
        for i in range(K)
    ]

    sim = LocationGamesSimulator(
        regions=regions, sources=sources, builders=builders,
        tx_generator=StochasticTransactionGenerator(),
        propagation_model=prop, sharing_rule=EqualSplitSharingRule(),
        delta=delta_slot, seed=seed, initial_placement=init_placement,
    )
    sim.run(n_slots)

    burn_in = int(n_slots * BURN_IN_FRACTION)
    per_round_welfares = []
    per_round_profiles = []
    for region_counts in sim.region_counts_history[burn_in:]:
        profile = [
            r for r, count in enumerate(region_counts.astype(int))
            for _ in range(int(count))
        ]
        per_round_profiles.append(profile)
        per_round_welfares.append(
            _compute_welfare_analytical(profile, sources, prop, delta_slot, n_t)
        )
    w_avg = float(np.mean(per_round_welfares))

    return {
        "w_avg": w_avg,
        "burn_in": burn_in,
        "n_reporting_rounds": len(per_round_welfares),
        "per_round_welfares": per_round_welfares,
        "per_round_profiles": per_round_profiles,
    }


def mode_profile(profiles):
    """Return the most frequent profile (as sorted tuple) and its count."""
    profile_keys = [tuple(sorted(p)) for p in profiles]
    counter = Counter(profile_keys)
    most_common, count = counter.most_common(1)[0]
    return list(most_common), count


def region_frequency(profiles, n_regions):
    """Count how many builder-rounds were spent at each region.
    Returns array of length n_regions with absolute counts."""
    freq = np.zeros(n_regions, dtype=int)
    for p in profiles:
        for r in p:
            freq[r] += 1
    return freq


def save_run_data(filepath, data):
    """Save run data as JSON. Convert numpy types to native Python."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj

    with open(filepath, "w") as f:
        json.dump(convert(data), f, indent=2)


def main():
    all_regions, full_mean, _ = load_gcp(latency_std_fraction=0.15)
    _, latency_mean, _ = subregion(all_regions, full_mean, full_mean*0.15, REGIONS)
    prop = FixedLatencyPropagationModel(latency_mean)

    min_lat = min_pairwise_latency(latency_mean)
    delta_slot = DELTA_LATENCY_RATIO * min_lat
    sources = build_sources(M, DELTA_MU)

    metadata = {
        "config": {
            "source_regions": SOURCE_REGIONS,
            "regions": REGIONS,
            "M": M,
            "N_REGIONS": N_REGIONS,
            "delta_latency_ratio": DELTA_LATENCY_RATIO,
            "min_pairwise_latency_s": min_lat,
            "delta_slot_s": delta_slot,
            "delta_mu": DELTA_MU,
            "lambda_rate": LAMBDA_RATE,
            "mu_val_peripheral": MU_VAL_PERIPHERAL,
            "sigma_val": SIGMA_VAL,
            "K_min": K_MIN,
            "K_max": K_MAX,
            "greedy_threshold": GREEDY_THRESHOLD,
            "exp3_eta": ETA,
            "exp3_gamma": GAMMA,
            "exp3_gamma_min": GAMMA_MIN,
            "exp3_gamma_decay": GAMMA_DECAY,
            "exp3_gamma_schedule": GAMMA_SCHEDULE,
            "exp3_norm_alpha": NORM_ALPHA,
            "n_slots": N_SLOTS,
            "burn_in_fraction": BURN_IN_FRACTION,
            "n_seeds": N_SEEDS,
            "init_placement": INIT_PLACEMENT,
        },
    }
    save_run_data(RESULTS_DIR / "metadata.json", metadata)

    print(f"M = {M} sources at: {SOURCE_REGIONS}")
    print(f"|R| = {N_REGIONS} regions")
    print(f"Min pairwise latency: {min_lat*1000:.1f} ms")
    print(f"Delta = {DELTA_LATENCY_RATIO} * min latency = {delta_slot*1000:.1f} ms")
    print(f"Init placement: {INIT_PLACEMENT}")
    print(f"EXP3: {N_SEEDS} seeds, {N_SLOTS} slots, burn-in = {BURN_IN_FRACTION:.0%}")
    print(f"Results dir: {RESULTS_DIR}")
    print()

    K_grid = list(range(K_MIN, K_MAX + 1))

    # Per-K aggregate results (for plotting)
    poa_abr = []
    poa_exp3_mean = []
    poa_exp3_std = []

    # Per-K profile summaries (for the summary file)
    summary_per_K = []

    for K in K_grid:
        opt_method = "brute" if K < GREEDY_THRESHOLD else "greedy"
        w_opt, opt_profile = compute_optimum(K, N_REGIONS, sources, prop, delta_slot)

        # ABR
        abr_profile, w_abr = run_abr(
            K, sources, prop, REGIONS, delta_slot, INIT_PLACEMENT,
        )
        poa_a = w_opt / w_abr if w_abr > 1e-12 else float("inf")
        poa_abr.append(poa_a)

        abr_run = {
            "algorithm": "ABR",
            "K": K,
            "w_opt": w_opt,
            "opt_profile": opt_profile,
            "opt_profile_names": [REGIONS[r] for r in opt_profile],
            "opt_method": opt_method,
            "init_placement": INIT_PLACEMENT,
            "w_converged": w_abr,
            "final_profile": abr_profile,
            "final_profile_names": [REGIONS[r] for r in abr_profile],
            "poa": poa_a,
        }

        # EXP3 (multi-seed)
        exp3_runs = []
        exp3_poas = []
        all_seed_profiles = []  # for global mode across all seeds
        per_seed_summary = []

        for seed in range(N_SEEDS):
            exp3_data = run_exp3_instance(
                K, N_REGIONS, sources, prop, delta_slot,
                init_placement=INIT_PLACEMENT, seed=seed,
            )
            poa_seed = (
                w_opt / exp3_data["w_avg"]
                if exp3_data["w_avg"] > 1e-12
                else float("inf")
            )
            exp3_poas.append(poa_seed)

            # Per-seed mode profile and frequency
            seed_mode_profile, seed_mode_count = mode_profile(
                exp3_data["per_round_profiles"],
            )
            seed_freq = region_frequency(
                exp3_data["per_round_profiles"], N_REGIONS,
            )
            n_rounds = exp3_data["n_reporting_rounds"]

            per_seed_summary.append({
                "seed": seed,
                "poa": poa_seed,
                "w_avg": exp3_data["w_avg"],
                "mode_profile": seed_mode_profile,
                "mode_profile_names": [REGIONS[r] for r in seed_mode_profile],
                "mode_count": seed_mode_count,
                "mode_fraction": seed_mode_count / n_rounds,
                "region_frequency_counts": seed_freq.tolist(),
                "region_frequency_names": {
                    REGIONS[r]: int(seed_freq[r])
                    for r in range(N_REGIONS) if seed_freq[r] > 0
                },
            })

            all_seed_profiles.extend(exp3_data["per_round_profiles"])

            exp3_runs.append({
                "algorithm": "EXP3",
                "K": K,
                "seed": seed,
                "w_opt": w_opt,
                "init_placement": INIT_PLACEMENT,
                "w_avg": exp3_data["w_avg"],
                "burn_in": exp3_data["burn_in"],
                "n_reporting_rounds": exp3_data["n_reporting_rounds"],
                "poa": poa_seed,
                "per_round_welfares": exp3_data["per_round_welfares"],
                "per_round_profiles": exp3_data["per_round_profiles"],
            })

        m = float(np.mean(exp3_poas))
        s = float(np.std(exp3_poas))
        poa_exp3_mean.append(m)
        poa_exp3_std.append(s)

        # Global mode and frequency across all EXP3 seeds
        global_mode_profile, global_mode_count = mode_profile(all_seed_profiles)
        global_freq = region_frequency(all_seed_profiles, N_REGIONS)
        total_rounds = len(all_seed_profiles)

        # Save full per-K data
        K_data = {
            "K": K,
            "w_opt": w_opt,
            "opt_profile": opt_profile,
            "opt_profile_names": [REGIONS[r] for r in opt_profile],
            "opt_method": opt_method,
            "abr": abr_run,
            "exp3_seeds": exp3_runs,
            "exp3_poa_mean": m,
            "exp3_poa_std": s,
        }
        save_run_data(RESULTS_DIR / f"K_{K:02d}.json", K_data)

        # Compact summary entry for this K
        summary_per_K.append({
            "K": K,
            "opt_profile": opt_profile,
            "opt_profile_names": [REGIONS[r] for r in opt_profile],
            "abr": {
                "poa": poa_a,
                "final_profile": abr_profile,
                "final_profile_names": [REGIONS[r] for r in abr_profile],
            },
            "exp3": {
                "poa_mean": m,
                "poa_std": s,
                "global_mode_profile": global_mode_profile,
                "global_mode_profile_names": [REGIONS[r] for r in global_mode_profile],
                "global_mode_count": global_mode_count,
                "global_mode_fraction": global_mode_count / total_rounds,
                "global_region_frequency_counts": global_freq.tolist(),
                "global_region_frequency_names": {
                    REGIONS[r]: int(global_freq[r])
                    for r in range(N_REGIONS) if global_freq[r] > 0
                },
                "per_seed": per_seed_summary,
            },
        })

        print(f"K={K:2d}  W*={w_opt:.5f} ({opt_method})  "
              f"ABR={poa_a:.3f}  EXP3={m:.3f}±{s:.3f}")
        print(f"     opt   = {[REGIONS[r] for r in opt_profile]}")
        print(f"     ABR   = {[REGIONS[r] for r in abr_profile]}")
        print(f"     EXP3 global mode = {[REGIONS[r] for r in global_mode_profile]} "
              f"({global_mode_count}/{total_rounds} = {global_mode_count/total_rounds:.1%})")
        print()

    summary = {
        "K_grid": K_grid,
        "poa_abr": poa_abr,
        "poa_exp3_mean": poa_exp3_mean,
        "poa_exp3_std": poa_exp3_std,
        "per_K": summary_per_K,
    }
    save_run_data(RESULTS_DIR / "summary.json", summary)

    poa_exp3_mean_arr = np.array(poa_exp3_mean)
    poa_exp3_std_arr = np.array(poa_exp3_std)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.plot(K_grid, poa_abr, "-", linewidth=2, color="#c0392b", label="ABR")
    ax.plot(K_grid, poa_exp3_mean_arr, "--", linewidth=2, color="#2c3e50",
            label=f"EXP3 (mean of {N_SEEDS} seeds)")
    ax.fill_between(
        K_grid,
        poa_exp3_mean_arr - poa_exp3_std_arr,
        poa_exp3_mean_arr + poa_exp3_std_arr,
        alpha=0.2, color="#2c3e50", linewidth=0,
    )

    ax.axhline(1.0, ls=":", color="black", alpha=0.5, linewidth=1)
    ax.axvline(M, ls=":", color="#27ae60", alpha=0.6, linewidth=1.5,
               label=f"$K = M = {M}$ (matched regime)")

    ax.set_xlabel("Number of builders $K$")
    ax.set_ylabel("Empirical PoA")
    ax.set_title(rf"PoA vs. $K$ at fixed $M={M}$ sources, $|\mathcal{{R}}|={N_REGIONS}$ regions "
                 rf"($\Delta = {DELTA_LATENCY_RATIO}\times$ min latency)")

    ymax = max(max(poa_abr), max(poa_exp3_mean_arr + poa_exp3_std_arr))
    ax.set_ylim(0.95, ymax * 1.05)
    ax.set_xticks(K_grid)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "plot.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "plot.png", dpi=200, bbox_inches="tight")
    fig.savefig("figures/exp_poa_vs_K_fixed_M.pdf", bbox_inches="tight")
    fig.savefig("figures/exp_poa_vs_K_fixed_M.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved plot to {RESULTS_DIR}/plot.pdf and figures/exp_poa_vs_K_fixed_M.pdf")
    print(f"All run data saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
