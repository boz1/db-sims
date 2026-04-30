# scripts/exp_poa_vs_delta_exp3.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Experiment: PoA vs slot duration Delta (EXP3 version)

Same worst-case construction as the ABR version (one dominant source of
K*Lambda mass, K-1 peripheral sources of Lambda each), but builders use EXP3
bandit learning instead of best response. Runs from concentrated and dispersed
initialisations, averaged across multiple seeds; reports time-averaged
expected welfare over last few rounds of each run (burn-in), plotting mean
PoA with a +/- 1 std shaded band.

EXP3 with bandit feedback converges to coarse correlated equilibria on
time-average rather than to pure NE in last-iterate, so the metric reported
here is W(s*) / W_avg where W_avg is the time-averaged expected welfare.
"""
import numpy as np
import matplotlib.pyplot as plt

from sim.datasets import load_gcp, subregion
from sim.simulator import (
    Source, Builder, Region, FixedLatencyPropagationModel,
    StochasticTransactionGenerator, EqualSplitSharingRule,
    LocationGamesSimulator, EXP3Policy,
)
from analysis.poa import _compute_welfare_analytical, optimal_welfare_brute_force
from scripts.poa_experiment_helpers import FIGURES_DIR


REGIONS = ["us-east1", "us-west1", "europe-west1", "asia-northeast1", "australia-southeast1"]
K = len(REGIONS)

# EXP3 hyperparameters
ETA = 0.05
GAMMA = 0.5
GAMMA_MIN = 0.002
GAMMA_DECAY = 0.0005
GAMMA_SCHEDULE = "exponential"
NORM_ALPHA = 0.01
N_SLOTS = 15000
N_T = 100
BURN_IN_FRACTION = 0.5

# Multi-seed sweep
N_SEEDS = 5
N_DELTA_POINTS = 100


def build_sources(K, delta_mu, mu_peripheral=0.0, sigma_val=0.0, lambda_rate=4.28):
    mu_0 = mu_peripheral + np.log(K * (1 + delta_mu))
    mus = [mu_0] + [mu_peripheral] * (K - 1)
    return [
        Source(id=i, name=f"src{i}", region=i,
               lambda_rate=lambda_rate, mu_val=mus[i], sigma_val=sigma_val)
        for i in range(K)
    ]



def initial_belief_estimate(sources, delta):
    """Per-builder expected total slot value, used to normalise EXP3 gains."""
    return sum(
        s.lambda_rate * delta * np.exp(s.mu_val + s.sigma_val ** 2 / 2)
        for s in sources
    )


def run_exp3_instance(K, delta_slot, delta_mu, latency_mean, init_placement,
                      region_names=None, n_slots=N_SLOTS, n_t=N_T, seed=42):
    """Run EXP3 dynamics; report time-averaged expected welfare with burn-in."""
    if region_names is None:
        region_names = REGIONS
    if len(region_names) != K:
        raise ValueError(f"region_names has {len(region_names)} entries but K={K}")

    sources = build_sources(K, delta_mu)
    prop = FixedLatencyPropagationModel(latency_mean)
    regions = [Region(id=i, name=region_names[i]) for i in range(K)]
    initial_belief = initial_belief_estimate(sources, delta_slot)

    builders = [
        Builder(id=i, policy=EXP3Policy(
            n_regions=K,
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

    # Time-averaged expected welfare over the post-burn-in window
    burn_in = int(n_slots * BURN_IN_FRACTION)
    post_burn_welfares = []
    for region_counts in sim.region_counts_history[burn_in:]:
        profile = [
            r for r, count in enumerate(region_counts.astype(int)) for _ in range(int(count))
        ]
        post_burn_welfares.append(
            _compute_welfare_analytical(profile, sources, prop, delta_slot, n_t)
        )
    w_avg = float(np.mean(post_burn_welfares))

    w_opt, opt_profile = optimal_welfare_brute_force(K, K, sources, prop, delta_slot, n_t)
    return {
        "opt_profile": opt_profile,
        "w_avg": w_avg,
        "w_opt": w_opt,
        "poa": w_opt / w_avg if w_avg > 1e-12 else float("inf"),
    }


def run_exp3_with_seeds(K, delta_slot, delta_mu, latency_mean, init_placement,
                         n_seeds=N_SEEDS):
    """Run EXP3 across multiple seeds; return mean and std of PoA."""
    poa_seeds = []
    for seed in range(n_seeds):
        result = run_exp3_instance(K, delta_slot, delta_mu, latency_mean,
                                   init_placement=init_placement, seed=seed)
        poa_seeds.append(result["poa"])
    return np.mean(poa_seeds), np.std(poa_seeds), poa_seeds


def main():
    all_regions, full_mean, _ = load_gcp(latency_std_fraction=0.15)
    _, latency_mean, _ = subregion(all_regions, full_mean, full_mean*0.15, REGIONS)

    delta_grid = np.geomspace(0.02, 12.0, N_DELTA_POINTS)
    delta_mu = 0.001

    poa_conc_mean = np.zeros(N_DELTA_POINTS)
    poa_conc_std = np.zeros(N_DELTA_POINTS)
    poa_disp_mean = np.zeros(N_DELTA_POINTS)
    poa_disp_std = np.zeros(N_DELTA_POINTS)

    for i, delta_slot in enumerate(delta_grid):
        m_c, s_c, all_c = run_exp3_with_seeds(K, delta_slot, delta_mu, latency_mean,
                                               init_placement="concentrated")
        m_d, s_d, all_d = run_exp3_with_seeds(K, delta_slot, delta_mu, latency_mean,
                                               init_placement="dispersed")
        poa_conc_mean[i] = m_c
        poa_conc_std[i] = s_c
        poa_disp_mean[i] = m_d
        poa_disp_std[i] = s_d
        print(f"delta={delta_slot:.3f}  "
              f"PoA(conc) = {m_c:.3f} ± {s_c:.3f}  "
              f"PoA(disp) = {m_d:.3f} ± {s_d:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Concentrated init
    ax.plot(delta_grid, poa_conc_mean, "-", color="#d62728",
            label="EXP3 from concentrated init")
    ax.fill_between(delta_grid,
                    poa_conc_mean - poa_conc_std,
                    poa_conc_mean + poa_conc_std,
                    alpha=0.2, color="#d62728")

    # Dispersed init
    ax.plot(delta_grid, poa_disp_mean, "-", color="#1f77b4",
            label="EXP3 from dispersed init")
    ax.fill_between(delta_grid,
                    poa_disp_mean - poa_disp_std,
                    poa_disp_mean + poa_disp_std,
                    alpha=0.2, color="#1f77b4")

    # Reference lines
    ax.axhline(2 - 1/K, ls="--", color="gray",
               label=f"Theoretical bound: $2 - 1/K = {2 - 1/K}$")
    ax.axhline(1.0, ls=":", color="black", alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel(r"Slot duration $\Delta$ (s)")
    ax.set_ylabel(r"Empirical PoA $= W(s^*) / \bar{W}_{\mathrm{EXP3}}$")
    ax.set_title(rf"PoA vs. slot duration under EXP3 ($K={K}$, $\delta_\mu={delta_mu}$, "
                 rf"{N_SEEDS} seeds)")
    ax.set_ylim(0.95, 2 - 1/K + 0.05)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_poa_vs_delta_exp3.pdf")
    print(f"\nSaved {FIGURES_DIR / 'exp_poa_vs_delta_exp3.pdf'}")


if __name__ == "__main__":
    main()
