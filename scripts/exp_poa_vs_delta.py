# scripts/exp_poa_vs_delta.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
Experiment: PoA vs slot duration Delta.
Sweeps Delta with the worst-case construction (one dominant source of K*Lambda
mass, K-1 peripheral sources of Lambda each). Runs ABR from concentrated and
dispersed initialisations + plots realised PoA vs Delta with the theoretical
bound 2 - 1/K as a horizontal reference line.
"""
import numpy as np
import matplotlib.pyplot as plt

from sim.datasets import load_gcp, subregion
from sim.simulator import (
    Source, Builder, Region, FixedLatencyPropagationModel,
    StochasticTransactionGenerator, EqualSplitSharingRule,
    LocationGamesSimulator, FixedPolicy,
)
from analysis.poa import _compute_welfare_analytical, optimal_welfare_brute_force
from scripts.poa_experiment_helpers import FIGURES_DIR


REGIONS = ["us-east1", "us-west1", "europe-west1", "asia-northeast1", "australia-southeast1"]
K = len(REGIONS)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def build_sources(K, delta_mu, mu_peripheral=0.0, sigma_val=0.0, lambda_rate=4.28):
    mu_0 = mu_peripheral + np.log(K * (1 + delta_mu))
    mus = [mu_0] + [mu_peripheral] * (K - 1)
    return [
        Source(id=i, name=f"src{i}", region=i,
               lambda_rate=lambda_rate, mu_val=mus[i], sigma_val=sigma_val)
        for i in range(K)
    ]



def run_abr_instance(K, delta_slot, delta_mu, latency_mean, init_placement,
                     region_names=None, n_slots=200, n_t=200, seed=42):
    """Run ABR to convergence; return final profile and welfare."""
    if region_names is None:
        region_names = REGIONS
    if len(region_names) != K:
        raise ValueError(f"region_names has {len(region_names)} entries but K={K}")

    sources = build_sources(K, delta_mu)
    prop = FixedLatencyPropagationModel(latency_mean)
    regions = [Region(id=i, name=region_names[i]) for i in range(K)]
    builders = [Builder(id=i, policy=FixedPolicy(K)) for i in range(K)]

    sim = LocationGamesSimulator(
        regions=regions, sources=sources, builders=builders,
        tx_generator=StochasticTransactionGenerator(),
        propagation_model=prop, sharing_rule=EqualSplitSharingRule(),
        delta=delta_slot, seed=seed, initial_placement=init_placement,
    )
    sim.run_abr(n_slots=n_slots, n_t=n_t)
    final_profile = [b.current_region for b in sim.builders]
    w_converged = _compute_welfare_analytical(final_profile, sources, prop, delta_slot, n_t)
    w_opt, opt_profile = optimal_welfare_brute_force(K, K, sources, prop, delta_slot, n_t)
    return {
        "final_profile": final_profile,
        "opt_profile": opt_profile,
        "w_converged": w_converged,
        "w_opt": w_opt,
        "poa": w_opt / w_converged if w_converged > 1e-12 else float("inf"),
    }


def main():
    all_regions, full_mean, _ = load_gcp(latency_std_fraction=0.15)
    _, latency_mean, _ = subregion(all_regions, full_mean, full_mean*0.15, REGIONS)

    # Slot durations spanning the regime where exclusivity breaks
    delta_grid = np.geomspace(0.02, 12.0, 100)
    delta_mu = 0.001  # tight worst-case dominance

    poa_concentrated = []
    poa_dispersed = []

    for delta_slot in delta_grid:
        r_conc = run_abr_instance(K, delta_slot, delta_mu, latency_mean,
                                  init_placement="concentrated")
        r_disp = run_abr_instance(K, delta_slot, delta_mu, latency_mean,
                                  init_placement="dispersed")
        poa_concentrated.append(r_conc["poa"])
        poa_dispersed.append(r_disp["poa"])
        print(f"delta={delta_slot:.3f}  PoA(conc)={r_conc['poa']:.3f}  "
              f"PoA(disp)={r_disp['poa']:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(delta_grid, poa_concentrated, "-", linewidth=2, color="#d62728",
            label="ABR from concentrated init")
    ax.plot(delta_grid, poa_dispersed, "-", linewidth=2, color="#1f77b4",
            label="ABR from dispersed init")

    ax.axhline(2 - 1/K, ls="--", color="gray", linewidth=1,
            label=f"Theoretical bound: $2 - 1/K = {2 - 1/K}$")
    ax.axhline(1.0, ls=":", color="black", alpha=0.5, linewidth=1)

    ax.set_xscale("log")
    ax.set_xlabel(r"Slot duration $\Delta$ (s)")
    ax.set_ylabel(r"Empirical PoA $= W(s^*) / W(s_{\mathrm{ABR}})$")
    ax.set_title(rf"PoA vs. slot duration ($K={K}$, $\delta_\mu={delta_mu}$)")
    ax.set_ylim(0.95, 2 - 1/K + 0.05)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_poa_vs_delta.pdf")
    print(f"Saved {FIGURES_DIR / 'exp_poa_vs_delta.pdf'}")


if __name__ == "__main__":
    main()
