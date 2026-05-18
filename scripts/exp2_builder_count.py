"""
Experiment 2 - Builder Count Sweep (with source-instance randomisation)

Fixed alpha=0.9, delta=50ms, 24 GCP regions, 10 sources
(5 high-value, 5 peripheral). Sweeps K from 3 to 12.

For each K we draw N_INSTANCES random source layouts and run
N_SEEDS_PER_INSTANCE ABR runs from random initial placements. Reported
curves are median + IQR over all (instance, seed) ABR runs.

Planner benchmarks are computed once per (K, instance), so planner curves
are also shown as median + IQR over source-layout instances.
"""
import argparse
import sys
import warnings
import json
import pickle
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from math import comb
from multiprocessing import Pool, cpu_count
import numpy as np

from scripts.exp_constants import REGIONS_DEFAULT
from scripts.plot_common import FIGURES_DIR, GCP_REGION_COORDS
from scripts.plot_exp2_builder_count import plot
from scripts.plot_results import load_exp2_results


_RUNTIME_DEPS_LOADED = False


def _load_runtime_deps():
    global _RUNTIME_DEPS_LOADED
    global load_propagation_model, build_two_cluster_sources, make_sliced_prop
    global compute_opt_sliced, run_abr_full, geo_hhi
    global mean_pairwise_distance_km, cluster_coverage_fraction
    global compute_all_builder_utilities, _hhi
    if _RUNTIME_DEPS_LOADED:
        return
    from scripts.exp_helpers import (
        load_propagation_model as _load_propagation_model,
        build_two_cluster_sources as _build_two_cluster_sources,
        make_sliced_prop as _make_sliced_prop,
        compute_opt_sliced as _compute_opt_sliced,
        run_abr_full as _run_abr_full,
        geo_hhi as _geo_hhi,
        mean_pairwise_distance_km as _mean_pairwise_distance_km,
        cluster_coverage_fraction as _cluster_coverage_fraction,
    )
    from sim.simulator import compute_all_builder_utilities as _compute_all_builder_utilities
    from sim.metrics import hhi as _metric_hhi

    load_propagation_model = _load_propagation_model
    build_two_cluster_sources = _build_two_cluster_sources
    make_sliced_prop = _make_sliced_prop
    compute_opt_sliced = _compute_opt_sliced
    run_abr_full = _run_abr_full
    geo_hhi = _geo_hhi
    mean_pairwise_distance_km = _mean_pairwise_distance_km
    cluster_coverage_fraction = _cluster_coverage_fraction
    compute_all_builder_utilities = _compute_all_builder_utilities
    _hhi = _metric_hhi
    _RUNTIME_DEPS_LOADED = True

REGIONS_EXP2 = list(REGIONS_DEFAULT) + ["europe-west2", "asia-northeast2", "asia-south2", "us-west2"]

# Experiment parameters
ALPHA = 0.9
DELTA = 0.05
TOTAL_VALUE = 10.0
K_GRID = list(range(3, 13))  # K = 3 .. 12
N_INSTANCES = 5  # random source-layout instances per K
N_SEEDS_PER_INSTANCE = 3  # random ABR initialisations per instance
N_T = 100
N_T_FINAL = 200
MAX_ROUNDS = 6000
N_HIGH = 6
N_PERI = 6

# Plausible high-value pool: regions near major financial / tech centers
HIGH_VALUE_POOL = [
    "us-east1", "us-east4", "us-central1",
    "europe-west1", "europe-west2", "europe-west3", "europe-west4",
    "europe-north1",
    "asia-northeast1", "asia-northeast2",
    "asia-southeast1",
]

# Plausible peripheral pool: regions geographically distant from the above
PERIPHERAL_POOL = [
    "southamerica-east1", "southamerica-west1",
    "africa-south1",
    "australia-southeast1", "australia-southeast2",
    "asia-south1", "asia-south2",
    "us-west1", "us-west2",
]

OPT_METHOD = "greedy"
BRUTE_FORCE_MAX_PROFILES = 10_000_000  # used only if OPT_METHOD == "auto"

def sample_source_layout(rng):
    """Return (high_value_regions, peripheral_regions) drawn from the pools."""
    high = list(rng.choice(HIGH_VALUE_POOL, size=N_HIGH, replace=False))
    peri = list(rng.choice(PERIPHERAL_POOL, size=N_PERI, replace=False))
    return high, peri

def _worker(args):
    _load_runtime_deps()
    K, instance_idx, seed_within_instance = args

    inst_rng = np.random.default_rng(instance_idx)
    high_regions, peri_regions = sample_source_layout(inst_rng)

    regions, prop, region_index_map = load_propagation_model(REGIONS_EXP2)
    sources = build_two_cluster_sources(
        ALPHA, TOTAL_VALUE, region_index_map,
        high_value_regions=high_regions,
        distant_regions=peri_regions,
    )

    # Source-sliced propagation model: shape is (n_regions, n_sources), so
    # source.id is the correct column index inside ABR and welfare routines.
    sliced_prop = make_sliced_prop(sources, prop)
    n_regions = len(regions)

    # ABR initialisation: deterministic per (K, instance, seed)
    init_rng = np.random.default_rng(K * 1_000_000 + instance_idx * 10_000 + seed_within_instance)
    init_regions = [int(init_rng.integers(0, n_regions)) for _ in range(K)]

    # Seed for ABR's internal randomness (candidate region shuffling)
    abr_seed = K * 1_000_000 + instance_idx * 10_000 + seed_within_instance + 1

    result = run_abr_full(
        K, sources, sliced_prop, regions, DELTA, init_regions, abr_seed,
        n_t=N_T, max_rounds=MAX_ROUNDS, n_t_final=N_T_FINAL,
        n_high_sources=N_HIGH,
    )
    result["K"] = K
    result["instance_idx"] = instance_idx
    result["seed_within_instance"] = seed_within_instance
    result["high_regions"] = high_regions
    result["peri_regions"] = peri_regions
    return result

def _opt_method_for(K, n_regions):
    if OPT_METHOD != "auto":
        return OPT_METHOD
    n_profiles = comb(n_regions + K - 1, K)
    return "brute" if n_profiles <= BRUTE_FORCE_MAX_PROFILES else "greedy"


def compute_planner_metrics_one(args):
    _load_runtime_deps()
    K, instance_idx, opt_method = args

    inst_rng = np.random.default_rng(instance_idx)
    high_regions, peri_regions = sample_source_layout(inst_rng)
    regions, prop, region_index_map = load_propagation_model(REGIONS_EXP2)
    sources = build_two_cluster_sources(
        ALPHA, TOTAL_VALUE, region_index_map,
        high_value_regions=high_regions,
        distant_regions=peri_regions,
    )

    # Source-sliced propagation model used by both planner and ABR
    sliced_prop = make_sliced_prop(sources, prop)
    n_regions = len(regions)

    w_opt, opt_profile = compute_opt_sliced(
        K, sources, sliced_prop, n_regions, DELTA,
        n_t=N_T_FINAL, method=opt_method,
    )
    high_ids = list(range(N_HIGH))
    peri_ids = list(range(N_HIGH, N_HIGH + N_PERI))
    planner_utilities = compute_all_builder_utilities(
        opt_profile, sources, sliced_prop, DELTA, N_T_FINAL,
    )
    planner_util_hhi = float(_hhi(planner_utilities))

    return {
        "K": K,
        "instance_idx": instance_idx,
        "w_opt": w_opt,
        "opt_profile": opt_profile,
        "geo_hhi_opt": geo_hhi(opt_profile, n_regions),
        "mean_pairwise_km_opt": mean_pairwise_distance_km(opt_profile, list(regions)),
        "utility_hhi_opt": planner_util_hhi,
        "cov_high_opt": cluster_coverage_fraction(
            opt_profile, sources, sliced_prop, DELTA, high_ids, n_t=N_T_FINAL),
        "cov_peripheral_opt": cluster_coverage_fraction(
            opt_profile, sources, sliced_prop, DELTA, peri_ids, n_t=N_T_FINAL),
        "high_regions": high_regions,
        "peri_regions": peri_regions,
    }


def save_results(K_grid, abr_runs_by_K, planner_runs_by_K,
                 region_names, n_runs_per_K, opt_method):
    """Save raw experiment outputs so figures can be inspected/replotted later."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = FIGURES_DIR / f"exp2_builder_count_results_{ts}"

    payload = {
        "metadata": {
            "experiment": "exp2_builder_count",
            "timestamp": ts,
            "ALPHA": ALPHA,
            "DELTA": DELTA,
            "TOTAL_VALUE": TOTAL_VALUE,
            "K_GRID": list(K_grid),
            "N_INSTANCES": N_INSTANCES,
            "N_SEEDS_PER_INSTANCE": N_SEEDS_PER_INSTANCE,
            "N_T": N_T,
            "N_T_FINAL": N_T_FINAL,
            "MAX_ROUNDS": MAX_ROUNDS,
            "N_HIGH": N_HIGH,
            "N_PERI": N_PERI,
            "OPT_METHOD": OPT_METHOD,
            "opt_method_used": opt_method,
            "n_runs_per_K": n_runs_per_K,
            "region_names": list(region_names),
            "high_value_pool": HIGH_VALUE_POOL,
            "peripheral_pool": PERIPHERAL_POOL,
        },
        "abr_runs_by_K": {
            str(K): abr_runs_by_K[K] for K in K_grid
        },
        "planner_runs_by_K": {
            str(K): planner_runs_by_K[K] for K in K_grid
        },
    }

    pkl_path = out_base.with_suffix(".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    def _json_default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    json_path = out_base.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    print(f"Saved raw results:")
    print(f"  {pkl_path}")
    print(f"  {json_path}")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load-results",
        help="Load a saved exp2 .pkl/.json payload and replot without rerunning.",
    )
    args = parser.parse_args(argv)
    if args.load_results:
        loaded = load_exp2_results(args.load_results)
        meta = loaded.metadata
        plot(
            loaded.K_grid,
            loaded.abr_runs_by_K,
            loaded.planner_runs_by_K,
            loaded.n_runs_per_K,
            region_names=loaded.region_names,
            alpha=meta.get("ALPHA", ALPHA),
            delta=meta.get("DELTA", DELTA),
            n_instances=meta.get("N_INSTANCES", N_INSTANCES),
            n_seeds_per_instance=meta.get("N_SEEDS_PER_INSTANCE", N_SEEDS_PER_INSTANCE),
        )
        return

    _load_runtime_deps()
    regions, _, _ = load_propagation_model(REGIONS_EXP2)
    n_regions = len(regions)

    pool_regions = set(HIGH_VALUE_POOL) | set(PERIPHERAL_POOL)
    missing_regions = [r for r in pool_regions if r not in REGIONS_EXP2]
    missing_coords = [r for r in pool_regions if r not in GCP_REGION_COORDS]
    if missing_regions or missing_coords:
        raise ValueError(
            f"Pool regions misconfigured: "
            f"missing from REGIONS_EXP2={missing_regions}, "
            f"missing from GCP_REGION_COORDS={missing_coords}"
        )

    opt_method = _opt_method_for(max(K_GRID), n_regions)
    n_runs_per_K = N_INSTANCES * N_SEEDS_PER_INSTANCE

    print(f"Exp 2: K sweep: {K_GRID}")
    print(f"alpha={ALPHA}, delta={DELTA*1000:.0f} ms")
    print(f"Per K: {N_INSTANCES} source instances x {N_SEEDS_PER_INSTANCE} "
          f"random inits = {n_runs_per_K} ABR runs")
    print(f"Optimum method: {opt_method}")
    print(f"Total ABR jobs: {len(K_GRID) * n_runs_per_K}")
    print(f"Total planner jobs: {len(K_GRID) * N_INSTANCES}")
    print()

    n_workers = max(1, cpu_count() - 1)

    # Planner benchmarks (one per (K, instance))
    planner_tasks = [
        (K, inst, opt_method)
        for K in K_GRID
        for inst in range(N_INSTANCES)
    ]
    print(f"Computing {len(planner_tasks)} planner benchmarks "
          f"({n_workers} workers) ...")
    planner_runs_by_K = {K: [] for K in K_GRID}
    with Pool(n_workers) as pool:
        for i, p in enumerate(pool.imap_unordered(compute_planner_metrics_one,
                                                    planner_tasks)):
            planner_runs_by_K[p["K"]].append(p)
            if (i + 1) % 10 == 0 or (i + 1) == len(planner_tasks):
                print(f"  planner [{i+1}/{len(planner_tasks)}] "
                      f"K={p['K']} inst={p['instance_idx']} "
                      f"W*={p['w_opt']:.4f}")
    abr_tasks = [
        (K, inst, seed)
        for K in K_GRID
        for inst in range(N_INSTANCES)
        for seed in range(N_SEEDS_PER_INSTANCE)
    ]
    print(f"\nRunning {len(abr_tasks)} ABR jobs ({n_workers} workers) ...")
    abr_runs_by_K = {K: [] for K in K_GRID}
    with Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, abr_tasks)):
            abr_runs_by_K[r["K"]].append(r)
            if (i + 1) % 25 == 0 or (i + 1) == len(abr_tasks):
                print(f"  abr [{i+1}/{len(abr_tasks)}] "
                      f"K={r['K']} inst={r['instance_idx']} "
                      f"seed={r['seed_within_instance']} "
                      f"welfare={r['welfare']:.4f}")
    n_truncated = sum(
        1 for runs in abr_runs_by_K.values()
        for r in runs
        if (r.get("converged") is False
            or (r.get("rounds_used", 0) >= MAX_ROUNDS))
    )
    if n_truncated > 0:
        warnings.warn(
            f"{n_truncated}/{len(abr_tasks)} ABR runs hit MAX_ROUNDS={MAX_ROUNDS} "
            f"without converging. Band width may reflect incomplete convergence."
        )

    print("\nGeographic HHI diagnostic:")
    for K in K_GRID:
        profiles = [r["final_profile"] for r in abr_runs_by_K[K]]
        unique_counts = [len(set(p)) for p in profiles]
        geo_vals = [r["geo_hhi"] for r in abr_runs_by_K[K]]
        print(
            f"  K={K}: "
            f"geo_hhi={geo_vals}, "
            f"unique_builder_regions={unique_counts}, "
            f"profiles={profiles}"
        )

    save_results(
        K_GRID,
        abr_runs_by_K,
        planner_runs_by_K,
        region_names=regions,
        n_runs_per_K=n_runs_per_K,
        opt_method=opt_method,
    )

    plot(
        K_GRID,
        abr_runs_by_K,
        planner_runs_by_K,
        n_runs_per_K,
        region_names=regions,
        alpha=ALPHA,
        delta=DELTA,
        n_instances=N_INSTANCES,
        n_seeds_per_instance=N_SEEDS_PER_INSTANCE,
    )


if __name__ == "__main__":
    main()
