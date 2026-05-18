"""
Experiment 1: Value Asymmetry Sweep (with source-instance randomization)

K=5 builders, 24 GCP regions, 10 sources (5 high-value, 5 low-value).
Sweeps the per-source value ratio between high-value and low-value sources.
Internally, this is converted to alpha: the fraction of total emitted value
assigned to the high-value cluster.

For each ratio we draw N_INSTANCES random source layouts and run
N_SEEDS_PER_INSTANCE ABR runs from random initial placements. Reported
curves are median + IQR over all (instance, seed) runs.
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
from scripts.plot_exp1_value_asymmetry import plot
from scripts.plot_results import load_exp1_results


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

REGIONS_EXP1 = list(REGIONS_DEFAULT) + ["europe-west2", "asia-northeast2", "asia-south2", "us-west2"]

# Experiment parameters
MASTER_SEED = 1234
K = 5
# DELTA = 0.2
DELTA = 0.05
TOTAL_VALUE = 10.0

VALUE_RATIO_GRID = np.array([1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0, 10.0, 20.0])

N_INSTANCES = 3  # random source-layout instances per ratio
N_SEEDS_PER_INSTANCE = 3   # random ABR initialisations per instance
N_T = 100
N_T_FINAL = 200
MAX_ROUNDS = 6000
N_HIGH = 5
N_PERI = 5


def alpha_from_value_ratio(ratio, n_high=N_HIGH, n_peri=N_PERI):
    """Convert per-source high/low value ratio into cluster-level alpha."""
    return float((ratio * n_high) / (ratio * n_high + n_peri))


ALPHA_GRID = np.array([alpha_from_value_ratio(r) for r in VALUE_RATIO_GRID])


def _format_ratio_label(r):
    if abs(r - round(r)) < 1e-9:
        return f"{int(round(r))}x"
    return f"{r:g}x"


VALUE_RATIO_LABELS = [_format_ratio_label(r) for r in VALUE_RATIO_GRID]

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


# Worker: run one (ratio_idx, ratio, alpha, instance, seed) job

def _worker(args):
    _load_runtime_deps()
    ratio_idx, value_ratio, alpha, instance_idx, seed_within_instance = args
    inst_rng = np.random.default_rng(MASTER_SEED + instance_idx)
    high_regions, peri_regions = sample_source_layout(inst_rng)

    regions, prop, region_index_map = load_propagation_model(REGIONS_EXP1)
    sources = build_two_cluster_sources(
        alpha, TOTAL_VALUE, region_index_map,
        high_value_regions=high_regions,
        distant_regions=peri_regions,
    )
    sliced_prop = make_sliced_prop(sources, prop)
    n_regions = len(regions)

    # ABR initialisation: deterministic for each (instance, seed), intentionally not
    # ratio-dependent so the same initial placements are used across the sweep.
    init_rng = np.random.default_rng(
        MASTER_SEED + 1_000_000 + instance_idx * 10_000 + seed_within_instance
    )
    init_regions = [int(init_rng.integers(0, n_regions)) for _ in range(K)]

    # Seed for ABR's internal randomness (candidate region shuffling).
    abr_seed = (
        MASTER_SEED + 2_000_000 + ratio_idx * 100_000
        + instance_idx * 10_000 + seed_within_instance
    )

    result = run_abr_full(
        K, sources, sliced_prop, regions, DELTA, init_regions, abr_seed,
        n_t=N_T, max_rounds=MAX_ROUNDS, n_t_final=N_T_FINAL,
        n_high_sources=N_HIGH,
    )
    result["ratio_idx"] = ratio_idx
    result["value_ratio"] = value_ratio
    result["alpha"] = alpha
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
    ratio_idx, value_ratio, alpha, instance_idx, opt_method = args

    inst_rng = np.random.default_rng(MASTER_SEED + instance_idx)
    high_regions, peri_regions = sample_source_layout(inst_rng)

    regions, prop, region_index_map = load_propagation_model(REGIONS_EXP1)
    sources = build_two_cluster_sources(
        alpha, TOTAL_VALUE, region_index_map,
        high_value_regions=high_regions,
        distant_regions=peri_regions,
    )
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
        "ratio_idx": ratio_idx,
        "value_ratio": value_ratio,
        "alpha": alpha,
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


def save_results(value_ratio_grid, alpha_grid, abr_runs_by_ratio, planner_runs_by_ratio,
                 region_names, n_runs_per_ratio, opt_method):
    """Save raw experiment outputs so figures can be inspected/replotted later."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = FIGURES_DIR / f"exp1_value_asymmetry_results_{ts}"

    payload = {
        "metadata": {
            "experiment": "exp1_value_asymmetry",
            "timestamp": ts,
            "MASTER_SEED": MASTER_SEED,
            "K": K,
            "DELTA": DELTA,
            "TOTAL_VALUE": TOTAL_VALUE,
            "VALUE_RATIO_GRID": value_ratio_grid.tolist(),
            "VALUE_RATIO_LABELS": VALUE_RATIO_LABELS,
            "ALPHA_GRID": alpha_grid.tolist(),
            "N_INSTANCES": N_INSTANCES,
            "N_SEEDS_PER_INSTANCE": N_SEEDS_PER_INSTANCE,
            "N_T": N_T,
            "N_T_FINAL": N_T_FINAL,
            "MAX_ROUNDS": MAX_ROUNDS,
            "N_HIGH": N_HIGH,
            "N_PERI": N_PERI,
            "OPT_METHOD": OPT_METHOD,
            "opt_method_used": opt_method,
            "n_runs_per_ratio": n_runs_per_ratio,
            "region_names": list(region_names),
            "high_value_pool": HIGH_VALUE_POOL,
            "peripheral_pool": PERIPHERAL_POOL,
            "value_ratio_definition": "per-source high-cluster expected value divided by per-source peripheral-cluster expected value",
        },
        "abr_runs_by_ratio": {
            str(r): abr_runs_by_ratio[r] for r in value_ratio_grid
        },
        "planner_runs_by_ratio": {
            str(r): planner_runs_by_ratio[r] for r in value_ratio_grid
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
        help="Load a saved exp1 .pkl/.json payload and replot without rerunning.",
    )
    args = parser.parse_args(argv)
    if args.load_results:
        loaded = load_exp1_results(args.load_results)
        meta = loaded.metadata
        plot(
            loaded.value_ratio_grid,
            loaded.abr_runs_by_ratio,
            loaded.planner_runs_by_ratio,
            loaded.K,
            loaded.n_runs_per_ratio,
            region_names=loaded.region_names,
            delta=meta.get("DELTA", DELTA),
            n_instances=meta.get("N_INSTANCES", N_INSTANCES),
            n_seeds_per_instance=meta.get("N_SEEDS_PER_INSTANCE", N_SEEDS_PER_INSTANCE),
        )
        return

    _load_runtime_deps()
    regions, _, _ = load_propagation_model(REGIONS_EXP1)
    n_regions = len(regions)

    pool_regions = set(HIGH_VALUE_POOL) | set(PERIPHERAL_POOL)
    missing_regions = [r for r in pool_regions if r not in REGIONS_EXP1]
    missing_coords = [r for r in pool_regions if r not in GCP_REGION_COORDS]
    if missing_regions or missing_coords:
        raise ValueError(
            f"Pool regions misconfigured: "
            f"missing from REGIONS_EXP1={missing_regions}, "
            f"missing from GCP_REGION_COORDS={missing_coords}"
        )

    opt_method = _opt_method_for(K, n_regions)
    n_profiles = comb(n_regions + K - 1, K)
    n_runs_per_ratio = N_INSTANCES * N_SEEDS_PER_INSTANCE

    print(f"Exp 1: per-source value ratio sweep: {VALUE_RATIO_GRID}")
    print(f"Corresponding alpha values: {ALPHA_GRID}")
    print(f"MASTER_SEED={MASTER_SEED}")
    print(f"K={K}, delta={DELTA*1000:.0f} ms")
    print(f"Per ratio: {N_INSTANCES} source instances x {N_SEEDS_PER_INSTANCE} "
          f"random inits = {n_runs_per_ratio} ABR runs")
    print(f"Optimum method: {opt_method} (profile count = {n_profiles:,})")
    print(f"Total ABR jobs: {len(VALUE_RATIO_GRID) * n_runs_per_ratio}")
    print(f"Total planner jobs: {len(VALUE_RATIO_GRID) * N_INSTANCES}")
    print()

    n_workers = max(1, cpu_count() - 1)

    planner_tasks = [
        (i, ratio, ALPHA_GRID[i], inst, opt_method)
        for i, ratio in enumerate(VALUE_RATIO_GRID)
        for inst in range(N_INSTANCES)
    ]
    print(f"Computing {len(planner_tasks)} planner benchmarks "
          f"({n_workers} workers) ...")
    planner_runs_by_ratio = {r: [] for r in VALUE_RATIO_GRID}
    with Pool(n_workers) as pool:
        for i, p in enumerate(pool.imap_unordered(compute_planner_metrics_one,
                                                  planner_tasks)):
            r_key = VALUE_RATIO_GRID[p["ratio_idx"]]
            planner_runs_by_ratio[r_key].append(p)
            if (i + 1) % 10 == 0 or (i + 1) == len(planner_tasks):
                print(f"  planner [{i+1}/{len(planner_tasks)}] "
                      f"ratio={p['value_ratio']:.2f}x alpha={p['alpha']:.3f} "
                      f"inst={p['instance_idx']} W*={p['w_opt']:.4f}")

    abr_tasks = [
        (i, ratio, ALPHA_GRID[i], inst, seed)
        for i, ratio in enumerate(VALUE_RATIO_GRID)
        for inst in range(N_INSTANCES)
        for seed in range(N_SEEDS_PER_INSTANCE)
    ]
    print(f"\nRunning {len(abr_tasks)} ABR jobs ({n_workers} workers) ...")
    abr_runs_by_ratio = {r: [] for r in VALUE_RATIO_GRID}
    with Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, abr_tasks)):
            r_key = VALUE_RATIO_GRID[r["ratio_idx"]]
            abr_runs_by_ratio[r_key].append(r)
            if (i + 1) % 25 == 0 or (i + 1) == len(abr_tasks):
                print(f"  abr [{i+1}/{len(abr_tasks)}] "
                      f"ratio={r['value_ratio']:.2f}x alpha={r['alpha']:.3f} "
                      f"inst={r['instance_idx']} "
                      f"seed={r['seed_within_instance']} "
                      f"welfare={r['welfare']:.4f}")

    n_truncated = sum(
        1 for runs in abr_runs_by_ratio.values()
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
    for ratio in VALUE_RATIO_GRID:
        profiles = [r["final_profile"] for r in abr_runs_by_ratio[ratio]]
        unique_counts = [len(set(p)) for p in profiles]
        geo_vals = [r["geo_hhi"] for r in abr_runs_by_ratio[ratio]]
        print(
            f"  ratio={ratio:.2f}x alpha={alpha_from_value_ratio(ratio):.3f}: "
            f"geo_hhi={geo_vals}, "
            f"unique_builder_regions={unique_counts}, "
            f"profiles={profiles}"
        )

    save_results(
        VALUE_RATIO_GRID,
        ALPHA_GRID,
        abr_runs_by_ratio,
        planner_runs_by_ratio,
        region_names=regions,
        n_runs_per_ratio=n_runs_per_ratio,
        opt_method=opt_method,
    )

    plot(
        VALUE_RATIO_GRID,
        abr_runs_by_ratio,
        planner_runs_by_ratio,
        K,
        n_runs_per_ratio,
        region_names=regions,
        delta=DELTA,
        n_instances=N_INSTANCES,
        n_seeds_per_instance=N_SEEDS_PER_INSTANCE,
    )


if __name__ == "__main__":
    main()
