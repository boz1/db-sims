import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Exp1PlotResults:
    metadata: dict
    value_ratio_grid: np.ndarray
    abr_runs_by_ratio: dict
    planner_runs_by_ratio: dict
    K: int
    n_runs_per_ratio: int
    region_names: list


@dataclass
class Exp2PlotResults:
    metadata: dict
    K_grid: list
    abr_runs_by_K: dict
    planner_runs_by_K: dict
    n_runs_per_K: int
    region_names: list


@dataclass
class Exp4PlotResults:
    metadata: dict
    delta_grid_ms: list
    delta_grid: np.ndarray
    abr_runs_by_delta: dict
    planner_runs_by_delta: dict
    K: int
    n_runs_per_delta: int


def load_payload(path):
    """Load a saved experiment payload from .pkl or .json."""
    path = Path(path)
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    raise ValueError(f"Unsupported results file extension: {path.suffix}")


def _coerce_keyed_runs(saved_runs, keys, key_type):
    """Return a dict keyed by typed sweep values from a JSON/pickle payload."""
    result = {}
    for key in keys:
        typed_key = key_type(key)
        candidates = [
            str(key),
            str(typed_key),
        ]
        if isinstance(typed_key, float):
            candidates.extend([f"{typed_key:.4f}", f"{typed_key:g}"])
        for candidate in candidates:
            if candidate in saved_runs:
                result[typed_key] = saved_runs[candidate]
                break
        else:
            raise KeyError(f"Missing saved runs for sweep key {typed_key!r}")
    return result


def load_exp1_results(path):
    payload = load_payload(path)
    metadata = payload["metadata"]
    if metadata.get("experiment") != "exp1_value_asymmetry":
        raise ValueError(f"Not an exp1 payload: {metadata.get('experiment')}")

    value_ratio_grid = np.asarray(metadata["VALUE_RATIO_GRID"], dtype=float)
    keys = value_ratio_grid.tolist()
    return Exp1PlotResults(
        metadata=metadata,
        value_ratio_grid=value_ratio_grid,
        abr_runs_by_ratio=_coerce_keyed_runs(
            payload["abr_runs_by_ratio"], keys, float),
        planner_runs_by_ratio=_coerce_keyed_runs(
            payload["planner_runs_by_ratio"], keys, float),
        K=int(metadata["K"]),
        n_runs_per_ratio=int(metadata["n_runs_per_ratio"]),
        region_names=list(metadata["region_names"]),
    )


def load_exp2_results(path):
    payload = load_payload(path)
    metadata = payload["metadata"]
    if metadata.get("experiment") != "exp2_builder_count":
        raise ValueError(f"Not an exp2 payload: {metadata.get('experiment')}")

    K_grid = [int(k) for k in metadata["K_GRID"]]
    return Exp2PlotResults(
        metadata=metadata,
        K_grid=K_grid,
        abr_runs_by_K=_coerce_keyed_runs(payload["abr_runs_by_K"], K_grid, int),
        planner_runs_by_K=_coerce_keyed_runs(
            payload["planner_runs_by_K"], K_grid, int),
        n_runs_per_K=int(metadata["n_runs_per_K"]),
        region_names=list(metadata["region_names"]),
    )


def load_exp4_results(path):
    payload = load_payload(path)
    metadata = payload["metadata"]
    if metadata.get("experiment") != "exp4_slot_duration":
        raise ValueError(f"Not an exp4 payload: {metadata.get('experiment')}")

    delta_grid_ms = [int(d) for d in metadata["DELTA_GRID_MS"]]
    delta_grid = np.asarray(metadata["DELTA_GRID"], dtype=float)
    keys = delta_grid.tolist()
    return Exp4PlotResults(
        metadata=metadata,
        delta_grid_ms=delta_grid_ms,
        delta_grid=delta_grid,
        abr_runs_by_delta=_coerce_keyed_runs(
            payload["abr_runs_by_delta"], keys, float),
        planner_runs_by_delta=_coerce_keyed_runs(
            payload["planner_runs_by_delta"], keys, float),
        K=int(metadata["K"]),
        n_runs_per_delta=int(metadata["n_runs_per_delta"]),
    )


def replot_saved_results(path):
    payload = load_payload(path)
    experiment = payload.get("metadata", {}).get("experiment")

    if experiment == "exp1_value_asymmetry":
        from scripts.plot_exp1_value_asymmetry import plot
        loaded = load_exp1_results(path)
        meta = loaded.metadata
        plot(
            loaded.value_ratio_grid,
            loaded.abr_runs_by_ratio,
            loaded.planner_runs_by_ratio,
            loaded.K,
            loaded.n_runs_per_ratio,
            loaded.region_names,
            delta=meta.get("DELTA", 0.05),
            n_instances=meta.get("N_INSTANCES", 3),
            n_seeds_per_instance=meta.get("N_SEEDS_PER_INSTANCE", 3),
        )
        return

    if experiment == "exp2_builder_count":
        from scripts.plot_exp2_builder_count import plot
        loaded = load_exp2_results(path)
        meta = loaded.metadata
        plot(
            loaded.K_grid,
            loaded.abr_runs_by_K,
            loaded.planner_runs_by_K,
            loaded.n_runs_per_K,
            loaded.region_names,
            alpha=meta.get("ALPHA", 0.9),
            delta=meta.get("DELTA", 0.05),
            n_instances=meta.get("N_INSTANCES", 5),
            n_seeds_per_instance=meta.get("N_SEEDS_PER_INSTANCE", 3),
        )
        return

    if experiment == "exp4_slot_duration":
        from scripts.plot_exp4_slot_duration import plot
        loaded = load_exp4_results(path)
        meta = loaded.metadata
        plot(
            loaded.delta_grid_ms,
            loaded.delta_grid,
            loaded.abr_runs_by_delta,
            loaded.planner_runs_by_delta,
            loaded.K,
            loaded.n_runs_per_delta,
            value_ratio=meta.get("VALUE_RATIO", 10.0),
            alpha=meta.get("ALPHA"),
            delta_anchor=meta.get("DELTA_ANCHOR", 0.05),
            n_instances=meta.get("N_INSTANCES", 5),
            n_seeds_per_instance=meta.get("N_SEEDS_PER_INSTANCE", 3),
        )
        return

    raise ValueError(f"Unsupported experiment payload: {experiment}")
