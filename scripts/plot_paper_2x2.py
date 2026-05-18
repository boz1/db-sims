#!/usr/bin/env python3
"""Generate compact 2x2 paper figures from saved experiment outputs."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scripts.plot_common import ABR_COLOR, FIGURES_DIR, PLAN_COLOR, format_ratio_label, plt, summarize
from scripts.plot_results import load_exp1_results, load_exp2_results, load_exp4_results


LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 16


def _latest(pattern: str, directory: Path) -> Path:
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files match {directory / pattern}")
    return paths[-1]


def _series(values):
    med, low, high = summarize(values)
    return med, low, high


def _empty_series(n):
    return {
        "median": np.full(n, np.nan),
        "low": np.full(n, np.nan),
        "high": np.full(n, np.nan),
    }


def _metric_series(keys, runs_by_key, metric):
    out = _empty_series(len(keys))
    for i, key in enumerate(keys):
        values = [run[metric] for run in runs_by_key.get(key, []) if metric in run]
        out["median"][i], out["low"][i], out["high"][i] = _series(values)
    return out


def _welfare_ratio_series(keys, abr_runs_by_key, planner_runs_by_key):
    out = _empty_series(len(keys))
    for i, key in enumerate(keys):
        w_opt_by_instance = {
            run["instance_idx"]: run["w_opt"]
            for run in planner_runs_by_key.get(key, [])
            if run.get("w_opt", 0.0) > 1e-12
        }
        ratios = []
        for run in abr_runs_by_key.get(key, []):
            w_opt = w_opt_by_instance.get(run.get("instance_idx"))
            if w_opt is not None:
                ratios.append(run["welfare"] / w_opt)
        out["median"][i], out["low"][i], out["high"][i] = _series(ratios)
    return out


def aggregate_sweep_metrics(keys, abr_runs_by_key, planner_runs_by_key):
    """Aggregate the four paper panels for a one-dimensional sweep."""
    return {
        "welfare_ratio": _welfare_ratio_series(keys, abr_runs_by_key, planner_runs_by_key),
        "geo_hhi": _metric_series(keys, abr_runs_by_key, "geo_hhi"),
        "geo_hhi_opt": _metric_series(keys, planner_runs_by_key, "geo_hhi_opt"),
        "utility_hhi": _metric_series(keys, abr_runs_by_key, "utility_hhi"),
        "utility_hhi_opt": _metric_series(keys, planner_runs_by_key, "utility_hhi_opt"),
        "cov_high": _metric_series(keys, abr_runs_by_key, "cov_high"),
        "cov_high_opt": _metric_series(keys, planner_runs_by_key, "cov_high_opt"),
        "cov_peripheral": _metric_series(keys, abr_runs_by_key, "cov_peripheral"),
        "cov_peripheral_opt": _metric_series(keys, planner_runs_by_key, "cov_peripheral_opt"),
    }


def _plot_band(ax, x, series, label, color, linestyle="-", marker="o", fill=True):
    ax.plot(
        x,
        series["median"],
        label=label,
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=3,
        markersize=6,
    )
    if fill and not np.allclose(series["low"], series["high"], equal_nan=True):
        ax.fill_between(x, series["low"], series["high"], color=color, alpha=0.16)


def _style_axis(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(True, alpha=0.25, linewidth=0.6)


def _baseline_arrays(x, K):
    if np.isscalar(K):
        k_arr = np.full(len(x), float(K))
    else:
        k_arr = np.asarray(K, dtype=float)
    return 1.0 / k_arr, 9.0 / (8.0 * k_arr)


def plot_sweep_2x2(x, data, xlabel, out_base, *, K, xscale=None, xtick_labels=None,
                   xticks=None, include_planner=True):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)
    ax_w, ax_geo, ax_util, ax_cov = axes.ravel()

    x = np.asarray(x, dtype=float)
    geo_floor, util_ceiling = _baseline_arrays(x, K)

    _plot_band(ax_w, x, data["welfare_ratio"], r"ABR", ABR_COLOR)
    ax_w.axhline(1.0, color="0.35", linewidth=1.0, linestyle=":")
    _style_axis(ax_w, xlabel, "Welfare Ratio")

    _plot_band(ax_geo, x, data["geo_hhi"], "ABR", ABR_COLOR)
    if include_planner:
        _plot_band(ax_geo, x, data["geo_hhi_opt"], "Planner", PLAN_COLOR, linestyle="--", marker="s")
    ax_geo.plot(x, geo_floor, color="0.35", linewidth=1.0, linestyle=":", label=r"$1/K$")
    _style_axis(ax_geo, xlabel, "Geographic HHI")

    _plot_band(ax_util, x, data["utility_hhi"], "ABR", ABR_COLOR)
    if include_planner:
        _plot_band(ax_util, x, data["utility_hhi_opt"], "Planner", PLAN_COLOR, linestyle="--", marker="s")
    ax_util.plot(x, geo_floor, color="0.35", linewidth=1.0, linestyle=":", label=r"$1/K$")
    ax_util.plot(x, util_ceiling, color="0.5", linewidth=1.0, linestyle="--", label=r"$9/(8K)$")
    _style_axis(ax_util, xlabel, "Utility HHI")

    _plot_band(ax_cov, x, data["cov_high"], "ABR high", ABR_COLOR)
    _plot_band(ax_cov, x, data["cov_peripheral"], "ABR peripheral", ABR_COLOR, linestyle="--", marker="D")
    if include_planner:
        _plot_band(ax_cov, x, data["cov_high_opt"], "Planner high", PLAN_COLOR, linestyle="-", marker="s")
        _plot_band(
            ax_cov,
            x,
            data["cov_peripheral_opt"],
            "Planner peripheral",
            PLAN_COLOR,
            linestyle="--",
            marker="^",
        )
    ax_cov.set_ylim(-0.03, 1.03)
    _style_axis(ax_cov, xlabel, "Cluster Coverage")

    for ax in axes.ravel():
        if xscale:
            ax.set_xscale(xscale)
        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels or [str(v) for v in xticks])
        elif xtick_labels is not None:
            ax.set_xticks(x)
            ax.set_xticklabels(xtick_labels)
        if len(ax.get_xticklabels()) > 6:
            for label in ax.get_xticklabels():
                label.set_rotation(35)
                label.set_ha("right")

    ax_w.legend(fontsize=16, frameon=False)
    ax_geo.legend(fontsize=16, frameon=False)
    ax_util.legend(fontsize=16, frameon=False)
    ax_cov.legend(fontsize=16, frameon=False, ncol=2)

    out_base = Path(out_base)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def _coerce_object_array(arr):
    return [item.item() if hasattr(item, "item") else item for item in arr]


def load_exp3_grids(path):
    payload = np.load(path, allow_pickle=True)
    abr_runs = _coerce_object_array(payload["abr_runs"])
    planner_runs = _coerce_object_array(payload["planner_runs"])
    value_ratio_grid = np.asarray(payload["value_ratio_grid"], dtype=float)
    delta_grid = np.asarray(payload["delta_grid"], dtype=float)

    shape = (len(value_ratio_grid), len(delta_grid))
    grids = {
        "welfare_ratio": np.full(shape, np.nan),
        "geo_hhi": np.full(shape, np.nan),
        "utility_hhi": np.full(shape, np.nan),
        "cov_high": np.full(shape, np.nan),
        "cov_peripheral": np.full(shape, np.nan),
    }

    planner_by_cell = {}
    for run in planner_runs:
        planner_by_cell.setdefault((run["ratio_idx"], run["delta_idx"]), {})[
            run["instance_idx"]
        ] = run["w_opt"]

    abr_by_cell = {}
    for run in abr_runs:
        abr_by_cell.setdefault((run["ratio_idx"], run["delta_idx"]), []).append(run)

    for (ratio_idx, delta_idx), runs in abr_by_cell.items():
        ratios = []
        w_opt_by_instance = planner_by_cell.get((ratio_idx, delta_idx), {})
        for run in runs:
            w_opt = w_opt_by_instance.get(run["instance_idx"])
            if w_opt is not None and w_opt > 1e-12:
                ratios.append(run["welfare"] / w_opt)
        grids["welfare_ratio"][ratio_idx, delta_idx] = summarize(ratios)[0]
        for metric in ("geo_hhi", "utility_hhi", "cov_high", "cov_peripheral"):
            grids[metric][ratio_idx, delta_idx] = summarize([run[metric] for run in runs])[0]

    return value_ratio_grid, delta_grid, grids


def _heatmap(ax, grid, xlabel, ylabel, cbar_label, *, xtick_labels, ytick_labels,
             vmin=None, vmax=None, cmap="viridis", colorbar=True):
    im = ax.imshow(grid, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(np.arange(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=TICK_FONT_SIZE)
    ax.set_yticks(np.arange(len(ytick_labels)))
    ax.set_yticklabels(ytick_labels, fontsize=TICK_FONT_SIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    return im


def plot_exp3_2x2(cache_path, out_base):
    value_ratio_grid, delta_grid, grids = load_exp3_grids(cache_path)
    delta_labels = [str(int(round(d * 1000))) for d in delta_grid]
    coverage_delta_labels = [label if i in (0, 2, 4, 6, 9) else "" for i, label in enumerate(delta_labels)]
    ratio_labels = [format_ratio_label(r) for r in value_ratio_grid]

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_w = fig.add_subplot(gs[0, 0])
    ax_geo = fig.add_subplot(gs[0, 1])
    ax_util = fig.add_subplot(gs[1, 0])
    cov_gs = gs[1, 1].subgridspec(1, 2, wspace=0.12)
    ax_high = fig.add_subplot(cov_gs[0, 0])
    ax_peri = fig.add_subplot(cov_gs[0, 1])

    xlabel = r"Slot duration $\Delta$ (ms)"
    ylabel = "Value ratio (high / low)"

    _heatmap(
        ax_w,
        grids["welfare_ratio"],
        xlabel,
        ylabel,
        "ratio",
        xtick_labels=delta_labels,
        ytick_labels=ratio_labels,
        vmin=0.5,
        vmax=1.0,
        cmap="viridis",
    )

    _heatmap(
        ax_geo,
        grids["geo_hhi"],
        xlabel,
        ylabel,
        "HHI",
        xtick_labels=delta_labels,
        ytick_labels=ratio_labels,
        vmin=0.2,
        vmax=1.0,
        cmap="magma",
    )

    _heatmap(
        ax_util,
        grids["utility_hhi"],
        xlabel,
        ylabel,
        "HHI",
        xtick_labels=delta_labels,
        ytick_labels=ratio_labels,
        vmin=0.2,
        vmax=0.225,
        cmap="magma",
    )

    im_high = _heatmap(
        ax_high,
        grids["cov_high"],
        xlabel,
        ylabel,
        "fraction",
        xtick_labels=coverage_delta_labels,
        ytick_labels=ratio_labels,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        colorbar=False,
    )
    ax_high.set_title("High-value", fontsize=9, pad=2)
    im_peri = _heatmap(
        ax_peri,
        grids["cov_peripheral"],
        xlabel,
        "",
        "fraction",
        xtick_labels=coverage_delta_labels,
        ytick_labels=[],
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        colorbar=False,
    )
    ax_peri.set_title("Peripheral", fontsize=9, pad=2)
    cbar = fig.colorbar(im_peri, ax=[ax_high, ax_peri], fraction=0.046, pad=0.02)
    cbar.set_label("fraction", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    out_base = Path(out_base)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_exp1(path, output_dir):
    loaded = load_exp1_results(path)
    x = loaded.value_ratio_grid.tolist()
    data = aggregate_sweep_metrics(x, loaded.abr_runs_by_ratio, loaded.planner_runs_by_ratio)
    plot_sweep_2x2(
        x,
        data,
        "Value ratio (high / low)",
        output_dir / "paper_exp1_2x2",
        K=loaded.K,
        xscale="log",
        xticks=[1.0, 2.0, 5.0, 10.0, 20.0],
        xtick_labels=["1x", "2x", "5x", "10x", "20x"],
    )


def plot_exp2(path, output_dir):
    loaded = load_exp2_results(path)
    x = loaded.K_grid
    data = aggregate_sweep_metrics(x, loaded.abr_runs_by_K, loaded.planner_runs_by_K)
    plot_sweep_2x2(
        x,
        data,
        r"Number of builders $K$",
        output_dir / "paper_exp2_2x2",
        K=x,
    )


def plot_exp4(path, output_dir):
    loaded = load_exp4_results(path)
    x = loaded.delta_grid_ms
    data = aggregate_sweep_metrics(
        loaded.delta_grid.tolist(),
        loaded.abr_runs_by_delta,
        loaded.planner_runs_by_delta,
    )
    plot_sweep_2x2(
        x,
        data,
        r"Slot duration $\Delta$ (ms)",
        output_dir / "paper_exp4_2x2",
        K=loaded.K,
        xscale="log",
        xticks=[10, 50, 100, 1000, 12000],
        xtick_labels=["10", "50", "100", "1000", "12000"],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate compact 2x2 paper figures from saved experiment results."
    )
    parser.add_argument("--exp", choices=["all", "exp1", "exp2", "exp3", "exp4"], default="all")
    parser.add_argument("--exp1-results", type=Path)
    parser.add_argument("--exp2-results", type=Path)
    parser.add_argument("--exp3-results", type=Path)
    parser.add_argument("--exp4-results", type=Path)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.exp in ("all", "exp1"):
        path = args.exp1_results or _latest("exp1_value_asymmetry_results_*.json", FIGURES_DIR)
        plot_exp1(path, args.output_dir)
        print(f"Saved {args.output_dir / 'paper_exp1_2x2.pdf'} and .png")

    if args.exp in ("all", "exp2"):
        path = args.exp2_results or _latest("exp2_builder_count_results_*.json", FIGURES_DIR)
        plot_exp2(path, args.output_dir)
        print(f"Saved {args.output_dir / 'paper_exp2_2x2.pdf'} and .png")

    if args.exp in ("all", "exp3"):
        results_dir = Path(__file__).resolve().parent.parent / "results"
        path = args.exp3_results or _latest("exp3_k*.npz", results_dir)
        plot_exp3_2x2(path, args.output_dir / "paper_exp3_2x2")
        print(f"Saved {args.output_dir / 'paper_exp3_2x2.pdf'} and .png")

    if args.exp in ("all", "exp4"):
        path = args.exp4_results or _latest("exp4_slot_duration_results_*.json", FIGURES_DIR)
        plot_exp4(path, args.output_dir)
        print(f"Saved {args.output_dir / 'paper_exp4_2x2.pdf'} and .png")


if __name__ == "__main__":
    main()
