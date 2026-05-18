import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/db-sims-matplotlib")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
ABR_COLOR = "#1f77b4"
PLAN_COLOR = "#2ca02c"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def summarize(values):
    """Median, 25th percentile, 75th percentile."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


def format_ratio_label(r):
    if abs(r - round(r)) < 1e-9:
        return f"{int(round(r))}x"
    return f"{r:g}x"


def builder_counts(profiles, region_names):
    """Aggregate converged profiles into mean per-region builder counts."""
    counts = {r: 0.0 for r in region_names}
    n_profiles = len(profiles)
    if n_profiles == 0:
        return counts
    for profile in profiles:
        for r_idx in profile:
            counts[region_names[r_idx]] += 1.0 / n_profiles
    return counts


def _load_region_coords():
    coords = {}
    path = Path(__file__).resolve().parent.parent / "sim/data/gcp_regions.csv"
    with open(path) as f:
        for row in csv.DictReader(f):
            coords[row["Region"]] = (
                float(row["Nearest City Latitude"]),
                float(row["Nearest City Longitude"]),
            )
    return coords


GCP_REGION_COORDS = _load_region_coords()


def plot_world_map(ax, builder_counts_by_region, region_names, title,
                   region_coords=GCP_REGION_COORDS, size_scale=80):
    """Equirectangular world map with markers sized by mean builder count."""
    ax.set_xlim(-180, 180)
    ax.set_ylim(-65, 80)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.25, lw=0.5, ls=":")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(left=False, bottom=False)

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("black")

    for r_name in region_names:
        if r_name not in region_coords:
            continue
        lat, lon = region_coords[r_name]
        ax.scatter(lon, lat, s=8, c="#cccccc", zorder=2, edgecolors="none")

    for r_name, count in builder_counts_by_region.items():
        if count <= 0 or r_name not in region_coords:
            continue
        lat, lon = region_coords[r_name]
        size = 30 + size_scale * count
        ax.scatter(lon, lat, s=size, c=ABR_COLOR, alpha=0.65,
                   edgecolors="white", linewidths=0.6, zorder=3)

    ax.set_title(title, fontsize=10)
