import numpy as np

from scripts.plot_common import (
    ABR_COLOR,
    PLAN_COLOR,
    FIGURES_DIR,
    builder_counts,
    format_ratio_label,
    plot_world_map,
    plt,
    summarize,
)


def set_ratio_axis(ax, x_vals):
    major_ticks = [1.0, 2.0, 5.0, 10.0, 20.0]
    major_labels = ["1x", "2x", "5x", "10x", "20x"]

    ax.set_xscale("log")
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_labels)
    ax.set_xlim(min(x_vals) * 0.92, max(x_vals) * 1.08)
    ax.set_xlabel("Expected per-source value ratio (high / low)")

def plot(value_ratio_grid, abr_runs_by_ratio, planner_runs_by_ratio, K,
         n_runs_per_ratio, region_names, delta=0.05, n_instances=3,
         n_seeds_per_instance=3):
    """abr_runs_by_ratio[ratio] = list of run dicts.
    planner_runs_by_ratio[ratio] = list of planner dicts (one per instance)."""

    def _abr_agg(key):
        meds, los, his = [], [], []
        for r in value_ratio_grid:
            vals = [x[key] for x in abr_runs_by_ratio[r]]
            m, lo, hi = summarize(vals)
            meds.append(m); los.append(lo); his.append(hi)
        return np.array(meds), np.array(los), np.array(his)

    def _plan_agg(key):
        meds, los, his = [], [], []
        for r in value_ratio_grid:
            vals = [x[key] for x in planner_runs_by_ratio[r]]
            m, lo, hi = summarize(vals)
            meds.append(m); los.append(lo); his.append(hi)
        return np.array(meds), np.array(los), np.array(his)


    wr_med, wr_lo, wr_hi = [], [], []
    for ratio in value_ratio_grid:
        w_opt_by_inst = {
            p["instance_idx"]: p["w_opt"] for p in planner_runs_by_ratio[ratio]
        }
        ratios = []
        for r in abr_runs_by_ratio[ratio]:
            w_opt = w_opt_by_inst.get(r["instance_idx"])
            if w_opt is not None and w_opt > 1e-12:
                ratios.append(r["welfare"] / w_opt)
        m, lo, hi = summarize(ratios)
        wr_med.append(m); wr_lo.append(lo); wr_hi.append(hi)
    wr_med = np.array(wr_med); wr_lo = np.array(wr_lo); wr_hi = np.array(wr_hi)

    welfare_med, welfare_lo, welfare_hi = _abr_agg("welfare")
    welfare_opt_med, welfare_opt_lo, welfare_opt_hi = _plan_agg("w_opt")

    geo_med, geo_lo, geo_hi = _abr_agg("geo_hhi")
    util_med, util_lo, util_hi = _abr_agg("utility_hhi")
    cov_hi_med, cov_hi_lo, cov_hi_hi = _abr_agg("cov_high")
    cov_pe_med, cov_pe_lo, cov_pe_hi = _abr_agg("cov_peripheral")

    geo_opt_med, geo_opt_lo, geo_opt_hi = _plan_agg("geo_hhi_opt")
    cov_hi_opt_med, cov_hi_opt_lo, cov_hi_opt_hi = _plan_agg("cov_high_opt")
    cov_pe_opt_med, cov_pe_opt_lo, cov_pe_opt_hi = _plan_agg("cov_peripheral_opt")

    # Layout: top row 3 panels, middle row 3 panels, bottom row 2 maps.
    fig = plt.figure(figsize=(14.5, 11))
    gs = fig.add_gridspec(
        nrows=3, ncols=6,
        height_ratios=[1.0, 1.0, 1.05],
        hspace=0.52, wspace=0.65,
    )

    # Row 1: panels A, B, C (cols 0-1, 2-3, 4-5)
    ax_A = fig.add_subplot(gs[0, 0:2])
    ax_B = fig.add_subplot(gs[0, 2:4])
    ax_C = fig.add_subplot(gs[0, 4:6])
    # Row 2: panels D, E, F (cols 0-1, 2-3, 4-5)
    ax_D = fig.add_subplot(gs[1, 0:2])
    ax_E = fig.add_subplot(gs[1, 2:4])
    ax_F = fig.add_subplot(gs[1, 4:6])
    # Row 3: world maps at smallest and largest ratio (cols 0-2, 3-5)
    ax_M1 = fig.add_subplot(gs[2, 0:3])
    ax_M2 = fig.add_subplot(gs[2, 3:6])

    x = value_ratio_grid
    x_idx = np.arange(len(value_ratio_grid))
    band_label = "ABR IQR"
    planner_band_label = "Planner IQR"
    abr_marker_kwargs = dict(marker="o", ms=3.5, mec=ABR_COLOR, mfc=ABR_COLOR)
    plan_marker_kwargs = dict(marker="s", ms=3.2, mec=PLAN_COLOR, mfc=PLAN_COLOR)

    # Panel A: welfare (bar chart)
    ax = ax_A
    width = 0.36
    ax.bar(
        x_idx - width / 2, welfare_med, width=width, color=ABR_COLOR, alpha=0.75,
        label="ABR median",
        yerr=np.vstack([welfare_med - welfare_lo, welfare_hi - welfare_med]),
        error_kw=dict(ecolor=ABR_COLOR, elinewidth=1.0, capsize=2),
    )
    ax.bar(
        x_idx + width / 2, welfare_opt_med, width=width, color=PLAN_COLOR, alpha=0.55,
        label="Planner median",
        yerr=np.vstack([welfare_opt_med - welfare_opt_lo, welfare_opt_hi - welfare_opt_med]),
        error_kw=dict(ecolor=PLAN_COLOR, elinewidth=1.0, capsize=2),
    )
    bar_tick_positions = [
        i for i, r in enumerate(value_ratio_grid)
        if r in {1.0, 2.0, 5.0, 10.0, 20.0}
    ]
    bar_tick_labels = [
        format_ratio_label(value_ratio_grid[i]) for i in bar_tick_positions
    ]
    ax.set_xticks(bar_tick_positions)
    ax.set_xticklabels(bar_tick_labels)
    ax.set_xlabel("Expected per-source value ratio (high / low)")
    ax.set_ylabel("Expected welfare")
    ax.set_title("(A) Welfare")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    # Panel B: welfare ratio
    ax = ax_B
    ax.plot(x, wr_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median", **abr_marker_kwargs)
    ax.fill_between(x, wr_lo, wr_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.axhline(0.5, ls="--", color="gray", lw=1.0, label="PoA floor = 0.5")
    ax.axhline(1.0, ls=":", color="black", lw=0.8, alpha=0.6)
    set_ratio_axis(ax, x)
    ax.set_ylabel(r"$W_{\rm ABR}\,/\,W^*$")
    ax.set_ylim(0.45, 1.05)
    ax.set_title("(B) Welfare ratio")
    ax.legend(fontsize=8, loc="lower left")

    # Panel C: geographic HHI
    ax = ax_C
    ax.plot(x, geo_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median", **abr_marker_kwargs)
    ax.fill_between(x, geo_lo, geo_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, geo_opt_med, "-.", color=PLAN_COLOR, lw=1.5, label="Planner median",
            **plan_marker_kwargs)
    ax.fill_between(x, geo_opt_lo, geo_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.axhline(1 / K, ls=":", color="black", lw=1.0, alpha=0.6,
               label=rf"Floor $1/K = {1/K:.3f}$")
    set_ratio_axis(ax, x)
    ax.set_ylabel("Geographic HHI")
    ax.set_title("(C) Geographic HHI")
    ax.legend(fontsize=8)

    # Panel D: utility HHI
    ax = ax_D
    ax.plot(x, util_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median", **abr_marker_kwargs)
    ax.fill_between(x, util_lo, util_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.axhline(9 / (8 * K), ls="--", color="gray", lw=1.0,
               label=rf"NE ceiling $9/(8K)$")
    ax.axhline(1 / K, ls=":", color="black", lw=1.0, alpha=0.6,
               label=rf"Egalitarian $1/K$")
    ax.set_ylim(1 / K - 0.005, 9 / (8 * K) + 0.005)
    set_ratio_axis(ax, x)
    ax.set_ylabel("Utility HHI")
    ax.set_title("(D) Utility HHI")
    ax.legend(fontsize=8)

    # Panel E: high-value cluster coverage
    ax = ax_E
    ax.plot(x, cov_hi_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median", **abr_marker_kwargs)
    ax.fill_between(x, cov_hi_lo, cov_hi_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, cov_hi_opt_med, "-.", color=PLAN_COLOR, lw=1.5, label="Planner median",
            **plan_marker_kwargs)
    ax.fill_between(x, cov_hi_opt_lo, cov_hi_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    set_ratio_axis(ax, x)
    ax.set_ylabel("Coverage fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(E) High-value cluster coverage")
    ax.legend(fontsize=8, loc="lower right")

    # Panel F: peripheral cluster coverage
    ax = ax_F
    ax.plot(x, cov_pe_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median", **abr_marker_kwargs)
    ax.fill_between(x, cov_pe_lo, cov_pe_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, cov_pe_opt_med, "-.", color=PLAN_COLOR, lw=1.5,
            label="Planner median", **plan_marker_kwargs)
    ax.fill_between(x, cov_pe_opt_lo, cov_pe_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    set_ratio_axis(ax, x)
    ax.set_ylabel("Coverage fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(F) Peripheral cluster coverage")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    # Bottom row: world maps at smallest and largest ratio with data
    ratios_with_data = [r for r in value_ratio_grid if abr_runs_by_ratio.get(r)]
    if len(ratios_with_data) >= 2:
        map_ratios = [ratios_with_data[0], ratios_with_data[-1]]
    elif len(ratios_with_data) == 1:
        map_ratios = [ratios_with_data[0], ratios_with_data[0]]
    else:
        map_ratios = []

    for ax_map, ratio in zip([ax_M1, ax_M2], map_ratios):
        profiles = [r["final_profile"] for r in abr_runs_by_ratio.get(ratio, [])]
        counts = builder_counts(profiles, region_names)
        plot_world_map(ax_map, counts, region_names,
                       title=rf"high/low value ratio = {ratio:g}x  (mean ABR placement)")

    fig.suptitle(
        rf"Experiment 1: Value asymmetry sweep "
        rf"($K={K}$, $\Delta={int(delta*1000)}\,\mathrm{{ms}}$, "
        rf"{n_instances} source instances $\times$ {n_seeds_per_instance} inits)",
        fontsize=12, y=0.995,
    )
    out = FIGURES_DIR / "exp1_value_asymmetry.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    print(f"Saved {out}")
