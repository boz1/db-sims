import numpy as np

from scripts.plot_common import ABR_COLOR, PLAN_COLOR, FIGURES_DIR, plt, summarize


def set_delta_axis(ax, x_vals_ms):
    """Log-scale delta axis with sensible tick labels."""
    ax.set_xscale("log")
    major_ticks = [10, 50, 100, 1000, 12000]
    major_labels = ["10", "50", "100", "1000", "12000"]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_labels)
    ax.set_xlim(min(x_vals_ms) * 0.85, max(x_vals_ms) * 1.18)
    ax.set_xlabel(r"Slot duration $\Delta$ (ms)")

def plot(delta_grid_ms, delta_grid, abr_runs_by_delta, planner_runs_by_delta, K,
         n_runs_per_delta, value_ratio=10.0, alpha=None, delta_anchor=0.05,
         n_instances=5, n_seeds_per_instance=3):
    """abr_runs_by_delta[delta] = list of run dicts.
    planner_runs_by_delta[delta] = list of planner dicts (one per instance)."""

    def _abr_agg(key):
        meds, los, his = [], [], []
        for d in delta_grid:
            vals = [x[key] for x in abr_runs_by_delta[d]]
            m, lo, hi = summarize(vals)
            meds.append(m); los.append(lo); his.append(hi)
        return np.array(meds), np.array(los), np.array(his)

    def _plan_agg(key):
        meds, los, his = [], [], []
        for d in delta_grid:
            vals = [x[key] for x in planner_runs_by_delta[d]]
            m, lo, hi = summarize(vals)
            meds.append(m); los.append(lo); his.append(hi)
        return np.array(meds), np.array(los), np.array(his)

    wr_med, wr_lo, wr_hi = [], [], []
    for d in delta_grid:
        w_opt_by_inst = {
            p["instance_idx"]: p["w_opt"] for p in planner_runs_by_delta[d]
        }
        ratios = []
        for r in abr_runs_by_delta[d]:
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
    mpd_med, mpd_lo, mpd_hi = _abr_agg("mean_pairwise_km")

    geo_opt_med, geo_opt_lo, geo_opt_hi = _plan_agg("geo_hhi_opt")
    cov_hi_opt_med, cov_hi_opt_lo, cov_hi_opt_hi = _plan_agg("cov_high_opt")
    cov_pe_opt_med, cov_pe_opt_lo, cov_pe_opt_hi = _plan_agg("cov_peripheral_opt")
    mpd_opt_med, mpd_opt_lo, mpd_opt_hi = _plan_agg("mean_pairwise_km_opt")

    fig = plt.figure(figsize=(14.5, 13.5))
    gs = fig.add_gridspec(
        nrows=3, ncols=6,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.52, wspace=0.65,
    )

    # Row 1: panels A, B, C
    ax_A = fig.add_subplot(gs[0, 0:2])
    ax_B = fig.add_subplot(gs[0, 2:4])
    ax_C = fig.add_subplot(gs[0, 4:6])
    # Row 2: panels D, E, F
    ax_D = fig.add_subplot(gs[1, 0:2])
    ax_E = fig.add_subplot(gs[1, 2:4])
    ax_F = fig.add_subplot(gs[1, 4:6])
    # Row 3: panel G (mean pairwise distance), centered, narrower
    ax_G = fig.add_subplot(gs[2, 1:5])

    x_ms = np.array(delta_grid_ms, dtype=float)
    anchor_x = delta_anchor * 1000

    band_label = "ABR IQR"
    planner_band_label = "Planner IQR"
    abr_marker_kwargs = dict(marker="o", ms=3.5, mec=ABR_COLOR, mfc=ABR_COLOR)
    plan_marker_kwargs = dict(marker="s", ms=3.2, mec=PLAN_COLOR, mfc=PLAN_COLOR)
    anchor_label = f"Anchor {int(anchor_x)} ms"

    # Panel A: welfare (bar chart)
    ax = ax_A
    x_idx = np.arange(len(delta_grid))
    width = 0.38
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
    bar_tick_positions = list(range(0, len(delta_grid), 2))
    if (len(delta_grid) - 1) not in bar_tick_positions:
        bar_tick_positions.append(len(delta_grid) - 1)
    ax.set_xticks(bar_tick_positions)
    ax.set_xticklabels([f"{delta_grid_ms[i]}" for i in bar_tick_positions])
    anchor_idx = delta_grid_ms.index(int(round(anchor_x)))
    ax.axvspan(anchor_idx - 0.5, anchor_idx + 0.5,
               color="black", alpha=0.06, label=anchor_label)
    ax.set_xlabel(r"Slot duration $\Delta$ (ms)")
    ax.set_ylabel("Expected welfare")
    ax.set_title("(A) Welfare")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    # Panel B: welfare ratio
    ax = ax_B
    ax.plot(x_ms, wr_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median",
            **abr_marker_kwargs)
    ax.fill_between(x_ms, wr_lo, wr_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.axhline(0.5, ls="--", color="gray", lw=1.0, label="PoA floor = 0.5")
    ax.axhline(1.0, ls=":", color="black", lw=0.8, alpha=0.6)
    ax.axvline(anchor_x, ls=":", color="black", lw=1.0, alpha=0.6, label=anchor_label)
    set_delta_axis(ax, x_ms)
    ax.set_ylabel(r"$W_{\rm ABR}\,/\,W^*$")
    ax.set_ylim(0.45, 1.05)
    ax.set_title("(B) Welfare ratio")
    ax.legend(fontsize=8, loc="lower left")

    # Panel C: geographic HHI
    ax = ax_C
    ax.plot(x_ms, geo_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median",
            **abr_marker_kwargs)
    ax.fill_between(x_ms, geo_lo, geo_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x_ms, geo_opt_med, "-.", color=PLAN_COLOR, lw=1.5, label="Planner median",
            **plan_marker_kwargs)
    ax.fill_between(x_ms, geo_opt_lo, geo_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.axhline(1 / K, ls=":", color="black", lw=1.0, alpha=0.6,
               label=rf"Floor $1/K = {1/K:.3f}$")
    ax.axvline(anchor_x, ls=":", color="black", lw=1.0, alpha=0.6)
    set_delta_axis(ax, x_ms)
    ax.set_ylabel("Geographic HHI")
    ax.set_title("(C) Geographic HHI")
    ax.legend(fontsize=8)

    # Panel D: utility HHI
    ax = ax_D
    ax.plot(x_ms, util_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median",
            **abr_marker_kwargs)
    ax.fill_between(x_ms, util_lo, util_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.axhline(9 / (8 * K), ls="--", color="gray", lw=1.0,
               label=rf"NE ceiling $9/(8K)$")
    ax.axhline(1 / K, ls=":", color="black", lw=1.0, alpha=0.6,
               label=rf"Egalitarian $1/K$")
    ax.axvline(anchor_x, ls=":", color="black", lw=1.0, alpha=0.6)
    ax.set_ylim(1 / K - 0.005, 9 / (8 * K) + 0.005)
    set_delta_axis(ax, x_ms)
    ax.set_ylabel("Utility HHI")
    ax.set_title("(D) Utility HHI")
    ax.legend(fontsize=8)

    # Panel E: high-value cluster coverage
    ax = ax_E
    ax.plot(x_ms, cov_hi_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median",
            **abr_marker_kwargs)
    ax.fill_between(x_ms, cov_hi_lo, cov_hi_hi, color=ABR_COLOR, alpha=0.18,
                    label=band_label)
    ax.plot(x_ms, cov_hi_opt_med, "-.", color=PLAN_COLOR, lw=1.5,
            label="Planner median", **plan_marker_kwargs)
    ax.fill_between(x_ms, cov_hi_opt_lo, cov_hi_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.axvline(anchor_x, ls=":", color="black", lw=1.0, alpha=0.6)
    set_delta_axis(ax, x_ms)
    ax.set_ylabel("Coverage fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(E) High-value cluster coverage")
    ax.legend(fontsize=8, loc="lower right")

    # Panel F: peripheral cluster coverage
    ax = ax_F
    ax.plot(x_ms, cov_pe_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median",
            **abr_marker_kwargs)
    ax.fill_between(x_ms, cov_pe_lo, cov_pe_hi, color=ABR_COLOR, alpha=0.18,
                    label=band_label)
    ax.plot(x_ms, cov_pe_opt_med, "-.", color=PLAN_COLOR, lw=1.5,
            label="Planner median", **plan_marker_kwargs)
    ax.fill_between(x_ms, cov_pe_opt_lo, cov_pe_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.axvline(anchor_x, ls=":", color="black", lw=1.0, alpha=0.6)
    set_delta_axis(ax, x_ms)
    ax.set_ylabel("Coverage fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(F) Peripheral cluster coverage")
    ax.legend(fontsize=8, loc="lower right")

    # Panel G: mean pairwise distance (km)
    ax = ax_G
    ax.plot(x_ms, mpd_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median",
            **abr_marker_kwargs)
    ax.fill_between(x_ms, mpd_lo, mpd_hi, color=ABR_COLOR, alpha=0.18,
                    label=band_label)
    ax.plot(x_ms, mpd_opt_med, "-.", color=PLAN_COLOR, lw=1.5,
            label="Planner median", **plan_marker_kwargs)
    ax.fill_between(x_ms, mpd_opt_lo, mpd_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.axvline(anchor_x, ls=":", color="black", lw=1.0, alpha=0.6, label=anchor_label)
    set_delta_axis(ax, x_ms)
    ax.set_ylabel("Mean pairwise distance (km)")
    ax.set_title("(G) Mean pairwise distance")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        rf"Experiment 4: Slot duration sweep "
        rf"($K={K}$, value ratio = {value_ratio:g}x [$\alpha \approx {(alpha if alpha is not None else 0.0):.3f}$], "
        rf"{n_instances} source instances $\times$ {n_seeds_per_instance} inits)",
        fontsize=12, y=0.995,
    )
    out = FIGURES_DIR / "exp4_slot_duration.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    print(f"Saved {out}")
