import numpy as np

from scripts.plot_common import (
    ABR_COLOR,
    PLAN_COLOR,
    FIGURES_DIR,
    builder_counts,
    plot_world_map,
    plt,
    summarize,
)


def plot(K_grid, abr_runs_by_K, planner_runs_by_K, n_runs_per_K, region_names,
         alpha=0.9, delta=0.05, n_instances=5, n_seeds_per_instance=3):
    """abr_runs_by_K[K] = list of run dicts.
    planner_runs_by_K[K] = list of planner dicts (one per instance)."""

    def _abr_agg(key):
        meds, los, his = [], [], []
        for K in K_grid:
            vals = [r[key] for r in abr_runs_by_K[K]]
            m, lo, hi = summarize(vals)
            meds.append(m); los.append(lo); his.append(hi)
        return np.array(meds), np.array(los), np.array(his)

    def _plan_agg(key):
        meds, los, his = [], [], []
        for K in K_grid:
            vals = [r[key] for r in planner_runs_by_K[K]]
            m, lo, hi = summarize(vals)
            meds.append(m); los.append(lo); his.append(hi)
        return np.array(meds), np.array(los), np.array(his)

    wr_med, wr_lo, wr_hi = [], [], []
    for K in K_grid:
        w_opt_by_inst = {p["instance_idx"]: p["w_opt"] for p in planner_runs_by_K[K]}
        ratios = []
        for r in abr_runs_by_K[K]:
            w_opt = w_opt_by_inst.get(r["instance_idx"])
            if w_opt is not None and w_opt > 1e-12:
                ratios.append(r["welfare"] / w_opt)
        m, lo, hi = summarize(ratios)
        wr_med.append(m); wr_lo.append(lo); wr_hi.append(hi)
    wr_med = np.array(wr_med); wr_lo = np.array(wr_lo); wr_hi = np.array(wr_hi)

    geo_med, geo_lo, geo_hi = _abr_agg("geo_hhi")
    util_med, util_lo, util_hi = _abr_agg("utility_hhi")
    cov_hi_med, cov_hi_lo, cov_hi_hi = _abr_agg("cov_high")
    cov_pe_med, cov_pe_lo, cov_pe_hi = _abr_agg("cov_peripheral")
    mpd_med, mpd_lo, mpd_hi = _abr_agg("mean_pairwise_km")

    geo_opt_med, geo_opt_lo, geo_opt_hi = _plan_agg("geo_hhi_opt")
    cov_hi_opt_med, cov_hi_opt_lo, cov_hi_opt_hi = _plan_agg("cov_high_opt")
    cov_pe_opt_med, cov_pe_opt_lo, cov_pe_opt_hi = _plan_agg("cov_peripheral_opt")
    mpd_opt_med, mpd_opt_lo, mpd_opt_hi = _plan_agg("mean_pairwise_km_opt")

    x = np.array(K_grid)
    welfare_floor = np.array([1.0 / (2.0 - 1.0 / K) for K in K_grid])
    geo_floor = np.array([1.0 / K for K in K_grid])
    util_ceiling = np.array([9.0 / (8.0 * K) for K in K_grid])
    util_egalitarian = np.array([1.0 / K for K in K_grid])

    band_label = (
        f"ABR IQR "
        f"({n_instances} source instances x {n_seeds_per_instance} init)"
    )
    planner_band_label = f"Planner IQR ({n_instances} source instances)"

    # Layout: top row 2 panels, middle row 3 panels, bottom row 3 panels.
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(
        nrows=3, ncols=6,
        height_ratios=[1.0, 1.0, 1.05],
        hspace=0.45, wspace=0.55,
    )

    # Row 1: panels A and B
    ax_A = fig.add_subplot(gs[0, 0:3])
    ax_B = fig.add_subplot(gs[0, 3:6])
    # Row 2: panels C, D, E
    ax_C = fig.add_subplot(gs[1, 0:2])
    ax_D = fig.add_subplot(gs[1, 2:4])
    ax_E = fig.add_subplot(gs[1, 4:6])
    # Row 3: mean pairwise distance and two maps
    ax_F = fig.add_subplot(gs[2, 0:2])
    ax_M1 = fig.add_subplot(gs[2, 2:4])
    ax_M2 = fig.add_subplot(gs[2, 4:6])

    # Panel A: welfare ratio
    ax = ax_A
    ax.plot(x, wr_med, "-o", color=ABR_COLOR, lw=1.5, ms=4, label="ABR (median)")
    ax.fill_between(x, wr_lo, wr_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, welfare_floor, "--", color="gray", lw=1.2, label=r"Floor $1/(2-1/K)$")
    ax.axhline(1.0, ls=":", color="black", lw=0.8, alpha=0.6)
    ax.set_xlabel("Number of builders $K$")
    ax.set_ylabel(r"$W_{\rm ABR}\,/\,W^*$")
    ax.set_ylim(0.45, 1.05)
    ax.set_title("(A) Welfare ratio")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Panel B: geographic HHI
    ax = ax_B
    ax.plot(x, geo_med, "-o", color=ABR_COLOR, lw=1.5, ms=4, label="ABR (median)")
    ax.fill_between(x, geo_lo, geo_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, geo_opt_med, "-.", color=PLAN_COLOR, lw=1.5, label="Planner median")
    ax.fill_between(x, geo_opt_lo, geo_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.plot(x, geo_floor, ":", color="black", lw=1.0, alpha=0.7, label=r"Floor $1/K$")
    ax.set_xlabel("Number of builders $K$")
    ax.set_ylabel("Geographic HHI")
    ax.set_title("(B) Geographic HHI")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Panel C: utility HHI
    ax = ax_C
    ax.plot(x, util_med, "-o", color=ABR_COLOR, lw=1.5, ms=4, label="ABR (median)")
    ax.fill_between(x, util_lo, util_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, util_ceiling, "--", color="gray", lw=1.2, label=r"NE ceiling $9/(8K)$")
    ax.plot(x, util_egalitarian, ":", color="black", lw=1.0, alpha=0.7,
            label=r"Egalitarian $1/K$")
    ax.set_xlabel("Number of builders $K$")
    ax.set_ylabel("Utility HHI")
    ax.set_title("(C) Utility HHI")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Panel D: high-value cluster coverage
    ax = ax_D
    ax.plot(x, cov_hi_med, "-o", color=ABR_COLOR, lw=1.5, ms=4, label="ABR (median)")
    ax.fill_between(x, cov_hi_lo, cov_hi_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, cov_hi_opt_med, "-.", color=PLAN_COLOR, lw=1.5, label="Planner median")
    ax.fill_between(x, cov_hi_opt_lo, cov_hi_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.set_xlabel("Number of builders $K$")
    ax.set_ylabel("Coverage fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(D) High-value cluster coverage")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)

    # Panel E: peripheral cluster coverage
    ax = ax_E
    ax.plot(x, cov_pe_med, "-o", color=ABR_COLOR, lw=1.5, ms=4, label="ABR (median)")
    ax.fill_between(x, cov_pe_lo, cov_pe_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, cov_pe_opt_med, "-.", color=PLAN_COLOR, lw=1.5, label="Planner median")
    ax.fill_between(x, cov_pe_opt_lo, cov_pe_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.set_xlabel("Number of builders $K$")
    ax.set_ylabel("Coverage fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(E) Peripheral cluster coverage")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)

    # Panel F: mean pairwise distance
    ax = ax_F
    ax.plot(x, mpd_med, "-o", color=ABR_COLOR, lw=1.5, ms=4, label="ABR (median)")
    ax.fill_between(x, mpd_lo, mpd_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, mpd_opt_med, "-.", color=PLAN_COLOR, lw=1.5, label="Planner median")
    ax.fill_between(x, mpd_opt_lo, mpd_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.set_xlabel("Number of builders $K$")
    ax.set_ylabel("Mean pairwise distance (km)")
    ax.set_title("(F) Mean pairwise distance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Bottom row: world maps at smallest and largest K with data.
    Ks_with_data = [K for K in K_grid if abr_runs_by_K.get(K)]
    if len(Ks_with_data) >= 2:
        map_Ks = [Ks_with_data[0], Ks_with_data[-1]]
    elif len(Ks_with_data) == 1:
        map_Ks = [Ks_with_data[0], Ks_with_data[0]]
    else:
        map_Ks = []

    for ax_map, K in zip([ax_M1, ax_M2], map_Ks):
        profiles = [r["final_profile"] for r in abr_runs_by_K.get(K, [])]
        counts = builder_counts(profiles, region_names)
        plot_world_map(ax_map, counts, region_names,
                       title=rf"$K = {K}$  (mean ABR placement)",
                       size_scale=55)

    fig.suptitle(
        rf"Experiment 2: Builder count sweep "
        rf"($\alpha={alpha}$, $\Delta={int(delta*1000)}\,\mathrm{{ms}}$, "
        rf"{n_instances} source instances $\times$ {n_seeds_per_instance} inits)",
        fontsize=12, y=0.995,
    )
    out = FIGURES_DIR / "exp2_builder_count.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    print(f"Saved {out}")
