"""
================================================================================
 Aachen Public Transport Coverage — PAPER FIGURE & TABLE GENERATION
================================================================================
 Single source of truth for every figure and every data-driven table that
 appears in the report. Each generator writes to the exact filename that the
 LaTeX sources reference via \\includegraphics, so a full pipeline run
 regenerates the paper assets one-to-one.

 Figures (PNG, written to figures/)
 ----------------------------------
   fig31_data_overview.png       input data: demand, platforms, POIs   (Ch. 3)
   fig32_cost_distribution.png   modelled construction-cost histogram  (Ch. 3)
   fig_underserved_map.png       baseline underserved central pockets  (Ch. 4)
   fig_pareto_cost_coverage.png  cost-blind vs cost-aware Pareto        (Ch. 4)
   fig_avgwalk_budget.png        average walk vs budget                 (Ch. 4)
   fig_price_of_equity.png       minimum budget vs underserved target   (Ch. 4)
   fig_poi_catchments.png        facility catchments + structural ceil. (Ch. 4)
   fig_plan_maps.png             two headline plans over the underserved(Ch. 5)
   final_who_benefits.png        coverage by group under each plan      (Ch. 5)
   fig_nkv_map.png               per-candidate access-related BCR       (Ch. 5)

 Tables (LaTeX, written to results/)
 -----------------------------------
   table_sweep.tex   cost-aware budget sweep            (Table "tab:sweep")
   table_equity.tex  price of equity                    (Table "tab:equity")
   table_final.tex   integrated four-plan comparison    (Table "tab:final")

 The TikZ roadmap figure (Ch. 6, "fig:roadmap") is drawn directly in LaTeX and
 is therefore intentionally not produced here.

 This module never solves a model. It receives the already-computed arrays and
 result tables from the pipeline and only renders them, so the figures cannot
 drift from the numbers the optimisation produced.
================================================================================
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ------------------------------------------------------------------ house style
# RWTH Aachen corporate palette, matching the trade-off plots already in the
# pipeline so all figures share one visual identity.
BLUE = "#00549F"
LBLUE = "#8EBAE5"
MAGENTA = "#E30066"
GREEN = "#57AB27"
RED = "#CC071E"
ORANGE = "#F6A800"
GREY = "#9C9E9F"
LGREY = "#D9DADB"
YELLOW = "#FFD200"
PINK = "#F4B7C4"


def _finish_map(ax, title):
    """Common cosmetic treatment for the geographic panels (UTM coordinates)."""
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ============================================================ CHAPTER 3 FIGURES
def fig_data_overview(demand, pop, exy, cxy, pois, path):
    """fig31_data_overview.png — spatial overview of all input datasets."""
    fig, ax = plt.subplots(figsize=(9, 9))

    order = np.argsort(pop)
    ax.scatter(
        demand[order, 0],
        demand[order, 1],
        c=pop[order],
        cmap="Greys",
        s=2,
        alpha=0.65,
        linewidths=0,
        vmax=float(np.quantile(pop, 0.99)),
        zorder=1,
    )
    ax.scatter(
        exy[:, 0], exy[:, 1],
        s=13, facecolors="none", edgecolors=GREY, linewidths=0.5,
        label=f"existing platforms (n={len(exy):,})", zorder=2,
    )
    ax.scatter(
        cxy[:, 0], cxy[:, 1],
        s=26, c=YELLOW, edgecolors="black", linewidths=0.4,
        label=f"candidate platforms (n={len(cxy)})", zorder=5,
    )

    hospitals = pois[pois["type"] == "hospital"]
    schools = pois[pois["type"] == "school"]
    ax.scatter(
        hospitals["x"], hospitals["y"],
        marker="+", c=RED, s=95, linewidths=2.2,
        label=f"hospitals & clinics (n={len(hospitals)})", zorder=6,
    )
    ax.scatter(
        schools["x"], schools["y"],
        marker="^", c=GREEN, s=55, edgecolors="black", linewidths=0.4,
        label=f"secondary schools (n={len(schools)})", zorder=6,
    )

    _finish_map(ax, "Input data: demand, platforms and points of interest")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_cost_distribution(cost, path):
    """fig32_cost_distribution.png — modelled construction-cost histogram."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bins = np.arange(40, 106, 5)
    ax.hist(cost, bins=bins, color=BLUE, edgecolor="white")
    ax.axvline(
        cost.mean(), color=MAGENTA, ls="--", lw=2,
        label=f"mean = \u20ac{cost.mean() * 1000:,.0f}",
    )
    ax.set_xlabel("Modelled construction cost per candidate [k EUR]")
    ax.set_ylabel("Number of candidates")
    ax.set_title(f"Modelled construction costs across {len(cost)} candidates")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================ CHAPTER 4 FIGURES
def fig_underserved_map(demand, pop, central, equity_mask, path):
    """fig_underserved_map.png — baseline underserved central residents."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        demand[central, 0], demand[central, 1],
        s=3, c=LGREY, linewidths=0, label="central demand cells", zorder=1,
    )
    residents = int(round(pop[equity_mask].sum()))
    ax.scatter(
        demand[equity_mask, 0], demand[equity_mask, 1],
        s=9, c=MAGENTA, linewidths=0, zorder=2,
        label=f"underserved (>300 m): {residents:,} residents",
    )
    _finish_map(ax, "Underserved central residents at baseline")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_pareto_cost_coverage(costblind_df, costaware_df, path, mean_cost_k=74.3):
    """fig_pareto_cost_coverage.png — central coverage @300 m vs expenditure."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        costblind_df["cost_kEUR"], costblind_df["ccov300"],
        "o--", color=LBLUE, label="cost-blind (mean price)",
    )
    ax.plot(
        costaware_df["cost_kEUR"], costaware_df["ccov300"],
        "o-", color=BLUE, label="cost-aware (true prices)",
    )
    ax.axvline(
        mean_cost_k, color=GREY, ls=":", lw=1.2,
        label=f"mean candidate cost (\u20ac{mean_cost_k:.1f}k)",
    )
    ax.set_xlabel("Total investment [k EUR]")
    ax.set_ylabel("Central coverage @300 m [%]")
    ax.set_title("Cost\u2013accessibility trade-off")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_avgwalk_budget(costaware_df, path):
    """fig_avgwalk_budget.png — average walking distance vs budget."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(costaware_df["cost_kEUR"], costaware_df["avg_walk_m"], "o-", color=BLUE)
    ax.set_xlabel("Total investment [k EUR]")
    ax.set_ylabel("Average walking distance [m]")
    ax.set_title("Diminishing returns of investment")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_price_of_equity(equity_df, ceiling_pct, path):
    """fig_price_of_equity.png — minimum budget vs required underserved coverage."""
    numeric = equity_df[
        pd.to_numeric(equity_df["min_budget_kEUR"], errors="coerce").notna()
    ].copy()
    numeric["min_budget_kEUR"] = numeric["min_budget_kEUR"].astype(float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        numeric["equity_tau"] * 100, numeric["min_budget_kEUR"],
        "o-", color=BLUE,
    )
    ax.axvline(
        ceiling_pct, color=RED, ls="--", lw=1.6,
        label=f"achievable ceiling ({ceiling_pct:.1f}%)",
    )
    ax.set_xlabel("Share of underserved covered @300 m [%]")
    ax.set_ylabel("Minimum required investment [k EUR]")
    ax.set_title("The price of equity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_poi_catchments(demand, catchment, underserved, fixable, cxy, path):
    """fig_poi_catchments.png — facility catchments and the structural ceiling."""
    covered = underserved & fixable
    uncovered = underserved & ~fixable
    total = float((demand[underserved]).shape[0])
    share_reachable = 100 * covered.sum() / total if total else 0.0
    share_unreachable = 100 * uncovered.sum() / total if total else 0.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        demand[catchment, 0], demand[catchment, 1],
        s=4, c=LBLUE, linewidths=0, zorder=1, label="POI catchment (\u2264400 m)",
    )
    ax.scatter(
        demand[covered, 0], demand[covered, 1],
        s=11, c=ORANGE, linewidths=0, zorder=3,
        label=f"POI-underserved, reachable ({share_reachable:.1f}% by cell)",
    )
    ax.scatter(
        demand[uncovered, 0], demand[uncovered, 1],
        s=11, c=RED, linewidths=0, zorder=3,
        label=f"POI-underserved, unreachable ({share_unreachable:.1f}% by cell)",
    )
    ax.scatter(
        cxy[:, 0], cxy[:, 1],
        s=12, marker="s", facecolors="none", edgecolors="black",
        linewidths=0.5, zorder=4, label="candidate platforms",
    )
    _finish_map(ax, "Facility catchments and the structural ceiling")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================ CHAPTER 5 FIGURES
def fig_plan_maps(demand, equity_mask, plan_a_xy, plan_b_xy, plan_b_ids, path):
    """fig_plan_maps.png — cost-aware (€360k) vs equity-first (€450k) plan."""
    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.scatter(
        demand[equity_mask, 0], demand[equity_mask, 1],
        s=6, c=PINK, linewidths=0, zorder=1, label="underserved residents",
    )
    ax.scatter(
        plan_a_xy[:, 0], plan_a_xy[:, 1],
        s=95, marker="s", c=BLUE, edgecolors="black", linewidths=0.5, zorder=5,
        label=f"cost-aware \u20ac360k ({len(plan_a_xy)} platforms)",
    )
    ax.scatter(
        plan_b_xy[:, 0], plan_b_xy[:, 1],
        s=110, marker="o", facecolors="none", edgecolors=MAGENTA, linewidths=2.2,
        zorder=6, label=f"equity-first \u20ac450k ({len(plan_b_xy)} platforms)",
    )
    for (x, y), stop_id in zip(plan_b_xy, plan_b_ids):
        ax.annotate(
            str(stop_id), (x, y), fontsize=7, fontweight="bold",
            xytext=(5, 5), textcoords="offset points", zorder=7,
        )
    _finish_map(ax, "Two headline plans over the underserved population")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_who_benefits(final_df, path):
    """final_who_benefits.png — coverage by group under each plan."""
    data = final_df.set_index("plan")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    positions = np.arange(len(data))
    width = 0.28

    ax.bar(
        positions - width, data["central_cov300_%"], width,
        label="central population", color=BLUE,
    )
    ax.bar(
        positions, data["underserved_cov300_%"].astype(float), width,
        label="underserved central population", color=MAGENTA,
    )
    ax.bar(
        positions + width, data["poi_underserved_cov300_%"], width,
        label="POI-underserved population", color=GREEN,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(data.index, rotation=15, ha="right")
    ax.set_ylabel("Coverage @300 m [%]")
    ax.set_title("Coverage under alternative planning priorities")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_nkv_map(cxy, cid, bcr_by_id, path):
    """fig_nkv_map.png — isolated access-related BCR of every candidate."""
    bcr = np.array(
        [bcr_by_id.get(int(cid[c]), np.nan) for c in range(len(cid))], dtype=float
    )
    has_value = ~np.isnan(bcr)
    below = has_value & (bcr < 1.0)
    above = has_value & (bcr >= 1.0)
    missing = ~has_value

    fig, ax = plt.subplots(figsize=(8.5, 8))
    if missing.any():
        ax.scatter(
            cxy[missing, 0], cxy[missing, 1],
            s=20, c=LGREY, linewidths=0, zorder=1,
            label="no improvement (excluded)",
        )
    scatter = ax.scatter(
        cxy[above, 0], cxy[above, 1],
        c=bcr[above], cmap="viridis", s=48, edgecolors="black", linewidths=0.3,
        vmin=0, zorder=3,
    )
    ax.scatter(
        cxy[below, 0], cxy[below, 1],
        marker="x", c=RED, s=85, linewidths=2.2, zorder=4,
        label=f"BCR < 1 ({int(below.sum())} candidates)",
    )
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.72)
    colorbar.set_label("Access-related BCR (candidate in isolation)")
    _finish_map(ax, "Per-candidate access-related benefit\u2013cost ratio")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================ LATEX TABLES
def _latex_header(caption, label, column_spec):
    return (
        "% Auto-generated by validation/paper_assets.py — do not edit by hand.\n"
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n"
        f"\\begin{{tabular}}{{{column_spec}}}\n\\toprule\n"
    )


_LATEX_FOOTER = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def table_sweep(costaware_df, path, budgets=(0, 180, 360, 450, 540, 900, 1200)):
    """table_sweep.tex — cost-aware budget sweep (subset of budgets in the paper)."""
    rows = []
    for budget in budgets:
        match = costaware_df.loc[costaware_df["budget_kEUR"] == budget]
        if match.empty:
            continue
        record = match.iloc[0]
        budget_label = "\\num{1200}" if budget == 1200 else f"{budget}"
        rows.append(
            f"\\euro{{}}{budget_label}k & {int(record['n_stops'])} & "
            f"\\SI{{{record['avg_walk_m']:.2f}}}{{\\metre}} & "
            f"\\SI{{{record['ccov200']:.2f}}}{{\\percent}} & "
            f"\\SI{{{record['ccov300']:.2f}}}{{\\percent}} \\\\"
        )

    body = (
        "\\textbf{Budget} & \\textbf{Platforms} & \\textbf{Avg.\\ walk} & "
        "\\textbf{Cov@200} & \\textbf{Cov@300} \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n"
    )
    caption = (
        "Cost-aware budget sweep (Model~I, $\\tau=0$). Counts are directional "
        "platforms; Cov@200 and Cov@300 are \\emph{central-area} coverage at 200 "
        "and \\SI{300}{\\metre}."
    )
    with open(path, "w") as handle:
        handle.write(
            _latex_header(caption, "tab:sweep", "r r r r r") + body + _LATEX_FOOTER
        )


def table_equity(equity_df, path):
    """table_equity.tex — minimum budget per underserved-coverage target."""
    rows = []
    for _, record in equity_df.iterrows():
        budget = record["min_budget_kEUR"]
        if not isinstance(budget, (int, float)) or pd.isna(budget):
            continue
        rows.append(
            f"\\SI{{{int(round(record['equity_tau'] * 100))}}}{{\\percent}} & "
            f"\\euro{{}}{int(round(float(budget)))}k & "
            f"{int(record['n_stops'])} & "
            f"\\SI{{{record['avg_walk_m']:.2f}}}{{\\metre}} \\\\"
        )

    body = (
        "\\textbf{Target $\\tau$} & \\textbf{Min.\\ budget} & "
        "\\textbf{Platforms} & \\textbf{System avg.\\ walk} \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n"
    )
    caption = (
        "The price of equity: minimum budget to cover a share $\\tau$ of the "
        "underserved population within \\SI{300}{\\metre}, and the resulting "
        "system-wide average walking distance."
    )
    with open(path, "w") as handle:
        handle.write(
            _latex_header(caption, "tab:equity", "r r r r") + body + _LATEX_FOOTER
        )


# Human-readable plan labels for the integrated comparison table.
_PLAN_LABELS = {
    "status_quo": "Status quo",
    "cost_aware_360k": "Cost-aware (\\euro{}360k)",
    "equity_first_50pct": "Equity-first (\\SI{50}{\\percent})",
    "poi_priority": "POI-priority",
    "net_benefit_opt": "Net-benefit-optimal",
}


def _money_million(value_k):
    if value_k is None or (isinstance(value_k, float) and pd.isna(value_k)):
        return "---"
    return f"\\euro{{}}{value_k / 1000:.2f}M"


def _money_k(value_k):
    if value_k is None or (isinstance(value_k, float) and pd.isna(value_k)):
        return "---"
    # The paper reports costs below EUR 1M in kEUR and larger costs in millions.
    if value_k >= 1000:
        return f"\\euro{{}}{value_k / 1000:.2f}M"
    return f"\\euro{{}}{int(round(value_k))}k"


def table_final(final_df, path):
    """table_final.tex — integrated four-plan comparison on one KPI panel."""
    rows = []
    for _, record in final_df.iterrows():
        label = _PLAN_LABELS.get(record["plan"], record["plan"])
        if record["plan"] == "status_quo":
            cost = benefit = net = ratio = "---"
        else:
            cost = _money_k(record["cost_kEUR"])
            benefit = _money_million(record["benefit_kEUR"])
            net = _money_million(record["net_benefit_kEUR"])
            ratio = f"{record['access_BCR']:.1f}"
        rows.append(
            f"{label} & {int(record['n_stops'])} & {cost} & {benefit} & {net} & "
            f"{ratio} & \\SI{{{record['avg_walk_m']:.2f}}}{{\\metre}} & "
            f"\\SI{{{record['central_cov300_%']:.1f}}}{{\\percent}} & "
            f"\\SI{{{float(record['underserved_cov300_%'] or 0):.1f}}}{{\\percent}} & "
            f"\\SI{{{float(record['poi_underserved_cov300_%'] or 0):.1f}}}{{\\percent}} \\\\"
        )

    body = (
        "\\textbf{Plan} & \\textbf{Platf.} & \\textbf{Cost} & \\textbf{Benefit} & "
        "\\textbf{Net ben.} & \\textbf{Ratio} & \\textbf{Avg.\\ walk} & "
        "\\textbf{Cov@300} & \\textbf{Unders.} & \\textbf{POI-unders.} \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
    )
    caption = (
        "Integrated comparison of the four planning philosophies. Counts are "
        "directional platforms. Coverage figures refer to the central area at "
        "\\SI{300}{\\metre}. The ratio is the access-related benefit--cost ratio, "
        "not the statutory NKV."
    )
    with open(path, "w") as handle:
        handle.write(
            _latex_header(caption, "tab:final", "l r r r r r r r r r")
            + body
            + _LATEX_FOOTER
        )


def table_poi(poi_detail_df, path):
    """table_poi.tex — appendix list of the 18 POIs and their baseline walking
    distance to the nearest existing directional platform (descending).

    Facility names are emitted exactly as they appear in pois_aachen.csv (ASCII
    transliteration of the German umlauts); the typeset umlauts in the report are
    a cosmetic choice and do not affect the numbers."""
    ordered = poi_detail_df.sort_values(
        "stop_walk_m_baseline", ascending=False
    )
    rows = [
        f"{record['name']} & {record['type']} & "
        f"\\SI{{{int(round(record['stop_walk_m_baseline']))}}}{{\\metre}} \\\\"
        for _, record in ordered.iterrows()
    ]
    body = (
        "\\textbf{Facility} & \\textbf{Type} & "
        "\\textbf{Baseline platform walk} \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n"
    )
    caption = (
        "The 18 points of interest and their baseline walking distance to the "
        "nearest existing directional platform. Distances are modelled as "
        "circuity-adjusted Euclidean distances."
    )
    with open(path, "w") as handle:
        handle.write(
            _latex_header(caption, "tab:poilist", "l l r") + body + _LATEX_FOOTER
        )
