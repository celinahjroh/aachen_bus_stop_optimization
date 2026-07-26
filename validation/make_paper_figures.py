"""
================================================================================
 Aachen Public Transport Coverage — REGENERATE EVERY PAPER FIGURE & TABLE
================================================================================
 Convenience entry point that (re)builds every figure and every data-driven
 table used in the report, using the exact filenames referenced by the LaTeX
 sources. It reads the cached result CSVs in results/ and only renders them, so
 it finishes in a few seconds.

 If the required result CSVs are not present yet, the full analysis pipeline is
 run first to produce them (this is the slow path). On a repository that has
 already been analysed once, regeneration is near-instant.

     python validation/make_paper_figures.py

 Produced (figures/  and  results/):
   fig31_data_overview.png        table_sweep.tex
   fig32_cost_distribution.png    table_equity.tex
   fig_underserved_map.png        table_final.tex
   fig_pareto_cost_coverage.png
   fig_avgwalk_budget.png
   fig_price_of_equity.png
   fig_poi_catchments.png
   fig_plan_maps.png
   final_who_benefits.png
   fig_nkv_map.png

 The TikZ roadmap (Ch. 6) is drawn in LaTeX and is deliberately not produced.
================================================================================
"""
from __future__ import annotations

import ast
import os

import numpy as np
import pandas as pd

import aachen_cba as CBA
import aachen_final as F
import aachen_model as A
import paper_assets

OUT = A.OUT
FIG = A.FIG

# Result CSVs required to render the figures/tables without re-solving.
_REQUIRED = [
    "01_pareto_costblind.csv",
    "02_pareto_costaware.csv",
    "03_equity_scenarios.csv",
    "07_per_stop_access_BCR.csv",
    "12_final_plan_comparison.csv",
]


def _results_present():
    return all(os.path.exists(f"{OUT}/{name}") for name in _REQUIRED)


def _plan_indices(stop_ids, cid):
    id_to_index = {int(cid[c]): c for c in range(len(cid))}
    return [id_to_index[int(s)] for s in stop_ids]


def _plan_from_final(final_df, plan_name, cid):
    row = final_df.loc[final_df["plan"] == plan_name]
    if row.empty:
        return []
    return _plan_indices(ast.literal_eval(row.iloc[0]["selected_stop_ids"]), cid)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    if not _results_present():
        print("[make_paper_figures] result CSVs missing; running full pipeline first.")
        F.main()  # produces every result CSV (slow path)

    # Recompute the lightweight spatial inputs (no optimisation involved).
    demand, pop, central, exy, cxy, cid = A.load()
    nearest_e, dC, improves = A.distances(demand, exy, cxy)
    cost, _ = A.build_cost_model(cxy, dC, improves, pop)
    equity_mask = central & (nearest_e > A.R_EQUITY)

    pois = F.load_pois()
    catchment, poi_underserved, _ = F.poi_masks(pois, demand, nearest_e)
    poi_fixable = poi_underserved & (dC.min(axis=1) <= F.R_ACCESS)

    underserved_ceiling = (
        100 * pop[equity_mask & (dC.min(axis=1) <= A.R_EQUITY)].sum()
        / pop[equity_mask].sum()
        if equity_mask.any() else 0.0
    )

    # Cached result tables.
    costblind_df = pd.read_csv(f"{OUT}/01_pareto_costblind.csv")
    costaware_df = pd.read_csv(f"{OUT}/02_pareto_costaware.csv")
    equity_df = pd.read_csv(f"{OUT}/03_equity_scenarios.csv")
    per_stop_df = pd.read_csv(f"{OUT}/07_per_stop_access_BCR.csv")
    final_df = pd.read_csv(f"{OUT}/12_final_plan_comparison.csv")

    # ---- Chapter 3 figures -------------------------------------------------
    paper_assets.fig_data_overview(
        demand, pop, exy, cxy, pois, f"{FIG}/fig31_data_overview.png"
    )
    paper_assets.fig_cost_distribution(cost, f"{FIG}/fig32_cost_distribution.png")

    # ---- Chapter 4 figures -------------------------------------------------
    paper_assets.fig_underserved_map(
        demand, pop, central, equity_mask, f"{FIG}/fig_underserved_map.png"
    )
    paper_assets.fig_pareto_cost_coverage(
        costblind_df, costaware_df, f"{FIG}/fig_pareto_cost_coverage.png",
        mean_cost_k=float(cost.mean()),
    )
    paper_assets.fig_avgwalk_budget(costaware_df, f"{FIG}/fig_avgwalk_budget.png")
    paper_assets.fig_price_of_equity(
        equity_df, underserved_ceiling, f"{FIG}/fig_price_of_equity.png"
    )
    paper_assets.fig_poi_catchments(
        demand, catchment, poi_underserved, poi_fixable, cxy,
        f"{FIG}/fig_poi_catchments.png",
    )

    # ---- Chapter 5 figures -------------------------------------------------
    plan_a = _plan_from_final(final_df, "cost_aware_360k", cid)
    plan_b = _plan_from_final(final_df, "equity_first_50pct", cid)
    paper_assets.fig_plan_maps(
        demand,
        equity_mask,
        cxy[plan_a] if plan_a else np.empty((0, 2)),
        cxy[plan_b] if plan_b else np.empty((0, 2)),
        [int(cid[c]) for c in plan_b],
        f"{FIG}/fig_plan_maps.png",
    )
    paper_assets.fig_who_benefits(final_df, f"{FIG}/final_who_benefits.png")

    bcr_by_id = dict(
        zip(
            per_stop_df["stop_id"].astype(int),
            pd.to_numeric(per_stop_df["access_BCR"], errors="coerce"),
        )
    )
    paper_assets.fig_nkv_map(cxy, cid, bcr_by_id, f"{FIG}/fig_nkv_map.png")

    # ---- Data-driven LaTeX tables -----------------------------------------
    paper_assets.table_sweep(costaware_df, f"{OUT}/table_sweep.tex")
    paper_assets.table_equity(equity_df, f"{OUT}/table_equity.tex")
    paper_assets.table_final(final_df, f"{OUT}/table_final.tex")

    # Appendix POI table: baseline walk from each facility to nearest existing
    # platform (recomputed here so the fast path needs no extra CSV).
    poi_detail = pois[["name", "type"]].copy()
    poi_detail["stop_walk_m_baseline"] = F.poi_stop_distances(
        pois, exy, cxy, []
    ).round(0)
    paper_assets.table_poi(poi_detail, f"{OUT}/table_poi.tex")

    print("Regenerated 10 figures -> figures/ and 4 LaTeX tables -> results/")


if __name__ == "__main__":
    main()
