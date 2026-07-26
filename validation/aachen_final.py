"""
================================================================================
 Aachen Public Transport Coverage — POI ANALYSIS + FINAL PLAN COMPARISON
================================================================================
 This file combines:
   - the corrected school/hospital coordinate data;
   - the POI-priority equity scenario;
   - the final comparison of the status quo, cost-aware, equity-first,
     POI-priority and net-benefit plans.

 The POI coordinates in pois_aachen.csv were re-geocoded from official facility
 addresses. The analysis uses the same 300 m accessibility benchmark and 400 m
 POI catchment definition as the report.

 Run:
     python aachen_final.py
     python aachen_final.py --poi-only
================================================================================
"""
from __future__ import annotations

import ast
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import aachen_cba as CBA
import aachen_model as A
import paper_assets  # figure/table generators (exact paper filenames)

# Reuse the repository-root-anchored output directories defined in aachen_model.
OUT = A.OUT
FIG = A.FIG
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

R_POI_CATCH = 400
R_ACCESS = 300  # locally interpretable accessibility benchmark used in the study


# ============================================================ POI machinery
def load_pois(path=None):
    if path is None:
        path = A.DATA / "pois_aachen.csv"
    pois = pd.read_csv(path, comment="#")
    if not {"x", "y"}.issubset(pois.columns):
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:25832", always_xy=True
        )
        pois["x"], pois["y"] = transformer.transform(
            pois["lon"].to_numpy(), pois["lat"].to_numpy()
        )
    return pois


def poi_masks(pois, demand, nearest_e):
    """Return POI catchment and POI-underserved demand masks."""
    distance_to_poi = (
        cKDTree(pois[["x", "y"]].to_numpy()).query(demand)[0] * A.F
    )
    catchment = distance_to_poi <= R_POI_CATCH
    underserved = catchment & (nearest_e > R_ACCESS)
    return catchment, underserved, distance_to_poi


def poi_stop_distances(pois, exy, cxy, opened_idx):
    """Walking distance from every POI to its nearest existing or opened stop."""
    stops = exy if not opened_idx else np.vstack([exy, cxy[opened_idx]])
    return (
        cKDTree(stops).query(pois[["x", "y"]].to_numpy())[0] * A.F
    )


# ============================================================ PART C: POI analysis
def part_c_poi(
    demand,
    pop,
    central,
    exy,
    cxy,
    cid,
    nearest_e,
    dC,
    improves,
    cost,
):
    del central  # retained in the signature for compatibility

    pois = load_pois()
    catchment, underserved, _ = poi_masks(pois, demand, nearest_e)

    # Chapter 3 data-overview map (needs the POI coordinates resolved above).
    paper_assets.fig_data_overview(
        demand, pop, exy, cxy, pois, f"{FIG}/fig31_data_overview.png"
    )

    print(
        f"[C] POIs: {len(pois)} "
        f"({(pois.type == 'hospital').sum()} hospitals, "
        f"{(pois.type == 'school').sum()} schools)"
    )
    print(
        f"[C] catchment population (<= {R_POI_CATCH} m of a POI): "
        f"{pop[catchment].sum():,.0f}; currently underserved: "
        f"{pop[underserved].sum():,.0f}"
    )

    fixable = underserved & (dC.min(axis=1) <= R_ACCESS)
    underserved_population = float(pop[underserved].sum())
    ceiling = (
        float(pop[fixable].sum()) / underserved_population
        if underserved_population > 0
        else 0.0
    )
    print(
        f"[C] achievable ceiling for the POI-underserved population: "
        f"{ceiling * 100:.1f}%"
    )

    # Chapter 4 facility-catchments map (POI catchment, reachable vs
    # unreachable POI-underserved, candidate platforms overlaid).
    paper_assets.fig_poi_catchments(
        demand, catchment, underserved, fixable, cxy,
        f"{FIG}/fig_poi_catchments.png",
    )

    # The dedicated target equals 80% of the candidate-set ceiling. It is kept
    # unrounded in the optimisation so the rule is implemented exactly.
    if ceiling > 0:
        target_share = min(0.5, 0.8 * ceiling)
        plan_poi, minimum_cost = A.solve_equity_lexicographic(
            pop,
            nearest_e,
            dC,
            improves,
            cost,
            equity_mask=underserved,
            equity_R=R_ACCESS,
            equity_tau=target_share,
        )
        plan_poi = plan_poi or []
    else:
        target_share = 0.0
        plan_poi = []
        minimum_cost = None

    if minimum_cost is not None:
        print(
            f"[C] POI-priority plan: target={target_share * 100:.2f}% of the "
            f"full POI-underserved population, cost={minimum_cost:.0f}k EUR, "
            f"stops={len(plan_poi)}"
        )

    # Facility-side accessibility
    facility_rows = []
    for plan_name, opened in [
        ("baseline", []),
        ("poi_priority", plan_poi),
    ]:
        distances = poi_stop_distances(pois, exy, cxy, opened)
        facility_rows.append(
            {
                "plan": plan_name,
                "mean_poi_stop_walk_m": round(float(distances.mean()), 1),
                "max_poi_stop_walk_m": round(float(distances.max()), 1),
                "pois_within_200m_%": round(100 * (distances <= 200).mean(), 1),
                "pois_within_300m_%": round(100 * (distances <= 300).mean(), 1),
            }
        )

    facility_access = pd.DataFrame(facility_rows)
    facility_access.to_csv(f"{OUT}/10_poi_stop_access.csv", index=False)

    baseline_distances = poi_stop_distances(pois, exy, cxy, [])
    detail = pois[["name", "type"]].copy()
    detail["stop_walk_m_baseline"] = baseline_distances.round(0)
    detail.sort_values(
        "stop_walk_m_baseline", ascending=False
    ).to_csv(f"{OUT}/10b_poi_detail.csv", index=False)

    # Appendix table (tab:poilist), generated from the same detail table.
    paper_assets.table_poi(detail, f"{OUT}/table_poi.tex")

    print(
        "[C] baseline facility-side access: "
        f"{facility_access.iloc[0]['pois_within_300m_%']}% within 300 m; "
        f"mean {facility_access.iloc[0]['mean_poi_stop_walk_m']} m"
    )

    return (
        pois,
        catchment,
        underserved,
        plan_poi,
        target_share,
        minimum_cost,
    )


# ============================================================ PART D: integrated comparison
def _plan_from_budget_file(path, budget, cid):
    selected = pd.read_csv(path)
    row = selected.loc[selected["budget_kEUR"] == budget]
    if row.empty:
        raise ValueError(f"Budget {budget} not found in {path}")

    stop_ids = ast.literal_eval(row.iloc[0]["stops"])
    id_to_index = {int(cid[c]): c for c in range(len(cid))}
    return [id_to_index[int(stop_id)] for stop_id in stop_ids]


def part_d_final(
    demand,
    pop,
    central,
    exy,
    cxy,
    cid,
    nearest_e,
    dC,
    improves,
    cost,
    catchment,
    poi_underserved,
    plan_poi,
    poi_target_share,
):
    del exy  # retained in the signature for compatibility; demand is used below

    equity_mask = central & (nearest_e > A.R_EQUITY)
    benefit = CBA.benefit_matrix(pop, nearest_e, dC, improves)

    plans = {
        "status_quo": [],
        "cost_aware_360k": _plan_from_budget_file(
            f"{OUT}/selected_stops_costaware.csv", 360, cid
        ),
    }

    equity_plan, equity_minimum_cost = A.solve_equity_lexicographic(
        pop,
        nearest_e,
        dC,
        improves,
        cost,
        equity_mask,
        equity_tau=0.5,
    )
    if equity_minimum_cost is not None:
        plans["equity_first_50pct"] = equity_plan

    if plan_poi:
        plans["poi_priority"] = plan_poi

    plans["net_benefit_opt"] = CBA.solve_netbenefit(
        pop, nearest_e, dC, improves, cost, benefit
    )

    catchment_population = float(pop[catchment].sum())
    poi_underserved_population = float(pop[poi_underserved].sum())

    comparison_rows = []
    for plan_name, opened in plans.items():
        indicators, nearest_open = A.kpis(
            opened, pop, nearest_e, dC, cost, central, equity_mask
        )
        appraisal = CBA.plan_appraisal(
            opened, pop, nearest_e, dC, cost, benefit
        )

        comparison_rows.append(
            {
                "plan": plan_name,
                "selected_stop_ids": [int(cid[c]) for c in opened],
                "n_stops": indicators["n_stops"],
                "cost_kEUR": indicators["cost_kEUR"],
                "benefit_kEUR": appraisal["benefit_kEUR"],
                "net_benefit_kEUR": appraisal["net_benefit_kEUR"],
                "access_BCR": appraisal["access_BCR"],
                "avg_walk_m": indicators["avg_walk_m"],
                "central_cov300_%": indicators["ccov300"],
                "underserved_cov300_%": indicators.get(
                    "underserved_now_cov300"
                ),
                "poi_catchment_cov300_%": (
                    round(
                        100
                        * pop[catchment & (nearest_open <= R_ACCESS)].sum()
                        / catchment_population,
                        2,
                    )
                    if catchment_population > 0
                    else None
                ),
                "poi_underserved_cov300_%": (
                    round(
                        100
                        * pop[poi_underserved & (nearest_open <= R_ACCESS)].sum()
                        / poi_underserved_population,
                        2,
                    )
                    if poi_underserved_population > 0
                    else None
                ),
            }
        )

    final = pd.DataFrame(comparison_rows)
    final.to_csv(f"{OUT}/12_final_plan_comparison.csv", index=False)
    print("\n[D] FINAL INTEGRATED PLAN COMPARISON:")
    print(final.to_string(index=False))

    # --- Chapter 5 figures + the integrated-comparison LaTeX table ----------
    # Plan map: cost-aware EUR 360k (blue squares) vs equity-first EUR 450k
    # (magenta circles, labelled with platform IDs), over the underserved.
    plan_a = plans.get("cost_aware_360k", [])
    plan_b = plans.get("equity_first_50pct", [])
    paper_assets.fig_plan_maps(
        demand,
        equity_mask,
        cxy[plan_a] if plan_a else np.empty((0, 2)),
        cxy[plan_b] if plan_b else np.empty((0, 2)),
        [int(cid[c]) for c in plan_b],
        f"{FIG}/fig_plan_maps.png",
    )

    paper_assets.fig_who_benefits(final, f"{FIG}/final_who_benefits.png")

    # Per-candidate access-related BCR map, driven by the isolated screening
    # produced by aachen_cba.main() (results/07_*). Skipped if absent.
    per_stop_path = f"{OUT}/07_per_stop_access_BCR.csv"
    if os.path.exists(per_stop_path):
        per_stop = pd.read_csv(per_stop_path)
        bcr_by_id = dict(
            zip(
                per_stop["stop_id"].astype(int),
                pd.to_numeric(per_stop["access_BCR"], errors="coerce"),
            )
        )
        paper_assets.fig_nkv_map(
            cxy, cid, bcr_by_id, f"{FIG}/fig_nkv_map.png"
        )
    else:
        print("[D] 07_per_stop_access_BCR.csv absent; skipped fig_nkv_map.png")

    paper_assets.table_final(final, f"{OUT}/table_final.tex")

    print(
        f"[D] figures -> {FIG}/fig_plan_maps.png, {FIG}/final_who_benefits.png, "
        f"{FIG}/fig_nkv_map.png; table -> {OUT}/table_final.tex"
    )
    return final


# ============================================================ MAIN
def main(poi_only=False):
    selected_file = f"{OUT}/selected_stops_costaware.csv"
    if not poi_only:
        A.main()
        CBA.main()
    elif not os.path.exists(selected_file):
        print("--poi-only requested without prerequisite Model I results; regenerating them.")
        A.main()

    demand, pop, central, exy, cxy, cid = A.load()
    nearest_e, dC, improves = A.distances(demand, exy, cxy)
    cost, _ = A.build_cost_model(cxy, dC, improves, pop)

    (
        _pois,
        catchment,
        poi_underserved,
        plan_poi,
        target_share,
        _minimum_cost,
    ) = part_c_poi(
        demand,
        pop,
        central,
        exy,
        cxy,
        cid,
        nearest_e,
        dC,
        improves,
        cost,
    )

    part_d_final(
        demand,
        pop,
        central,
        exy,
        cxy,
        cid,
        nearest_e,
        dC,
        improves,
        cost,
        catchment,
        poi_underserved,
        plan_poi,
        target_share,
    )
    print("\nALL DONE — results/ and figures/ contain the regenerated analysis outputs.")


if __name__ == "__main__":
    main(poi_only="--poi-only" in sys.argv)
