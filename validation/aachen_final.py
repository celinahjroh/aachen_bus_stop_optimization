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
    del demand, exy  # retained in the signature for compatibility

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

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_data = final.set_index("plan")
    figure, axis = plt.subplots(figsize=(10, 5.5))
    positions = np.arange(len(chart_data))
    width = 0.28

    axis.bar(
        positions - width,
        chart_data["central_cov300_%"],
        width,
        label="central population",
        color="#00549F",
    )
    axis.bar(
        positions,
        chart_data["underserved_cov300_%"].astype(float),
        width,
        label="underserved central population",
        color="#E30066",
    )
    axis.bar(
        positions + width,
        chart_data["poi_underserved_cov300_%"],
        width,
        label="POI-underserved population",
        color="#57AB27",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(chart_data.index, rotation=15, ha="right")
    axis.set_ylabel("Coverage @300 m [%]")
    axis.set_title("Coverage under alternative planning priorities")
    axis.legend()
    axis.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{FIG}/final_who_benefits.png", dpi=200)
    plt.close()

    print(f"[D] figure -> {FIG}/final_who_benefits.png")
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
