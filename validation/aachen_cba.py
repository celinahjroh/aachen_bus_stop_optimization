"""
================================================================================
 Aachen Bus Stops — ACCESS-TIME COST-BENEFIT LAYER
================================================================================
 This file implements Model II in PuLP and evaluates plans using the report's
 access-time benefit assumptions. The resulting ratio is an access-related
 benefit-cost ratio, not the statutory German NKV and not a funding decision.
================================================================================
"""
from __future__ import annotations

import ast
import os

import numpy as np
import pandas as pd
import pulp

import aachen_model as A

# Reuse the repository-root-anchored output directory defined in aachen_model,
# so all scripts write to the same results/ folder regardless of the working
# directory they are launched from.
OUT = A.OUT
os.makedirs(OUT, exist_ok=True)

# Appraisal parameters
VTTS_EUR_H = 7.10
WALK_MS = 1.25
TRIPS_YEAR = 150.0
LIFE_YEARS = 30
DISCOUNT = 0.017
PV_FACTOR = (1 - (1 + DISCOUNT) ** -LIFE_YEARS) / DISCOUNT


def benefit_matrix(
    pop,
    nearest_e,
    dC,
    improves,
    vtts=VTTS_EUR_H,
    trips=TRIPS_YEAR,
    pv=PV_FACTOR,
):
    """Return discounted lifetime access-time benefit b_ij in kEUR."""
    saved_metres = np.where(improves, nearest_e[:, None] - dC, 0.0)
    saved_hours = saved_metres / WALK_MS / 3600.0
    benefit = pop[:, None] * trips * saved_hours * vtts * pv / 1000.0
    return np.where(improves, benefit, 0.0)


def solve_netbenefit(pop, nearest_e, dC, improves, cost, benefit):
    """Solve Model II and return opened candidate indices.

    The result is returned only when CBC proves optimality. A candidate enters
    according to its marginal contribution to system-wide benefit conditional on
    the other selected candidates; isolated candidate ratios are not additive.
    """
    relevant = np.where(improves.any(axis=1))[0]
    prob = pulp.LpProblem("netbenefit", pulp.LpMaximize)

    x = {c: pulp.LpVariable(f"x_{c}", cat="Binary") for c in range(len(cost))}
    y = {
        (i, c): pulp.LpVariable(f"y_{i}_{c}", lowBound=0, upBound=1)
        for i in relevant
        for c in np.where(improves[i])[0]
    }

    prob += (
        pulp.lpSum(benefit[i, c] * y[i, c] for i, c in y)
        - pulp.lpSum(cost[c] * x[c] for c in x)
    )

    for i in relevant:
        prob += pulp.lpSum(y[i, c] for c in np.where(improves[i])[0]) <= 1
    for i, c in y:
        prob += y[i, c] <= x[c]

    A._solve_cbc(prob)
    opened = A._opened_indices(x)

    # Independent accounting check: the solver objective must equal the
    # nearest-opened-platform appraisal, up to numerical tolerance.
    appraisal = plan_appraisal(opened, pop, nearest_e, dC, cost, benefit)
    solver_net = float(pulp.value(prob.objective))
    posthoc_net = float(appraisal["net_benefit_kEUR_unrounded"])
    if abs(solver_net - posthoc_net) > 0.1:
        raise AssertionError(
            "Model II objective and post-hoc appraisal disagree: "
            f"solver={solver_net:.6f} kEUR, posthoc={posthoc_net:.6f} kEUR"
        )

    return opened


def plan_appraisal(opened, pop, nearest_e, dC, cost, benefit=None):
    """Evaluate an opened plan using access-time benefit only.

    When a benefit matrix is supplied, the appraisal uses that matrix directly,
    which keeps sensitivity scenarios consistent with the optimisation.
    """
    if not opened:
        return {
            "n_stops": 0,
            "cost_kEUR": 0.0,
            "benefit_kEUR": 0.0,
            "net_benefit_kEUR": 0.0,
            "access_BCR": None,
            "benefit_kEUR_unrounded": 0.0,
            "net_benefit_kEUR_unrounded": 0.0,
        }

    if benefit is not None:
        benefit_k = float(benefit[:, opened].max(axis=1).sum())
    else:
        best_candidate_distance = dC[:, opened].min(axis=1)
        served = best_candidate_distance < nearest_e - 1e-9
        saved_metres = np.where(
            served, nearest_e - best_candidate_distance, 0.0
        )
        benefit_k = (
            pop
            * TRIPS_YEAR
            * (saved_metres / WALK_MS / 3600.0)
            * VTTS_EUR_H
            * PV_FACTOR
        ).sum() / 1000.0
    cost_k = float(sum(cost[c] for c in opened))
    net_k = benefit_k - cost_k

    return {
        "n_stops": len(opened),
        "cost_kEUR": round(cost_k, 1),
        "benefit_kEUR": round(benefit_k, 1),
        "net_benefit_kEUR": round(net_k, 1),
        "access_BCR": round(benefit_k / cost_k, 2) if cost_k > 0 else None,
        "benefit_kEUR_unrounded": float(benefit_k),
        "net_benefit_kEUR_unrounded": float(net_k),
    }


def _plan_from_budget_file(path, budget, cid):
    selected = pd.read_csv(path)
    row = selected.loc[selected["budget_kEUR"] == budget]
    if row.empty:
        raise ValueError(f"Budget {budget} not found in {path}")

    stop_ids = ast.literal_eval(row.iloc[0]["stops"])
    id_to_index = {int(cid[c]): c for c in range(len(cid))}
    return [id_to_index[int(stop_id)] for stop_id in stop_ids]


def main():
    # The plan comparison below reads the cost-aware selection produced by
    # aachen_model.main(). Regenerate it first if this script is run on its own.
    if not os.path.exists(f"{OUT}/selected_stops_costaware.csv"):
        print(
            "[CBA] Model I selection not found; running aachen_model.main() first."
        )
        A.main()

    demand, pop, central, exy, cxy, cid = A.load()
    nearest_e, dC, improves = A.distances(demand, exy, cxy)
    cost, _ = A.build_cost_model(cxy, dC, improves, pop)
    benefit = benefit_matrix(pop, nearest_e, dC, improves)
    equity_mask = central & (nearest_e > A.R_EQUITY)

    print(
        f"PV factor={PV_FACTOR:.1f} VTTS={VTTS_EUR_H} EUR/h "
        f"trips/year={TRIPS_YEAR}"
    )

    net_benefit_plan = solve_netbenefit(
        pop, nearest_e, dC, improves, cost, benefit
    )
    net_benefit_appraisal = plan_appraisal(
        net_benefit_plan, pop, nearest_e, dC, cost, benefit
    )
    print(
        "[NB] net-benefit optimum: "
        f"{net_benefit_appraisal['n_stops']} stops, "
        f"cost {net_benefit_appraisal['cost_kEUR']}k, "
        f"benefit {net_benefit_appraisal['benefit_kEUR']}k, "
        f"access BCR={net_benefit_appraisal['access_BCR']}"
    )

    # Isolated candidate screening
    per_stop_rows = []
    for candidate in np.where(improves.any(axis=0))[0]:
        appraisal = plan_appraisal(
            [candidate], pop, nearest_e, dC, cost, benefit
        )
        per_stop_rows.append(
            {
                "stop_id": int(cid[candidate]),
                "cost_kEUR": appraisal["cost_kEUR"],
                "benefit_kEUR": appraisal["benefit_kEUR"],
                "access_BCR": appraisal["access_BCR"],
            }
        )

    per_stop = pd.DataFrame(per_stop_rows).sort_values(
        "access_BCR", ascending=False
    )
    per_stop.to_csv(f"{OUT}/07_per_stop_access_BCR.csv", index=False)
    above_one = int((per_stop["access_BCR"] >= 1.0).sum())
    print(
        f"[07] {above_one}/{len(per_stop)} candidates have an isolated "
        "access-related BCR >= 1"
    )

    plans = {
        "cost_aware_360k": _plan_from_budget_file(
            f"{OUT}/selected_stops_costaware.csv", 360, cid
        ),
        "net_benefit_opt": net_benefit_plan,
    }

    equity_plan, minimum_equity_cost = A.solve_equity_lexicographic(
        pop,
        nearest_e,
        dC,
        improves,
        cost,
        equity_mask,
        equity_tau=0.5,
    )
    if minimum_equity_cost is not None:
        plans["equity_first_50pct"] = equity_plan

    comparison_rows = []
    for name, opened in plans.items():
        appraisal = plan_appraisal(
            opened, pop, nearest_e, dC, cost, benefit
        )
        # Internal-only unrounded fields are not written to the results table.
        appraisal = {
            key: value
            for key, value in appraisal.items()
            if not key.endswith("_unrounded")
        }
        indicators, _ = A.kpis(
            opened, pop, nearest_e, dC, cost, central, equity_mask
        )
        comparison_rows.append(
            {
                "plan": name,
                "selected_stop_ids": [int(cid[c]) for c in opened],
                **appraisal,
                "central_cov300": indicators["ccov300"],
                "avg_walk_m": indicators["avg_walk_m"],
                "underserved_cov300": indicators.get(
                    "underserved_now_cov300"
                ),
            }
        )

    pd.DataFrame(comparison_rows).to_csv(
        f"{OUT}/08_plan_comparison_CBA.csv", index=False
    )
    print("[08] plan comparison with access-related BCR written")

    # Behavioural and appraisal sensitivity
    sensitivity_rows = []
    for trips in (75, 150, 300):
        for discount in (0.01, 0.017, 0.03):
            pv = (1 - (1 + discount) ** -LIFE_YEARS) / discount
            scenario_benefit = benefit_matrix(
                pop, nearest_e, dC, improves, trips=trips, pv=pv
            )
            opened = solve_netbenefit(
                pop,
                nearest_e,
                dC,
                improves,
                cost,
                scenario_benefit,
            )

            if opened:
                scenario_appraisal = plan_appraisal(
                    opened, pop, nearest_e, dC, cost, scenario_benefit
                )
                benefit_k = scenario_appraisal["benefit_kEUR_unrounded"]
                cost_k = float(sum(cost[c] for c in opened))
            else:
                benefit_k = 0.0
                cost_k = 0.0

            sensitivity_rows.append(
                {
                    "trips_year": trips,
                    "discount": discount,
                    "n_stops": len(opened),
                    "cost_kEUR": round(cost_k, 1),
                    "benefit_kEUR": round(benefit_k, 1),
                    "access_BCR": (
                        round(benefit_k / cost_k, 2) if cost_k else None
                    ),
                }
            )

    pd.DataFrame(sensitivity_rows).to_csv(
        f"{OUT}/09_CBA_sensitivity.csv", index=False
    )
    print("[09] CBA sensitivity written")
    print("\nDONE (CBA). See results/07_*, 08_*, 09_*.")


if __name__ == "__main__":
    main()
