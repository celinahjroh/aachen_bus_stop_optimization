"""
================================================================================
 Aachen Public Transport Coverage — GAMSPy IMPLEMENTATION (handout deliverable)
 Cost-Budgeted p-Median with Binary Equity Coverage + Lexicographic Equity Plan
================================================================================
 This file implements both report models in GAMSPy:

   Model I
     - solve_budget(): cost-budgeted p-median without an equity floor
     - min_cost_for_equity_gamspy(): minimum cost for a required equity target
     - solve_equity_lexicographic_gamspy(): minimum-cost stage followed by
       distance minimisation at the minimum cost

   Model II
     - solve_netbenefit_gamspy(): net-benefit maximisation / uncapacitated
       facility location

 The headline budget, equity and net-benefit scenarios are checked automatically
 against the PuLP reference values in main(). The assignment variables are limited
 to the dominance-admissible demand-candidate pairs only.

 Input files are resolved relative to this script, so it may be launched from any directory:

     python aachen_model_gamspy.py
================================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from gamspy import (
    Container,
    Set,
    Parameter,
    Variable,
    Equation,
    Model,
    Sum,
    ModelStatus,
    Options,
)


# Require the solver to close the relative MIP gap. Optimality is accepted only
# when GAMSPy reports ModelStatus.OptimalGlobal.
_PROVE_OPT = Options(relative_optimality_gap=0.0)

# Repository root used for all input paths.
ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------------ constants
F = 1.3
R_EQUITY = 300
C_BASE = 45.0
C_SOLO = 15.0
C_LOAD = (0.0, 20.0, 40.0)
PAIR_M = 40.0

# Model II appraisal parameters
VTTS_EUR_H = 7.10
WALK_MS = 1.25
TRIPS_YEAR = 150.0
LIFE_YEARS = 30
DISCOUNT = 0.017
PV_FACTOR = (1 - (1 + DISCOUNT) ** -LIFE_YEARS) / DISCOUNT


# ============================================================ 1. DATA + DISTANCES
def load():
    ex = pd.read_csv(ROOT / "existing.csv")
    ca = pd.read_csv(ROOT / "candidates.csv")
    dem = pd.read_csv(ROOT / "demand.csv")

    demand = dem[["x", "y"]].to_numpy(float)
    pop = dem["population"].to_numpy(float)
    central = dem["central"].to_numpy() == 1
    exy = ex[["x", "y"]].to_numpy(float)
    cxy = ca[["x", "y"]].to_numpy(float)
    cid = ca["stop_id"].to_numpy()
    did = dem["demand_id"].to_numpy()
    return demand, pop, central, exy, cxy, cid, did


def distances(demand, exy, cxy):
    nearest_e = cKDTree(exy).query(demand)[0] * F
    dC = np.sqrt(((demand[:, None, :] - cxy[None, :, :]) ** 2).sum(-1)) * F
    improves = dC < nearest_e[:, None] - 1e-9
    return nearest_e, dC, improves


def build_cost_model(cxy, dC, improves, pop):
    nn2 = cKDTree(cxy).query(cxy, k=2)[0][:, 1]
    solo = nn2 > PAIR_M

    load = np.array([pop[improves[:, c]].sum() for c in range(len(cxy))])
    tier = np.zeros(len(cxy), dtype=int)
    positive_load = load[load > 0]

    if len(positive_load):
        q1, q2 = np.quantile(positive_load, [1 / 3, 2 / 3])
        tier = np.where(load <= q1, 0, np.where(load <= q2, 1, 2))

    cost = (
        C_BASE
        + np.where(solo, C_SOLO, 0.0)
        + np.array([C_LOAD[t] for t in tier])
    )
    return cost, load


def benefit_matrix(
    pop,
    nearest_e,
    dC,
    improves,
    vtts=VTTS_EUR_H,
    trips=TRIPS_YEAR,
    pv=PV_FACTOR,
):
    """Discounted lifetime access-time benefit b_ij in kEUR."""
    saved_m = np.where(improves, nearest_e[:, None] - dC, 0.0)
    hours = saved_m / WALK_MS / 3600.0
    benefit = pop[:, None] * trips * hours * vtts * pv / 1000.0
    return np.where(improves, benefit, 0.0)


# ============================================================ 2. HELPERS
def _sparse_records(pop, nearest_e, dC, improves, cid, did):
    """Create the dominance-admissible demand-candidate mapping.

    Returns only the improving pairs. The mapping is later passed through
    Model(limited_variables=[y[a]]) so the solver contains one y-variable per
    admissible pair rather than the full demand-candidate Cartesian product.
    """
    relevant = np.where(improves.any(axis=1))[0]
    i_ids = [str(did[i]) for i in relevant]
    j_ids = [str(s) for s in cid]

    pop_records = [(str(did[i]), float(pop[i])) for i in relevant]
    nearest_records = [(str(did[i]), float(nearest_e[i])) for i in relevant]

    pair_records = []
    distance_records = []
    for i_idx in relevant:
        for c_idx in np.where(improves[i_idx])[0]:
            pair = (str(did[i_idx]), str(cid[c_idx]))
            pair_records.append(pair)
            distance_records.append((*pair, float(dC[i_idx, c_idx])))

    return (
        relevant,
        i_ids,
        j_ids,
        pop_records,
        nearest_records,
        pair_records,
        distance_records,
    )


def _equity_records(pop, dC, improves, cid, did, equity_mask, equity_R=R_EQUITY):
    """Create equity-group population records and candidate coverage pairs."""
    underserved = np.where(equity_mask)[0]
    u_ids = [str(did[i]) for i in underserved]
    u_pop_records = [(str(did[i]), float(pop[i])) for i in underserved]

    coverage_records = []
    for i_idx in underserved:
        for c_idx in np.where(improves[i_idx])[0]:
            if dC[i_idx, c_idx] <= equity_R:
                coverage_records.append(
                    (str(did[i_idx]), str(cid[c_idx]), 1.0)
                )

    return u_ids, u_pop_records, coverage_records


def _opened_ids(variable, column="j_c"):
    """Read opened candidate IDs safely, including the empty-plan case."""
    if variable.records is None or variable.records.empty:
        return []

    return sorted(
        variable.records.loc[variable.records["level"] > 0.5, column]
        .astype(int)
        .tolist()
    )


def _model_status(model):
    if model.status == ModelStatus.OptimalGlobal:
        return "optimal"
    if model.status == ModelStatus.Integer:
        return "feasible"
    return f"{model.status} / {model.solve_status}"


def _assert_close(actual, expected, tolerance, label):
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{label}: expected {expected}, obtained {actual} "
            f"(tolerance {tolerance})"
        )


def _post_plan_distance(opened_ids, nearest_e, dC, cid):
    if not opened_ids:
        return nearest_e.copy()

    id_to_index = {int(cid[c]): c for c in range(len(cid))}
    candidate_indices = [id_to_index[int(stop_id)] for stop_id in opened_ids]
    return np.minimum(nearest_e, dC[:, candidate_indices].min(axis=1))


# ============================================================ 3. MODEL I: BUDGET
def solve_budget(budget, pop, nearest_e, dC, improves, cost, cid, did):
    """Cost-budgeted p-median without an equity floor.

    Returns
    -------
    opened_stop_ids : list[int]
    status : str
    """
    (
        _relevant,
        i_ids,
        j_ids,
        pop_records,
        nearest_records,
        pair_records,
        distance_records,
    ) = _sparse_records(pop, nearest_e, dC, improves, cid, did)

    m = Container()
    i = Set(m, "i", records=i_ids)
    j_c = Set(m, "j_c", records=j_ids)
    a = Set(m, "a", domain=[i, j_c], records=pair_records)

    pop_p = Parameter(m, "pop_p", domain=[i], records=pop_records)
    nearest_p = Parameter(m, "nearest_p", domain=[i], records=nearest_records)
    distance_p = Parameter(
        m, "distance_p", domain=[i, j_c], records=distance_records
    )
    cost_p = Parameter(
        m,
        "cost_p",
        domain=[j_c],
        records=list(zip(j_ids, cost)),
    )

    base = float((pop * nearest_e).sum())

    x = Variable(m, "x", domain=[j_c], type="binary")
    y = Variable(m, "y", domain=[i, j_c], type="positive")

    assign = Equation(m, "assign", domain=[i])
    assign[i] = Sum(a[i, j_c], y[i, j_c]) <= 1

    link = Equation(m, "link", domain=[i, j_c])
    link[i, j_c].where[a[i, j_c]] = y[i, j_c] <= x[j_c]

    budget_eq = Equation(m, "budget_eq")
    budget_eq[...] = Sum(j_c, cost_p[j_c] * x[j_c]) <= budget

    noghost = Equation(m, "noghost", domain=[j_c])
    noghost[j_c] = x[j_c] <= Sum(a[i, j_c], y[i, j_c])

    objective = base + Sum(
        a[i, j_c],
        pop_p[i] * (distance_p[i, j_c] - nearest_p[i]) * y[i, j_c],
    )

    model = Model(
        m,
        "aachen_budget",
        equations=[assign, link, budget_eq, noghost],
        limited_variables=[y[a]],
        problem="mip",
        sense="min",
        objective=objective,
    )
    model.solve(options=_PROVE_OPT)

    if model.status != ModelStatus.OptimalGlobal:
        return [], _model_status(model)

    return _opened_ids(x), "optimal"


# ============================================================ 4. MODEL I: EQUITY STAGE 1
def min_cost_for_equity_gamspy(
    pop,
    nearest_e,
    dC,
    improves,
    cost,
    equity_mask,
    cid,
    did,
    equity_R=R_EQUITY,
    equity_tau=0.5,
):
    """Minimum construction cost required to meet an equity target.

    The returned value is treated as C*, so it is returned only when global
    optimality is proven. A merely integer-feasible solution is not accepted as
    the lexicographic first-stage optimum.
    """
    del nearest_e  # not needed in the minimum-cost first stage

    j_ids = [str(s) for s in cid]
    u_ids, u_pop_records, coverage_records = _equity_records(
        pop, dC, improves, cid, did, equity_mask, equity_R
    )

    if not coverage_records:
        return None

    m = Container()
    j_c = Set(m, "j_c", records=j_ids)
    u = Set(m, "u", records=u_ids)

    cost_p = Parameter(
        m,
        "cost_p",
        domain=[j_c],
        records=list(zip(j_ids, cost)),
    )
    u_pop_p = Parameter(m, "u_pop_p", domain=[u], records=u_pop_records)
    coverage_p = Parameter(
        m,
        "coverage_p",
        domain=[u, j_c],
        records=coverage_records,
    )
    target_population = float(sum(value for _, value in u_pop_records))

    x = Variable(m, "x", domain=[j_c], type="binary")
    z = Variable(m, "z", domain=[u], type="binary")

    coverage_link = Equation(m, "coverage_link", domain=[u])
    coverage_link[u] = z[u] <= Sum(
        j_c, coverage_p[u, j_c] * x[j_c]
    )

    equity_eq = Equation(m, "equity_eq")
    equity_eq[...] = Sum(u, u_pop_p[u] * z[u]) >= (
        equity_tau * target_population
    )

    model = Model(
        m,
        "mincost_equity",
        equations=[coverage_link, equity_eq],
        problem="mip",
        sense="min",
        objective=Sum(j_c, cost_p[j_c] * x[j_c]),
    )
    model.solve(options=_PROVE_OPT)

    if model.status != ModelStatus.OptimalGlobal:
        return None

    return float(model.objective_value)


# ============================================================ 5. MODEL I: EQUITY STAGE 2
def solve_equity_lexicographic_gamspy(
    pop,
    nearest_e,
    dC,
    improves,
    cost,
    equity_mask,
    cid,
    did,
    equity_R=R_EQUITY,
    equity_tau=0.5,
):
    """Lexicographic equity plan.

    Stage 1 minimises cost. Stage 2 fixes the budget at the proven minimum cost
    and minimises population-weighted walking distance.
    """
    minimum_cost = min_cost_for_equity_gamspy(
        pop,
        nearest_e,
        dC,
        improves,
        cost,
        equity_mask,
        cid,
        did,
        equity_R,
        equity_tau,
    )
    if minimum_cost is None:
        return None, None

    (
        _relevant,
        i_ids,
        j_ids,
        pop_records,
        nearest_records,
        pair_records,
        distance_records,
    ) = _sparse_records(pop, nearest_e, dC, improves, cid, did)

    u_ids, u_pop_records, coverage_records = _equity_records(
        pop, dC, improves, cid, did, equity_mask, equity_R
    )

    base = float((pop * nearest_e).sum())
    target_population = float(sum(value for _, value in u_pop_records))

    m = Container()
    i = Set(m, "i", records=i_ids)
    j_c = Set(m, "j_c", records=j_ids)
    u = Set(m, "u", records=u_ids)
    a = Set(m, "a", domain=[i, j_c], records=pair_records)

    pop_p = Parameter(m, "pop_p", domain=[i], records=pop_records)
    nearest_p = Parameter(m, "nearest_p", domain=[i], records=nearest_records)
    distance_p = Parameter(
        m, "distance_p", domain=[i, j_c], records=distance_records
    )
    cost_p = Parameter(
        m,
        "cost_p",
        domain=[j_c],
        records=list(zip(j_ids, cost)),
    )
    u_pop_p = Parameter(m, "u_pop_p", domain=[u], records=u_pop_records)
    coverage_p = Parameter(
        m,
        "coverage_p",
        domain=[u, j_c],
        records=coverage_records,
    )

    x = Variable(m, "x", domain=[j_c], type="binary")
    y = Variable(m, "y", domain=[i, j_c], type="positive")
    z = Variable(m, "z", domain=[u], type="binary")

    assign = Equation(m, "assign", domain=[i])
    assign[i] = Sum(a[i, j_c], y[i, j_c]) <= 1

    link = Equation(m, "link", domain=[i, j_c])
    link[i, j_c].where[a[i, j_c]] = y[i, j_c] <= x[j_c]

    budget_eq = Equation(m, "budget_eq")
    budget_eq[...] = Sum(j_c, cost_p[j_c] * x[j_c]) <= minimum_cost + 1e-6

    noghost = Equation(m, "noghost", domain=[j_c])
    noghost[j_c] = x[j_c] <= Sum(a[i, j_c], y[i, j_c])

    coverage_link = Equation(m, "coverage_link", domain=[u])
    coverage_link[u] = z[u] <= Sum(
        j_c, coverage_p[u, j_c] * x[j_c]
    )

    equity_eq = Equation(m, "equity_eq")
    equity_eq[...] = Sum(u, u_pop_p[u] * z[u]) >= (
        equity_tau * target_population
    )

    objective = base + Sum(
        a[i, j_c],
        pop_p[i] * (distance_p[i, j_c] - nearest_p[i]) * y[i, j_c],
    )

    model = Model(
        m,
        "equity_stage2",
        equations=[
            assign,
            link,
            budget_eq,
            noghost,
            coverage_link,
            equity_eq,
        ],
        limited_variables=[y[a]],
        problem="mip",
        sense="min",
        objective=objective,
    )
    model.solve(options=_PROVE_OPT)

    if model.status != ModelStatus.OptimalGlobal:
        return None, minimum_cost

    return _opened_ids(x), minimum_cost


# ============================================================ 6. MODEL II: NET BENEFIT
def solve_netbenefit_gamspy(
    pop, nearest_e, dC, improves, cost, benefit, cid, did
):
    """Net-benefit-maximising uncapacitated facility-location model.

    A candidate is selected according to its marginal contribution to the
    system-wide objective conditional on the other selected candidates.
    """
    (
        _relevant,
        i_ids,
        j_ids,
        _pop_records,
        _nearest_records,
        pair_records,
        _distance_records,
    ) = _sparse_records(pop, nearest_e, dC, improves, cid, did)

    benefit_records = []
    for i_idx in np.where(improves.any(axis=1))[0]:
        for c_idx in np.where(improves[i_idx])[0]:
            benefit_records.append(
                (
                    str(did[i_idx]),
                    str(cid[c_idx]),
                    float(benefit[i_idx, c_idx]),
                )
            )

    m = Container()
    i = Set(m, "i", records=i_ids)
    j_c = Set(m, "j_c", records=j_ids)
    a = Set(m, "a", domain=[i, j_c], records=pair_records)

    benefit_p = Parameter(
        m,
        "benefit_p",
        domain=[i, j_c],
        records=benefit_records,
    )
    cost_p = Parameter(
        m,
        "cost_p",
        domain=[j_c],
        records=list(zip(j_ids, cost)),
    )

    x = Variable(m, "x", domain=[j_c], type="binary")
    y = Variable(m, "y", domain=[i, j_c], type="positive")

    assign = Equation(m, "assign", domain=[i])
    assign[i] = Sum(a[i, j_c], y[i, j_c]) <= 1

    link = Equation(m, "link", domain=[i, j_c])
    link[i, j_c].where[a[i, j_c]] = y[i, j_c] <= x[j_c]

    objective = (
        Sum(a[i, j_c], benefit_p[i, j_c] * y[i, j_c])
        - Sum(j_c, cost_p[j_c] * x[j_c])
    )

    model = Model(
        m,
        "aachen_netbenefit",
        equations=[assign, link],
        limited_variables=[y[a]],
        problem="mip",
        sense="max",
        objective=objective,
    )
    model.solve(options=_PROVE_OPT)

    if model.status != ModelStatus.OptimalGlobal:
        return [], _model_status(model)

    return _opened_ids(x), "optimal"


def _appraise(opened_ids, pop, nearest_e, dC, cost, cid):
    """Post-hoc appraisal of an opened stop list in kEUR."""
    if not opened_ids:
        return 0, 0.0, 0.0, 0.0, None

    id_to_index = {int(cid[c]): c for c in range(len(cid))}
    candidate_indices = [id_to_index[int(stop_id)] for stop_id in opened_ids]

    candidate_distances = dC[:, candidate_indices]
    best_candidate = candidate_distances.min(axis=1)
    served = best_candidate < nearest_e - 1e-9
    saved_m = np.where(served, nearest_e - best_candidate, 0.0)

    benefit_k = (
        pop
        * TRIPS_YEAR
        * (saved_m / WALK_MS / 3600.0)
        * VTTS_EUR_H
        * PV_FACTOR
    ).sum() / 1000.0

    cost_k = float(sum(cost[c] for c in candidate_indices))
    net_k = benefit_k - cost_k

    return (
        len(candidate_indices),
        round(cost_k, 1),
        round(benefit_k, 1),
        round(net_k, 1),
        round(benefit_k / cost_k, 2),
    )


# ============================================================ 7. AUTOMATIC VERIFICATION
def main():
    demand, pop, central, exy, cxy, cid, did = load()
    nearest_e, dC, improves = distances(demand, exy, cxy)
    cost, _load_values = build_cost_model(cxy, dC, improves, pop)
    equity_mask = central & (nearest_e > R_EQUITY)

    (
        relevant,
        _i_ids,
        _j_ids,
        _pop_records,
        _nearest_records,
        pair_records,
        _distance_records,
    ) = _sparse_records(pop, nearest_e, dC, improves, cid, did)

    # Input and preprocessing checks
    _assert_close((pop * nearest_e).sum(), 47_228_220.00, 0.1, "Baseline Z")
    _assert_close(pop.sum(), 245_489.37, 0.1, "Total population")
    _assert_close(pop[equity_mask].sum(), 6_032.18, 0.1, "Underserved population")
    assert len(relevant) == 850, f"Expected 850 relevant nodes, got {len(relevant)}"
    assert len(pair_records) == 5_197, (
        f"Expected 5,197 admissible pairs, got {len(pair_records)}"
    )

    # Model I: headline budget scenarios
    expected_budget_plans = {
        360: {1050, 1072, 1082, 1089},
        450: {1050, 1072, 1082, 1089, 1093},
    }

    for budget, expected_stops in expected_budget_plans.items():
        stops, status = solve_budget(
            budget, pop, nearest_e, dC, improves, cost, cid, did
        )
        assert status == "optimal", f"Budget {budget}: status was {status}"
        assert set(stops) == expected_stops, (
            f"Budget {budget}: expected {sorted(expected_stops)}, got {stops}"
        )

    # Model I: lexicographic equity plan
    equity_stops, minimum_cost = solve_equity_lexicographic_gamspy(
        pop,
        nearest_e,
        dC,
        improves,
        cost,
        equity_mask,
        cid,
        did,
        equity_tau=0.5,
    )
    assert equity_stops is not None, "Equity model did not return an optimal plan"
    _assert_close(minimum_cost, 450.0, 1e-6, "Equity minimum cost")
    assert set(equity_stops) == {1050, 1065, 1089, 1092, 1123, 1159}, (
        f"Unexpected equity plan: {equity_stops}"
    )

    # Model II: net-benefit optimum
    benefit = benefit_matrix(pop, nearest_e, dC, improves)
    net_benefit_stops, net_benefit_status = solve_netbenefit_gamspy(
        pop, nearest_e, dC, improves, cost, benefit, cid, did
    )
    assert net_benefit_status == "optimal", (
        f"Net-benefit status was {net_benefit_status}"
    )

    n_stops, cost_k, benefit_k, net_k, ratio = _appraise(
        net_benefit_stops, pop, nearest_e, dC, cost, cid
    )
    assert n_stops == 43, f"Expected 43 net-benefit stops, got {n_stops}"
    _assert_close(cost_k, 3_440.0, 0.1, "Net-benefit plan cost")
    _assert_close(benefit_k, 16_128.9, 0.2, "Net-benefit plan benefit")
    _assert_close(net_k, 12_688.9, 0.2, "Net-benefit plan net benefit")
    _assert_close(ratio, 4.69, 0.01, "Net-benefit plan ratio")

    print("ALL VERIFICATION TESTS PASSED")
    print(f"Relevant nodes: {len(relevant):,}")
    print(f"Admissible assignment pairs: {len(pair_records):,}")
    print(f"Budget 360k: {sorted(expected_budget_plans[360])}")
    print(f"Budget 450k: {sorted(expected_budget_plans[450])}")
    print(f"Equity 50%: {equity_stops}, C*={minimum_cost:.1f}k")
    print(
        "Net benefit: "
        f"{n_stops} stops, cost={cost_k:.1f}k, benefit={benefit_k:.1f}k, "
        f"net={net_k:.1f}k, access_BCR={ratio:.2f}"
    )


if __name__ == "__main__":
    main()
