# Cost-Aware Bus-Stop Placement in Aachen

Optimisation code for a study that turns a validated *p*-median analysis of
bus-stop placement in Aachen into a decision-ready investment framework. The
baseline *p*-median model is generalised in two directions:

- **Model I — Cost-budgeted *p*-median with an optional equity floor.** The
  stop-count cap of the baseline is replaced by a monetary budget with
  heterogeneous, source-based per-candidate construction costs. An optional
  minimum-coverage requirement guarantees that a chosen share of a spatially
  defined underserved group reaches a stop within 300 m.
- **Model II — Net-benefit maximisation.** Walking-time savings are monetised
  using parameters from the German *Standardisierte Bewertung* framework, and
  the model selects the candidate set that maximises discounted access-time
  benefit net of construction cost.

Both models share the same data, candidate set and distance model, so every
difference between the resulting plans stems from the changed decision logic
alone.

> The monetised ratio computed here is an **access-related** benefit–cost ratio
> (`access_BCR`). It counts walking-time savings only and is **not** the
> statutory German NKV; it establishes nothing about GVFG funding eligibility.

## Repository layout

```
aachen_bus_stop_optimization/
├── aachen_model_gamspy.py        # GAMSPy implementation of both models (+ self-check)
├── existing.csv                  # 1,037 existing directional platforms (stop_id,x,y)
├── candidates.csv                #   141 screened candidate platforms (stop_id,x,y)
├── demand.csv                    # 16,984 demand nodes (demand_id,x,y,population,central)
├── pois_aachen.csv               #    18 schools & hospitals (name,type,lat,lon)
├── requirements.txt              # dependencies for the PuLP pipeline (no GAMS licence)
├── requirements-gamspy.txt       # extra dependency for the GAMSPy implementation
└── validation/                   # pure-Python PuLP pipeline (full analysis + figures)
    ├── aachen_model.py           #   Model I, KPI panel, budget/equity sweeps, figures
    ├── aachen_cba.py             #   Model II (net benefit) + access-related BCR screening
    └── aachen_final.py           #   POI analysis + integrated four-plan comparison
```

All spatial data use EPSG:25832 (ETRS89 / UTM zone 32N); POI coordinates are
converted from WGS84 on load. Euclidean distances are scaled by a circuity
factor `f = 1.3` to approximate walking distance. All monetary quantities are
handled in kEUR internally (the cost constant `45.0` denotes EUR 45,000).

## Two implementations

The optimisation core is provided twice. Both encode the same models and, on the
scenarios cross-validated below, return identical objective values and platform
selections.

| | `aachen_model_gamspy.py` | `validation/` (PuLP) |
|---|---|---|
| Solver | GAMSPy default MIP | CBC via PuLP |
| Needs a GAMS licence | yes | no |
| Scope | both models + built-in numeric self-check | full pipeline: sweeps, POI analysis, figures, CSV outputs |

The PuLP pipeline is the portable reference: it reproduces the published
Katsioupis (2026) baseline exactly and runs without a GAMS licence. The GAMSPy
file exists to satisfy the project's GAMSPy requirement; its `main()` solves the
headline **budget**, **underserved-equity**, and **net-benefit** scenarios and
asserts them against the same reference values the PuLP pipeline produces. The
POI-priority scenario is produced by the PuLP pipeline (`aachen_final.py`) and is
not part of the GAMSPy self-check.

## Installation

```bash
pip install -r requirements.txt
```

To additionally run the GAMSPy implementation (requires a working GAMS
installation / licence):

```bash
pip install -r requirements-gamspy.txt
```

## How to run

Scripts resolve all input and output paths against the repository root, so they
can be launched from anywhere. From the repository root:

**Full analysis (recommended).** Runs Model I, Model II and the POI /
integrated comparison, and writes all CSVs and figures:

```bash
python validation/aachen_final.py
```

Individual stages (each regenerates its prerequisites if needed):

```bash
python validation/aachen_model.py   # Model I: baseline validation, budget & equity sweeps
python validation/aachen_cba.py     # Model II: net-benefit optimum + per-candidate screening
```

**GAMSPy implementation with built-in verification.** Solves the headline
budget, equity and net-benefit scenarios and asserts them against the reference
values:

```bash
python aachen_model_gamspy.py
```

Outputs are written to `results/` (CSV tables) and `figures/` (PNG plots) in the
repository root. Both directories are regenerated on each run and are
git-ignored.

## Reproduced key figures

Running the pipeline reproduces the values reported in the study, including:

| Quantity | Value |
|---|---|
| Total residential population *P* | 245,489 |
| Baseline population-weighted average walk | 192.38 m |
| Global coverage @200 / 400 / 600 m | 61.44 % / 94.71 % / 98.39 % |
| Central coverage @200 / 300 m | 68.15 % / 91.36 % |
| Underserved central residents (>300 m) | 6,032 |
| Admissible demand–candidate pairs | 5,197 |
| Candidate construction cost min / mean / max | €45k / €74.3k / €100k |
| Cost-aware plan @€360k | platforms {1050, 1072, 1082, 1089}, central cov@300 = 94.5 % |
| Cost-aware plan @€450k | platforms {1050, 1072, 1082, 1089, 1093}, central cov@300 = 95.2 % |
| Equity-first plan (τ = 50 %) | platforms {1050, 1065, 1089, 1092, 1123, 1159}, C\* = €450k, cov@300 = 95.7 % |
| Net-benefit optimum | 43 platforms, cost €3.44M, benefit €16.13M, access BCR = 4.7 |
| Candidates with isolated access-BCR ≥ 1 | 136 of 141 |

## Output files

| File | Contents |
|---|---|
| `00_baseline_validation.csv` | Status-quo KPIs; reproduces the Katsioupis (2026) baseline |
| `01_pareto_costblind.csv` / `02_pareto_costaware.csv` | Budget sweeps under homogeneous vs heterogeneous costs |
| `03_equity_scenarios.csv` | Minimum budget per underserved-coverage target τ |
| `05_cost_effectiveness.csv` | Marginal €/coverage-point along the sweep (reference scale, not a unique knee) |
| `06_cost_sensitivity.csv` | Robustness of the €360k plan to the ±50 % surcharge test |
| `07_per_stop_access_BCR.csv` | Isolated access-related BCR of every candidate |
| `08_plan_comparison_CBA.csv` | Access-related appraisal of the headline plans |
| `09_CBA_sensitivity.csv` | Net-benefit optimum across trip-rate and discount-rate scenarios |
| `10_poi_stop_access.csv`, `10b_poi_detail.csv` | Facility-side POI accessibility |
| `12_final_plan_comparison.csv` | Integrated four-plan comparison on one KPI panel |
| `selected_stops_*.csv` | Which candidates are opened in each scenario |

## Notes and caveats

- **Directional platforms.** All counts are directional platforms, the unit of
  the underlying AVV data; the 1,037 existing platforms correspond to 461 stop
  facilities. A bidirectional facility would require two platforms.
- **Modelled costs.** Per-candidate costs are a transparent planning proxy
  calibrated to documented German municipal figures, not a surveyed schedule.
  Plan *compositions* are considerably better supported than plan *prices*; the
  ±50 % surcharge sensitivity (`06_*`) bounds the latter.
- **Access-time appraisal only.** The BCR monetises walking-time savings alone.
  It omits effects on both sides of the ledger (induced ridership and emissions
  on the benefit side; operating, in-vehicle-time and maintenance cost on the
  cost side), so its net direction is undetermined and it is not the statutory
  NKV.
- **Scope.** The models site platforms only. Routing, frequencies, timetables
  and the operating consequences of new stopping events are out of scope, as in
  the baseline.

## Data sources

- Existing and candidate platforms: AVV open data and the four-stage screening
  of Katsioupis (2026).
- Demand grid: census-disaggregated, Earth-observation-refined gridded
  population, aggregated to 50 × 50 m cells.
- POIs: OpenStreetMap facilities re-geocoded from official postal addresses.
- Cost levels: documented barrier-free stop-construction projects in German
  municipalities.
- Appraisal parameters: German *Standardisierte Bewertung* (Version 2016+).
