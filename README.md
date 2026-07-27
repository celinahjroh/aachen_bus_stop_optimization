# Optimizing Public Transport Coverage in Aachen: Budget-Constrained and Equity-Constrained Bus Stop Location Models

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

> The monetised ratio computed here is an **access-related** benefit–cost ratio.
> It counts walking-time savings only and is **not** the statutory German NKV; it
> establishes nothing about GVFG funding eligibility.

## Repository layout

```
aachen_bus_stop_optimization/
├── aachen_model_gamspy.py        # GAMSPy implementation of both models (+ verification)
├── existing.csv                  # 1,037 existing directional platforms (stop_id,x,y)
├── candidates.csv                #   141 screened candidate platforms (stop_id,x,y)
├── demand.csv                    # 16,984 demand nodes (demand_id,x,y,population,central)
├── pois_aachen.csv               #    18 schools & hospitals (name,type,lat,lon)
├── requirements.txt              # dependencies for the PuLP pipeline (no GAMS licence)
├── requirements-gamspy.txt       # extra dependency for the GAMSPy implementation
├── validation/                   # independent PuLP/CBC implementation + analysis pipeline
│   ├── aachen_model.py           #   Model I, KPI panel, budget & equity sweeps
│   ├── aachen_cba.py             #   Model II (net benefit) + access-related BCR screening
│   ├── aachen_final.py           #   POI analysis + integrated four-plan comparison
│   ├── make_paper_figures.py     #   regenerates every figure and LaTeX table of the report
│   └── paper_assets.py           #   rendering routines used by make_paper_figures.py
├── figures/                      # rendered report figures (committed, see below)
└── results/                      # LaTeX tables (committed) + CSV outputs (generated)
```

All spatial data use EPSG:25832 (ETRS89 / UTM zone 32N); POI coordinates are
converted from WGS84 on load. Euclidean distances are scaled by a circuity
factor `f = 1.3` to approximate walking distance.

## Two implementations

The optimisation core is provided twice, and the two agree on every reported
plan and objective value.

`aachen_model_gamspy.py` is the GAMSPy implementation of both models and the
reference implementation for the optimisation results reported in the study. It
encodes Model I (budget scenarios and the lexicographic minimum-cost/minimum-
distance equity construction) and Model II, requires the solver to close the
relative MIP gap, and accepts a result only when GAMSPy reports
`OptimalGlobal` — a time-limited feasible incumbent is rejected rather than
reported.

The `validation/` directory holds a fully independent re-implementation of the
same models in PuLP, solved with CBC. Its primary purpose is cross-validation:
the headline budget, equity and net-benefit scenarios are solved in both
implementations, which return identical objective values and select the same
candidate platforms, so the reported results are not an artefact of one
solver or one algebraic encoding. Because it needs no GAMS licence, this
implementation also carries the surrounding analysis pipeline — the budget and
equity sweeps, the KPI panel, the sensitivity analyses, and the figure and table
rendering — which makes the whole study reproducible on any machine.

| | `aachen_model_gamspy.py` | `validation/` (PuLP) |
|---|---|---|
| Role | reference implementation of Model I and Model II | independent cross-check + analysis pipeline |
| Solver | GAMSPy default MIP | CBC via PuLP |
| Needs a GAMS licence | yes | no |
| Scope | both models, headline scenarios, built-in numeric verification | both models, full sweeps, POI analysis, KPIs, figures, tables |

The two also validate different things. The cross-implementation agreement tests
the *model*; the status-quo run (no new platforms) reproduces the published
baseline reference values exactly and thereby tests the *data pipeline*.

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

**Everything at once (recommended).** Regenerates every figure and every
data-driven table of the report. If the result CSVs are not present yet it runs
the full analysis pipeline first (the slow path); on a repository that has
already been analysed once it finishes in seconds:

```bash
python validation/make_paper_figures.py
```

**Analysis only**, without re-rendering figures and tables:

```bash
python validation/aachen_final.py   # Model I + Model II + POI + integrated comparison
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

Runtime depends on hardware: the full PuLP pipeline takes roughly three minutes
on an Apple Silicon MacBook Air, and each GAMSPy scenario a few seconds.

## Reproduced figures and tables

`validation/make_paper_figures.py` regenerates every figure of the report that
contains data, under the exact filenames the LaTeX sources reference:

| Report | File | Content |
|---|---|---|
| Figure 3.1 | `figures/fig31_data_overview.png` | Spatial overview of all input datasets |
| Figure 3.2 | `figures/fig32_cost_distribution.png` | Modelled construction-cost distribution |
| Figure 5.1 | `figures/fig_underserved_map.png` | Underserved central residents at baseline |
| Figure 5.2 | `figures/fig_pareto_cost_coverage.png` | Central coverage @300 m vs actual expenditure |
| Figure 5.3 | `figures/fig_avgwalk_budget.png` | Average walking distance vs budget |
| Figure 5.4 | `figures/fig_price_of_equity.png` | Minimum budget vs required underserved coverage τ |
| Figure 5.5 | `figures/fig_poi_catchments.png` | Facility catchments and the structural ceiling |
| Figure 6.1 | `figures/fig_plan_maps.png` | The two headline plans over the underserved population |
| Figure 6.2 | `figures/final_who_benefits.png` | Coverage by population group under each plan |
| Figure 6.3 | `figures/fig_nkv_map.png` | Per-candidate access-related benefit–cost ratio |
| Table 5.1 | `results/table_sweep.tex` | Cost-aware budget sweep |
| Table 5.2 | `results/table_equity.tex` | The price of equity |
| Table 6.1 | `results/table_final.tex` | Integrated four-plan comparison |
| Table A.1 | `results/table_poi.tex` | Points of interest and baseline platform walk |

Figures 4.1 (analysis framework) and 7.1 (staged investment recommendation) are
schematic diagrams drawn directly in LaTeX/TikZ and contain no computed data, so
they are deliberately not produced here. Every other figure and every
data-driven table in the report comes out of the pipeline above.

A rendered copy of the ten figures and four LaTeX tables is committed to the
repository so they can be inspected without running anything. The numbered CSV
outputs listed below are intermediate artefacts, regenerated on each run and
git-ignored.

## Reproduced key values

| Quantity | Value |
|---|---|
| Total residential population *P* | 245,489 |
| Baseline population-weighted average walk | 192.38 m |
| Global coverage @200 / 400 / 600 m | 61.44 % / 94.71 % / 98.39 % |
| Central coverage @200 / 300 m | 68.15 % / 91.36 % |
| Underserved central residents (>300 m) | 6,032 |
| Admissible demand–candidate pairs | 5,197 |
| Demand cells admitting any improvement | 850 |
| Candidate construction cost min / mean / max | €45k / €74.3k / €100k |
| Cost-aware plan @€360k | platforms {1050, 1072, 1082, 1089}, spend €350k, central cov@300 = 94.5 % |
| Equity-first plan (τ = 50 %) | platforms {1050, 1065, 1089, 1092, 1123, 1159}, C\* = €450k, cov@300 = 95.7 % |
| POI-priority plan | platforms {1063, 1092}, spend €160k, POI-underserved cov = 14.4 % |
| Net-benefit optimum | 43 platforms, cost €3.44M, benefit €16.13M, access BCR = 4.7 |
| Candidates with isolated access-BCR ≥ 1 | 136 of 141 |
| Structural coverage ceilings (underserved / POI-underserved) | 77.9 % / 18.0 % |

## Output files

CSV outputs written to `results/`:

| File | Contents |
|---|---|
| `00_baseline_validation.csv` | Status-quo KPIs; reproduces the baseline reference values |
| `01_pareto_costblind.csv` / `02_pareto_costaware.csv` | Budget sweeps under homogeneous vs heterogeneous costs |
| `03_equity_scenarios.csv` | Minimum budget per underserved-coverage target τ |
| `05_cost_effectiveness.csv` | Marginal €/coverage-point along the sweep |
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
- **Pairing rule.** The €15k solo surcharge is exempted whenever another
  candidate lies within 40 m, evaluated on the candidate *set* rather than on the
  selected plan. Where a plan opens only one member of a pair, the shared
  mobilisation does not occur and the cost is understated; 54 of the 141
  candidates are classified as paired.
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

## License and reuse

This repository accompanies an academic project report and is provided for
review, reproduction and educational use in connection with that report. No
separate software license is granted; please contact the authors for any other
use.

The input datasets are covered by their own terms in any case. They retain the
license/usage conditions of their original sources (AVV open data,
OpenStreetMap/ODbL, and the gridded population dataset of Schug et al. [2021])
and are redistributed here only to support reproducibility of this study. See
Section 3 of the report for full provenance and citation of each dataset.
