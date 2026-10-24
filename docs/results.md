# Results: next-sighting forecast for *Vespa velutina* (Asian hornet) in Western Europe

All numbers below come from the real downloads of 24 Oct 2026 (DOIs in `docs/gbif_downloads.md`)
and are reproducible with `make data`. Phases 1-2 (data, panel, bias diagnostic) come first,
then the Phase 3-5 forecast evaluation, the single 2026 holdout, the frozen models, backtest,
SHAP drivers and the dashboard; limitations are at the end.

## Occurrence data

* Raw GBIF download: **464,199** *V. velutina* records in the region. By country: BE 247,565,
  NL 90,643, FR 55,170, CH 41,605, DE 21,430, ES 4,067, PT 2,724, LU 836, JE 78, IE 34, GB 29,
  GG 11.
* Top datasets: Waarnemingen.be non-native animals (221,794), Observation.org (96,798),
  iNaturalist research-grade (21,483), Swiss National Fauna Databank (21,431), Swiss National
  Vespoidea Databank (21,373).
* After cleaning: **332,976** records (`reports/cleaning_log.csv`, `reports/figures/cleaning_log.png`).
  The largest removals were outside the 2004-2025 period (70,992; 70,964 of them from the
  partial year 2026), duplicates with the same coordinates and date (34,919), coordinates
  rounded to 1 decimal place or fewer (16,924), outside the region polygons (5,333) and
  coordinate uncertainty above 5 km (2,637). Only 28 records sat on country centroids.

## Grid and panel

* 5 km EPSG:3035 grid: **80,098** land cells.
* Annual panel: 80,098 cells x 22 windows (2004-2025) = 1,762,156 rows. Half-year panel:
  44 windows, 3,524,312 rows.
* Cumulative occupied cells: 5 (2004), 202 (2006), 2,251 (2010), 4,390 (2015), 8,170 (2020),
  **15,691 (2025)**. Newly occupied cells per year rose from about 500-750 (2009-2021) to
  1,809 / 1,992 / 2,047 in 2023 / 2024 / 2025 (`reports/panel_summary_year.csv`).
* The first-detection map (`reports/figures/first_detection_map.png`) shows the expected
  pattern: the 2004 introduction in south-west France, then a front through France, a
  separate wave into Belgium and the Netherlands from about 2016-2022, and recent (2023-2025)
  colonisation of western Germany and the Swiss Plateau, as well as northern Iberia.

## Caveat: reporting intensity is not stationary

Clean records per year were roughly 1-5k from 2007 to 2021, then 16k (2022), 42k (2023),
52k (2024) and **181k (2025)**. In 2025, 114,698 of those records are from Belgium, 110,341 of
them from a single platform (Waarnemingen.be). Reporting in CH, NL and DE also rose 4-10x
between 2023 and 2025, while France, the core of the range, stayed flat at about 1.4-2.2k records
per year. Record counts therefore track reporting schemes more than range size. That is why Phase 2 uses presence/absence per cell (thinning) instead of
record counts, and why it adds an effort layer.

## Effort layer

* GBIF Insecta (focal species excluded): **130.4 million** records aggregated to 918,803
  non-zero cell-year rows covering 76,528 of the 80,098 cells. Annual totals grew from 2.7M
  (2004) to 8.0M (2025) (`reports/figures/effort_map.png`).

## Bias-correction diagnostic (split year 2019)

Static presence/background logistic GLM on bio1-19 and elevation plus land cover.
Training presences: 7,527 cells occupied up to 2019. Test: 72,571 cells not yet occupied,
of which 8,164 were first detected in 2020-2025 (base rate 11.2%). 65,708 test cells were
surveyed, i.e. had at least 1 Insecta record after 2019 (base rate 12.4%).

| Variant | ROC-AUC | PR-AUC | ROC-AUC surveyed | rho(pop) | rho(roads) | rho(effort) |
|---|---|---|---|---|---|---|
| naive (uniform background) | 0.743 | 0.205 | 0.723 | 0.352 | 0.371 | 0.382 |
| target-group background | 0.717 | 0.190 | 0.701 | 0.234 | 0.271 | 0.184 |
| inverse-effort weights | 0.727 | 0.186 | 0.705 | 0.359 | 0.375 | 0.346 |
| bias covariates (raw prediction) | **0.763** | **0.221** | **0.742** | 0.415 | 0.424 | 0.454 |
| bias covariates (neutralised) | 0.716 | 0.180 | 0.696 | 0.292 | 0.319 | 0.340 |
| target-group + inverse-effort | 0.705 | 0.183 | 0.687 | 0.256 | 0.292 | **0.153** |

(rho = Spearman correlation of the predicted map with population density, road density and
recorder effort. Source: `reports/bias_diagnostic.csv`; maps in `reports/figures/bias_variant_maps.png`.)

**Interpretation.**
* **Do the uncorrected maps trace roads and cities? Partly, yes.** The naive map correlates
  with effort at 0.38 and with population at 0.35. Adding the bias covariates without
  neutralising them gives the best AUC (0.763) but also the strongest correlation with
  effort (0.45): it scores well by predicting where people record.
* Target-group background halves the correlation with effort (0.38 -> 0.18). Combined with
  inverse-effort weights it reaches 0.15, the lowest of all variants. Inverse-effort weights
  on their own barely change anything (0.35).
* The correction costs about 0.03-0.04 ROC-AUC on later detections. That is expected: future
  detections are themselves effort-biased, so a model that follows effort is rewarded by a
  biased test set. Restricting to surveyed cells does not remove the gap, because almost all
  cells are surveyed at least once. A sharper test (for example stratifying by effort decile)
  is planned for Phase 4.
* Default for Phases 3-4: **target-group background + inverse-effort weights**, with bias
  covariates neutralised, plus effort as an explicit feature of the spread model so that
  "new detection" is modelled as occupancy times detection.

## Known limitations of the Phase 1-2 data

* The 2026 records (70,964) are excluded because the year is incomplete. They can serve as a
  true out-of-sample check in Phase 4.
* The WorldCover 2021 snapshot is applied to all years, and the climate layers are
  1970-2000 normals, i.e. static covariates.
* The effort proxy (all GBIF insects) is dominated by the same Belgian and Dutch platforms;
  it measures recording activity, not detection probability for hornets specifically.

## Phases 3-5: next-window spread forecast (test 2022-2025)

**Task:** for each 5 km cell not yet confirmed by the end of year *t*, predict a first *V. velutina*
sighting in *t+1* (h=1), or within *t+1..t+2* / *t+1..t+3* (h=2, h=3). Features use windows <= *t*
only (see the leakage test below). The split is strictly chronological by label year: training labels
end by 2018, validation 2019-2021 (expanding-window Optuna tuning with 25 trials, plus isotonic
calibration on out-of-fold scores), test 2022-2025. The 2026 holdout has **not** been evaluated.

Test set (h=1): 276,524 cell-years, 7,633 first sightings (base rate 2.6%). Effort deciles come from
Insecta records in the label year; **low-effort = deciles 1-3** (82,959 cell-years, 169 first
sightings, base rate 0.20%). Precision@k is per forecast year, then averaged over the 4 years.
The 95% CIs come from a 50 km spatial block bootstrap (200 resamples).

| model (h=1) | PR-AUC [95% CI] | ROC-AUC | P@100 | P@500 | P@2000 | low-effort PR-AUC | low-effort P@100 | rho(pred, effort) |
|---|---|---|---|---|---|---|---|---|
| **spread GBM, approved default** (TG negatives + inverse-effort weights + neutralised pop/road + effort features) | 0.046 [0.039, 0.054] | 0.747 | 0.055 | 0.039 | 0.035 | 0.010 | 0.028 | -0.03 |
| ablation: first-run TG (negatives only from recorded cells) | 0.044 [0.038, 0.051] | 0.735 | 0.018 | 0.021 | 0.031 | 0.007 | 0.005 | -0.04 |
| ablation: effort features also neutralised | 0.075 [0.060, 0.088] | 0.799 | 0.218 | 0.146 | 0.100 | 0.012 | 0.033 | -0.04 |
| ablation: no bias correction | **0.205 [0.165, 0.240]** | 0.888 | 0.553 | 0.367 | 0.249 | 0.026 | 0.055 | 0.26 |
| ablation: dynamic features only (TG + weights, no effort/env) | 0.174 [0.141, 0.205] | 0.856 | 0.393 | 0.327 | 0.225 | 0.021 | 0.045 | 0.11 |
| ablation: no dynamic features (static GBM) | 0.036 [0.032, 0.041] | 0.680 | 0.023 | 0.016 | 0.020 | 0.008 | 0.015 | 0.04 |
| static SDM, logistic (poly-2) | 0.038 [0.034, 0.043] | 0.665 | 0.070 | 0.061 | 0.046 | 0.007 | 0.013 | 0.00 |
| static SDM, random forest | 0.034 [0.030, 0.038] | 0.604 | 0.063 | 0.054 | 0.046 | 0.010 | 0.018 | -0.20 |
| adjacency, binary (next to an occupied cell) | 0.061 [0.052, 0.071] | 0.734 | 0.083 | 0.088 | 0.078 | 0.006 | 0.010 | 0.03 |
| **adjacency, ranked** (number of occupied neighbours) | 0.083 [0.067, 0.098] | 0.794 | 0.168 | 0.128 | 0.109 | 0.009 | 0.023 | 0.04 |
| distance to front | 0.078 [0.066, 0.090] | 0.831 | 0.093 | 0.091 | 0.087 | 0.008 | 0.010 | 0.12 |
| random | 0.026 | 0.505 | 0.020 | 0.026 | 0.027 | 0.002 | 0.005 | 0.00 |

**Does the model beat adjacency?** PR-AUC difference vs ranked adjacency (paired block bootstrap):

- Approved default: **no**, -0.037 [-0.047, -0.026].
- No bias correction: yes, +0.122 [0.091, 0.148]. In low-effort cells, P@100 is 0.055 vs 0.023 (difference CI [+0.008, +0.058]).
- Dynamic only: yes, +0.091 [0.071, 0.113].
- Effort also neutralised: no clear difference, -0.008 [-0.017, +0.002].

On validation (out-of-fold 2019-2021, base rate 0.8%) only the no-bias-correction model beats
ranked adjacency (PR-AUC 0.150 vs 0.044; approved default 0.031; dynamic only 0.041).

**Why the approved default fails.** Within the test set, its mean raw score *falls* with effort
(0.076 in decile 1 vs 0.014 in decile 10), while the observed rate *rises* 250x (0.04% vs 10.1%).
Target-group sampling and inverse-effort weights already remove the effort-detection link from the
training labels, and with effort also given as an explicit feature the model learns that link
backwards ("well-recorded means no hornet"). Sampling floors and weight caps did not change this
(0.044 -> 0.046).

**Longer horizons (default model vs baselines, PR-AUC):** h=2: 0.087 vs adjacency 0.106 and
distance 0.126. h=3: 0.123 vs adjacency 0.114 (diff CI [-0.001, +0.019]) and distance 0.153.

**Lead time** (a cell counts as flagged when it is in a year's top 2,000):
- No bias correction: flags 30.6% of newly sighted test cells before their first record (median lead 1 year).
- Ranked adjacency: 12.6% (median lead 1 year).
- Approved default: 6.0% (median lead 2 years).

**Calibration caveat:** all isotonic calibrators were fit on the validation years (base rate 0.8%),
so on the test years (base rate 2.6%) every model under-predicts. ECE is 0.014-0.017 across models,
which is about two-thirds of the base rate.

Full numbers: `results/metrics.json`, `results/summary_h1.csv`, `results/metrics_by_effort_decile.csv`;
figures: `reports/figures/reliability_h1.png`, `reports/figures/effort_decile_metrics.png`.
The leakage test output is in `results/leakage_test_output.txt`.

## 2026 holdout (evaluated once, pre-registered)

The model list, metrics and comparisons were committed in `results/holdout_preregistration.json`
before the run. Every model was refit on all labels up to 2025 using the frozen hyperparameters and
the 2019-2021 calibrators, then scored once on first sightings in 2026.

The first attempt crashed with NaN labels (a nullable-integer bug) before any metric was computed or
printed. It was fixed and re-run once.

2026 is a partial year (GBIF snapshot from October 2026): 64,407 at-risk cells, **508 first
sightings** (base rate 0.79%). The low-effort strata (deciles 1-3) contain only **9** first
sightings, so the low-effort holdout numbers carry almost no information.

| model (h=1, 2026) | PR-AUC [95% CI] | ROC-AUC | P@100 | P@500 | P@2000 | PR-AUC minus ranked adjacency [95% CI] |
|---|---|---|---|---|---|---|
| spread GBM, approved default | 0.049 [0.031, 0.077] | 0.831 | 0.12 | 0.108 | 0.066 | -0.001 [-0.018, +0.025] |
| ablation: no bias correction | **0.091 [0.061, 0.137]** | 0.899 | 0.18 | 0.168 | 0.102 | **+0.041 [+0.018, +0.082]** |
| ablation: first-run TG (recorded cells only) | 0.046 [0.031, 0.074] | 0.826 | 0.13 | 0.114 | 0.061 | -0.003 [-0.018, +0.020] |
| ablation: effort features also neutralised | 0.082 [0.052, 0.130] | 0.875 | 0.20 | 0.158 | 0.091 | +0.032 [+0.005, +0.079] |
| ablation: dynamic features only | 0.070 [0.047, 0.107] | 0.847 | 0.13 | 0.158 | 0.090 | +0.020 [+0.002, +0.051] |
| ablation: no dynamic features (static GBM) | 0.018 [0.015, 0.022] | 0.770 | 0.00 | 0.008 | 0.013 | -0.032 [-0.048, -0.022] |
| static SDM, logistic / random forest | 0.023 / 0.022 | 0.739 / 0.723 | 0.02 / 0.04 | 0.056 / 0.046 | 0.037 / 0.037 | -0.027 / -0.028 |
| adjacency, binary | 0.021 | 0.788 | 0.02 | 0.020 | 0.017 | -0.029 [-0.045, -0.019] |
| **adjacency, ranked** | 0.050 [0.039, 0.067] | 0.863 | 0.12 | 0.084 | 0.070 | - |
| distance to front | 0.026 | 0.837 | 0.02 | 0.020 | 0.024 | -0.024 [-0.039, -0.015] |
| random | 0.008 | 0.499 | 0.01 | 0.008 | 0.006 | -0.042 |

Reading the holdout:
- It confirms the test-period picture. The approved default only ties ranked adjacency, while the no-bias-correction model beats it (CI excludes 0).
- The no-bias-correction model has the strongest correlation with 2026 recording effort (Spearman 0.42 vs 0.13 for the default and for adjacency).
- On the 9 low-effort first sightings no model can be distinguished from adjacency.

Outputs: `results/holdout_metrics.json`, `results/holdout_by_effort_decile.csv`, and
`results/forecast_from_2025.parquet` (per-cell P(first sighting within 1/2/3 years after 2025) from
the default model; not committed, regenerate with `make holdout --force`).

## Frozen model files (post-run addendum)

The holdout run fitted its models in memory. `make freeze` (`src/models/freeze.py`) re-created them
afterwards from the same data, hyperparameters and seeds, and saved them to `models/frozen/`:
LightGBM boosters, training medians, SDMs and isotonic calibrators. The SHA-256 of every file and of
`results/tuning.json` is in `models/frozen/MANIFEST.json`. The same hashes were appended to
`results/holdout_preregistration.json` as a `post_run_addendum`, in a separate commit **after** the
holdout run; they were not part of the original pre-registration.

Five boosters reproduce per-cell outputs saved by the holdout run (max abs. difference <= 1.5e-12):
layers A, B and C at h=1, and the default model at h=2 and h=3. The other ablations and the SDMs have
no saved per-cell output to compare with unless 2026 is re-scored, so they are marked unverified.
Multithreaded LightGBM fits are not always bit-reproducible (two back-to-back h=3 refits in one
diagnostic process differed), so the flags record the check that was actually made.

## Backtest replay 2022-2025 (dashboard "Backtest" tab)

These are the saved Phase 5 test predictions, not re-scored. The models were fit once on labels up
to 2021, and features were updated each year. Hits are cells in a layer's top 2,000 that were first
sighted in the forecast year:

| forecast for | new cells | C top-2000 hits | A | B | adjacency (ranked) |
|---|---|---|---|---|---|
| 2022 | 1,114 | 309 | 419 | 136 | 205 |
| 2023 | 1,809 | 335 | 446 | 76 | 168 |
| 2024 | 1,992 | 424 | 538 | 34 | 180 |
| 2025 | 2,047 | 733 | 592 | 35 | 317 |

## SHAP drivers (layers C and A)

Exact TreeSHAP contributions (LightGBM `pred_contrib`, log-odds) for the top 200 cells of each layer
in the forecast issued at the end of 2025 (`results/dashboard_shap.parquet`).

- **C:** the drivers are all spread features: distance to the nearest occupied cell (mean |SHAP|
  2.20), new occupied cells within 10 cells this year (1.09), active cells within 10 (0.72),
  occupied cells within 5 (0.71) and within 10 (0.62).
- **A:** also led by distance (1.86), active cells within 10 (1.35) and occupied cells within 10
  (0.98). Population density appears among its top five drivers (0.58), consistent with A tracking
  where people are and record.

## Dashboard probabilities

The isotonic calibrators were fit on 2019-2021, when 0.80% of at-risk cells were newly sighted each
year. On the 2022-2025 test, layers A, B and C predicted on average 0.9-1.0% against an observed
2.5%. The dashboard therefore shows rank bands and relative risk (a cell's predicted risk divided by
the average at-risk cell), states this under-prediction in the UI, and does not show absolute
probabilities. No recalibration on test-era base rates was done.

## Retraining

`make retrain` (`src/pipeline/retrain.py`, also `.github/workflows/retrain.yml`) runs the whole
pipeline: GBIF download, cleaning, grid, panels, bias layers, features, tuning/training,
evaluation, and dashboard layers plus backtest. Everything is written under `runs/<timestamp>/`.
It never writes to `results/` or `models/frozen/` and never runs the holdout. The study period
stays as configured. View a run with `ISF_RESULTS=runs/<id>/results make dashboard`.

## Limitations

- **Partial 2026.** The holdout year is a GBIF snapshot from October 2026, so it has fewer first
  sightings than a full year (508; base rate 0.79% vs 1.6-3.1% in 2022-2025). Its numbers are
  comparable across models, but not to full years.
- **Static covariates.** Climate, land cover, elevation, population and roads are single static
  layers (`docs/data_sources.md`). Year-to-year weather, land-use change and control measures
  (nest destruction) are not modelled.
- **Platform-dominated data.** Recording intensity is not stationary: in 2025, 110k of the 181k
  records came from one Belgian platform (Waarnemingen.be). The labels are first *recorded*
  sightings, so every model partly forecasts where recording happens. A's correlation with effort
  (0.26 on test, 0.42 on 2026) shows this most.
- **Low-effort holdout is uninformative.** Only 9 of the 508 first sightings in 2026 fell in the
  low-effort cells (deciles 1-3). No low-effort holdout comparison can separate the models.
- **Default chosen after seeing test results.** Layer C did not beat adjacency on validation
  (PR-AUC 0.041 vs 0.044), so its selection as the dashboard default is informed by test results.
