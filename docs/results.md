# Results: Phases 1 and 2

All numbers below come from the real downloads of 24 Oct 2026 (DOIs in `docs/gbif_downloads.md`)
and are reproducible with `make data`. No model forecasts yet; those are Phases 3-4.

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

## Known limitations at this stage

* The 2026 records (70,964) are excluded because the year is incomplete. They can serve as a
  true out-of-sample check in Phase 4.
* The WorldCover 2021 snapshot is applied to all years, and the climate layers are
  1970-2000 normals, i.e. static covariates.
* The effort proxy (all GBIF insects) is dominated by the same Belgian and Dutch platforms;
  it measures recording activity, not detection probability for hornets specifically.
