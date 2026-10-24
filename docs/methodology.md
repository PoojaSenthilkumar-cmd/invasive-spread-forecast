# Methodology: Phases 1 and 2

## Phase 1: data acquisition (`src/ingest/`)

* **Occurrences** (`gbif.py`). One GBIF asynchronous download (SIMPLE_CSV) for taxon key 1311477
  (*Vespa velutina*, GBIF Backbone), limited to the region's countries and bbox, with
  coordinates present, no geospatial issues, and years from 2000 onwards. The predicate is
  POSTed directly to `/occurrence/download/request` with the Backbone as the root
  `checklistKey`. pygbif 0.6 adds the COL XR checklist to nested taxon predicates, and that
  produced two downloads with 0 records. Downloads are tracked in `data/raw/gbif/*_download.json`
  and cached as `data/raw/gbif/*.zip`. A download that succeeds with 0 records is treated as an
  error. DOIs and citations go to `docs/gbif_downloads.md`.
* **Target-group effort** (`gbif.py`, `effort_sql`). One GBIF SQL download that aggregates every
  Insecta record except the focal species, with the same spatial filters and observation-type
  basisOfRecord, by `year x half-year x EEA 1 km reference-grid cell` (`GBIF_EEARGCode`). It
  returns `n_insecta`, `n_hymenoptera` and `n_observers`. Taxa are filtered by *name*, because
  numeric keys in GBIF SQL downloads resolve against COL XR (`classKey = 216` returned 0 rows).
* **Covariates** (`covariates.py`). The rasters listed in `docs/data_sources.md` are downloaded
  once and clipped to the bbox in `data/interim/covariates/`.
* **iNaturalist** (`inaturalist.py`). Disabled by default, because iNaturalist research-grade
  observations are already in GBIF. When enabled, observations whose URL already appears as a
  GBIF `occurrenceID` are dropped.

## Phase 2: cleaning, grid, panel, bias (`src/preprocess/`, `src/features/`)

### Cleaning (`clean.py`)
The steps run in this order, and every step records its removal count in `reports/cleaning_log.csv`:
missing coordinates; out-of-range coordinates; lat or lon equal to 0; GBIF geospatial issue
flags; ABSENT records; fossil, preserved or living specimens and material citations; wrong
species key; missing or unparseable eventDate; outside the study period; coordinate
uncertainty greater than the cell size (5 km); coordinates rounded to 1 decimal place or fewer
(about 11 km or coarser); within 1 km of a country centroid (whole-country and largest-part
centroids, computed in EPSG:3035); outside the region polygons; duplicates with the same
coordinates and date. For half-year windows, records with year-only date precision are also
dropped.

### Grid (`grid.py`)
Regular 5 km cells in EPSG:3035 (ETRS89-LAEA, equal-area), snapped to multiples of the cell
size so they nest within the EEA reference grid. Only cells that intersect the region's land
polygons are kept. `cell_id` is `"{size}m_E{col}N{row}"`, with `col` and `row` in cell units
(the EEA convention).

### Panel (`dataset_builder.py`)
Each cell x window (annual, or Jan-Jun / Jul-Dec) row holds `n_records`, `present` (at least 1
record), `cumulative_present` (any record so far), `first_detection_window` and
`new_detection` (the first window with a record). Presence is thinned spatially by
construction: one bit per cell per window, however many records fall in it.

### Cell covariates (`covariates_join.py`)
Each raster is resampled onto the 5 km grid. Continuous variables are averaged; WorldCover is
converted to class fractions. Distance to the nearest Natural Earth major road is measured in
EPSG:3035.

### Bias correction (`bias.py`)
1. **Effort layer**: the 1 km effort cube is aggregated to 5 km cells x windows
   (`effort_insecta`, `effort_hymenoptera`, `effort_observers`).
2. **Target-group background**: background cells are drawn with probability proportional to
   the Insecta effort before the split year, so presences and background share the same
   recorder bias.
3. **Inverse-effort weights** for presences, `1 / max(effort, q05)`, rescaled to mean 1.
4. **Bias covariates** (log population, road density, distance to road, log effort) are
   included as features during training and fixed at their medians at prediction time
   ("neutralised").

`bias_diagnostic` measures the effect of each correction with a static presence/background
logistic GLM (standardised linear plus quadratic environmental terms). Training presences are
cells occupied up to and including the split year. The test set is every cell not occupied by
then: positives are cells first detected after the split year, negatives are cells never
detected. Metrics are ROC-AUC and PR-AUC on all test cells and on *surveyed* test cells (at
least 1 Insecta record after the split), plus the Spearman correlation between each predicted
map and population, road density and effort. A strong correlation suggests the model is
reproducing where people record rather than where the hornet is. This is a deliberately simple
static check of the corrections. The forecasting models belong to Phases 3-4.
