# Spatiotemporal forecasting of invasive species spread

This project forecasts where *Vespa velutina* (Asian hornet) will be sighted **next** in Western
Europe, using time-ordered GBIF citizen-science records combined with climate, land-cover and
observer-bias covariates.

**Status: Phases 1-5 are done** (data, cleaning/grid/panel/bias correction, causal features,
models and baselines, chronological evaluation, and the single 2026 holdout run); see
`docs/results.md`. Run `make features train evaluate`, then `make holdout` once (it refuses to
re-run). Dashboard (Phase 6): `make dashboard` (Streamlit, http://localhost:8501). It shows the frozen 1-year forecast layers from the end of 2025: C (bias-corrected, spread inputs only; default), A (no bias correction), B (approved default) and the adjacency baseline, each labelled with validation/test precision@k. An Evaluation tab shows the 2026 holdout (scored once, frozen). `make dashboard-data` rebuilds `results/dashboard_layers.parquet` from the frozen fits and checks them against the holdout forecast; the committed file lets the dashboard run without the raw data.

## Quick start

```bash
uv sync                               # Python 3.11 environment from pyproject/uv.lock
cp .env.example .env                  # fill in GBIF_USER, GBIF_PWD, GBIF_EMAIL (never commit .env)
make data                             # GBIF downloads + covariates -> clean -> grid -> panel -> bias diagnostic -> figures
make test lint
```

`make data` is cached: GBIF downloads are tracked by key in `data/raw/gbif/*_download.json` and
are not resubmitted, and rasters already on disk are not downloaded again. A fresh run needs
about 2 GB of disk. Most of the time is spent waiting for GBIF to build the files and
downloading WorldPop.

## Layout

| Path | Contents |
|---|---|
| `configs/default.yaml` | species, region, grid (EPSG:3035, 5 km), time windows, cleaning thresholds, bias settings |
| `src/ingest/` | `gbif.py` (occurrence download + SQL effort cube), `covariates.py`, `inaturalist.py` (optional, deduplicated) |
| `src/preprocess/` | `clean.py`, `grid.py`, `dataset_builder.py` (cell x window panel), `bias.py` |
| `src/features/covariates_join.py` | per-cell climate, elevation, land-cover fractions, population, roads |
| `src/viz/` | cleaning/panel plots and maps |
| `reports/` | cleaning log, panel summaries, bias diagnostic (CSV/JSON) and figures |
| `docs/` | `species_selection.md`, `data_sources.md`, `gbif_downloads.md` (DOIs), `methodology.md`, `results.md` |

## Main outputs (`data/processed/`, gitignored)

* `grid_5000m.parquet`: 80,098 land cells
* `panel_5000m_year.parquet` / `panel_5000m_half.parquet`: `cell_id, window, n_records, present,
  cumulative_present, first_detection_window, new_detection`
* `cell_covariates_5000m.parquet`, `effort_5000m_year.parquet`

See `docs/results.md` for the numbers, and cite the GBIF DOIs in `docs/gbif_downloads.md`.
