# Data sources

Everything is downloaded by `make data` (`src/ingest/*.py`) into `data/raw/` (gitignored) and
cached: a file that is already present is not downloaded again. GBIF download keys, DOIs and
citations are written to [`docs/gbif_downloads.md`](gbif_downloads.md) automatically.

Current downloads (24 Oct 2026):

* Occurrences: GBIF.org (24 October 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.93qv4t (key `0001807-261023074918325`, 464,199 records)
* Effort cube: GBIF.org (24 October 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.s367gk (key `0001813-261023074918325`, 5,449,469 aggregated rows of cell x year x half-year)

| Layer | Source / version | Resolution | Access | Licence / terms | Used for |
|---|---|---|---|---|---|
| Focal-species occurrences | GBIF occurrence download (SIMPLE_CSV), taxon key 1311477, GBIF Backbone | points | GBIF async download API (needs a GBIF account: `GBIF_USER`, `GBIF_PWD`, `GBIF_EMAIL`) | per-dataset CC0 / CC BY / CC BY-NC; cite the download DOI | presences, panel |
| Target-group effort | GBIF SQL download (SQL_TSV_ZIP): Insecta (classKey 216) minus focal species, grouped by EEA 1 km cell x year x half-year (`GBIF_EEARGCode`) | 1 km -> 5 km | GBIF SQL download API (same account) | as above; cite DOI | sampling-effort layer, target-group background, inverse-effort weights |
| Climate | WorldClim 2.1 bio1-bio19 (1970-2000 normals), Fick & Hijmans 2017 | 2.5 arc-min | direct HTTPS | free for academic and other non-commercial use (WorldClim terms) | covariates |
| Elevation | WorldClim 2.1 elevation (derived from SRTM) | 2.5 arc-min | direct HTTPS | WorldClim terms | covariate |
| Land cover | ESA WorldCover 2021 v200 (Zanaga et al. 2022, doi:10.5281/zenodo.7254221) | 10 m, read at COG overview level 4 (320 m) | public AWS S3 COGs, read remotely (no full download) | CC BY 4.0 | land-cover class fractions per cell |
| Population | WorldPop 2020 unconstrained global mosaic, 1 km (doi:10.5258/SOTON/WP00647) | 30 arc-sec | direct HTTPS | CC BY 4.0 | population density (bias proxy) |
| Road density | GRIP4 total road density (Meijer et al. 2018, Environ. Res. Lett. 13 064006) | 5 arc-min | PBL data portal | see PBL GRIP4 terms; cite the paper | road density (bias proxy) |
| Major roads | Natural Earth 10m roads v5 | 1:10M vector | naciscdn.org | public domain | distance to the nearest major road (bias proxy) |
| Country polygons | Natural Earth 10m admin-0 countries | 1:10M vector | naciscdn.org | public domain | region mask, grid, country-centroid cleaning |

## Deviations from the brief

| Brief asked for | What we used | Why |
|---|---|---|
| iNaturalist merge | GBIF only. `src/ingest/inaturalist.py` exists but is disabled by default | iNaturalist research-grade observations are already in GBIF (dataset `50c9509d-22c7-4a22-a47d-8c48425ef4a7`), so merging both would count them twice. When enabled, the module keeps only observations whose URL is not already a GBIF `occurrenceID`, i.e. ones not yet synced. |
| CHELSA | not used | WorldClim already provides bio1-bio19. CHELSA can be added as an extra source later. |
| MODIS MCD12Q1 / CGLS-LC100 | ESA WorldCover 2021 | MODIS needs a NASA Earthdata login, which we don't have. WorldCover is open (CC BY 4.0) and its COG overviews can be read over HTTP without downloading full tiles. |
| GPW population | WorldPop 2020 | GPW needs a NASA Earthdata login. WorldPop is open. |
| OSM distance-to-road (osmnx / Geofabrik) | Natural Earth major roads (distance) + GRIP4 road density | Geofabrik extracts for about 2 million km^2 come to tens of GB. GRIP4 is a harmonised, OSM-derived road-density product, and Natural Earth gives a light distance-to-major-road layer. |
| SRTM / Copernicus DEM | WorldClim elevation (SRTM-derived) | 2.5 arc-min already matches the 5 km cells, so a 30 m DEM would add nothing. |

## Credentials

The only credentials needed are a GBIF account, for the occurrence and SQL downloads. They are
read from the environment variables `GBIF_USER`, `GBIF_PWD` and `GBIF_EMAIL`, falling back to a
local `.env` file (gitignored; template in `.env.example`). They are never hardcoded or committed.
