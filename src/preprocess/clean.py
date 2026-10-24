"""Clean raw GBIF occurrences and log how many records each step removes.

Steps (in order): missing/invalid/zero coordinates, GBIF geospatial issue flags, absence
records, fossil/preserved/living specimens, taxon check, missing date, outside study period,
coordinate uncertainty larger than the grid cell, coordinates rounded to <= 1 decimal place
(precision coarser than a cell), country-centroid artefacts, records outside the region
polygons, duplicates (same coordinates + date), and - for sub-year windows only - records with
year-only date precision.
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import load_config

log = logging.getLogger(__name__)

USECOLS = [
    "gbifID",
    "datasetKey",
    "occurrenceID",
    "speciesKey",
    "taxonKey",
    "countryCode",
    "occurrenceStatus",
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters",
    "eventDate",
    "day",
    "month",
    "year",
    "basisOfRecord",
    "issue",
    "recordedBy",
]


class CleaningLog:
    """Accumulates (step, removed, remaining) rows while filtering a DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Start the log from the raw record count."""
        self.rows: list[dict[str, Any]] = [
            {"step": "raw download", "removed": 0, "remaining": len(df)}
        ]

    def apply(self, df: pd.DataFrame, drop_mask: pd.Series, step: str) -> pd.DataFrame:
        """Drop rows where ``drop_mask`` is True and record the step."""
        drop_mask = drop_mask.fillna(False).astype(bool)
        kept = df.loc[~drop_mask]
        self.rows.append({"step": step, "removed": int(drop_mask.sum()), "remaining": len(kept)})
        log.info("%-60s removed %8d remaining %8d", step, drop_mask.sum(), len(kept))
        return kept

    def frame(self) -> pd.DataFrame:
        """Return the log as a DataFrame."""
        return pd.DataFrame(self.rows)


def read_simple_csv(zip_path: Path) -> pd.DataFrame:
    """Read a GBIF SIMPLE_CSV download (tab-separated, unquoted) from its zip."""
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as fh:
            return pd.read_csv(
                fh,
                sep="\t",
                usecols=lambda c: c in USECOLS,
                quoting=3,
                dtype={
                    "gbifID": "int64",
                    "occurrenceID": "string",
                    "datasetKey": "string",
                    "issue": "string",
                    "recordedBy": "string",
                    "eventDate": "string",
                },
                low_memory=False,
                on_bad_lines="warn",
            )


def has_any_issue(issues: pd.Series, flags: list[str]) -> pd.Series:
    """True where the semicolon-separated GBIF ``issue`` field contains any of ``flags``."""
    s = issues.fillna("")
    out = pd.Series(False, index=issues.index)
    for f in flags:
        out |= s.str.contains(rf"(?:^|;){f}(?:;|$)", regex=True)
    return out


def rounded_to(v: pd.Series, decimals: int) -> pd.Series:
    """True where values have at most ``decimals`` decimal places."""
    scaled = v.to_numpy(dtype=float) * 10**decimals
    return pd.Series(np.isclose(scaled, np.round(scaled), atol=1e-6), index=v.index)


def country_centroids(world: gpd.GeoDataFrame, crs: str) -> gpd.GeoSeries:
    """Centroids of all country polygons (largest part and whole country) in ``crs``."""
    w = world.to_crs(crs)
    whole = w.geometry.centroid
    parts = w.explode(index_parts=False)
    largest = parts.loc[parts.geometry.area.groupby(level=0).idxmax()].geometry.centroid
    return gpd.GeoSeries(pd.concat([whole, largest], ignore_index=True), crs=crs)


def clean(
    raw: pd.DataFrame, cfg: dict[str, Any], region: gpd.GeoDataFrame, world: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply all cleaning steps; return (clean records, cleaning log)."""
    c = cfg["cleaning"]
    crs = cfg["grid"]["crs"]
    t = cfg["time"]
    df = raw.copy()
    lg = CleaningLog(df)
    lat, lon = df["decimalLatitude"], df["decimalLongitude"]
    df = lg.apply(df, lat.isna() | lon.isna(), "missing coordinates")
    df = lg.apply(
        df,
        (df["decimalLatitude"].abs() > 90) | (df["decimalLongitude"].abs() > 180),
        "invalid coordinates (out of range)",
    )
    df = lg.apply(
        df,
        (df["decimalLatitude"] == 0) | (df["decimalLongitude"] == 0),
        "zero coordinates (lat or lon == 0)",
    )
    df = lg.apply(df, has_any_issue(df["issue"], c["drop_issues"]), "GBIF geospatial issue flags")
    df = lg.apply(
        df,
        df["occurrenceStatus"].isin(c["drop_occurrence_status"]),
        "absence records (occurrenceStatus=ABSENT)",
    )
    df = lg.apply(
        df,
        df["basisOfRecord"].isin(c["drop_basis_of_record"]),
        "fossil/preserved/living specimens, material citations",
    )
    df = lg.apply(
        df, df["speciesKey"] != cfg["species"]["gbif_taxon_key"], "not matched to focal species key"
    )
    df = lg.apply(df, df["year"].isna(), "missing/unparseable eventDate")
    last = int(t.get("holdout_year") or t["end_year"])
    df = lg.apply(
        df,
        (df["year"] < t["start_year"]) | (df["year"] > last),
        f"outside data period {t['start_year']}-{last}",
    )
    unc = df["coordinateUncertaintyInMeters"]
    df = lg.apply(
        df,
        unc > c["max_coordinate_uncertainty_m"],
        f"coordinate uncertainty > {c['max_coordinate_uncertainty_m']} m",
    )
    df = lg.apply(
        df,
        rounded_to(df["decimalLatitude"], 1) & rounded_to(df["decimalLongitude"], 1),
        "coordinates rounded to <= 1 decimal place",
    )

    pts = gpd.GeoSeries(
        gpd.points_from_xy(df["decimalLongitude"], df["decimalLatitude"]), index=df.index, crs=4326
    ).to_crs(crs)
    cents = country_centroids(world, crs)
    near = gpd.sjoin_nearest(
        gpd.GeoDataFrame(geometry=pts),
        gpd.GeoDataFrame(geometry=cents),
        max_distance=c["country_centroid_buffer_m"],
        how="left",
    )
    near_mask = near["index_right"].notna().groupby(level=0).any().reindex(df.index)
    df = lg.apply(df, near_mask, f"within {c['country_centroid_buffer_m']} m of a country centroid")
    pts = pts.loc[df.index]

    reg = region.to_crs(crs)
    inside = gpd.sjoin(
        gpd.GeoDataFrame(geometry=pts), reg[["geometry"]], predicate="within", how="left"
    )
    in_mask = inside["index_right"].notna().groupby(level=0).any().reindex(df.index)
    df = lg.apply(df, ~in_mask, "outside region polygons")
    pts = pts.loc[df.index]
    df = df.assign(x=pts.x.to_numpy(), y=pts.y.to_numpy())

    date_key = df["eventDate"].fillna("").str.slice(0, 10)
    dup = df.assign(
        _d=date_key, _lat=df["decimalLatitude"].round(5), _lon=df["decimalLongitude"].round(5)
    ).duplicated(["_lat", "_lon", "_d", "year", "month", "day"])
    df = lg.apply(df, dup, "duplicates (same coordinates + date)")
    if t["window"] == "half":
        df = lg.apply(df, df["month"].isna(), "year-only date precision (sub-year windows)")

    out = pd.DataFrame(
        {
            "gbif_id": df["gbifID"].astype("int64"),
            "dataset_key": df["datasetKey"],
            "occurrence_id": df["occurrenceID"],
            "country": df["countryCode"],
            "lon": df["decimalLongitude"].astype(float),
            "lat": df["decimalLatitude"].astype(float),
            "x": df["x"],
            "y": df["y"],
            "uncertainty_m": df["coordinateUncertaintyInMeters"],
            "year": df["year"].astype(int),
            "month": df["month"].astype("Int64"),
            "day": df["day"].astype("Int64"),
            "basis_of_record": df["basisOfRecord"],
        }
    ).reset_index(drop=True)
    return out, lg.frame()


def run(cfg: dict[str, Any]) -> pd.DataFrame:
    """Read the cached GBIF download, clean it and write parquet + log."""
    raw_zip = Path(cfg["paths"]["raw"]) / "gbif" / "occurrences.zip"
    cov = Path(cfg["paths"]["interim"]) / "covariates"
    raw = read_simple_csv(raw_zip)
    region = gpd.read_file(cov / "region.gpkg")
    world = gpd.read_file(cov / "ne_countries.gpkg")
    out, lg = clean(raw, cfg, region, world)
    interim = Path(cfg["paths"]["interim"])
    out.to_parquet(interim / "occurrences_clean.parquet", index=False)
    reports = Path(cfg["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    lg.to_csv(reports / "cleaning_log.csv", index=False)
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
