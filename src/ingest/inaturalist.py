"""Optional iNaturalist ingestion with de-duplication against GBIF.

iNaturalist research-grade observations are already published to GBIF (dataset
``50c9509d-22c7-4a22-a47d-8c48425ef4a7``), where ``occurrenceID`` is the observation URL. The
pipeline therefore uses GBIF as the single source and this module is disabled by default
(``inaturalist.enabled: false``). When enabled, it fetches research-grade observations for the
focal species inside the region bbox and keeps only those whose URL is *not* already present as
a GBIF ``occurrenceID`` (i.e. observations not yet synced to GBIF), avoiding double counting.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pyinaturalist import get_observations

from src.config import load_config

log = logging.getLogger(__name__)
INAT_GBIF_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"


def fetch_observations(cfg: dict[str, Any], per_page: int = 200) -> pd.DataFrame:
    """Fetch research-grade iNaturalist observations for the focal species in the bbox."""
    min_lon, min_lat, max_lon, max_lat = cfg["region"]["bbox"]
    rows, id_above = [], 0
    while True:
        resp = get_observations(
            taxon_name=cfg["species"]["scientific_name"],
            quality_grade="research",
            swlat=min_lat,
            swlng=min_lon,
            nelat=max_lat,
            nelng=max_lon,
            per_page=per_page,
            order_by="id",
            order="asc",
            id_above=id_above,
        )
        res = resp.get("results", [])
        if not res:
            break
        for o in res:
            loc = o.get("location")
            if not loc:
                continue
            lat, lon = (float(v) for v in str(loc).split(","))
            rows.append(
                {
                    "occurrence_id": f"https://www.inaturalist.org/observations/{o['id']}",
                    "lat": lat,
                    "lon": lon,
                    "observed_on": o.get("observed_on"),
                    "positional_accuracy": o.get("positional_accuracy"),
                }
            )
        id_above = res[-1]["id"]
    return pd.DataFrame(rows)


def deduplicate_against_gbif(inat: pd.DataFrame, gbif_occurrence_ids: pd.Series) -> pd.DataFrame:
    """Drop iNaturalist observations whose URL already appears as a GBIF ``occurrenceID``."""
    known = set(gbif_occurrence_ids.dropna().astype(str).str.replace("http://", "https://"))
    return inat[~inat["occurrence_id"].isin(known)].reset_index(drop=True)


def run(cfg: dict[str, Any]) -> pd.DataFrame | None:
    """Fetch + de-duplicate if enabled; write ``data/raw/inaturalist_not_in_gbif.parquet``."""
    if not cfg["inaturalist"]["enabled"]:
        log.info("iNaturalist ingestion disabled (data already reaches the pipeline via GBIF)")
        return None
    inat = fetch_observations(cfg)
    gbif = pd.read_parquet(Path(cfg["paths"]["interim"]) / "occurrences_clean.parquet")
    new = deduplicate_against_gbif(inat, gbif["occurrence_id"])
    log.info("iNaturalist: %d fetched, %d not yet in GBIF", len(inat), len(new))
    new.to_parquet(Path(cfg["paths"]["raw"]) / "inaturalist_not_in_gbif.parquet", index=False)
    return new


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
