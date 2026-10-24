"""GBIF ingestion via the asynchronous download API.

Two downloads are made and cached in ``data/raw/gbif``:

* ``occurrences``: all georeferenced records of the focal species in the region
  (``SIMPLE_CSV`` via :func:`pygbif.occurrences.download`).
* ``effort``: a GBIF SQL download (``SQL_TSV_ZIP``) aggregating all Insecta records
  (excluding the focal species) per EEA 1 km reference-grid cell x year x half-year.
  This is the target-group sampling-effort layer used for bias correction.

Download keys/DOIs are persisted in ``<name>_download.json`` so an interrupted run resumes
polling instead of resubmitting, and completed zips are never re-downloaded.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from src.config import gbif_credentials, load_config

log = logging.getLogger(__name__)

TERMINAL_FAIL = {"CANCELLED", "FAILED", "KILLED", "FILE_ERASED"}
# Numeric taxon keys refer to the GBIF Backbone; pygbif otherwise defaults to the COL XR
# checklist, under which numeric keys match nothing (0-record download).
GBIF_BACKBONE = "d7dddbf4-2cf0-4f39-9b2a-bb099caae36c"


def occurrence_predicate(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the GBIF download predicate for the focal species and region."""
    min_lon, min_lat, max_lon, max_lat = cfg["region"]["bbox"]
    wkt = (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},"
        f"{min_lon} {max_lat},{min_lon} {min_lat}))"
    )
    return {
        "type": "and",
        "predicates": [
            {"type": "equals", "key": "TAXON_KEY", "value": str(cfg["species"]["gbif_taxon_key"])},
            {"type": "in", "key": "COUNTRY", "values": list(cfg["region"]["countries"])},
            {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
            {"type": "within", "geometry": wkt},
        ],
    }


def effort_sql(cfg: dict[str, Any]) -> str:
    """Build the GBIF SQL query for the target-group (Insecta) effort cube."""
    e = cfg["gbif"]["effort"]
    countries = ", ".join(f"'{c}'" for c in cfg["region"]["countries"])
    bor = ", ".join(f"'{b}'" for b in e["basis_of_record"])
    min_lon, min_lat, max_lon, max_lat = cfg["region"]["bbox"]
    key = cfg["species"]["gbif_taxon_key"]
    return (
        'SELECT "year", '
        'CASE WHEN "month" IS NULL THEN 0 WHEN "month" <= 6 THEN 1 ELSE 2 END AS half, '
        "GBIF_EEARGCode(1000, decimalLatitude, decimalLongitude, 0) AS eeaCellCode, "
        "COUNT(*) AS n_insecta, "
        f"SUM(CASE WHEN orderKey = {e['hymenoptera_order_key']} THEN 1 ELSE 0 END) "
        "AS n_hymenoptera, "
        "COUNT(DISTINCT recordedBy) AS n_observers "
        "FROM occurrence "
        f"WHERE classKey = {e['class_key']} "
        f"AND countryCode IN ({countries}) "
        f'AND "year" >= {e["min_year"]} '
        "AND hasCoordinate = TRUE AND hasGeospatialIssues = FALSE "
        f"AND decimalLongitude >= {min_lon} AND decimalLongitude <= {max_lon} "
        f"AND decimalLatitude >= {min_lat} AND decimalLatitude <= {max_lat} "
        f"AND (speciesKey IS NULL OR speciesKey <> {key}) "
        f"AND basisOfRecord IN ({bor}) "
        'GROUP BY "year", half, eeaCellCode'
    )


def _state_path(raw_dir: Path, name: str) -> Path:
    return raw_dir / f"{name}_download.json"


def _load_state(raw_dir: Path, name: str) -> dict[str, Any] | None:
    p = _state_path(raw_dir, name)
    return json.loads(p.read_text()) if p.exists() else None


def _save_state(raw_dir: Path, name: str, state: dict[str, Any]) -> None:
    _state_path(raw_dir, name).write_text(json.dumps(state, indent=2, default=str))


def _meta(api: str, key: str) -> dict[str, Any]:
    r = requests.get(f"{api}/occurrence/download/{key}", timeout=60)
    r.raise_for_status()
    return r.json()


def submit_occurrences(cfg: dict[str, Any]) -> str:
    """Submit the focal-species occurrence download and return its key.

    The request is posted directly (same endpoint pygbif uses) because pygbif 0.6.x injects the
    COL XR checklist key into each taxon predicate even when the root ``checklistKey`` is the
    GBIF Backbone, which makes numeric backbone keys match nothing.
    """
    creds = gbif_credentials()
    body = {
        "creator": creds.user,
        "notificationAddresses": [creds.email],
        "sendNotification": False,
        "format": cfg["gbif"]["occurrence_format"],
        "checklistKey": GBIF_BACKBONE,
        "predicate": occurrence_predicate(cfg),
    }
    r = requests.post(
        f"{cfg['gbif']['api_url']}/occurrence/download/request",
        json=body,
        auth=(creds.user, creds.pwd),
        timeout=120,
    )
    r.raise_for_status()
    return r.text.strip()


def submit_effort(cfg: dict[str, Any]) -> str:
    """Submit the target-group effort SQL download and return its key."""
    creds = gbif_credentials()
    body = {
        "sendNotification": False,
        "notificationAddresses": [creds.email],
        "format": "SQL_TSV_ZIP",
        "sql": effort_sql(cfg),
    }
    r = requests.post(
        f"{cfg['gbif']['api_url']}/occurrence/download/request",
        json=body,
        auth=(creds.user, creds.pwd),
        timeout=120,
    )
    r.raise_for_status()
    return r.text.strip()


def ensure_download(cfg: dict[str, Any], name: str, wait: bool = True) -> Path | None:
    """Submit (if needed), poll and fetch a named download; return the local zip path."""
    raw_dir = Path(cfg["paths"]["raw"]) / "gbif"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{name}.zip"
    state = _load_state(raw_dir, name)
    if zip_path.exists() and state and state.get("status") == "SUCCEEDED":
        log.info("%s: cached at %s (DOI %s)", name, zip_path, state.get("doi"))
        return zip_path

    if not state or state.get("status") in TERMINAL_FAIL:
        key = submit_occurrences(cfg) if name == "occurrences" else submit_effort(cfg)
        state = {"name": name, "key": key, "submitted": datetime.now(UTC).isoformat()}
        _save_state(raw_dir, name, state)
        log.info("%s: submitted download %s", name, key)

    api = cfg["gbif"]["api_url"]
    t0 = time.time()
    while True:
        meta = _meta(api, state["key"])
        state.update(
            status=meta["status"],
            doi=meta.get("doi"),
            total_records=meta.get("totalRecords"),
            size_bytes=meta.get("size"),
            download_link=meta.get("downloadLink"),
            created=meta.get("created"),
            modified=meta.get("modified"),
            request=meta.get("request"),
        )
        _save_state(raw_dir, name, state)
        log.info("%s: %s status=%s", name, state["key"], meta["status"])
        if meta["status"] == "SUCCEEDED":
            if not meta.get("totalRecords"):
                raise RuntimeError(
                    f"GBIF download {state['key']} succeeded with 0 records; "
                    "check the predicate/checklist and delete the state file."
                )
            break
        if meta["status"] in TERMINAL_FAIL:
            raise RuntimeError(f"GBIF download {state['key']} ended with {meta['status']}")
        if not wait or time.time() - t0 > cfg["gbif"]["max_wait_s"]:
            return None
        time.sleep(cfg["gbif"]["poll_interval_s"])

    tmp = zip_path.with_suffix(".zip.part")
    with requests.get(state["download_link"], stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.rename(zip_path)
    state["downloaded"] = datetime.now(UTC).isoformat()
    _save_state(raw_dir, name, state)
    write_citations(cfg)
    return zip_path


def citation(state: dict[str, Any]) -> str:
    """Format the GBIF-recommended citation for a download."""
    created = state.get("created") or state.get("submitted")
    date = datetime.fromisoformat(str(created).replace("Z", "+00:00")).strftime("%d %B %Y")
    return f"GBIF.org ({date}) GBIF Occurrence Download https://doi.org/{state['doi']}"


def write_citations(cfg: dict[str, Any]) -> Path:
    """Write ``docs/gbif_downloads.md`` with key, DOI, record count and citation per download."""
    raw_dir = Path(cfg["paths"]["raw"]) / "gbif"
    lines = [
        "# GBIF downloads",
        "",
        "Generated by `src/ingest/gbif.py`. Cite these DOIs when using the data.",
        "",
    ]
    for name in ("occurrences", "effort"):
        st = _load_state(raw_dir, name)
        if not st:
            continue
        lines += [
            f"## {name}",
            "",
            f"- Download key: `{st['key']}`",
            f"- Status: {st.get('status')}",
            f"- DOI: https://doi.org/{st.get('doi')}",
            f"- Records: {st.get('total_records')}",
            f"- Created: {st.get('created')}",
            f"- Citation: {citation(st) if st.get('doi') else 'pending'}",
            "",
        ]
        if name == "effort":
            lines += ["SQL:", "", "```sql", effort_sql(cfg), "```", ""]
        else:
            lines += [
                "Predicate:",
                "",
                "```json",
                json.dumps(occurrence_predicate(cfg), indent=2),
                "```",
                "",
            ]
    out = Path(cfg["paths"]["docs"]) / "gbif_downloads.md"
    out.write_text("\n".join(lines))
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", choices=["occurrences", "effort"], default=None)
    ap.add_argument("--no-wait", action="store_true", help="submit/poll once and exit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    names = [args.only] if args.only else ["occurrences", "effort"]
    for name in names:
        ensure_download(cfg, name, wait=False)
    if not args.no_wait:
        for name in names:
            ensure_download(cfg, name, wait=True)
    write_citations(cfg)


if __name__ == "__main__":
    main()
