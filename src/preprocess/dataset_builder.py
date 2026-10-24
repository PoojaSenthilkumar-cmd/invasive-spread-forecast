"""Build the cell x time-window panel of sightings.

For every grid cell ``c`` and window ``t`` the panel stores:

* ``n_records``: cleaned sightings in (c, t)
* ``present``: 1 if any sighting in (c, t) (equivalent to spatial thinning to one record per
  cell per window)
* ``cumulative_present``: 1 if c has had a sighting in any window <= t
* ``first_detection_window``: index of the first window with a sighting in c (NaN if never)
* ``new_detection``: 1 if t is the first detection window of c
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.preprocess.grid import assign_cells, cell_ids, load_or_build_grid

log = logging.getLogger(__name__)


def window_index(year: pd.Series, month: pd.Series, start_year: int, window: str) -> pd.Series:
    """Map (year, month) to an integer window index starting at 0 for ``start_year``.

    ``window`` is ``"year"`` (12-month windows) or ``"half"`` (Jan-Jun, Jul-Dec). For ``"half"``
    records without a month get ``<NA>`` (they must be removed during cleaning).
    """
    year = pd.Series(year).astype("Int64")
    if window == "year":
        return (year - start_year).astype("Int64")
    if window == "half":
        month = pd.Series(month).astype("Int64")
        return ((year - start_year) * 2 + (month > 6).astype("Int64")).astype("Int64")
    raise ValueError(f"unknown window type {window!r}")


def window_labels(n: int, start_year: int, window: str) -> list[str]:
    """Human-readable labels for window indices 0..n-1 (``2004`` or ``2004H1``)."""
    if window == "year":
        return [str(start_year + i) for i in range(n)]
    return [f"{start_year + i // 2}H{i % 2 + 1}" for i in range(n)]


def n_windows(cfg: dict[str, Any]) -> int:
    """Number of windows in the configured study period."""
    t = cfg["time"]
    years = t["end_year"] - t["start_year"] + 1
    return years if t["window"] == "year" else 2 * years


def build_panel(
    occ: pd.DataFrame, cell_index: pd.Index, n_win: int, size: int, start_year: int, window: str
) -> pd.DataFrame:
    """Build the dense cell x window panel from cleaned occurrences (with projected x, y)."""
    col, row = assign_cells(occ["x"].to_numpy(), occ["y"].to_numpy(), size)
    occ = occ.assign(
        cell_id=cell_ids(col, row, size),
        window=window_index(occ["year"], occ["month"], start_year, window),
    )
    occ = occ[occ["window"].notna() & occ["cell_id"].isin(cell_index)]
    counts = occ.groupby(["cell_id", "window"]).size().rename("n_records")

    idx = pd.MultiIndex.from_product([cell_index, np.arange(n_win)], names=["cell_id", "window"])
    panel = pd.DataFrame(index=idx).join(
        counts.set_axis(counts.index.set_levels(counts.index.levels[1].astype(int), level=1))
    )
    panel["n_records"] = panel["n_records"].fillna(0).astype(np.int32)
    panel["present"] = (panel["n_records"] > 0).astype(np.int8)
    panel = panel.reset_index()
    panel["cumulative_present"] = (
        panel.groupby("cell_id", sort=False)["present"].cummax().astype(np.int8)
    )
    first = panel.loc[panel["present"] == 1].groupby("cell_id")["window"].min()
    panel["first_detection_window"] = panel["cell_id"].map(first).astype("Int64")
    panel["new_detection"] = (panel["window"] == panel["first_detection_window"]).fillna(False)
    panel["new_detection"] = panel["new_detection"].astype(np.int8)
    return panel


def panel_path(cfg: dict[str, Any]) -> Path:
    """Path of the panel parquet for the configured cell size and window type."""
    return (
        Path(cfg["paths"]["processed"])
        / f"panel_{cfg['grid']['cell_size_m']}m_{cfg['time']['window']}.parquet"
    )


def run(cfg: dict[str, Any]) -> pd.DataFrame:
    """Build and save the panel; also write a per-window summary to ``reports/``."""
    grid = load_or_build_grid(cfg)
    occ = pd.read_parquet(Path(cfg["paths"]["interim"]) / "occurrences_clean.parquet")
    t = cfg["time"]
    if t["window"] == "half":
        occ = occ[occ["month"].notna()]
    nw = n_windows(cfg)
    panel = build_panel(
        occ,
        pd.Index(grid["cell_id"]),
        nw,
        int(cfg["grid"]["cell_size_m"]),
        t["start_year"],
        t["window"],
    )
    labels = window_labels(nw, t["start_year"], t["window"])
    panel["window_label"] = pd.Categorical.from_codes(panel["window"], labels)
    panel.to_parquet(panel_path(cfg), index=False)

    summary = (
        panel.groupby("window_label", observed=True)
        .agg(
            records=("n_records", "sum"),
            occupied_cells=("present", "sum"),
            new_cells=("new_detection", "sum"),
            cumulative_cells=("cumulative_present", "sum"),
        )
        .reset_index()
    )
    out = Path(cfg["paths"]["reports"]) / f"panel_summary_{t['window']}.csv"
    summary.to_csv(out, index=False)
    log.info(
        "panel: %d rows (%d cells x %d windows) -> %s", len(panel), len(grid), nw, panel_path(cfg)
    )
    return panel


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--window",
        choices=["year", "half"],
        default=None,
        help="override time.window from the config",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.window:
        cfg["time"]["window"] = args.window
    run(cfg)


if __name__ == "__main__":
    main()
