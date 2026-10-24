"""Equal-area regular grid (default 5 km in EPSG:3035) and record-to-cell assignment.

Cells are aligned to multiples of ``cell_size_m`` from the CRS origin, so a 5 km cell nests
exactly 25 EEA 1 km reference-grid cells. Cell IDs follow the EEA reference-grid convention,
e.g. ``5kmE3500N2800`` for the cell whose lower-left corner is (3 500 km E, 2 800 km N).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from src.config import load_config

log = logging.getLogger(__name__)


def assign_cells(x: np.ndarray, y: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return integer (col, row) indices of the cells containing projected points (x, y)."""
    col = np.floor(np.asarray(x, dtype=float) / size).astype(np.int64)
    row = np.floor(np.asarray(y, dtype=float) / size).astype(np.int64)
    return col, row


def cell_ids(col: np.ndarray, row: np.ndarray, size: int) -> np.ndarray:
    """Build EEA-style cell IDs (e.g. ``5kmE3500N2800``) from integer indices."""
    km = size / 1000
    label = f"{km:g}km"
    e = (np.asarray(col) * km).astype(np.int64).astype(str)
    n = (np.asarray(row) * km).astype(np.int64).astype(str)
    return np.char.add(np.char.add(np.char.add(np.char.add(label, "E"), e), "N"), n)


def build_grid(region: gpd.GeoDataFrame, crs: str, size: int) -> gpd.GeoDataFrame:
    """Build all grid cells intersecting the (projected) region polygons.

    Returns one row per cell with ``cell_id``, ``col``, ``row``, centroid ``x``/``y`` (CRS units)
    and ``lon``/``lat``, the majority ``country`` and the ``land_frac`` inside the region.
    """
    reg = region.to_crs(crs)
    minx, miny, maxx, maxy = reg.total_bounds
    c0, r0 = int(np.floor(minx / size)), int(np.floor(miny / size))
    c1, r1 = int(np.ceil(maxx / size)), int(np.ceil(maxy / size))
    cols, rows = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
    cols, rows = cols.ravel(), rows.ravel()
    geoms = [
        box(c * size, r * size, (c + 1) * size, (r + 1) * size)
        for c, r in zip(cols, rows, strict=True)
    ]
    cand = gpd.GeoDataFrame({"col": cols, "row": rows}, geometry=geoms, crs=crs)
    inter = gpd.overlay(cand, reg[["iso2", "geometry"]], how="intersection", keep_geom_type=True)
    inter["area"] = inter.geometry.area
    by_cell = inter.groupby(["col", "row"])
    land = by_cell["area"].sum() / (size * size)
    major = inter.loc[by_cell["area"].idxmax(), ["col", "row", "iso2"]].set_index(["col", "row"])
    out = pd.DataFrame({"land_frac": land}).join(major).reset_index()
    out = out.rename(columns={"iso2": "country"})
    out["cell_id"] = cell_ids(out["col"].to_numpy(), out["row"].to_numpy(), size)
    out["x"] = (out["col"] + 0.5) * size
    out["y"] = (out["row"] + 0.5) * size
    geom = [
        box(c * size, r * size, (c + 1) * size, (r + 1) * size)
        for c, r in zip(out["col"], out["row"], strict=True)
    ]
    grid = gpd.GeoDataFrame(out, geometry=geom, crs=crs)
    ll = gpd.GeoSeries(gpd.points_from_xy(grid["x"], grid["y"]), crs=crs).to_crs(4326)
    grid["lon"], grid["lat"] = ll.x.to_numpy(), ll.y.to_numpy()
    grid = grid.sort_values(["row", "col"]).reset_index(drop=True)
    return grid[
        ["cell_id", "col", "row", "x", "y", "lon", "lat", "country", "land_frac", "geometry"]
    ]


def grid_path(cfg: dict[str, Any]) -> Path:
    """Location of the cached grid GeoParquet for the configured cell size."""
    return Path(cfg["paths"]["processed"]) / f"grid_{cfg['grid']['cell_size_m']}m.parquet"


def load_or_build_grid(cfg: dict[str, Any]) -> gpd.GeoDataFrame:
    """Load the cached grid or build it from ``data/interim/covariates/region.gpkg``."""
    path = grid_path(cfg)
    if path.exists():
        return gpd.read_parquet(path)
    region = gpd.read_file(Path(cfg["paths"]["interim"]) / "covariates" / "region.gpkg")
    grid = build_grid(region, cfg["grid"]["crs"], int(cfg["grid"]["cell_size_m"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(path)
    log.info("grid: %d cells of %d m -> %s", len(grid), cfg["grid"]["cell_size_m"], path)
    return grid


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_or_build_grid(load_config(args.config))


if __name__ == "__main__":
    main()
