"""Aggregate covariate rasters/vectors onto the equal-area grid (one row per cell).

Rasters are warped onto a raster aligned exactly with the grid (same CRS, origin and cell
size) so that each pixel is one grid cell:

* WorldClim bio1-bio19, elevation, GRIP4 road density: area-weighted mean.
* ESA WorldCover: fraction of each land-cover class (mean of class indicators).
* WorldPop: population sum per cell -> density (people / km^2).
* Natural Earth major roads: distance from cell centroid to nearest road (km).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from src.config import load_config
from src.preprocess.grid import load_or_build_grid

log = logging.getLogger(__name__)

LANDCOVER_CLASSES = {
    10: "lc_tree",
    20: "lc_shrub",
    30: "lc_grass",
    40: "lc_crop",
    50: "lc_built",
    60: "lc_bare",
    80: "lc_water",
    90: "lc_wetland",
}


class GridRaster:
    """A raster definition aligned with the grid cells (one pixel per cell)."""

    def __init__(self, grid: pd.DataFrame, crs: str, size: int) -> None:
        """Derive the raster extent from the grid's integer (col, row) indices."""
        self.size, self.crs = size, crs
        self.c0, self.c1 = int(grid["col"].min()), int(grid["col"].max())
        self.r0, self.r1 = int(grid["row"].min()), int(grid["row"].max())
        self.width, self.height = self.c1 - self.c0 + 1, self.r1 - self.r0 + 1
        self.transform = from_origin(self.c0 * size, (self.r1 + 1) * size, size, size)
        self.i = (self.r1 - grid["row"].to_numpy()).astype(int)
        self.j = (grid["col"].to_numpy() - self.c0).astype(int)

    def warp(
        self,
        src: np.ndarray,
        src_transform: Any,
        src_crs: Any,
        src_nodata: float | None,
        resampling: Resampling,
    ) -> np.ndarray:
        """Warp a source array onto the grid and return one value per grid cell."""
        dst = np.full((self.height, self.width), np.nan, dtype=np.float32)
        reproject(
            source=src.astype(np.float32),
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=src_nodata,
            dst_transform=self.transform,
            dst_crs=self.crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        return dst[self.i, self.j]


def _read(path: Path) -> tuple[np.ndarray, Any, Any, float | None]:
    with rasterio.open(path) as s:
        return s.read(1), s.transform, s.crs, s.nodata


def build_cell_covariates(cfg: dict[str, Any]) -> pd.DataFrame:
    """Compute the static per-cell covariate table."""
    grid = load_or_build_grid(cfg)
    size, crs = int(cfg["grid"]["cell_size_m"]), cfg["grid"]["crs"]
    cov = Path(cfg["paths"]["interim"]) / "covariates"
    gr = GridRaster(grid, crs, size)
    out = pd.DataFrame({"cell_id": grid["cell_id"]})

    for name in [f"bio{i}" for i in range(1, 20)] + ["elev"]:
        a, t, c, nd = _read(cov / f"{name}.tif")
        out[name] = gr.warp(a, t, c, nd, Resampling.average)
    a, t, c, nd = _read(cov / "road_density.tif")
    out["road_density"] = gr.warp(a, t, c, nd, Resampling.average)

    a, t, c, nd = _read(cov / "population.tif")
    a = np.where((a < 0) | (a == nd), 0, a)
    pop = gr.warp(a, t, c, None, Resampling.sum)
    out["pop_density"] = pop / (size * size / 1e6)

    lc, t, c, _ = _read(cov / "landcover.tif")
    valid = lc > 0
    for code, col in LANDCOVER_CLASSES.items():
        ind = np.where(valid, (lc == code).astype(np.float32), np.nan)
        out[col] = gr.warp(ind, t, c, np.nan, Resampling.average)

    roads = gpd.read_file(cov / "roads_major.gpkg").to_crs(crs)
    pts = gpd.GeoDataFrame(
        {"cell_id": grid["cell_id"]}, geometry=gpd.points_from_xy(grid["x"], grid["y"]), crs=crs
    )
    near = gpd.sjoin_nearest(pts, roads[["geometry"]], distance_col="d").groupby(level=0)["d"].min()
    out["dist_road_km"] = (near.reindex(pts.index).to_numpy() / 1000).astype(np.float32)

    out["land_frac"] = grid["land_frac"].to_numpy()
    n_missing = out.isna().sum()
    if n_missing.any():
        log.info(
            "cells with missing covariates (filled with the country median):\n%s",
            n_missing[n_missing > 0].to_string(),
        )
    num = out.columns.drop("cell_id")
    out[num] = out[num].fillna(out[num].groupby(grid["country"].to_numpy()).transform("median"))
    out[num] = out[num].fillna(out[num].median())
    return out


def covariates_path(cfg: dict[str, Any]) -> Path:
    """Path of the per-cell covariate parquet."""
    size = cfg["grid"]["cell_size_m"]
    return Path(cfg["paths"]["processed"]) / f"cell_covariates_{size}m.parquet"


def run(cfg: dict[str, Any]) -> pd.DataFrame:
    """Build (or load cached) per-cell covariates."""
    path = covariates_path(cfg)
    if path.exists():
        return pd.read_parquet(path)
    out = build_cell_covariates(cfg)
    out.to_parquet(path, index=False)
    log.info("cell covariates: %s -> %s", out.shape, path)
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
