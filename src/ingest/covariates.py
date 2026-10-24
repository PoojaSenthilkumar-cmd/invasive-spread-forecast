"""Download and clip environmental and sampling-bias covariates to the study region.

All downloads are cached in ``data/raw/covariates`` and skipped when present.
Clipped rasters/vectors (WGS84, region bbox) are written to ``data/interim/covariates``.

Sources (see ``docs/data_sources.md`` for versions and licences):

* WorldClim 2.1 bioclim bio1-bio19 and elevation (2.5 arc-min).
* ESA WorldCover 2021 v200 land cover, read remotely from the public COGs at an overview
  level (default 320 m) and mosaicked.
* WorldPop 2020 1 km population counts (global mosaic).
* GRIP4 total road density (5 arc-min, m/km^2).
* Natural Earth 10m roads (major roads, for distance-to-road) and admin-0 countries
  (region mask, country centroids for cleaning).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.errors import RasterioIOError
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_origin
from shapely.geometry import box
from tqdm import tqdm

from src.config import load_config

log = logging.getLogger(__name__)

WORLDCLIM_BASE = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"
WORLDCOVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
SOURCES: dict[str, str] = {
    "worldclim_bio": f"{WORLDCLIM_BASE}/wc2.1_{{res}}_bio.zip",
    "worldclim_elev": f"{WORLDCLIM_BASE}/wc2.1_{{res}}_elev.zip",
    "worldpop": (
        "https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/0_Mosaicked/"
        "ppp_{year}_1km_Aggregated.tif"
    ),
    "grip4": "https://dataportaal.pbl.nl/downloads/GRIP4/GRIP4_density_total.zip",
    "ne_roads": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_roads.zip",
    "ne_countries": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip",
}
WGS84 = CRS.from_epsg(4326)


def download(url: str, dest_dir: Path) -> Path:
    """Download ``url`` into ``dest_dir`` unless the file already exists."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / url.rsplit("/", 1)[-1]
    if dest.exists():
        log.info("cached: %s", dest.name)
        return dest
    tmp = dest.with_name(dest.name + ".part")
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(tmp, "wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False) as bar,
        ):
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)
    return dest


def clip_raster(src_path: str, out_path: Path, bbox: list[float], crs: CRS | None = None) -> Path:
    """Clip a raster (any GDAL path, incl. ``/vsizip/``) to a WGS84 bbox and save as GeoTIFF."""
    if out_path.exists():
        return out_path
    with rasterio.open(src_path) as src:
        data, transform = rio_mask(src, [box(*bbox)], crop=True, all_touched=True, filled=True)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
            compress="deflate",
            crs=crs or src.crs or WGS84,
        )
        if profile.get("nodata") is None and np.issubdtype(data.dtype, np.floating):
            profile["nodata"] = np.nan
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
    return out_path


def worldclim(cfg: dict[str, Any], raw: Path, out: Path) -> list[Path]:
    """Download WorldClim 2.1 bio1-19 + elevation and clip them to the region."""
    res = cfg["covariates"]["worldclim_resolution"]
    bbox = cfg["region"]["bbox"]
    bio_zip = download(SOURCES["worldclim_bio"].format(res=res), raw)
    elev_zip = download(SOURCES["worldclim_elev"].format(res=res), raw)
    paths = []
    for i in range(1, 20):
        src = f"/vsizip/{bio_zip}/wc2.1_{res}_bio_{i}.tif"
        paths.append(clip_raster(src, out / f"bio{i}.tif", bbox))
    paths.append(clip_raster(f"/vsizip/{elev_zip}/wc2.1_{res}_elev.tif", out / "elev.tif", bbox))
    return paths


def worldpop(cfg: dict[str, Any], raw: Path, out: Path) -> Path:
    """Download the WorldPop global 1 km population mosaic and clip it."""
    url = SOURCES["worldpop"].format(year=cfg["covariates"]["worldpop_year"])
    return clip_raster(str(download(url, raw)), out / "population.tif", cfg["region"]["bbox"])


def grip4(cfg: dict[str, Any], raw: Path, out: Path) -> Path:
    """Download GRIP4 total road density (ESRI ASCII grid, WGS84) and clip it."""
    z = download(SOURCES["grip4"], raw)
    return clip_raster(
        f"/vsizip/{z}/grip4_total_dens_m_km2.asc",
        out / "road_density.tif",
        cfg["region"]["bbox"],
        crs=WGS84,
    )


def natural_earth(cfg: dict[str, Any], raw: Path, out: Path) -> tuple[Path, Path, Path]:
    """Extract the region mask, all-country polygons and major roads from Natural Earth."""
    bbox = cfg["region"]["bbox"]
    countries_zip = download(SOURCES["ne_countries"], raw)
    roads_zip = download(SOURCES["ne_roads"], raw)
    world = gpd.read_file(f"zip://{countries_zip}")
    world["iso2"] = world["ISO_A2_EH"].where(world["ISO_A2_EH"] != "-99", world["ISO_A2"])
    world_path = out / "ne_countries.gpkg"
    if not world_path.exists():
        world[["iso2", "ADMIN", "geometry"]].to_file(world_path, driver="GPKG")
    region_path = out / "region.gpkg"
    if not region_path.exists():
        reg = world[world["iso2"].isin(cfg["region"]["countries"])][["iso2", "ADMIN", "geometry"]]
        reg = gpd.clip(reg, box(*bbox))
        reg.to_file(region_path, driver="GPKG")
    roads_path = out / "roads_major.gpkg"
    if not roads_path.exists():
        roads = gpd.read_file(f"zip://{roads_zip}", bbox=tuple(bbox))
        roads = roads[roads["featurecla"].str.contains("Road", na=False)]
        roads[["type", "featurecla", "geometry"]].to_file(roads_path, driver="GPKG")
    return region_path, world_path, roads_path


def worldcover_tiles(bbox: list[float]) -> list[str]:
    """List WorldCover 3x3 degree tile names whose extent intersects ``bbox``."""
    min_lon, min_lat, max_lon, max_lat = bbox
    names = []
    for lat in range(int(np.floor(min_lat / 3) * 3), int(np.ceil(max_lat / 3) * 3), 3):
        for lon in range(int(np.floor(min_lon / 3) * 3), int(np.ceil(max_lon / 3) * 3), 3):
            ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            names.append(f"{ns}{ew}")
    return names


def worldcover(cfg: dict[str, Any], out: Path) -> Path:
    """Mosaic ESA WorldCover 2021 over the region from remote COG overviews (no full download)."""
    out_path = out / "landcover.tif"
    if out_path.exists():
        return out_path
    level = cfg["covariates"]["worldcover_overview_level"]
    px_per_tile = 36000 // (2 ** (level + 1))
    res = 3.0 / px_per_tile
    min_lon, min_lat, max_lon, max_lat = cfg["region"]["bbox"]
    x0, y1 = np.floor(min_lon / 3) * 3, np.ceil(max_lat / 3) * 3
    x1, y0 = np.ceil(max_lon / 3) * 3, np.floor(min_lat / 3) * 3
    width, height = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    mosaic = np.zeros((height, width), dtype=np.uint8)
    n_ok = 0
    for name in tqdm(worldcover_tiles(cfg["region"]["bbox"]), desc="worldcover"):
        url = f"/vsicurl/{WORLDCOVER_BASE}/ESA_WorldCover_10m_2021_v200_{name}_Map.tif"
        try:
            with rasterio.open(url, overview_level=level) as src:
                arr = src.read(1)
                left, top = src.bounds.left, src.bounds.top
        except RasterioIOError:
            continue  # ocean-only tiles do not exist
        r0, c0 = int(round((y1 - top) / res)), int(round((left - x0) / res))
        mosaic[r0 : r0 + arr.shape[0], c0 : c0 + arr.shape[1]] = arr
        n_ok += 1
    log.info("worldcover: %d tiles mosaicked at %.5f deg", n_ok, res)
    profile = dict(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=WGS84,
        transform=from_origin(x0, y1, res, res),
        nodata=0,
        compress="deflate",
        tiled=True,
    )
    tmp = out / "landcover_full.tif"
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(mosaic, 1)
    clip_raster(str(tmp), out_path, cfg["region"]["bbox"])
    tmp.unlink()
    return out_path


def run(cfg: dict[str, Any]) -> None:
    """Fetch and clip all covariates."""
    raw = Path(cfg["paths"]["raw"]) / "covariates"
    out = Path(cfg["paths"]["interim"]) / "covariates"
    out.mkdir(parents=True, exist_ok=True)
    natural_earth(cfg, raw, out)
    worldclim(cfg, raw, out)
    grip4(cfg, raw, out)
    worldpop(cfg, raw, out)
    worldcover(cfg, out)
    log.info("covariates ready in %s", out)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
