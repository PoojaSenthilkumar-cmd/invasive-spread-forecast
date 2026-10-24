import geopandas as gpd
import numpy as np
from shapely.geometry import box

from src.preprocess.grid import assign_cells, build_grid, cell_ids


def test_assign_cells_floor_semantics():
    col, row = assign_cells(np.array([0, 4999.9, 5000, -1]), np.array([0, 0, 10001, -5000]), 5000)
    assert col.tolist() == [0, 0, 1, -1]
    assert row.tolist() == [0, 0, 2, -1]


def test_cell_ids_follow_eea_convention():
    ids = cell_ids(np.array([700, 701]), np.array([560, 561]), 5000)
    assert ids.tolist() == ["5kmE3500N2800", "5kmE3505N2805"]


def test_1km_cells_nest_in_5km_cells():
    # every 1 km EEA cell lower-left corner falls in exactly one 5 km cell
    x = np.arange(3_500_000, 3_510_000, 1000)
    col5, _ = assign_cells(x, np.zeros_like(x), 5000)
    assert col5.tolist() == [700] * 5 + [701] * 5


def test_build_grid_covers_region_and_land_fraction():
    crs = "EPSG:3035"
    region = gpd.GeoDataFrame(
        {"iso2": ["AA", "BB"]},
        geometry=[box(0, 0, 10_000, 10_000), box(10_000, 0, 12_500, 5_000)],
        crs=crs,
    )
    g = build_grid(region, crs, 5000)
    assert len(g) == 5  # 4 full cells in AA + 1 half cell in BB
    assert set(g["country"]) == {"AA", "BB"}
    half = g[g["country"] == "BB"].iloc[0]
    assert np.isclose(half["land_frac"], 0.5)
    assert np.allclose(g.loc[g["country"] == "AA", "land_frac"], 1.0)
    assert g["cell_id"].is_unique
