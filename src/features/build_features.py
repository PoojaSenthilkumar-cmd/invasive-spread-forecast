"""Phase 3: time-ordered (cell, window) feature table and spread labels.

Task: for every cell ``c`` and window ``t`` such that ``c`` has **not** been confirmed present by
the end of ``t``, predict whether ``c`` is first sighted in ``t+1`` (``y_h1``) or within
``t+1..t+k`` (``y_h{k}``).

Leakage rule: features of row ``(c, t)`` are computed only from panel and effort windows
``<= t`` (each window's inputs are sliced ``[: t + 1]`` before any feature is computed). Columns
prefixed ``y_`` (labels) and ``meta_`` (target-window effort, used only for sample weights and
evaluation strata) look ahead by design and are never model features. The holdout year
(windows after ``end_window``) is dropped before anything else, so it cannot reach this table.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from src.config import load_config
from src.preprocess.bias import build_effort
from src.preprocess.dataset_builder import end_window, panel_path
from src.preprocess.grid import load_or_build_grid

log = logging.getLogger(__name__)

EFFORT_COLS = ["effort_insecta", "effort_hymenoptera", "effort_observers"]
NOT_FEATURES = {"cell_id", "window", "year"}
NO_PRESENCE_DIST_KM = 2000.0


def is_feature(col: str) -> bool:
    """True for model-feature columns (not ids, labels ``y_*`` or look-ahead ``meta_*``)."""
    return col not in NOT_FEATURES and not col.startswith(("y_", "meta_"))


def box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Sum of ``a`` over the (2r+1) x (2r+1) square around each pixel (zero outside)."""
    ap = np.pad(a.astype(np.float64), r)
    s = np.zeros((ap.shape[0] + 1, ap.shape[1] + 1))
    s[1:, 1:] = ap.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]


class Raster:
    """Maps grid cells (col, row) onto a dense 2-D array for neighbourhood operations."""

    def __init__(self, col: np.ndarray, row: np.ndarray, size: int):
        """Index cells by (row, col) offsets from the grid's minimum column/row."""
        self.ri = (row - row.min()).astype(int)
        self.ci = (col - col.min()).astype(int)
        self.shape = (int(self.ri.max()) + 1, int(self.ci.max()) + 1)
        self.size = size

    def to2d(self, v: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """Scatter a per-cell vector onto the raster."""
        out = np.full(self.shape, fill, dtype=np.float64)
        out[self.ri, self.ci] = v
        return out

    def to1d(self, a: np.ndarray) -> np.ndarray:
        """Gather raster values back to the per-cell vector."""
        return a[self.ri, self.ci]


def _dense(panel: pd.DataFrame, effort: pd.DataFrame, cells: pd.Index, n_win: int) -> dict:
    """Dense (window, cell) arrays of records, presence and effort for windows < ``n_win``."""
    shape = (n_win, len(cells))
    out = {"records": np.zeros(shape), **{c: np.zeros(shape) for c in EFFORT_COLS}}
    p = panel[(panel["window"] < n_win) & (panel["n_records"] > 0)]
    pi = cells.get_indexer(p["cell_id"])
    out["records"][p["window"].to_numpy(), pi] = p["n_records"].to_numpy()
    e = effort[(effort["window"] < n_win) & effort["cell_id"].isin(cells)]
    ei = cells.get_indexer(e["cell_id"])
    for c in EFFORT_COLS:
        out[c][e["window"].to_numpy(), ei] = e[c].to_numpy()
    out["present"] = out["records"] > 0
    return out


def _distance(r: Raster, occupied: np.ndarray, first_w: np.ndarray, t: int) -> dict:
    """Distance (km), bearing and age (windows) of the nearest occupied cell."""
    if not occupied.any():
        n = len(occupied)
        return {
            "dist_km": np.full(n, NO_PRESENCE_DIST_KM),
            "bearing_sin": np.zeros(n),
            "bearing_cos": np.zeros(n),
            "nearest_age": np.full(n, -1.0),
        }
    occ2 = r.to2d(occupied.astype(float)) > 0
    d, (iy, ix) = distance_transform_edt(~occ2, sampling=r.size, return_indices=True)
    ny, nx = r.to1d(iy), r.to1d(ix)
    dy, dx = (ny - r.ri).astype(float), (nx - r.ci).astype(float)
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    first2 = r.to2d(first_w, fill=np.nan)
    return {
        "dist_km": r.to1d(d) / 1000.0,
        "bearing_sin": dy / norm,
        "bearing_cos": dx / norm,
        "nearest_age": t - first2[ny, nx],
    }


def window_features(
    hist: dict[str, np.ndarray], r: Raster, t: int, rings: list[int]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Features for every cell at window ``t`` from history arrays covering windows ``0..t``.

    Returns (features, occupied_by_t). ``hist`` arrays must have exactly ``t + 1`` rows.
    """
    assert all(v.shape[0] == t + 1 for v in hist.values()), "history must end at t"
    pres = hist["present"]
    cum = pres.any(axis=0)
    ever = pres.cumsum(axis=0) > 0
    first_w = np.where(cum, ever.argmax(axis=0), np.nan).astype(float)
    f: dict[str, np.ndarray] = {}

    cur = _distance(r, cum, first_w, t)
    f.update(cur)
    for lag in (1, 2):
        if t - lag >= 0:
            prev = _distance(r, pres[: t - lag + 1].any(axis=0), first_w, t)["dist_km"]
        else:
            prev = np.full(len(cum), NO_PRESENCE_DIST_KM)
        f[f"front_speed_{lag}w_kmpw"] = (prev - cur["dist_km"]) / lag

    cum2 = r.to2d(cum.astype(float))
    act2 = r.to2d(pres[t].astype(float))
    for k in rings:
        f[f"occ_ring{k}"] = r.to1d(box_sum(cum2, k) - cum2)
    for k in (1, 3, 10):
        f[f"active_ring{k}"] = r.to1d(box_sum(act2, k) - act2)
    for lag in range(3):
        new = (first_w == t - lag).astype(float)
        new2 = r.to2d(new)
        for k in (3, 10):
            f[f"new_ring{k}_lag{lag}"] = r.to1d(box_sum(new2, k))
    rec_cum2 = r.to2d(hist["records"].sum(axis=0))
    f["log_records_ring3"] = np.log1p(r.to1d(box_sum(rec_cum2, 3)))

    n_occ = ever.sum(axis=1).astype(float)  # occupied cells by each window <= t
    f["region_occupied"] = np.full(len(cum), n_occ[t])
    for lag in (1, 3):
        base = n_occ[t - lag] if t - lag >= 0 else 0.0
        f[f"region_growth_{lag}w"] = np.full(len(cum), np.log1p(n_occ[t]) - np.log1p(base))
    any_rec = np.flatnonzero(pres.any(axis=1))
    f["windows_since_first_record"] = np.full(
        len(cum), float(t - any_rec[0]) if any_rec.size else -1.0
    )

    e = hist["effort_insecta"]
    f["log_effort_t"] = np.log1p(e[t])
    f["log_effort_cum"] = np.log1p(e.sum(axis=0))
    f["log_effort_mean3"] = np.log1p(e[max(0, t - 2) :].mean(axis=0))
    f["log_effort_hym_t"] = np.log1p(hist["effort_hymenoptera"][t])
    f["log_observers_t"] = np.log1p(hist["effort_observers"][t])
    f["log_effort_ring3_t"] = np.log1p(r.to1d(box_sum(r.to2d(e[t]), 3)))
    return pd.DataFrame(f), cum


def build_features(
    panel: pd.DataFrame,
    effort: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    size: int,
    start_year: int,
    end_w: int,
    horizons: list[int],
    rings: list[int],
    first_t: int = 0,
) -> pd.DataFrame:
    """Build the (cell, window) table for ``t = first_t..end_w``; only unoccupied cells kept.

    Inputs after ``end_w`` (the holdout) are discarded first. ``y_h{k}`` is NaN where
    ``t + k > end_w`` (label not observable inside the study period).
    """
    cells = pd.Index(grid["cell_id"])
    dense = _dense(panel, effort, cells, end_w + 1)
    r = Raster(grid["col"].to_numpy(), grid["row"].to_numpy(), size)
    pres = dense["present"]
    ever = pres.cumsum(axis=0) > 0
    first = np.where(ever[-1], ever.argmax(axis=0), np.iinfo(np.int32).max)
    parts = []
    for t in range(first_t, end_w + 1):
        hist = {k: v[: t + 1] for k, v in dense.items()}
        feats, occupied = window_features(hist, r, t, rings)
        keep = ~occupied
        df = feats[keep].astype(np.float32)
        df.insert(0, "cell_id", cells[keep])
        df.insert(1, "window", np.int16(t))
        df.insert(2, "year", np.int16(start_year + t))
        fk = first[keep]
        for k in horizons:
            y = ((fk > t) & (fk <= t + k)).astype(np.float32)
            df[f"y_h{k}"] = y if t + k <= end_w else np.float32(np.nan)
        df["meta_effort_target"] = (
            dense["effort_insecta"][t + 1][keep].astype(np.float32)
            if t + 1 <= end_w
            else np.float32(np.nan)
        )
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def features_path(cfg: dict[str, Any]) -> Path:
    """Path of the feature table parquet."""
    return (
        Path(cfg["paths"]["processed"])
        / f"features_{cfg['grid']['cell_size_m']}m_{cfg['time']['window']}.parquet"
    )


def run(cfg: dict[str, Any]) -> pd.DataFrame:
    """Build and save the feature table for the study period (holdout excluded)."""
    fc = cfg["features"]
    grid = load_or_build_grid(cfg)
    df = build_features(
        pd.read_parquet(panel_path(cfg)),
        build_effort(cfg),
        grid,
        size=int(cfg["grid"]["cell_size_m"]),
        start_year=cfg["time"]["start_year"],
        end_w=end_window(cfg),
        horizons=fc["horizons"],
        rings=fc["rings"],
    )
    df.to_parquet(features_path(cfg), index=False)
    n_feat = sum(is_feature(c) for c in df.columns)
    log.info("features: %d rows x %d dynamic features -> %s", len(df), n_feat, features_path(cfg))
    return df


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
