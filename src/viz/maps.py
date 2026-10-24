"""Static maps of the gridded panel, effort layer and bias-diagnostic predictions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.preprocess.bias import effort_path
from src.preprocess.dataset_builder import panel_path
from src.preprocess.grid import load_or_build_grid

log = logging.getLogger(__name__)


def _base(ax: plt.Axes, region: gpd.GeoDataFrame) -> None:
    region.boundary.plot(ax=ax, color="#888888", linewidth=0.3)
    ax.set_axis_off()


def first_detection_map(
    grid: gpd.GeoDataFrame,
    panel: pd.DataFrame,
    region: gpd.GeoDataFrame,
    start_year: int,
    out: Path,
) -> Path:
    """Map of the year each cell was first detected (the invasion front over time)."""
    first = panel.groupby("cell_id")["first_detection_window"].first()
    g = grid.assign(first=grid["cell_id"].map(first).astype(float) + start_year)
    fig, ax = plt.subplots(figsize=(8, 8))
    _base(ax, region)
    g[g["first"].notna()].plot(
        column="first",
        ax=ax,
        cmap="viridis",
        legend=True,
        legend_kwds={"label": "first detection year", "shrink": 0.6},
    )
    ax.set_title("Vespa velutina: first detection year per 5 km cell")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def effort_map(
    grid: gpd.GeoDataFrame, effort: pd.DataFrame, region: gpd.GeoDataFrame, out: Path
) -> Path:
    """Map of total target-group (Insecta) records per cell over the study period."""
    tot = effort.groupby("cell_id")["effort_insecta"].sum()
    g = grid.assign(e=np.log10(grid["cell_id"].map(tot).fillna(0) + 1))
    fig, ax = plt.subplots(figsize=(8, 8))
    _base(ax, region)
    g.plot(
        column="e",
        ax=ax,
        cmap="magma",
        legend=True,
        legend_kwds={"label": "log10(1 + Insecta records)", "shrink": 0.6},
    )
    ax.set_title("Target-group sampling effort (GBIF Insecta, excl. focal species)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def bias_variant_maps(
    grid: gpd.GeoDataFrame, preds: pd.DataFrame, region: gpd.GeoDataFrame, out: Path
) -> Path:
    """Small multiples of predicted suitability (percentile rank) per bias-correction variant."""
    variants = [c for c in preds.columns if c != "cell_id"]
    n = len(variants)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
    p = preds.set_index("cell_id").reindex(grid["cell_id"])
    for ax, v in zip(axes.ravel(), variants, strict=False):
        _base(ax, region)
        grid.assign(s=p[v].rank(pct=True).to_numpy()).plot(
            column="s", ax=ax, cmap="RdYlBu_r", vmin=0, vmax=1
        )
        ax.set_title(v, fontsize=10)
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    fig.suptitle(
        "Static presence/background GLM: predicted suitability percentile by "
        "bias-correction variant"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def run(cfg: dict[str, Any]) -> list[Path]:
    """Write all Phase 1-2 maps to ``reports/figures``."""
    figs = Path(cfg["paths"]["reports"]) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    crs = cfg["grid"]["crs"]
    grid = load_or_build_grid(cfg)
    region = gpd.read_file(Path(cfg["paths"]["interim"]) / "covariates" / "region.gpkg").to_crs(crs)
    outs = [
        first_detection_map(
            grid,
            pd.read_parquet(panel_path(cfg)),
            region,
            cfg["time"]["start_year"],
            figs / "first_detection_map.png",
        )
    ]
    if effort_path(cfg).exists():
        outs.append(
            effort_map(grid, pd.read_parquet(effort_path(cfg)), region, figs / "effort_map.png")
        )
    pred_path = Path(cfg["paths"]["processed"]) / "bias_diagnostic_predictions.parquet"
    if pred_path.exists():
        outs.append(
            bias_variant_maps(
                grid, pd.read_parquet(pred_path), region, figs / "bias_variant_maps.png"
            )
        )
    return outs


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
