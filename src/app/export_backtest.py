"""Export the 2022-2025 backtest replay and the calibration gap for the dashboard.

Reads the saved Phase 5 test predictions only (``results/predictions_test.parquet``); nothing is
refit or re-scored. For each forecast issued at the end of year t (2021-2024) and each layer, the
cell's rank among at-risk cells is stored with the cell's later confirmed first-sighting year
(study period only, <= 2025; 2026 holdout labels are not used).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.app.export_layers import LAYERS, _rank
from src.config import load_config
from src.models.train import results_dir
from src.preprocess.dataset_builder import end_window, panel_path

log = logging.getLogger(__name__)


def run(cfg: dict[str, Any]) -> pd.DataFrame:
    """Write ``results/dashboard_backtest.parquet`` and ``results/dashboard_calibration.json``."""
    out = results_dir(cfg)
    rows = pd.read_parquet(out / "test_rows.parquet")
    preds = pd.read_parquet(out / "predictions_test.parquet")
    preds = preds[preds["horizon"] == 1]
    start, end_w = cfg["time"]["start_year"], end_window(cfg)
    panel = pd.read_parquet(panel_path(cfg), columns=["cell_id", "first_detection_window"])
    first = panel.drop_duplicates("cell_id").set_index("cell_id")["first_detection_window"]
    first = first.astype(float)
    first = first.where(first <= end_w) + start  # study period only; 2026 stays sealed

    bt = rows[["cell_id", "year", "y_h1"]].copy()
    bt["first_year"] = bt["cell_id"].map(first).astype("float32")
    calib: dict[str, Any] = {}
    for key, (name, title, _) in LAYERS.items():
        p = preds[preds["model"] == name].set_index("idx").reindex(bt.index)
        assert p["score"].notna().all(), name
        rank = np.empty(len(bt), dtype=np.int32)
        for _, idx in bt.groupby("year").indices.items():
            rank[idx] = _rank(p["score"].to_numpy()[idx])
        bt[f"rank_{key}"] = rank
        if key != "ADJ":
            g = pd.DataFrame({"year": bt["year"] + 1, "prob": p["prob"], "y": bt["y_h1"]})
            per = g.groupby("year").agg(mean_predicted=("prob", "mean"), observed=("y", "mean"))
            calib[key] = {
                "model": name,
                "title": title,
                "mean_predicted": float(g["prob"].mean()),
                "observed_rate": float(g["y"].mean()),
                "per_target_year": {int(y): v for y, v in per.round(5).to_dict("index").items()},
            }
    oof = pd.read_parquet(out / "predictions_oof.parquet", columns=["y", "model", "horizon"])
    o = oof[(oof["horizon"] == 1) & (oof["model"] == LAYERS["C"][0])]
    calib["_validation_observed_rate"] = float(o["y"].mean())
    bt = bt.rename(columns={"year": "issued_after"}).astype({"issued_after": "int16"})
    bt.reset_index(drop=True).to_parquet(out / "dashboard_backtest.parquet", index=False)
    (out / "dashboard_calibration.json").write_text(json.dumps(calib, indent=2))
    return bt


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bt = run(load_config())
    print(bt.shape)
    print(
        json.dumps(
            json.loads((results_dir(load_config()) / "dashboard_calibration.json").read_text()),
            indent=1,
        )[:1500]
    )


if __name__ == "__main__":
    main()
