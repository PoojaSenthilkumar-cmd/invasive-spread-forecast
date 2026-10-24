"""Final out-of-sample check on the 2026 holdout. Run ONCE, after Phase 5 is frozen.

Forecast issued at the end of 2025 (features from windows <= 2025) for first sightings in 2026.
Models are refit with the frozen Phase 4 hyperparameters on every label available in the study
period (label windows <= 2025); calibrators are the isotonic maps fit on the 2019-2021
expanding-window OOF scores (``results/predictions_oof.parquet``). No tuning or model choice
happens here. 2026 is a partial year (GBIF snapshot taken in October 2026), so base rates are
lower than in full years; ranking metrics are the most comparable.

Refuses to run if ``results/holdout_metrics.json`` exists (pass ``--force`` to override).
Also writes ``results/forecast_from_2025.parquet``: per-cell P(first sighting within 1, 2, 3
years of 2025) for the dashboard.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.config import load_config
from src.evaluation.evaluate import _json, compute_metrics, prepare_rows
from src.models.spread import (
    BASELINES,
    fixed_scorer,
    lgbm_fitter,
    load_table,
    static_sdm_fitter,
)
from src.models.train import DEFAULT, ROW_COLS, results_dir, variants
from src.preprocess.bias import build_effort
from src.preprocess.dataset_builder import end_window, panel_path

log = logging.getLogger(__name__)
SDMS = {"static SDM (logistic, poly-2)": "lr", "static SDM (random forest)": "rf"}
PREREG = "holdout_preregistration.json"


def _calibrator(oof: pd.DataFrame, name: str, k: int) -> IsotonicRegression:
    o = oof[(oof["model"] == name) & (oof["horizon"] == k)]
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(o["score"], o["y"])


def holdout_rows(cfg: dict[str, Any], table: pd.DataFrame) -> pd.DataFrame:
    """Rows at the last study window with 2026 labels and 2026 effort attached."""
    end_w = end_window(cfg)
    rows = table[table["window"] == end_w].copy()
    panel = pd.read_parquet(panel_path(cfg), columns=["cell_id", "first_detection_window"])
    first = panel.drop_duplicates("cell_id").set_index("cell_id")["first_detection_window"]
    rows["y_h1"] = (rows["cell_id"].map(first) == end_w + 1).astype(np.float32)
    eff = build_effort(cfg)
    e = eff[eff["window"] == end_w + 1].set_index("cell_id")["effort_insecta"]
    rows["meta_effort_target"] = rows["cell_id"].map(e).fillna(0).astype(np.float32)
    return rows


def run(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Refit on all study-period labels, score the 2026 holdout once, save metrics."""
    out = results_dir(cfg)
    dest = out / "holdout_metrics.json"
    if dest.exists() and not force:
        raise SystemExit(f"{dest} exists: the holdout has already been evaluated.")
    mcfg, seed = cfg["model"], int(cfg["model"]["seed"])
    last = int(cfg["time"]["end_year"])
    params = json.loads((out / "tuning.json").read_text())["best_params"]
    oof = pd.read_parquet(out / "predictions_oof.parquet")
    table = load_table(cfg)
    table = table[table["year"] + 1 >= mcfg["first_target_year"]].reset_index(drop=True)
    vs = {v.name: v for v in variants(table)}
    rows = holdout_rows(cfg, table)
    effort = build_effort(cfg)

    prereg = json.loads((out / PREREG).read_text())
    fitters = {n: lgbm_fitter(table, 1, v, params, mcfg, seed) for n, v in vs.items()}
    fitters.update({n: static_sdm_fitter(cfg, table, kind, effort) for n, kind in SDMS.items()})
    fitters.update({n: fixed_scorer(fn) for n, fn in BASELINES.items()})
    if sorted(fitters) != sorted(prereg["models"]):
        raise SystemExit("holdout model set differs from the pre-registration")

    parts, scorers = [], {}
    for name, fit in fitters.items():
        scorers[name] = fit(last)  # all labels ending <= 2025
        s = scorers[name](rows)
        prob = _calibrator(oof, name, 1).predict(s)
        parts.append(pd.DataFrame({"idx": rows.index, "score": s, "prob": prob, "model": name}))
        log.info("holdout scored: %s", name)
    preds = pd.concat(parts, ignore_index=True).assign(horizon=1)
    ev_rows = prepare_rows(rows[ROW_COLS + ["y_h1"]], cfg)
    metrics, deciles = compute_metrics(preds, ev_rows, cfg)
    metrics["holdout"] = {
        "forecast_issued_after": last,
        "label_year": last + 1,
        "n_rows": int(len(rows)),
        "n_new_cells": int(rows["y_h1"].sum()),
        "note": "partial year (GBIF snapshot Oct 2026); evaluated once",
        "preregistration": prereg,
    }
    dest.write_text(json.dumps(metrics, indent=2, default=_json))
    deciles.to_csv(out / "holdout_by_effort_decile.csv", index=False)

    fc = rows[["cell_id"]].copy()
    fc["p_within_1y"] = preds[preds["model"] == DEFAULT]["prob"].to_numpy()
    for k in cfg["features"]["horizons"][1:]:
        s = lgbm_fitter(table, k, vs[DEFAULT], params, mcfg, seed)(last)(rows)
        fc[f"p_within_{k}y"] = _calibrator(oof, DEFAULT, k).predict(s)
    p = fc.filter(like="p_within").to_numpy()
    fc[fc.filter(like="p_within").columns] = np.maximum.accumulate(p, axis=1)
    fc.to_parquet(out / "forecast_from_2025.parquet", index=False)
    return metrics


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_config(args.config), force=args.force)


if __name__ == "__main__":
    main()
