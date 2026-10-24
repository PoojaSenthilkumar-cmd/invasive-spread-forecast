"""Export the frozen 1-year forecast layers (issued at the end of 2025) and their labels.

Reproduces the frozen holdout fits of ``src.evaluation.holdout`` (same data, hyperparameters,
seed and 2019-2021 OOF isotonic calibrators). Nothing is tuned, changed or re-evaluated, and no
2026 label is read. The default model's output is checked against the saved
``results/forecast_from_2025.parquet`` to confirm the refit is identical to the frozen run.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import load_config
from src.evaluation.evaluate import per_year_at_k
from src.evaluation.holdout import _calibrator
from src.models.spread import BASELINES, lgbm_fitter, load_table
from src.models.train import DEFAULT, results_dir, variants
from src.preprocess.dataset_builder import end_window, panel_path
from src.preprocess.grid import grid_path

log = logging.getLogger(__name__)
ADJ = "adjacency (ranked)"
# key -> (model name in results, short title, what it is)
LAYERS: dict[str, tuple[str, str, str]] = {
    "C": (
        "ablation: dynamic features only",
        "Bias-corrected spread model, spread inputs only (dashboard default)",
        "LightGBM on neighbour and spread-front features only. Target-group negatives and "
        "inverse-effort weights; no effort, climate, land-use, population or road inputs.",
    ),
    "A": (
        "ablation: no bias correction",
        "Spread model without bias correction",
        "LightGBM on all inputs (spread features, recording effort, climate, land use, "
        "population, roads). Uniform negatives, no weights, no neutralisation.",
    ),
    "B": (
        DEFAULT,
        "Approved bias-corrected default",
        "LightGBM on all inputs. Target-group negatives, inverse-effort weights, population "
        "and road covariates neutralised, recording effort as an explicit feature.",
    ),
    "ADJ": (
        ADJ,
        "Adjacency baseline (not a model)",
        "Cells ranked by the number of occupied neighbouring 5 km cells (1-cell ring, then the "
        "2-cell ring as tie-break). Gives a ranking, not a probability.",
    ),
}
KS = [100, 500, 2000]


def _rank(score: np.ndarray, seed: int = 0) -> np.ndarray:
    """1 = highest score; ties broken by a seeded random draw (as in the evaluation)."""
    order = np.lexsort((np.random.default_rng(seed).random(len(score)), -score))
    rank = np.empty(len(score), dtype=np.int64)
    rank[order] = np.arange(1, len(score) + 1)
    return rank


def _git_commit(path: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%h %cI", "--", path],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out or None
    except (OSError, subprocess.CalledProcessError):
        return None


def labels(cfg: dict[str, Any], table: pd.DataFrame, oof: pd.DataFrame) -> dict[str, Any]:
    """Validation (OOF 2019-2021), test (2022-2025) and 2026 holdout numbers per layer."""
    out = results_dir(cfg)
    test = json.loads((out / "metrics.json").read_text())["models"]
    hold = json.loads((out / "holdout_metrics.json").read_text())
    o = oof[oof["horizon"] == 1].copy()
    o["year"] = table.loc[o["idx"], "year"].to_numpy()
    assert (table.loc[o["idx"], "y_h1"].to_numpy() == o["y"].to_numpy()).all()
    val = {}
    for name, g in o.groupby("model"):
        val[name] = {"pr_auc": float(average_precision_score(g["y"], g["score"]))}
        val[name].update(per_year_at_k(g, "y", KS))
    res: dict[str, Any] = {"layers": {}, "ks": KS}
    for key, (name, title, what) in LAYERS.items():
        t, h = test[name]["h1"], hold["models"][name]["h1"]
        res["layers"][key] = {
            "model": name,
            "title": title,
            "what": what,
            "validation": {m: val[name][m] for m in ["pr_auc"] + [f"precision@{k}" for k in KS]},
            "validation_pr_auc_minus_adjacency": val[name]["pr_auc"] - val[ADJ]["pr_auc"],
            "test": {m: t[m] for m in ["pr_auc"] + [f"precision@{k}" for k in KS]},
            "test_low_effort_precision@100": t["low_effort"]["precision@100"],
            "test_pr_auc_minus_adjacency_ci95": t.get("pr_auc_minus_adjacency_ci95"),
            "test_spearman_effort": t["spearman_with"]["meta_effort_target"],
            "holdout_spearman_effort": h["spearman_with"]["meta_effort_target"],
        }
    res["holdout_frozen"] = _git_commit("results/holdout_metrics.json")
    res["preregistration_commit"] = _git_commit("results/holdout_preregistration.json")
    return res


def run(cfg: dict[str, Any]) -> pd.DataFrame:
    """Write ``results/dashboard_layers.parquet`` and ``results/dashboard_labels.json``."""
    out = results_dir(cfg)
    mcfg, seed, last = cfg["model"], int(cfg["model"]["seed"]), int(cfg["time"]["end_year"])
    params = json.loads((out / "tuning.json").read_text())["best_params"]
    oof = pd.read_parquet(out / "predictions_oof.parquet")
    table = load_table(cfg)
    table = table[table["year"] + 1 >= mcfg["first_target_year"]].reset_index(drop=True)
    vs = {v.name: v for v in variants(table)}
    end_w = end_window(cfg)
    rows = table[table["window"] == end_w]  # features <= 2025 only; no 2026 labels attached

    lay = rows[["cell_id", "occ_ring1", "occ_ring2", "dist_km"]].reset_index(drop=True)
    for key, (name, _, _) in LAYERS.items():
        if name in BASELINES:
            s = BASELINES[name](rows)
            lay[f"prob_{key}"] = np.nan
        else:
            s = lgbm_fitter(table, 1, vs[name], params, mcfg, seed)(last)(rows)
            lay[f"prob_{key}"] = _calibrator(oof, name, 1).predict(s)
        lay[f"rank_{key}"] = _rank(np.asarray(s, dtype=float))
        log.info("layer %s (%s) exported", key, name)

    saved = pd.read_parquet(out / "forecast_from_2025.parquet")
    assert (saved["cell_id"].to_numpy() == lay["cell_id"].to_numpy()).all()
    diff = float(np.abs(saved["p_within_1y"].to_numpy() - lay["prob_B"].to_numpy()).max())
    if diff > 1e-9:
        raise SystemExit(f"refit differs from the frozen holdout forecast (max diff {diff})")
    log.info("frozen-model check passed: default layer equals saved forecast (max diff %g)", diff)

    panel = pd.read_parquet(panel_path(cfg), columns=["cell_id", "first_detection_window"])
    first = panel.drop_duplicates("cell_id").set_index("cell_id")["first_detection_window"]
    first = first.astype(float)
    occ = first[first <= end_w]
    grid = pd.read_parquet(grid_path(cfg), columns=["cell_id", "lon", "lat", "country"])
    occ_df = pd.DataFrame(
        {"cell_id": occ.index, "first_year": (occ.to_numpy() + cfg["time"]["start_year"])}
    )
    lay["status"] = "at risk"
    occ_df["status"] = "occupied by 2025"
    df = pd.concat([lay, occ_df], ignore_index=True).merge(grid, on="cell_id", how="left")
    df.to_parquet(out / "dashboard_layers.parquet", index=False)
    (out / "dashboard_labels.json").write_text(json.dumps(labels(cfg, table, oof), indent=2))
    return df


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = run(load_config())
    print(df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
