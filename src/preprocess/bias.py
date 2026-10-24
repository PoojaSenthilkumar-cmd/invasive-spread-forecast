"""Observation / sampling-bias correction.

Implemented corrections (ablated in :func:`bias_diagnostic`):

a. **Target-group effort layer**: all GBIF Insecta records (focal species excluded) per cell
   and window, from the GBIF SQL effort cube. Used to draw *target-group background* points
   (background sampled proportionally to effort, so background shares the presences' bias).
b. **Spatial thinning** (one record per cell per window - the panel's ``present`` flag) and
   **inverse-effort sample weights** for presences.
c. **Bias covariates** (population density, road density, distance to major road, effort) as
   model features, neutralised at prediction time by fixing them at their median.
d. :func:`bias_diagnostic` fits a static presence/background GLM for each variant on early
   presences, scores later new detections and reports how strongly each predicted map still
   tracks people, roads and recorder effort.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.config import load_config
from src.features.covariates_join import LANDCOVER_CLASSES
from src.features.covariates_join import run as cell_covariates
from src.preprocess.dataset_builder import panel_path
from src.preprocess.grid import cell_ids, load_or_build_grid

log = logging.getLogger(__name__)

EEA_RE = re.compile(r"^(\d+)(k?m)E(\d+)N(\d+)$")
ENV_FEATURES = [f"bio{i}" for i in range(1, 20)] + ["elev"] + list(LANDCOVER_CLASSES.values())
BIAS_FEATURES = ["log_pop_density", "road_density", "dist_road_km", "log_effort"]


def parse_eea_code(codes: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Parse EEA reference-grid codes like ``1kmE3500N2800`` into lower-left x, y (metres)."""
    m = codes.astype(str).str.extract(EEA_RE)
    if m.isna().any().any():
        bad = codes[m.isna().any(axis=1)].head().tolist()
        raise ValueError(f"unparseable EEA codes, e.g. {bad}")
    unit = np.where(m[1] == "km", 1000, 1)
    x = m[2].astype(np.int64).to_numpy() * unit
    y = m[3].astype(np.int64).to_numpy() * unit
    return x, y


def read_effort_cube(zip_path: Path) -> pd.DataFrame:
    """Read the GBIF SQL effort cube (TSV inside a zip)."""
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith((".csv", ".tsv", ".txt")))
        with zf.open(name) as fh:
            df = pd.read_csv(fh, sep="\t")
    df.columns = [c.lower() for c in df.columns]
    return df


def aggregate_effort(
    cube: pd.DataFrame, size: int, start_year: int, end_year: int, window: str
) -> pd.DataFrame:
    """Aggregate the 1 km effort cube to grid cells x windows.

    Records with unknown month (``half == 0``) count towards yearly windows only.
    ``effort_observers`` sums distinct recorders per 1 km cell (an upper bound on distinct
    recorders per grid cell).
    """
    cube = cube.dropna(subset=["eeacellcode", "year"])
    cube = cube[(cube["year"] >= start_year) & (cube["year"] <= end_year)]
    x, y = parse_eea_code(cube["eeacellcode"])
    col, row = np.floor_divide(x, size), np.floor_divide(y, size)
    cube = cube.assign(cell_id=cell_ids(col, row, size))
    if window == "year":
        cube = cube.assign(window=(cube["year"] - start_year).astype(int))
    else:
        cube = cube[cube["half"] > 0]
        cube = cube.assign(window=((cube["year"] - start_year) * 2 + cube["half"] - 1).astype(int))
    agg = (
        cube.groupby(["cell_id", "window"])
        .agg(
            effort_insecta=("n_insecta", "sum"),
            effort_hymenoptera=("n_hymenoptera", "sum"),
            effort_observers=("n_observers", "sum"),
        )
        .reset_index()
    )
    return agg


def inverse_effort_weights(effort: np.ndarray, floor_quantile: float = 0.05) -> np.ndarray:
    """Inverse-effort weights ``1 / max(effort, floor)`` normalised to mean 1.

    ``floor`` is the ``floor_quantile`` of strictly positive effort values, so cells with
    zero or tiny recorded effort do not get unbounded weights.
    """
    effort = np.asarray(effort, dtype=float)
    pos = effort[effort > 0]
    floor = np.quantile(pos, floor_quantile) if pos.size else 1.0
    w = 1.0 / np.maximum(effort, max(floor, 1.0))
    return w / w.mean()


def target_group_background(
    cells: pd.DataFrame, effort_col: str, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample ``n`` background cell indices with probability proportional to effort."""
    p = cells[effort_col].to_numpy(dtype=float)
    p = p / p.sum()
    return rng.choice(len(cells), size=n, replace=True, p=p)


def effort_path(cfg: dict[str, Any]) -> Path:
    """Path of the aggregated effort parquet."""
    return (
        Path(cfg["paths"]["processed"])
        / f"effort_{cfg['grid']['cell_size_m']}m_{cfg['time']['window']}.parquet"
    )


def build_effort(cfg: dict[str, Any]) -> pd.DataFrame:
    """Aggregate the effort cube to the grid for the configured window; cache as parquet."""
    path = effort_path(cfg)
    if path.exists():
        return pd.read_parquet(path)
    cube = read_effort_cube(Path(cfg["paths"]["raw"]) / "gbif" / "effort.zip")
    t = cfg["time"]
    eff = aggregate_effort(
        cube, int(cfg["grid"]["cell_size_m"]), t["start_year"], t["end_year"], t["window"]
    )
    grid = load_or_build_grid(cfg)
    eff = eff[eff["cell_id"].isin(grid["cell_id"])]
    eff.to_parquet(path, index=False)
    log.info("effort: %d non-zero cell-windows -> %s", len(eff), path)
    return eff


def _glm() -> Any:
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(2, interaction_only=False, include_bias=False),
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=3000),
    )


def _fit_predict(
    X: pd.DataFrame,
    presence_idx: np.ndarray,
    background_idx: np.ndarray,
    features: list[str],
    presence_w: np.ndarray | None,
    X_pred: pd.DataFrame,
) -> np.ndarray:
    """Fit a presence/background GLM (classes balanced by weight) and score ``X_pred``."""
    Xt = pd.concat([X.iloc[presence_idx][features], X.iloc[background_idx][features]])
    y = np.r_[np.ones(len(presence_idx)), np.zeros(len(background_idx))]
    wp = np.ones(len(presence_idx)) if presence_w is None else presence_w
    wb = np.ones(len(background_idx)) * wp.sum() / len(background_idx)
    model = _glm()
    model.fit(Xt.to_numpy(), y, logisticregression__sample_weight=np.r_[wp, wb])
    return model.predict_proba(X_pred[features].to_numpy())[:, 1]


def bias_diagnostic(cfg: dict[str, Any], split_year: int) -> dict[str, Any]:
    """Ablate bias corrections on a static presence/background model.

    Train presences: cells occupied in any window up to ``split_year``. Test positives: cells
    first detected after ``split_year``; test negatives: cells never detected. Metrics are
    reported on all test cells and on *surveyed* test cells only (>= 1 Insecta record after
    ``split_year``), since never-surveyed cells are uninformative absences.
    """
    rng = np.random.default_rng(cfg["bias"]["seed"])
    t = cfg["time"]
    grid = load_or_build_grid(cfg)
    cov = cell_covariates(cfg).set_index("cell_id").loc[grid["cell_id"]].reset_index()
    panel = pd.read_parquet(panel_path(cfg))
    eff = build_effort(cfg)
    split_w = (
        split_year - t["start_year"]
        if t["window"] == "year"
        else (split_year - t["start_year"]) * 2 + 1
    )

    eff_train = eff[eff["window"] <= split_w].groupby("cell_id")["effort_insecta"].sum()
    eff_test = eff[eff["window"] > split_w].groupby("cell_id")["effort_insecta"].sum()
    cov["effort_train"] = cov["cell_id"].map(eff_train).fillna(0).to_numpy()
    cov["effort_test"] = cov["cell_id"].map(eff_test).fillna(0).to_numpy()
    cov["log_pop_density"] = np.log1p(cov["pop_density"])
    cov["log_effort"] = np.log1p(cov["effort_train"])

    first = panel.groupby("cell_id")["first_detection_window"].first().reindex(cov["cell_id"])
    first = first.to_numpy(dtype=float)
    train_pos = np.flatnonzero(first <= split_w)
    test_mask = ~(first <= split_w)
    y_test = (first > split_w)[test_mask].astype(int)
    surveyed = (cov["effort_test"].to_numpy() > 0)[test_mask]
    not_pres = np.flatnonzero(~(first <= split_w))
    n_bg = int(cfg["bias"]["background_n"])

    uniform_bg = rng.choice(not_pres, size=n_bg, replace=True)
    tg_pool = cov.iloc[not_pres].reset_index(drop=True)
    tg_pool["effort_train_p"] = tg_pool["effort_train"] + 1e-9
    tg_bg = not_pres[target_group_background(tg_pool, "effort_train_p", n_bg, rng)]
    w_inv = inverse_effort_weights(
        cov["effort_train"].to_numpy()[train_pos], cfg["bias"]["effort_floor_quantile"]
    )
    neutral = cov.copy()
    for f in BIAS_FEATURES:
        neutral[f] = cov[f].median()

    env, full = ENV_FEATURES, ENV_FEATURES + BIAS_FEATURES
    variants = {
        "naive (uniform background)": _fit_predict(cov, train_pos, uniform_bg, env, None, cov),
        "target-group background": _fit_predict(cov, train_pos, tg_bg, env, None, cov),
        "inverse-effort weights": _fit_predict(cov, train_pos, uniform_bg, env, w_inv, cov),
        "bias covariates (raw prediction)": _fit_predict(
            cov, train_pos, uniform_bg, full, None, cov
        ),
        "bias covariates (neutralised)": _fit_predict(
            cov, train_pos, uniform_bg, full, None, neutral
        ),
        "target-group + inverse-effort weights": _fit_predict(
            cov, train_pos, tg_bg, env, w_inv, cov
        ),
    }

    rows = []
    for name, score in variants.items():
        s_test = score[test_mask]
        rows.append(
            {
                "variant": name,
                "roc_auc_all": roc_auc_score(y_test, s_test),
                "pr_auc_all": average_precision_score(y_test, s_test),
                "roc_auc_surveyed": roc_auc_score(y_test[surveyed], s_test[surveyed]),
                "pr_auc_surveyed": average_precision_score(y_test[surveyed], s_test[surveyed]),
                "spearman_pop_density": spearmanr(score, cov["pop_density"]).statistic,
                "spearman_road_density": spearmanr(score, cov["road_density"]).statistic,
                "spearman_effort": spearmanr(score, cov["effort_train"]).statistic,
            }
        )
    res = pd.DataFrame(rows)
    preds = pd.DataFrame({"cell_id": cov["cell_id"], **variants})
    return {
        "split_year": split_year,
        "n_train_presence_cells": int(len(train_pos)),
        "n_test_new_cells": int(y_test.sum()),
        "n_test_cells": int(len(y_test)),
        "n_test_surveyed_cells": int(surveyed.sum()),
        "n_test_new_cells_surveyed": int(y_test[surveyed].sum()),
        "base_rate_all": float(y_test.mean()),
        "base_rate_surveyed": float(y_test[surveyed].mean()),
        "metrics": res,
        "predictions": preds,
    }


def run(cfg: dict[str, Any], split_year: int) -> dict[str, Any]:
    """Build the effort layer, run the bias diagnostic and save outputs."""
    build_effort(cfg)
    out = bias_diagnostic(cfg, split_year)
    reports = Path(cfg["paths"]["reports"])
    out["predictions"].to_parquet(
        Path(cfg["paths"]["processed"]) / "bias_diagnostic_predictions.parquet", index=False
    )
    out["metrics"].to_csv(reports / "bias_diagnostic.csv", index=False)
    summary = {k: v for k, v in out.items() if k not in ("metrics", "predictions")}
    summary["metrics"] = out["metrics"].to_dict(orient="records")
    (reports / "bias_diagnostic.json").write_text(json.dumps(summary, indent=2))
    log.info("bias diagnostic:\n%s", out["metrics"].round(3).to_string(index=False))
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split-year", type=int, default=2019)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_config(args.config), args.split_year)


if __name__ == "__main__":
    main()
