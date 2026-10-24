"""Phase 4 runner: tune, fit and score every model on the chronological splits.

Outputs (``results/``): ``tuning.json``, ``predictions_test.parquet`` (long format, one row per
model x horizon x test row), ``predictions_oof.parquet`` (validation-year OOF scores, reused by
the holdout step to rebuild calibrators) and ``feature_importance.csv``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.models.spread import (
    BASELINES,
    BIAS_COVS,
    EFFORT_FEATS,
    ENV_FEATURES,
    Variant,
    dynamic_features,
    fixed_scorer,
    lgbm_fitter,
    load_table,
    oof_and_test,
    static_sdm_fitter,
    tune_lgbm,
)
from src.preprocess.bias import build_effort

log = logging.getLogger(__name__)

DEFAULT = "spread GBM (default)"
ROW_COLS = ["cell_id", "window", "year", "meta_effort_target", "log_effort_t", "log_effort_cum"]


def variants(table: pd.DataFrame) -> list[Variant]:
    """Default spread model and its ablations."""
    dyn = dynamic_features(table)
    full = dyn + EFFORT_FEATS + ENV_FEATURES + BIAS_COVS
    return [
        Variant(DEFAULT, full),
        Variant("ablation: no bias correction", full, "uniform", False, []),
        Variant(
            "ablation: TG negatives from recorded cells only (first run)",
            full,
            "target_group_strict",
        ),
        Variant(
            "ablation: effort features also neutralised",
            full,
            neutralise=BIAS_COVS + EFFORT_FEATS,
        ),
        Variant(
            "ablation: no dynamic features (static GBM)", EFFORT_FEATS + ENV_FEATURES + BIAS_COVS
        ),
        Variant("ablation: dynamic features only", dyn, neutralise=[]),
    ]


def results_dir(cfg: dict[str, Any]) -> Path:
    """``results/`` directory (created if missing)."""
    out = Path(cfg["paths"].get("results", "results"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def run(cfg: dict[str, Any]) -> None:
    """Tune the default model, then fit/score all models and baselines."""
    mcfg = cfg["model"]
    seed = int(mcfg["seed"])
    val_years, test_years = mcfg["val_target_years"], mcfg["test_target_years"]
    last_year = int(cfg["time"]["end_year"])
    table = load_table(cfg)
    table = table[table["year"] + 1 >= mcfg["first_target_year"]].reset_index(drop=True)
    out = results_dir(cfg)
    vs = variants(table)

    params, trials = tune_lgbm(table, vs[0], cfg)
    log.info("best params %s (CV PR-AUC %.4f)", params, trials["value"].max())
    (out / "tuning.json").write_text(
        json.dumps(
            {"best_params": params, "trials": trials.to_dict("records")}, indent=2, default=str
        )
    )

    effort = build_effort(cfg)
    fitters: dict[tuple[str, int], Any] = {}
    for v in vs:
        fitters[(v.name, 1)] = lgbm_fitter(table, 1, v, params, mcfg, seed)
    for k in cfg["features"]["horizons"][1:]:
        fitters[(DEFAULT, k)] = lgbm_fitter(table, k, vs[0], params, mcfg, seed)
    for kind, name in (
        ("lr", "static SDM (logistic, poly-2)"),
        ("rf", "static SDM (random forest)"),
    ):
        fitters[(name, 1)] = static_sdm_fitter(cfg, table, kind, effort)
    for name, fn in BASELINES.items():
        for k in cfg["features"]["horizons"]:
            if k == 1 or name in ("adjacency (ranked)", "distance to front"):
                fitters[(name, k)] = fixed_scorer(fn)

    test_parts, oof_parts, importance = [], [], None
    for (name, k), fit in fitters.items():
        res = oof_and_test(fit, table, k, val_years, test_years, last_year)
        test_parts.append(res["test"].assign(model=name, horizon=k))
        oof_parts.append(res["oof"].assign(model=name, horizon=k))
        if name == DEFAULT and k == 1:
            m = res["final"].model
            importance = pd.DataFrame(
                {"feature": m.feature_name_, "gain": m.booster_.feature_importance("gain")}
            ).sort_values("gain", ascending=False)
        log.info("%-48s h%d  test rows %d", name, k, len(res["test"]))

    test = pd.concat(test_parts, ignore_index=True)
    rows = table.loc[
        test["idx"].unique(), ROW_COLS + [f"y_h{k}" for k in cfg["features"]["horizons"]]
    ]
    test.to_parquet(out / "predictions_test.parquet", index=False)
    rows.to_parquet(out / "test_rows.parquet")
    pd.concat(oof_parts, ignore_index=True).to_parquet(out / "predictions_oof.parquet", index=False)
    importance.to_csv(out / "feature_importance.csv", index=False)
    log.info("wrote %d test predictions", len(test))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.seterr(all="ignore")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
