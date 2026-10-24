"""Save the frozen 2026-holdout models to disk, verify them, hash them, and export SHAP drivers.

The holdout run (``src.evaluation.holdout``) fitted its models in memory only. This script
re-creates exactly those fits (same data, frozen hyperparameters, seeds and OOF isotonic
calibrators), checks them against the outputs saved by the holdout run and the dashboard export,
and writes them to ``models/frozen/`` with a SHA-256 manifest. Nothing is tuned, changed or
re-scored on 2026 labels.

SHAP drivers: exact TreeSHAP contributions (LightGBM ``pred_contrib``) on the log-odds scale for
the top-ranked at-risk cells of layers C and A, for the forecast issued at the end of 2025.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.evaluation.holdout import SDMS, _calibrator
from src.models.spread import _neutralised, lgbm_fitter, load_table, static_sdm_fitter
from src.models.train import DEFAULT, results_dir, variants
from src.preprocess.bias import build_effort
from src.preprocess.dataset_builder import end_window

log = logging.getLogger(__name__)
FROZEN = ROOT / "models" / "frozen"
SHAP_LAYERS = {"C": "ablation: dynamic features only", "A": "ablation: no bias correction"}
SHAP_TOP = 200


def slug(name: str) -> str:
    """File-name-safe model name."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def sha256(path: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check(name: str, got: np.ndarray, want: np.ndarray, checks: dict[str, float]) -> None:
    diff = float(np.nanmax(np.abs(got - want)))
    if diff > 1e-9:
        raise SystemExit(f"{name}: refit differs from the saved holdout output (max diff {diff})")
    checks[name] = diff


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Refit, verify, save and hash the frozen models; write SHAP drivers for C and A."""
    out = results_dir(cfg)
    mcfg, seed, last = cfg["model"], int(cfg["model"]["seed"]), int(cfg["time"]["end_year"])
    params = json.loads((out / "tuning.json").read_text())["best_params"]
    oof = pd.read_parquet(out / "predictions_oof.parquet")
    table = load_table(cfg)
    table = table[table["year"] + 1 >= mcfg["first_target_year"]].reset_index(drop=True)
    vs = {v.name: v for v in variants(table)}
    rows = table[table["window"] == end_window(cfg)]
    layers = pd.read_parquet(out / "dashboard_layers.parquet")
    layers = layers[layers["status"] == "at risk"].reset_index(drop=True)
    saved_fc = pd.read_parquet(out / "forecast_from_2025.parquet")
    assert (layers["cell_id"].to_numpy() == rows["cell_id"].to_numpy()).all()
    FROZEN.mkdir(parents=True, exist_ok=True)

    files: dict[str, dict[str, Any]] = {}
    checks: dict[str, float] = {}
    calibrators: dict[str, Any] = {}
    scorers: dict[str, Any] = {}
    jobs = [(n, 1) for n in vs] + [(DEFAULT, k) for k in cfg["features"]["horizons"][1:]]
    for name, k in jobs:
        sc = lgbm_fitter(table, k, vs[name], params, mcfg, seed)(last)
        cal = _calibrator(oof, name, k)
        calibrators[f"{name}|h{k}"] = cal
        x = _neutralised(rows, vs[name], sc.medians)
        f = FROZEN / f"lgbm_h{k}__{slug(name)}.txt"
        sc.model.booster_.save_model(str(f))
        reloaded = lgb.Booster(model_file=str(f)).predict(x)
        _check(f"{name} h{k}: saved file reproduces in-memory model", reloaded, sc(rows), checks)
        prob = cal.predict(sc(rows))
        key = next(
            (
                c
                for c, n in [("A", SHAP_LAYERS["A"]), ("B", DEFAULT), ("C", SHAP_LAYERS["C"])]
                if n == name
            ),
            None,
        )
        if k == 1 and key:
            _check(f"{name} h1 vs dashboard layer {key}", prob, layers[f"prob_{key}"], checks)
        verified = bool(k == 1 and key)
        if name == DEFAULT and k > 1:
            want = saved_fc[f"p_within_{k}y"].to_numpy()
            # the saved forecast is made monotone across horizons; compare where it was not lifted
            ok = want > saved_fc[f"p_within_{k - 1}y"].to_numpy()
            diff = float(np.abs(prob[ok] - want[ok]).max())
            verified = diff <= 1e-9
            checks[f"{name} h{k} vs forecast_from_2025 ({'ok' if verified else 'MISMATCH'})"] = diff
        mf = FROZEN / f"lgbm_h{k}__{slug(name)}.medians.json"
        mf.write_text(json.dumps(sc.medians.to_dict(), indent=2))
        files[f.name] = {
            "model": name,
            "horizon": k,
            "kind": "LightGBM booster",
            "verified_against_saved_holdout_output": verified,
        }
        files[mf.name] = {"model": name, "horizon": k, "kind": "training medians (neutralisation)"}
        if k == 1:
            scorers[name] = sc
        log.info("frozen: %s h%d", name, k)

    effort = build_effort(cfg)
    for name, kind in SDMS.items():
        sc = static_sdm_fitter(cfg, table, kind, effort)(last)
        calibrators[f"{name}|h1"] = _calibrator(oof, name, 1)
        f = FROZEN / f"sdm__{slug(name)}.joblib"
        joblib.dump(sc.model, f, compress=3)
        files[f.name] = {
            "model": name,
            "horizon": 1,
            "kind": f"sklearn {kind} SDM",
            "verified_against_saved_holdout_output": False,
        }
        log.info("frozen: %s", name)
    cf = FROZEN / "isotonic_calibrators.joblib"
    joblib.dump(calibrators, cf, compress=3)
    files[cf.name] = {"model": "all", "kind": "isotonic calibrators (2019-2021 OOF)"}

    for fname, meta in files.items():
        p = FROZEN / fname
        meta.update(sha256=sha256(p), bytes=p.stat().st_size)
    manifest = {
        "note": (
            "Re-created after the single 2026 holdout run by refitting with the frozen data, "
            "hyperparameters and seeds. Files marked verified reproduce the per-cell outputs "
            "saved by that run (layers A, B, C at h1). The other h1 models have no saved per-cell "
            "output to compare against without re-scoring 2026, so they are unverified. "
            "Multithreaded LightGBM fits are not always bit-reproducible (two back-to-back h3 "
            "refits in one diagnostic process differed), so each file's verified flag is set "
            "from the comparison actually made when it was written."
        ),
        "files": files,
        "tuning.json": {"sha256": sha256(out / "tuning.json")},
        "verification_max_abs_diff": checks,
    }
    (FROZEN / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    shap_drivers(rows, vs, scorers, layers).to_parquet(out / "dashboard_shap.parquet", index=False)
    return manifest


def shap_drivers(rows: pd.DataFrame, vs: dict, scorers: dict, layers: pd.DataFrame) -> pd.DataFrame:
    """TreeSHAP contributions for the top ``SHAP_TOP`` cells of layers C and A (long format)."""
    parts = []
    for key, name in SHAP_LAYERS.items():
        top = np.flatnonzero(layers[f"rank_{key}"].to_numpy() <= SHAP_TOP)
        r = rows.iloc[top]
        x = _neutralised(r, vs[name], scorers[name].medians)
        contrib = scorers[name].model.booster_.predict(x, pred_contrib=True)
        raw = scorers[name].model.booster_.predict(x, raw_score=True)
        assert np.allclose(contrib.sum(axis=1), raw, atol=1e-6)
        feats = list(x.columns)
        df = pd.DataFrame(contrib[:, :-1], columns=feats)
        df["cell_id"] = r["cell_id"].to_numpy()
        df["rank"] = layers[f"rank_{key}"].to_numpy()[top]
        df["base_value"] = contrib[:, -1]
        long = df.melt(["cell_id", "rank", "base_value"], var_name="feature", value_name="shap")
        vals = x.assign(cell_id=r["cell_id"].to_numpy()).melt(
            "cell_id", var_name="feature", value_name="value"
        )
        parts.append(long.merge(vals, on=["cell_id", "feature"]).assign(layer=key))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    m = run(load_config())
    for f, meta in m["files"].items():
        print(f"{meta['sha256'][:16]}  {meta['bytes']:>10,}  {f}")
    print("tuning.json", m["tuning.json"]["sha256"][:16])
    for c, d in m["verification_max_abs_diff"].items():
        print(f"ok {d:.1e}  {c}")


if __name__ == "__main__":
    main()
