"""Retrain end to end into a separate run directory: pull GBIF, rebuild features, refit, export.

Steps (each a module run with ``ISF_CONFIG`` pointing at a per-run config):
GBIF download -> cleaning -> grid -> panels -> covariate join + bias layers -> features ->
tuning/training -> Phase 5 evaluation -> dashboard layers + backtest.

Everything is written under ``--run-dir`` (default ``runs/<UTC timestamp>``); the frozen
``results/`` and ``models/frozen/`` are never touched, and the holdout is never re-scored. Static
covariates are linked from ``data/raw/covariates`` and ``data/interim/covariates`` (download and
clip once with ``make covariates``).
The study period stays as in the config; roll ``time.end_year``/``holdout_year`` forward in a
copy of the config once a year is complete.

    uv run python -m src.pipeline.retrain                       # fresh GBIF download
    uv run python -m src.pipeline.retrain --skip-download --trials 2   # quick local check
    ISF_RESULTS=runs/<id>/results uv run streamlit run src/app/dashboard.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

from src.config import DEFAULT_CONFIG, ROOT

log = logging.getLogger(__name__)
STEPS: list[list[str]] = [
    ["src.preprocess.clean"],
    ["src.preprocess.grid"],
    ["src.preprocess.dataset_builder", "--window", "year"],
    ["src.preprocess.dataset_builder", "--window", "half"],
    ["src.features.covariates_join"],
    ["src.preprocess.bias", "--split-year", "2019"],
    ["src.features.build_features"],
    ["src.models.train"],
    ["src.evaluation.evaluate"],
    ["src.app.export_layers"],
    ["src.app.export_backtest"],
]


def run_config(base: Path, run_dir: Path, trials: int | None) -> Path:
    """Write the per-run config: same settings, all outputs under ``run_dir``."""
    cfg = yaml.safe_load(base.read_text())
    for key in ["raw", "interim", "processed", "docs", "reports", "results"]:
        cfg["paths"][key] = str(run_dir / key)
    if trials is not None:
        cfg["model"]["n_trials"] = trials
    path = run_dir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def prepare(run_dir: Path, skip_download: bool) -> None:
    """Create the run tree; link static covariates (and, with --skip-download, GBIF files)."""
    raw = run_dir / "raw"
    (raw / "gbif").mkdir(parents=True, exist_ok=True)
    for d in ["interim", "processed", "docs", "reports/figures", "results"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)
    cov = ROOT / "data" / "raw" / "covariates"
    if not cov.exists():
        raise SystemExit("data/raw/covariates missing: run `make covariates` first")
    if not (raw / "covariates").exists():
        (raw / "covariates").symlink_to(cov.resolve(), target_is_directory=True)
    clipped = ROOT / "data" / "interim" / "covariates"  # clipped static layers (make covariates)
    if not clipped.exists():
        raise SystemExit("data/interim/covariates missing: run `make covariates` first")
    if not (run_dir / "interim" / "covariates").exists():
        (run_dir / "interim" / "covariates").symlink_to(clipped.resolve(), target_is_directory=True)
    if skip_download:
        for f in (ROOT / "data" / "raw" / "gbif").glob("*"):
            dest = raw / "gbif" / f.name
            if not dest.exists():
                dest.symlink_to(f.resolve())


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--skip-download", action="store_true", help="reuse data/raw/gbif downloads")
    ap.add_argument("--trials", type=int, default=None, help="override model.n_trials")
    a = ap.parse_args()
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (a.run_dir or ROOT / "runs" / stamp).resolve()
    if run_dir in {(ROOT / "results").resolve(), ROOT.resolve(), (ROOT / "models").resolve()}:
        raise SystemExit("refusing to write a retrain run over the frozen outputs")
    prepare(run_dir, a.skip_download)
    cfg = run_config(a.config, run_dir, a.trials)
    env = {**os.environ, "ISF_CONFIG": str(cfg)}
    steps = ([] if a.skip_download else [["src.ingest.gbif"]]) + STEPS
    for step in steps:
        log.info("retrain step: %s", " ".join(step))
        subprocess.run([sys.executable, "-m", *step], env=env, cwd=ROOT, check=True)
    log.info("retrain finished: %s", run_dir)
    print(run_dir)


if __name__ == "__main__":
    main()
