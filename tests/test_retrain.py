"""The retrain script writes a per-run config whose outputs all live under the run directory."""

import yaml

from src.config import DEFAULT_CONFIG
from src.pipeline.retrain import STEPS, run_config


def test_run_config_redirects_every_output(tmp_path):
    cfg = yaml.safe_load(run_config(DEFAULT_CONFIG, tmp_path, trials=3).read_text())
    for key in ["raw", "interim", "processed", "docs", "reports", "results"]:
        assert cfg["paths"][key].startswith(str(tmp_path))
    assert cfg["model"]["n_trials"] == 3
    base = yaml.safe_load(DEFAULT_CONFIG.read_text())
    assert cfg["time"] == base["time"]  # study period / sealed holdout year unchanged


def test_retrain_never_runs_the_holdout():
    assert all("holdout" not in " ".join(s) for s in STEPS)
