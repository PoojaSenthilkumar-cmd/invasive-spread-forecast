"""Configuration and credential loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config (defaults to ``configs/default.yaml``) and resolve paths."""
    path = Path(path or os.environ.get("ISF_CONFIG", DEFAULT_CONFIG))
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {k: (ROOT / v) for k, v in cfg["paths"].items()}
    if cfg["cleaning"].get("max_coordinate_uncertainty_m") is None:
        cfg["cleaning"]["max_coordinate_uncertainty_m"] = cfg["grid"]["cell_size_m"]
    return cfg


@dataclass(frozen=True)
class GbifCredentials:
    """GBIF account credentials read from the environment."""

    user: str
    pwd: str
    email: str


def gbif_credentials() -> GbifCredentials:
    """Read ``GBIF_USER``, ``GBIF_PWD`` and ``GBIF_EMAIL`` from the environment (or ``.env``)."""
    load_dotenv(ROOT / ".env")
    missing = [k for k in ("GBIF_USER", "GBIF_PWD", "GBIF_EMAIL") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing GBIF credentials: {', '.join(missing)}. "
            "Set them as environment variables or in .env (see .env.example)."
        )
    env = os.environ
    return GbifCredentials(env["GBIF_USER"], env["GBIF_PWD"], env["GBIF_EMAIL"])
