"""Static EDA / diagnostic plots for Phases 1-2."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import load_config

log = logging.getLogger(__name__)


def plot_cleaning_log(log_df: pd.DataFrame, out: Path) -> Path:
    """Horizontal bar chart of records removed per cleaning step."""
    d = log_df.iloc[1:]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(d) + 1))
    sns.barplot(data=d, y="step", x="removed", ax=ax, color="#5b7a99")
    ax.set_xscale("symlog")
    ax.set_xlabel("records removed (symlog)")
    ax.set_ylabel("")
    ax.set_title(
        f"Cleaning: {log_df['remaining'].iloc[0]:,} raw -> "
        f"{log_df['remaining'].iloc[-1]:,} clean records"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_panel_summary(summary: pd.DataFrame, out: Path) -> Path:
    """Records and occupied / newly occupied / cumulative cells per window."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = summary["window_label"].astype(str)
    axes[0].bar(x, summary["records"], color="#5b7a99")
    axes[0].set_yscale("log")
    axes[0].set_title("Clean records per window")
    axes[1].plot(x, summary["cumulative_cells"], marker="o", label="cumulative occupied cells")
    axes[1].plot(x, summary["occupied_cells"], marker="o", label="occupied in window")
    axes[1].bar(x, summary["new_cells"], color="#c9a66b", alpha=0.7, label="newly occupied")
    axes[1].set_title("Occupied 5 km cells")
    axes[1].legend()
    for ax in axes:
        ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run(cfg: dict[str, Any]) -> list[Path]:
    """Write all Phase 1-2 summary plots to ``reports/figures``."""
    reports = Path(cfg["paths"]["reports"])
    figs = reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    outs = [plot_cleaning_log(pd.read_csv(reports / "cleaning_log.csv"), figs / "cleaning_log.png")]
    for w in ("year", "half"):
        p = reports / f"panel_summary_{w}.csv"
        if p.exists():
            outs.append(plot_panel_summary(pd.read_csv(p), figs / f"panel_summary_{w}.png"))
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
