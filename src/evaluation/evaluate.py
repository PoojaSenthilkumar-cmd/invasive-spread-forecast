"""Phase 5: chronological test-set evaluation of all models and baselines.

Reads ``results/predictions_test.parquet`` (written by ``src.models.train``) and writes
``results/metrics.json`` plus CSV tables and figures. The 2026 holdout is *not* touched here; see
``src.evaluation.holdout``.

Effort strata: within each forecast year, at-risk cells are ranked by Insecta effort recorded in
the label window (ties broken by cumulative past effort, then randomly) and cut into deciles
(1 = least recorded). "Low-effort" = the deciles in ``evaluation.low_effort_deciles``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.config import load_config  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    block_bootstrap_ci,
    classification_metrics,
    precision_recall_at_k,
    reliability_curve,
    weighted_ap,
)
from src.features.covariates_join import covariates_path  # noqa: E402
from src.models.train import DEFAULT, results_dir  # noqa: E402
from src.preprocess.grid import load_or_build_grid  # noqa: E402

log = logging.getLogger(__name__)
ADJ = "adjacency (ranked)"
PROBABILISTIC_BASELINES = {"adjacency (binary)", ADJ, "distance to front", "random"}


def effort_deciles(rows: pd.DataFrame, seed: int) -> np.ndarray:
    """Per-year effort decile (1..10) of each row, from label-window effort."""
    tie = np.random.default_rng(seed).random(len(rows))
    key = pd.DataFrame(
        {
            "year": rows["year"].to_numpy(),
            "e": rows["meta_effort_target"].to_numpy(),
            "c": rows["log_effort_cum"].to_numpy(),
            "tie": tie,
        }
    )
    key["rank"] = key.sort_values(["e", "c", "tie"]).groupby("year").cumcount()
    n = key.groupby("year")["rank"].transform("size")
    return (np.floor(key["rank"] * 10 / n) + 1).astype(int).to_numpy()


def per_year_at_k(
    df: pd.DataFrame, y: str, ks: list[int], w: np.ndarray | None = None
) -> dict[str, float]:
    """Mean over forecast years of precision@k and recall@k (one ranked list per year)."""
    out: dict[str, list[float]] = {}
    w = np.ones(len(df)) if w is None else w
    for _, idx in df.groupby("year").indices.items():
        yy, ss, ww = df[y].to_numpy()[idx], df["score"].to_numpy()[idx], w[idx]
        for k in ks:
            p, r = precision_recall_at_k(yy, ss, k, ww)
            out.setdefault(f"precision@{k}", []).append(p)
            out.setdefault(f"recall@{k}", []).append(r)
    return {key: float(np.nanmean(v)) for key, v in out.items()}


def lead_times(df: pd.DataFrame, y: str, k: int) -> dict[str, float]:
    """Lead time of top-``k`` flags before first sighting, over newly sighted test cells.

    For each cell first sighted in a test year, lead = first-sighting year minus the earliest
    test-period forecast-issue year in which the cell was in that year's top-``k``.
    """
    flagged = []
    for _, g in df.groupby("year"):
        top = g.nlargest(k, "score")
        flagged.append(top[["cell_id", "year"]])
    flagged = pd.concat(flagged).groupby("cell_id")["year"].min()
    pos = df[df[y] == 1][["cell_id", "year"]]
    first_issue = pos.groupby("cell_id")["year"].max() + 1  # sighted in year after last at-risk
    earliest = flagged.reindex(first_issue.index)
    lead = first_issue - earliest
    lead = lead[lead >= 1]
    return {
        "n_new_cells": int(len(first_issue)),
        "frac_flagged_before_sighting": float(len(lead) / len(first_issue)),
        "median_lead_years": float(lead.median()) if len(lead) else np.nan,
        "mean_lead_years": float(lead.mean()) if len(lead) else np.nan,
    }


def prepare_rows(rows: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add effort decile, spatial bootstrap block and bias covariates to evaluation rows."""
    ecfg, seed = cfg["evaluation"], int(cfg["model"]["seed"])
    rows = rows.copy()
    grid = load_or_build_grid(cfg).set_index("cell_id")
    cov = pd.read_parquet(covariates_path(cfg)).set_index("cell_id")
    rows["decile"] = effort_deciles(rows, seed)
    b = int(ecfg["block_cells"])
    g = grid.loc[rows["cell_id"]]
    rows["block"] = (g["col"].to_numpy() // b) * 100000 + g["row"].to_numpy() // b
    for c in ("pop_density", "road_density"):
        rows[c] = cov[c].reindex(rows["cell_id"]).to_numpy()
    return rows


def compute_metrics(
    preds: pd.DataFrame, rows: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """All metrics for long-format ``preds`` (model, horizon, idx, score, prob) on ``rows``."""
    ecfg, seed = cfg["evaluation"], int(cfg["model"]["seed"])
    low = ecfg["low_effort_deciles"]

    metrics: dict[str, Any] = {"models": {}}
    decile_rows, curves = [], {}
    adj_frames = {}
    for (name, k), p in preds.groupby(["model", "horizon"], sort=False):
        y = f"y_h{k}"
        d = rows.loc[p["idx"]].reset_index(drop=True)
        d["score"], d["prob"] = p["score"].to_numpy(), p["prob"].to_numpy()
        if name == ADJ:
            adj_frames[k] = d
        yv, sv = d[y].to_numpy(), d["score"].to_numpy()
        m = classification_metrics(yv, sv, d["prob"].to_numpy())
        m["pr_auc_ci95"] = block_bootstrap_ci(
            lambda w, yv=yv, sv=sv: weighted_ap(yv, sv, w),
            d["block"].to_numpy(),
            int(ecfg["n_bootstrap"]),
            seed,
        )
        m["per_year"] = {
            int(yr) + 1: classification_metrics(gg[y].to_numpy(), gg["score"].to_numpy(), None)
            for yr, gg in d.groupby("year")
        }
        m.update(per_year_at_k(d, y, ecfg["k"]))
        lo = d[d["decile"].isin(low)].reset_index(drop=True)
        m["low_effort"] = {
            "deciles": low,
            **classification_metrics(lo[y].to_numpy(), lo["score"].to_numpy(), None),
            **per_year_at_k(lo, y, ecfg["k_low_effort"]),
        }
        m["lead_time"] = lead_times(d, y, int(ecfg["lead_time_k"]))
        m["spearman_with"] = {
            c: float(spearmanr(d["prob"], d[c], nan_policy="omit").statistic)
            for c in ("pop_density", "road_density", "meta_effort_target")
        }
        resid = yv - d["prob"].to_numpy()
        m["residual_spearman_with"] = {
            c: float(spearmanr(resid, d[c], nan_policy="omit").statistic)
            for c in ("pop_density", "road_density", "meta_effort_target")
        }
        for dec, gg in d.groupby("decile"):
            r = classification_metrics(gg[y].to_numpy(), gg["score"].to_numpy(), None)
            r.update(per_year_at_k(gg, y, [50]))
            decile_rows.append({"model": name, "horizon": int(k), "decile": int(dec), **r})
        curves[f"{name} | h{k}"] = reliability_curve(yv, d["prob"].to_numpy())
        metrics["models"].setdefault(name, {})[f"h{k}"] = m
        log.info("%-48s h%d PR-AUC %.4f  P@500 %.3f", name, k, m["pr_auc"], m["precision@500"])

    # paired block-bootstrap difference vs ranked adjacency (same rows, same blocks)
    for name, per_h in metrics["models"].items():
        for hk, m in per_h.items():
            k = int(hk[1:])
            if name == ADJ or k not in adj_frames:
                continue
            p = preds[(preds["model"] == name) & (preds["horizon"] == k)]
            a = adj_frames[k]
            s_m = p["score"].to_numpy()
            yv, blocks = a[f"y_h{k}"].to_numpy(), a["block"].to_numpy()
            if not np.array_equal(
                p["idx"].to_numpy(),
                preds[(preds["model"] == ADJ) & (preds["horizon"] == k)]["idx"].to_numpy(),
            ):
                continue
            s_a = a["score"].to_numpy()
            m["pr_auc_minus_adjacency"] = m["pr_auc"] - per_h_adj(metrics, k)
            m["pr_auc_minus_adjacency_ci95"] = block_bootstrap_ci(
                lambda w, yv=yv, s_m=s_m, s_a=s_a: (
                    weighted_ap(yv, s_m, w) - weighted_ap(yv, s_a, w)
                ),
                blocks,
                int(ecfg["n_bootstrap"]),
                seed,
            )
            lo = a["decile"].isin(low).to_numpy()
            yl, bl = yv[lo], blocks[lo]
            al = a[lo].reset_index(drop=True)
            kk = int(ecfg["k_low_effort"][1])

            def diff_low(w: np.ndarray, s_m=s_m[lo], s_a=s_a[lo], al=al, kk=kk, k=k) -> float:
                dm = per_year_at_k(al.assign(score=s_m), f"y_h{k}", [kk], w)
                da = per_year_at_k(al.assign(score=s_a), f"y_h{k}", [kk], w)
                return dm[f"precision@{kk}"] - da[f"precision@{kk}"]

            m["low_effort"][f"precision@{kk}_minus_adjacency_ci95"] = block_bootstrap_ci(
                diff_low, bl, int(ecfg["n_bootstrap"]), seed
            )
            del yl

    metrics["reliability"] = curves
    return metrics, pd.DataFrame(decile_rows)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute every Phase 5 test metric and write metrics.json, tables and figures."""
    out = results_dir(cfg)
    figs = Path(cfg["paths"]["reports"]) / "figures"
    preds = pd.read_parquet(out / "predictions_test.parquet")
    rows = prepare_rows(pd.read_parquet(out / "test_rows.parquet"), cfg)
    metrics, deciles = compute_metrics(preds, rows, cfg)
    metrics["split"] = {
        "val_target_years": cfg["model"]["val_target_years"],
        "test_target_years": cfg["model"]["test_target_years"],
    }
    deciles.to_csv(out / "metrics_by_effort_decile.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json))
    _summary_table(metrics).to_csv(out / "summary_h1.csv", index=False)
    _figures(metrics, deciles, figs)
    return metrics


def per_h_adj(metrics: dict[str, Any], k: int) -> float:
    """PR-AUC of the ranked-adjacency baseline at horizon ``k``."""
    return metrics["models"][ADJ][f"h{k}"]["pr_auc"]


def _json(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def _summary_table(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, per_h in metrics["models"].items():
        m = per_h.get("h1")
        if m is None:
            continue
        lo = m["low_effort"]
        rows.append(
            {
                "model": name,
                "pr_auc": m["pr_auc"],
                "pr_auc_ci_lo": m["pr_auc_ci95"][0],
                "pr_auc_ci_hi": m["pr_auc_ci95"][1],
                "roc_auc": m["roc_auc"],
                "brier": m["brier"],
                "log_loss": m["log_loss"],
                "ece": m["ece"],
                **{c: m[c] for c in m if c.startswith(("precision@", "recall@"))},
                "low_pr_auc": lo["pr_auc"],
                **{f"low_{c}": lo[c] for c in lo if c.startswith("precision@")},
                "median_lead_years": m["lead_time"]["median_lead_years"],
                "frac_flagged": m["lead_time"]["frac_flagged_before_sighting"],
                "rho_effort": m["spearman_with"]["meta_effort_target"],
                "rho_pop": m["spearman_with"]["pop_density"],
                "rho_road": m["spearman_with"]["road_density"],
            }
        )
    return pd.DataFrame(rows)


def _figures(metrics: dict[str, Any], deciles: pd.DataFrame, figs: Path) -> None:
    figs.mkdir(parents=True, exist_ok=True)
    show = [DEFAULT, "ablation: no bias correction", "static SDM (random forest)", ADJ]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name in show:
        c = metrics["reliability"].get(f"{name} | h1")
        if c:
            ax.plot(c["mean_predicted"], c["observed"], "o-", ms=3, label=name)
    lim = max(max(v["observed"] + v["mean_predicted"]) for v in metrics["reliability"].values())
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set(
        xlabel="mean predicted probability",
        ylabel="observed first-sighting rate",
        title="Reliability (test 2022-2025, h=1)",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "reliability_h1.png", dpi=150)
    plt.close(fig)

    h1 = deciles[deciles["horizon"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in [DEFAULT, ADJ, "distance to front", "static SDM (random forest)"]:
        g = h1[h1["model"] == name]
        axes[0].plot(g["decile"], g["pr_auc"], "o-", label=name)
        axes[1].plot(g["decile"], g["precision@50"], "o-", label=name)
    base = h1[h1["model"] == DEFAULT]
    axes[0].plot(base["decile"], base["base_rate"], "k:", label="base rate")
    axes[0].set(xlabel="effort decile (label window, 1 = lowest)", ylabel="PR-AUC", yscale="log")
    axes[1].set(xlabel="effort decile (label window, 1 = lowest)", ylabel="precision@50 per year")
    axes[0].legend(fontsize=7)
    fig.suptitle("Test 2022-2025 by recording-effort decile (h=1)")
    fig.tight_layout()
    fig.savefig(figs / "effort_decile_metrics.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
