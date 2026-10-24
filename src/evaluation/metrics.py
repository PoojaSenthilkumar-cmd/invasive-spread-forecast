"""Ranking, probabilistic and calibration metrics for the next-window spread task."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def rank_order(score: np.ndarray, seed: int = 0) -> np.ndarray:
    """Indices sorted by descending score; ties broken by a seeded random permutation.

    Random tie-breaking keeps coarse scores (e.g. binary adjacency) from benefiting from row order.
    """
    tie = np.random.default_rng(seed).random(len(score))
    return np.lexsort((tie, -np.asarray(score, dtype=float)))


def precision_recall_at_k(
    y: np.ndarray, score: np.ndarray, k: int, w: np.ndarray | None = None, seed: int = 0
) -> tuple[float, float]:
    """Precision@k and recall@k of the top-``k`` ranked rows (optionally row-weighted)."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    if w.sum() == 0:
        return np.nan, np.nan
    order = rank_order(score, seed)
    cw = np.cumsum(w[order])
    n_top = int(np.searchsorted(cw, k, side="left")) + 1
    top = order[:n_top]
    hits = (w[top] * y[top]).sum()
    total_pos = (w * y).sum()
    denom = min(float(k), float(cw[-1]))
    return float(hits / denom), float(hits / total_pos) if total_pos > 0 else np.nan


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-count (quantile) bins, suited to very low base rates."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)) / len(y))


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> dict[str, list[float]]:
    """Mean predicted vs observed frequency in equal-count bins."""
    order = np.argsort(p, kind="stable")
    bins = [b for b in np.array_split(order, n_bins) if len(b)]
    return {
        "mean_predicted": [float(p[b].mean()) for b in bins],
        "observed": [float(y[b].mean()) for b in bins],
        "n": [len(b) for b in bins],
    }


def _safe(fn: Callable[[], float]) -> float:
    try:
        return float(fn())
    except ValueError:
        return np.nan


def classification_metrics(y: np.ndarray, score: np.ndarray, prob: np.ndarray | None) -> dict:
    """PR-AUC (primary), ROC-AUC and, for calibrated probabilities, Brier, log-loss and ECE."""
    y = np.asarray(y, dtype=int)
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "base_rate": float(y.mean()) if len(y) else np.nan,
        "pr_auc": _safe(lambda: average_precision_score(y, score)),
        "roc_auc": _safe(lambda: roc_auc_score(y, score)),
    }
    if prob is not None:
        p = np.clip(np.asarray(prob, float), 1e-6, 1 - 1e-6)
        out["brier"] = float(brier_score_loss(y, p))
        out["log_loss"] = _safe(lambda: log_loss(y, p, labels=[0, 1]))
        out["ece"] = expected_calibration_error(y, p)
    return out


def block_bootstrap_ci(
    stat: Callable[[np.ndarray], float],
    blocks: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI of ``stat(row_weights)`` under a spatial block bootstrap.

    Blocks (e.g. 50 km squares) are resampled with replacement; each row gets the multiplicity of
    its block as weight, which respects spatial autocorrelation better than a row bootstrap.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(blocks, return_inverse=True)
    vals = []
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        vals.append(stat(mult[inv].astype(float)))
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        return np.nan, np.nan
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def weighted_ap(y: np.ndarray, score: np.ndarray, w: np.ndarray) -> float:
    """Average precision with row weights (NaN if a resample has no positives)."""
    m = w > 0
    if (y[m] * w[m]).sum() == 0:
        return np.nan
    return float(average_precision_score(y[m], score[m], sample_weight=w[m]))
