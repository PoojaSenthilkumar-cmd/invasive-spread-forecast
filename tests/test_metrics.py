"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    block_bootstrap_ci,
    expected_calibration_error,
    precision_recall_at_k,
    weighted_ap,
)


def test_precision_recall_at_k() -> None:
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.05])
    p, r = precision_recall_at_k(y, s, 3)
    assert p == 2 / 3 and r == 2 / 3
    assert precision_recall_at_k(y, s, 1) == (1.0, 1 / 3)


def test_precision_at_k_random_tie_break_is_fair() -> None:
    y = np.r_[np.ones(500), np.zeros(500)]
    s = np.zeros(1000)  # all tied: expected precision = base rate, not 1.0 from row order
    p, _ = precision_recall_at_k(y, s, 200)
    assert 0.4 < p < 0.6


def test_weighted_precision_at_k_matches_duplication() -> None:
    rng = np.random.default_rng(1)
    y, s = rng.integers(0, 2, 50), rng.random(50)
    w = rng.integers(0, 3, 50).astype(float)
    rep = np.repeat(np.arange(50), w.astype(int))
    p_w, r_w = precision_recall_at_k(y, s, 20, w)
    p_d, r_d = precision_recall_at_k(y[rep], s[rep], 20)
    assert np.isclose(p_w, p_d) and np.isclose(r_w, r_d)


def test_ece_zero_for_perfect_calibration() -> None:
    p = np.repeat([0.1, 0.5], 1000)
    y = np.r_[np.r_[np.ones(100), np.zeros(900)], np.r_[np.ones(500), np.zeros(500)]]
    assert expected_calibration_error(y, p, n_bins=2) < 1e-12


def test_block_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    s = y + rng.normal(0, 1, 2000)
    blocks = np.arange(2000) // 20
    lo, hi = block_bootstrap_ci(lambda w: weighted_ap(y, s, w), blocks, 100, seed=0)
    assert lo < weighted_ap(y, s, np.ones(2000)) < hi
