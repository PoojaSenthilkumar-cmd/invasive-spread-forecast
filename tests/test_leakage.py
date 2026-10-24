"""Leakage tests: features at window t must not depend on any data after t."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import box_sum, build_features, is_feature

SIZE, N, W = 5000, 15, 8  # 15 x 15 cells, 8 windows (0..7); window 7 plays the holdout
END_W = 6


def _toy(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    col, row = np.meshgrid(np.arange(100, 100 + N), np.arange(200, 200 + N))
    col, row = col.ravel(), row.ravel()
    grid = pd.DataFrame({"cell_id": [f"c{i}" for i in range(N * N)], "col": col, "row": row})
    cid = np.repeat(grid["cell_id"].to_numpy(), W)
    win = np.tile(np.arange(W), N * N)
    n_rec = rng.poisson(0.15, size=len(cid))
    panel = pd.DataFrame({"cell_id": cid, "window": win, "n_records": n_rec})
    effort = pd.DataFrame(
        {
            "cell_id": cid,
            "window": win,
            "effort_insecta": rng.poisson(20, size=len(cid)),
            "effort_hymenoptera": rng.poisson(3, size=len(cid)),
            "effort_observers": rng.poisson(2, size=len(cid)),
        }
    )
    return panel, effort, grid


def _build(panel: pd.DataFrame, effort: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    return build_features(
        panel, effort, grid, size=SIZE, start_year=2000, end_w=END_W, horizons=[1, 2], rings=[1, 3]
    )


def _scramble_after(df: pd.DataFrame, t0: int, cols: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    m = out["window"] > t0
    for c in cols:
        out.loc[m, c] = rng.permutation(out.loc[m, c].to_numpy()) + rng.integers(0, 3, m.sum())
    return out


@pytest.mark.parametrize("t0", [0, 2, 4, 5])
def test_features_at_t_ignore_all_future_data(t0: int) -> None:
    """Rewrite every panel and effort value after window t0; features for t <= t0 must not move."""
    panel, effort, grid = _toy()
    base = _build(panel, effort, grid)
    panel2 = _scramble_after(panel, t0, ["n_records"], seed=1)
    effort2 = _scramble_after(
        effort, t0, ["effort_insecta", "effort_hymenoptera", "effort_observers"], seed=2
    )
    pert = _build(panel2, effort2, grid)
    feats = [c for c in base.columns if is_feature(c)]
    a = base[base["window"] <= t0].set_index(["cell_id", "window"])
    b = pert[pert["window"] <= t0].set_index(["cell_id", "window"])
    assert a.index.equals(b.index)  # the set of at-risk rows at t <= t0 is also unchanged
    pd.testing.assert_frame_equal(a[feats], b[feats])
    # sanity: the perturbation really changed the future (labels/meta are allowed to move)
    assert not base[base["window"] > t0][feats].equals(pert[pert["window"] > t0][feats])


def test_holdout_window_never_reaches_table() -> None:
    """Changing only the holdout window (after end_w) leaves the whole table identical."""
    panel, effort, grid = _toy()
    base = _build(panel, effort, grid)
    pert = _build(
        _scramble_after(panel, END_W, ["n_records"], seed=3),
        _scramble_after(effort, END_W, ["effort_insecta"], seed=4),
        grid,
    )
    pd.testing.assert_frame_equal(base, pert)
    assert base["window"].max() == END_W


def test_labels_are_first_detection_in_next_windows() -> None:
    """y_h1 = first sighting in t+1; y_h2 = within t+1..t+2; rows are cells unoccupied by t."""
    panel, effort, grid = _toy()
    df = _build(panel, effort, grid)
    pos = panel[panel["n_records"] > 0]
    first = pos.groupby("cell_id")["window"].min()
    f = df["cell_id"].map(first).fillna(np.inf).to_numpy()
    t = df["window"].to_numpy()
    assert (f > t).all()  # never a row for an already-occupied cell
    h1 = df["y_h1"].to_numpy()
    ok1 = t + 1 <= END_W
    assert np.array_equal(h1[ok1], (f[ok1] == t[ok1] + 1).astype(np.float32))
    assert np.isnan(h1[~ok1]).all()
    ok2 = t + 2 <= END_W
    assert np.array_equal(df["y_h2"].to_numpy()[ok2], (f[ok2] <= t[ok2] + 2).astype(np.float32))


def test_no_feature_column_is_a_label_or_lookahead() -> None:
    panel, effort, grid = _toy()
    df = _build(panel, effort, grid)
    feats = [c for c in df.columns if is_feature(c)]
    assert not any(c.startswith(("y_", "meta_")) for c in feats)
    assert "meta_effort_target" in df.columns and not is_feature("meta_effort_target")


def test_box_sum_matches_brute_force() -> None:
    a = np.random.default_rng(0).integers(0, 3, (9, 11)).astype(float)
    out = box_sum(a, 2)
    for i in range(9):
        for j in range(11):
            assert out[i, j] == a[max(0, i - 2) : i + 3, max(0, j - 2) : j + 3].sum()
