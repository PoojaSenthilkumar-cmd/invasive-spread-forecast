"""Tests for chronological splits and bias-corrected training samples."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.spread import Variant, label_window, sample_training

MCFG = {
    "min_neg_per_window": 50,
    "neg_per_pos": 5,
    "effort_floor_quantile": 0.05,
    "max_weight": 10.0,
}


def _rows(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 4000
    eff = np.where(rng.random(n) < 0.4, 0.0, rng.lognormal(3, 2, n))
    return pd.DataFrame(
        {
            "window": rng.integers(0, 4, n),
            "year": 2010,
            "y_h1": (rng.random(n) < 0.03).astype(float),
            "meta_effort_target": eff,
            "x": rng.random(n),
        }
    )


def test_target_group_keeps_unrecorded_cells_eligible() -> None:
    rows = _rows()
    tr, _ = sample_training(rows, "y_h1", Variant("d", ["x"]), MCFG, np.random.default_rng(0))
    neg = tr[tr["y_h1"] == 0]
    assert (neg["meta_effort_target"] == 0).any()
    # but sampling still favours recorded cells (target-group background)
    assert neg["meta_effort_target"].median() > rows["meta_effort_target"].median()


def test_strict_variant_only_samples_recorded_negatives() -> None:
    rows = _rows()
    v = Variant("s", ["x"], negatives="target_group_strict")
    tr, _ = sample_training(rows, "y_h1", v, MCFG, np.random.default_rng(0))
    assert (tr.loc[tr["y_h1"] == 0, "meta_effort_target"] > 0).all()


def test_weights_capped_and_classes_balanced() -> None:
    rows = _rows()
    tr, w = sample_training(rows, "y_h1", Variant("d", ["x"]), MCFG, np.random.default_rng(0))
    pos = tr["y_h1"].to_numpy() == 1
    assert np.isclose(w[pos].sum(), (~pos).sum())
    assert w[pos].max() / w[pos].mean() <= MCFG["max_weight"] + 1e-9
    assert (w[~pos] == 1).all()


def test_label_window_years() -> None:
    df = pd.DataFrame({"year": [2018, 2020]})
    start, end = label_window(df, 3)
    assert start.tolist() == [2019, 2021] and end.tolist() == [2021, 2023]


def test_oof_and_test_is_strictly_chronological() -> None:
    """Fold y is fit with cutoff y-1; the final model with the last val year; test rows later."""
    from src.models.spread import oof_and_test

    rng = np.random.default_rng(0)
    years = np.repeat(np.arange(2015, 2025), 200)
    table = pd.DataFrame(
        {
            "year": years,
            "y_h1": (rng.random(len(years)) < 0.1).astype(float),
            "x": rng.random(len(years)),
        }
    )
    calls: list[int] = []

    def fit(cutoff: int):  # noqa: ANN202
        calls.append(cutoff)

        def score(r: pd.DataFrame) -> np.ndarray:
            assert (r["year"] + 1 > cutoff).all()  # never scores rows whose label it trained on
            return r["x"].to_numpy()

        return score

    res = oof_and_test(fit, table, 1, [2019, 2020, 2021], [2022, 2023, 2024, 2025], 2025)
    assert calls == [2018, 2019, 2020, 2021]
    test_years = table.loc[res["test"]["idx"], "year"] + 1
    assert test_years.min() == 2022 and test_years.max() == 2025
    oof_years = table.loc[res["oof"]["idx"], "year"] + 1
    assert set(oof_years) == {2019, 2020, 2021}
