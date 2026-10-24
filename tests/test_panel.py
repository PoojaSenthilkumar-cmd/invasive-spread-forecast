import numpy as np
import pandas as pd
import pytest

from src.preprocess.dataset_builder import build_panel, window_index, window_labels


def _occ(rows):
    return pd.DataFrame(rows, columns=["x", "y", "year", "month"])


def test_window_index_year_and_half():
    y = pd.Series([2004, 2005, 2005])
    m = pd.Series([3, 6, 7])
    assert window_index(y, m, 2004, "year").tolist() == [0, 1, 1]
    assert window_index(y, m, 2004, "half").tolist() == [0, 2, 3]
    assert window_labels(3, 2004, "half") == ["2004H1", "2004H2", "2005H1"]
    with pytest.raises(ValueError):
        window_index(y, m, 2004, "month")


def test_panel_presence_cumulative_and_first_detection():
    cells = pd.Index(["5kmE0N0", "5kmE5N0"])
    occ = _occ([(100, 100, 2005, 5), (200, 200, 2005, 6), (6000, 10, 2007, 1), (100, 100, 2006, 1)])
    p = build_panel(occ, cells, n_win=4, size=5000, start_year=2004, window="year")
    assert len(p) == 8
    a = p[p.cell_id == "5kmE0N0"].set_index("window")
    b = p[p.cell_id == "5kmE5N0"].set_index("window")
    assert a["n_records"].tolist() == [0, 2, 1, 0]
    assert a["present"].tolist() == [0, 1, 1, 0]
    assert a["cumulative_present"].tolist() == [0, 1, 1, 1]
    assert a["new_detection"].tolist() == [0, 1, 0, 0]
    assert int(a["first_detection_window"].iloc[0]) == 1
    assert b["cumulative_present"].tolist() == [0, 0, 0, 1]
    assert int(b["first_detection_window"].iloc[0]) == 3


def test_panel_cumulative_is_monotone_and_never_precedes_first_detection():
    rng = np.random.default_rng(0)
    occ = _occ(
        [
            (rng.uniform(0, 20000), rng.uniform(0, 5000), int(rng.integers(2004, 2010)), 6)
            for _ in range(200)
        ]
    )
    cells = pd.Index([f"5kmE{5 * c}N0" for c in range(4)])
    p = build_panel(occ, cells, n_win=6, size=5000, start_year=2004, window="year")
    for _, g in p.groupby("cell_id"):
        assert (np.diff(g["cumulative_present"].to_numpy()) >= 0).all()
        fdw = g["first_detection_window"].iloc[0]
        assert (g.loc[g["window"] < fdw, "cumulative_present"] == 0).all()


def test_records_outside_grid_are_ignored():
    occ = _occ([(100, 100, 2004, 1), (99_000, 99_000, 2004, 1)])
    p = build_panel(occ, pd.Index(["5kmE0N0"]), n_win=1, size=5000, start_year=2004, window="year")
    assert p["n_records"].tolist() == [1]
