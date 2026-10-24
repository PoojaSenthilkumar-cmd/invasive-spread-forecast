"""Smoke test: the dashboard renders with C as the default layer, Backtest and Evaluation tabs."""

from pathlib import Path

import pytest

RES = Path(__file__).resolve().parents[1] / "results"


@pytest.mark.skipif(
    not (RES / "dashboard_backtest.parquet").exists(), reason="run make dashboard-data first"
)
def test_dashboard_renders_default_layer_c():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("../src/app/dashboard.py", default_timeout=180).run()
    assert not at.exception
    assert at.radio(key="layer").value == "C"
    assert [t.label for t in at.tabs] == ["Forecast map", "Backtest 2022-2025", "Evaluation"]
    assert "SCORED ONCE AND FROZEN" in at.error[0].value
    warn = " ".join(w.value for w in at.warning)
    assert "did not beat adjacency on validation" in warn
    assert "tracks recording effort" in warn
    assert "under-predict" in warn
    page = " ".join(m.value for m in at.markdown)
    assert "What drives the top-ranked cells?" in page
    for key in ["A", "B", "ADJ"]:
        at.radio(key="layer").set_value(key).run()
        assert not at.exception
        assert f"Layer {'Adjacency baseline' if key == 'ADJ' else key}" in at.info[0].value
    at.selectbox(key="bt_layer").set_value("A").run()
    assert not at.exception
