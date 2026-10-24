"""Smoke test: the dashboard renders with C as the default layer and an Evaluation tab."""

from pathlib import Path

import pytest

RES = Path(__file__).resolve().parents[1] / "results"


@pytest.mark.skipif(
    not (RES / "dashboard_layers.parquet").exists(), reason="run make dashboard-data first"
)
def test_dashboard_renders_default_layer_c():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("../src/app/dashboard.py", default_timeout=120).run()
    assert not at.exception
    assert at.radio[0].value == "C"
    assert [t.label for t in at.tabs] == ["Forecast map", "Evaluation"]
    assert "SCORED ONCE AND FROZEN" in at.error[0].value
    assert any("did not beat adjacency on validation" in w.value for w in at.warning)
    assert any("tracks recording effort" in w.value for w in at.warning)
    for key in ["A", "B", "ADJ"]:
        at.radio[0].set_value(key).run()
        assert not at.exception
        assert f"Layer {'Adjacency baseline' if key == 'ADJ' else key}" in at.info[0].value
