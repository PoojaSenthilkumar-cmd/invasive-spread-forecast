import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from src.preprocess.bias import aggregate_effort, inverse_effort_weights, parse_eea_code
from src.preprocess.clean import clean, has_any_issue, rounded_to


def _cfg():
    return {
        "species": {"gbif_taxon_key": 1},
        "grid": {"crs": "EPSG:4326", "cell_size_m": 5000},
        "time": {"window": "year", "start_year": 2004, "end_year": 2025},
        "cleaning": {
            "max_coordinate_uncertainty_m": 5000,
            "country_centroid_buffer_m": 0.01,
            "drop_basis_of_record": ["FOSSIL_SPECIMEN", "PRESERVED_SPECIMEN"],
            "drop_issues": ["ZERO_COORDINATE", "COUNTRY_COORDINATE_MISMATCH"],
            "drop_occurrence_status": ["ABSENT"],
        },
    }


def test_issue_matching_is_exact_token():
    s = pd.Series(["A;ZERO_COORDINATE", "ZERO_COORDINATE_X", None, "COUNTRY_COORDINATE_MISMATCH"])
    assert has_any_issue(s, ["ZERO_COORDINATE", "COUNTRY_COORDINATE_MISMATCH"]).tolist() == [
        True,
        False,
        False,
        True,
    ]


def test_rounded_to():
    assert rounded_to(pd.Series([1.0, 1.1, 1.12, -3.0]), 1).tolist() == [True, True, False, True]


def test_clean_drops_each_bad_record_once_and_logs_steps():
    base = dict(
        gbifID=0,
        datasetKey="d",
        occurrenceID="o",
        speciesKey=1,
        taxonKey=1,
        countryCode="AA",
        occurrenceStatus="PRESENT",
        decimalLatitude=0.5123,
        decimalLongitude=0.5321,
        coordinateUncertaintyInMeters=10.0,
        eventDate="2010-06-01",
        day=1,
        month=6,
        year=2010,
        basisOfRecord="HUMAN_OBSERVATION",
        issue="",
        recordedBy="x",
    )
    bad = [
        dict(decimalLatitude=np.nan),
        dict(decimalLatitude=95.0),
        dict(decimalLatitude=0.0),
        dict(issue="COUNTRY_COORDINATE_MISMATCH"),
        dict(occurrenceStatus="ABSENT"),
        dict(basisOfRecord="PRESERVED_SPECIMEN"),
        dict(speciesKey=2),
        dict(year=np.nan),
        dict(year=1999),
        dict(coordinateUncertaintyInMeters=10_000.0),
        dict(decimalLatitude=0.3, decimalLongitude=0.7),
        dict(decimalLatitude=1.50003, decimalLongitude=1.50002),  # near centroid of BB
        dict(decimalLatitude=5.123, decimalLongitude=5.321),  # outside region
    ]
    rows = [base | {"gbifID": 1}, base | {"gbifID": 2}]  # second is a duplicate
    rows += [base | b | {"gbifID": 10 + i} for i, b in enumerate(bad)]
    rows += [base | {"gbifID": 99, "decimalLatitude": 0.6111}]
    raw = pd.DataFrame(rows)
    region = gpd.GeoDataFrame(
        {"iso2": ["AA", "BB"]}, geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)], crs=4326
    )
    out, log = clean(raw, _cfg(), region, region)
    assert sorted(out["gbif_id"]) == [1, 99]
    assert log["removed"].sum() == len(raw) - len(out)
    assert (log["removed"].iloc[1:] >= 1).all()  # every step removed its planted record


def test_parse_eea_code_and_aggregate_effort():
    x, y = parse_eea_code(pd.Series(["1kmE3500N2800", "1kmE3504N2809"]))
    assert x.tolist() == [3_500_000, 3_504_000] and y.tolist() == [2_800_000, 2_809_000]
    cube = pd.DataFrame(
        {
            "year": [2005, 2005, 2005, 2006],
            "half": [1, 2, 0, 2],
            "eeacellcode": ["1kmE3500N2800", "1kmE3504N2804", "1kmE3505N2800", "1kmE3500N2800"],
            "n_insecta": [3, 4, 5, 6],
            "n_hymenoptera": [1, 1, 1, 1],
            "n_observers": [1, 2, 3, 4],
        }
    )
    yr = aggregate_effort(cube, 5000, 2004, 2025, "year").set_index(["cell_id", "window"])
    assert yr.loc[("5kmE3500N2800", 1), "effort_insecta"] == 7
    assert yr.loc[("5kmE3505N2800", 1), "effort_insecta"] == 5
    assert yr.loc[("5kmE3500N2800", 2), "effort_insecta"] == 6
    hf = aggregate_effort(cube, 5000, 2004, 2025, "half").set_index(["cell_id", "window"])
    assert ("5kmE3505N2800", 2) not in hf.index  # unknown month excluded from half windows
    assert hf.loc[("5kmE3500N2800", 2), "effort_insecta"] == 3  # 2005H1
    assert hf.loc[("5kmE3500N2800", 3), "effort_insecta"] == 4  # 2005H2
    assert hf.loc[("5kmE3500N2800", 5), "effort_insecta"] == 6  # 2006H2


def test_inverse_effort_weights():
    w = inverse_effort_weights(np.array([0, 1, 10, 100, 1000]), floor_quantile=0.0)
    assert np.isclose(w.mean(), 1.0)
    assert w[0] == w[1]  # zero effort floored to 1
    assert (np.diff(w[1:]) < 0).all()


def test_inaturalist_dedup_against_gbif_occurrence_ids():
    from src.ingest.inaturalist import deduplicate_against_gbif

    inat = pd.DataFrame(
        {
            "occurrence_id": [
                "https://www.inaturalist.org/observations/1",
                "https://www.inaturalist.org/observations/2",
            ]
        }
    )
    gbif = pd.Series(["http://www.inaturalist.org/observations/1", None, "urn:x"])
    out = deduplicate_against_gbif(inat, gbif)
    assert out["occurrence_id"].tolist() == ["https://www.inaturalist.org/observations/2"]
