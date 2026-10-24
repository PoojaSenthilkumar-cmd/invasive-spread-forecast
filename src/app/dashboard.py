"""Streamlit dashboard: frozen forecast layers from the end of 2025, backtest and evaluation.

Run with ``make dashboard`` (after ``make dashboard-data``). Reads only files in ``results/``
(or ``$ISF_RESULTS`` for a retrain run), ``models/frozen`` outputs exported there and
``reports/figures/``; it never fits, tunes or re-scores a model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydeck as pdk  # noqa: E402
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = Path(os.environ.get("ISF_RESULTS", ROOT / "results"))
FIG = ROOT / "reports" / "figures"
COUNTRIES = ROOT / "data" / "interim" / "covariates" / "ne_countries.gpkg"
ORDER = ["C", "A", "B", "ADJ"]
NAMES = {"C": "C (default)", "A": "A", "B": "B", "ADJ": "Adjacency baseline"}
REQUIRED = [
    "dashboard_layers.parquet",
    "dashboard_labels.json",
    "dashboard_backtest.parquet",
    "dashboard_calibration.json",
    "summary_h1.csv",
]
# rank bands shared by all layers (rank 1 = highest forecast among at-risk cells)
CLASSES = [
    (100, "top 100", [128, 0, 38]),
    (500, "101-500", [227, 26, 28]),
    (2000, "501-2,000", [253, 141, 60]),
    (10000, "2,001-10,000", [254, 217, 118]),
    (10**9, "rest", [255, 255, 204]),
]
OCC_RGB = [90, 90, 90]
BT_KS = [100, 500, 2000]
NOTE_A = (
    "**A tracks recording effort.** Its forecasts correlate with Insecta recording effort "
    "(Spearman {a_t:.2f} on 2022-2025, {a_h} on 2026), more than B ({b_t:.2f} / {b_h}) "
    "or C ({c_t:.2f} / {c_h}). Part of its accuracy comes from predicting where people record."
)
NOTE_C = (
    "**C did not beat adjacency on validation** (2019-2021 PR-AUC {c_v:.3f} vs adjacency "
    "{adj_v:.3f}). It beat adjacency on the 2022-2025 test and the 2026 holdout, but it was chosen "
    "as the dashboard default after the test results were seen."
)
NOTE_PROB = (
    "**Absolute probabilities under-predict, so they are not shown.** The isotonic calibration was "
    "fit on 2019-2021, when the yearly first-sighting rate was {val_rate:.1%}. On the 2022-2025 "
    "test the mean predicted 1-year probability was {pred_c:.1%} (C), {pred_a:.1%} (A) and "
    "{pred_b:.1%} (B), against an observed rate of {obs:.1%}: roughly {ratio:.1f}x too low. The "
    "map shows rank bands and relative risk (a cell's predicted risk / the average at-risk cell), "
    "which do not depend on that offset. No recalibration on test-era base rates was done."
)


@st.cache_data
def load() -> dict:
    """Cached read of the results files the dashboard shows."""
    d = {
        "layers": pd.read_parquet(RES / "dashboard_layers.parquet"),
        "labels": json.loads((RES / "dashboard_labels.json").read_text()),
        "backtest": pd.read_parquet(RES / "dashboard_backtest.parquet"),
        "calib": json.loads((RES / "dashboard_calibration.json").read_text()),
        "summary": pd.read_csv(RES / "summary_h1.csv"),
        "hold": None,
        "shap": None,
    }
    if (RES / "holdout_metrics.json").exists():
        d["hold"] = json.loads((RES / "holdout_metrics.json").read_text())
    if (RES / "dashboard_shap.parquet").exists():
        d["shap"] = pd.read_parquet(RES / "dashboard_shap.parquet")
    return d


@st.cache_data
def countries():
    """Country outlines for the static maps (None if the clipped layer is unavailable)."""
    if not COUNTRIES.exists():
        return None
    import geopandas as gpd

    return gpd.read_file(COUNTRIES).to_crs(4326)


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def layer_label(key: str, labels: dict) -> str:
    """Visible one-line label: what the layer is and its validation/test precision@k."""
    lab, ks = labels["layers"][key], labels["ks"]
    v, t = lab["validation"], lab["test"]
    pk = " · ".join(f"P@{k} {v[f'precision@{k}']:.3f} / {t[f'precision@{k}']:.3f}" for k in ks)
    return f"**Layer {NAMES[key]}: {lab['title']}.** Precision@k, validation / test: {pk}"


def notes(labels: dict) -> tuple[str, str]:
    """The two caveat notes (A tracks effort; C did not beat adjacency on validation)."""
    L = labels["layers"]
    a = NOTE_A.format(
        a_t=L["A"]["test_spearman_effort"],
        a_h=_fmt(L["A"]["holdout_spearman_effort"]),
        b_t=L["B"]["test_spearman_effort"],
        b_h=_fmt(L["B"]["holdout_spearman_effort"]),
        c_t=L["C"]["test_spearman_effort"],
        c_h=_fmt(L["C"]["holdout_spearman_effort"]),
    )
    c = NOTE_C.format(c_v=L["C"]["validation"]["pr_auc"], adj_v=L["ADJ"]["validation"]["pr_auc"])
    return a, c


def prob_note(calib: dict, summary: pd.DataFrame) -> str:
    """Under-prediction statement with the test-era numbers."""
    obs = calib["C"]["observed_rate"]
    pred = {k: calib[k]["mean_predicted"] for k in ["A", "B", "C"]}
    val_rate = calib["_validation_observed_rate"]
    return NOTE_PROB.format(
        val_rate=val_rate,
        pred_c=pred["C"],
        pred_a=pred["A"],
        pred_b=pred["B"],
        obs=obs,
        ratio=obs / np.mean(list(pred.values())),
    )


def bands(rank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Band index and label for each rank."""
    cls = np.searchsorted([c[0] for c in CLASSES], rank, side="left")
    return cls, np.array([c[1] for c in CLASSES])[cls]


def colour(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """At-risk cells with rank bands, relative risk and tooltip fields for one layer."""
    risk = df[df["status"] == "at risk"].copy()
    r = risk[f"rank_{key}"].to_numpy()
    cls, band = bands(r)
    rgb = np.array([c[2] for c in CLASSES])[cls]
    risk["rgba"] = [list(x) + [210] for x in rgb]
    risk["band"] = band
    risk["rank"] = r
    if key == "ADJ":
        risk["risk_txt"] = risk["occ_ring1"].map(lambda n: f"{int(n)} occupied neighbours")
    else:
        p = risk[f"prob_{key}"]
        risk["rel_risk"] = p / p.mean()
        risk["risk_txt"] = risk["rel_risk"].map(lambda x: f"{x:.1f}x the average at-risk cell")
    return risk


def _outline(ax) -> None:
    gdf = countries()
    if gdf is not None:
        gdf.boundary.plot(ax=ax, color="#9a9a9a", linewidth=0.4)


def _frame(ax, title: str) -> None:
    ax.set_aspect(1 / np.cos(np.deg2rad(47.0)))
    ax.set_xlim(-11, 15.5)
    ax.set_ylim(35.5, 58.5)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title, fontsize=10)


@st.cache_data
def static_map(layers: pd.DataFrame, key: str, show_rest: bool, show_occ: bool, title: str):
    """Static map image (no WebGL needed) of one layer's rank bands."""
    risk = colour(layers, key)
    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    ax.scatter(layers["lon"], layers["lat"], s=0.6, c="#ececec", marker="s", linewidths=0)
    if show_occ:
        occ = layers[layers["status"] != "at risk"]
        ax.scatter(
            occ["lon"],
            occ["lat"],
            s=0.8,
            c=[np.array(OCC_RGB) / 255],
            marker="s",
            linewidths=0,
            label="occupied by 2025",
        )
    for upper, name, rgb in reversed(CLASSES):
        if name == "rest" and not show_rest:
            continue
        d = risk[risk["band"] == name]
        ax.scatter(
            d["lon"],
            d["lat"],
            s=2.5 if upper <= 2000 else 1.2,
            c=[np.array(rgb) / 255],
            marker="s",
            linewidths=0,
            label=name,
        )
    _outline(ax)
    _frame(ax, title)
    ax.legend(loc="lower right", markerscale=6, fontsize=8, title="rank band (2026 forecast)")
    fig.tight_layout()
    return fig


def _scatter(df: pd.DataFrame) -> pdk.Layer:
    cols = ["lon", "lat", "rgba", "cell_id", "country", "band", "risk_txt", "rank"]
    d = df[cols].copy()
    d["rank"] = d["rank"].astype(str)
    return pdk.Layer(
        "ScatterplotLayer",
        d,
        get_position=["lon", "lat"],
        get_fill_color="rgba",
        get_radius=2600,
        radius_units="meters",
        pickable=True,
    )


def legend() -> None:
    """Rank-band legend."""
    st.markdown("**Rank band in the 2026 forecast**")
    for _, name, rgb in CLASSES:
        st.markdown(
            f"<span style='background:rgb{tuple(rgb)};padding:0 10px;"
            f"border:1px solid #999'>&nbsp;</span> {name}",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<span style='background:rgb{tuple(OCC_RGB)};padding:0 10px'>&nbsp;</span> "
        "occupied by 2025",
        unsafe_allow_html=True,
    )


def shap_section(shap: pd.DataFrame | None, key: str, layers: pd.DataFrame) -> None:
    """SHAP drivers of the top-ranked cells (layers C and A only)."""
    st.markdown("#### What drives the top-ranked cells?")
    if shap is None or key not in set(shap["layer"]):
        st.caption("SHAP drivers are provided for layers C and A (run `make freeze`).")
        return
    s = shap[shap["layer"] == key]
    n = s["cell_id"].nunique()
    st.caption(
        f"Exact TreeSHAP contributions (LightGBM, log-odds scale) for the top {n} cells of layer "
        f"{NAMES[key]} in the forecast issued at the end of 2025, from the frozen model in "
        "`models/frozen/`. Positive values push a cell up the ranking."
    )
    agg = (
        s.assign(abs_shap=s["shap"].abs())
        .groupby("feature")
        .agg(mean_abs_shap=("abs_shap", "mean"), mean_shap=("shap", "mean"))
        .sort_values("mean_abs_shap", ascending=False)
        .head(12)
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"Mean |SHAP| over the top {n} cells")
        st.bar_chart(agg["mean_abs_shap"], horizontal=True)
    with c2:
        cells = s[["cell_id", "rank"]].drop_duplicates().sort_values("rank").head(25)
        geo = layers.set_index("cell_id")
        pick = st.selectbox(
            "Cell (top 25)",
            cells["cell_id"],
            format_func=lambda c: (
                f"rank {int(cells.set_index('cell_id').loc[c, 'rank'])}: {c} "
                f"({geo.loc[c, 'country']}, {geo.loc[c, 'lat']:.2f}N {geo.loc[c, 'lon']:.2f}E)"
            ),
            key=f"shap_cell_{key}",
        )
        one = s[s["cell_id"] == pick].assign(abs_shap=lambda d: d["shap"].abs())
        one = one.sort_values("abs_shap", ascending=False).head(10)
        st.dataframe(one[["feature", "value", "shap"]].round(3), hide_index=True, width="stretch")


def map_tab(d: dict) -> None:
    """Forecast map with the layer toggle, its label, caveats and SHAP drivers."""
    layers, labels = d["layers"], d["labels"]
    key = st.radio(
        "Forecast layer",
        ORDER,
        index=0,
        format_func=lambda k: (
            labels["layers"][k]["title"] if k == "ADJ" else f"{k}: {labels['layers'][k]['title']}"
        ),
        key="layer",
    )
    lab = labels["layers"][key]
    st.info(layer_label(key, labels) + f"\n\n{lab['what']}")
    note_a, note_c = notes(labels)
    st.warning(note_a + "\n\n" + note_c)
    st.warning(prob_note(d["calib"], d["summary"]))

    c1, c2 = st.columns([1, 3])
    with c1:
        show_rest = st.checkbox("Show cells ranked below 10,000", value=False)
        show_occ = st.checkbox("Show cells occupied by 2025", value=True)
        render = st.radio(
            "Map renderer",
            ["Static image", "Interactive (WebGL)"],
            help="The interactive map needs WebGL in the browser; the static image always works.",
        )
        legend()
        st.caption(
            "Forecast issued at the end of 2025 for a first sighting in 2026 (5 km cells). "
            "Hover (interactive map) shows the band, rank and relative risk."
        )
    title = f"Layer {NAMES[key]}: {lab['title']}"
    with c2:
        st.markdown(f"#### {title}")
        if render == "Static image":
            st.pyplot(static_map(layers, key, show_rest, show_occ, title))
        else:
            risk = colour(layers, key)
            if not show_rest:
                risk = risk[risk["rank"] <= 10000]
            deck = []
            if show_occ:
                occ = layers[layers["status"] != "at risk"].copy()
                occ["rgba"] = [OCC_RGB + [160]] * len(occ)
                occ["band"], occ["risk_txt"], occ["rank"] = "occupied by 2025", "-", "-"
                deck.append(_scatter(occ))
            deck.append(_scatter(risk))
            st.pydeck_chart(
                pdk.Deck(
                    layers=deck,
                    initial_view_state=pdk.ViewState(latitude=47.0, longitude=2.0, zoom=4.2),
                    map_style=None,
                    tooltip={"text": "{cell_id} ({country})\n{band} | rank {rank}\n{risk_txt}"},
                ),
                height=640,
            )
    shap_section(d["shap"], key, layers)


def backtest_table(bt: pd.DataFrame) -> pd.DataFrame:
    """Per layer and forecast year: hits among the top-k ranked cells."""
    out = []
    for key in ORDER:
        for yr, g in bt.groupby("issued_after"):
            new = int(g["y_h1"].sum())
            row = {"layer": NAMES[key], "forecast for": int(yr) + 1, "new cells": new}
            for k in BT_KS:
                top = g[g[f"rank_{key}"] <= k]
                hits = int(top["y_h1"].sum())
                row[f"hits top {k}"] = hits
                row[f"P@{k}"] = round(hits / k, 3)
            top = g[g[f"rank_{key}"] <= 2000]
            row["top 2000 confirmed by 2025"] = int(top["first_year"].notna().sum())
            out.append(row)
    return pd.DataFrame(out)


@st.cache_data
def backtest_map(bt: pd.DataFrame, layers: pd.DataFrame, key: str, year: int, k: int, title: str):
    """Top-k cells of one backtest forecast vs the cells first sighted the following year."""
    g = bt[bt["issued_after"] == year].merge(
        layers[["cell_id", "lon", "lat"]], on="cell_id", how="left"
    )
    occ = layers[layers["cell_id"].isin(set(layers["cell_id"]) - set(g["cell_id"]))]
    top = g[g[f"rank_{key}"] <= k]
    hit = top[top["y_h1"] == 1]
    miss = g[(g["y_h1"] == 1) & (g[f"rank_{key}"] > k)]
    later = top[(top["y_h1"] == 0) & top["first_year"].notna()]
    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    ax.scatter(layers["lon"], layers["lat"], s=0.6, c="#ececec", marker="s", linewidths=0)
    ax.scatter(
        occ["lon"],
        occ["lat"],
        s=0.8,
        c="#5a5a5a",
        marker="s",
        linewidths=0,
        label=f"occupied by end of {year}",
    )
    ax.scatter(
        top["lon"],
        top["lat"],
        s=2.5,
        c="#fdae61",
        marker="s",
        linewidths=0,
        label=f"top {k:,} forecast, not sighted by 2025",
    )
    ax.scatter(
        later["lon"],
        later["lat"],
        s=3,
        c="#2c7bb6",
        marker="s",
        linewidths=0,
        label=f"top {k:,}, first sighted later (by 2025)",
    )
    ax.scatter(
        hit["lon"],
        hit["lat"],
        s=4,
        c="#d7191c",
        marker="s",
        linewidths=0,
        label=f"top {k:,}, first sighted in {year + 1} (hit)",
    )
    ax.scatter(
        miss["lon"],
        miss["lat"],
        s=6,
        c="black",
        marker="x",
        linewidths=0.4,
        label=f"first sighted in {year + 1}, outside top {k:,} (miss)",
    )
    _outline(ax)
    _frame(ax, title)
    ax.legend(loc="lower right", markerscale=4, fontsize=7)
    fig.tight_layout()
    return fig


def backtest_tab(d: dict) -> None:
    """Replay of the 2022-2025 test forecasts against later confirmed first sightings."""
    bt, labels = d["backtest"], d["labels"]
    st.markdown(
        "Replay of the Phase 5 test forecasts: at the end of each year 2021-2024 every layer "
        "ranked the cells not yet occupied, and the ranking is compared with the first sightings "
        "confirmed in the following year (and up to 2025). The models were fit once on labels up "
        "to 2021 (no refit per year), with features updated each year; these are the saved test "
        "predictions, not re-scored. 2026 sightings are not used here."
    )
    years = sorted(bt["issued_after"].unique())
    c1, c2, c3 = st.columns(3)
    key = c1.selectbox(
        "Layer",
        ORDER,
        format_func=lambda k: f"{NAMES[k]}: {labels['layers'][k]['title']}",
        key="bt_layer",
    )
    year = c2.selectbox(
        "Forecast issued at the end of",
        years,
        index=len(years) - 1,
        format_func=lambda y: f"{y} (for {y + 1})",
        key="bt_year",
    )
    k = c3.selectbox("Top-k cells", BT_KS, index=2, key="bt_k")
    tab = backtest_table(bt)
    one = tab[(tab["layer"] == NAMES[key]) & (tab["forecast for"] == year + 1)].iloc[0]
    st.info(
        f"**Layer {NAMES[key]}, forecast for {year + 1}:** {one[f'hits top {k}']} of the top "
        f"{k:,} cells were first sighted in {year + 1} (precision {one[f'P@{k}']:.3f}); "
        f"{one['new cells']:,} cells were newly sighted that year in total."
    )
    st.pyplot(
        backtest_map(
            bt,
            d["layers"],
            key,
            int(year),
            int(k),
            f"Layer {NAMES[key]}: forecast issued end of {year}, top {k:,}",
        )
    )
    st.markdown("**All layers and years** (hits = cells first sighted in the forecast year)")
    st.dataframe(tab, hide_index=True, width="stretch")


def eval_tab(d: dict) -> None:
    """Evaluation tab: the frozen 2026 holdout plus the Phase 5 test/validation tables."""
    labels, hold, summary = d["labels"], d["hold"], d["summary"]
    if hold is None:
        st.info("This results directory (a retrain run) has no 2026 holdout evaluation.")
    else:
        h = hold["holdout"]
        frozen = labels.get("holdout_frozen") or "commit unknown"
        prereg = labels.get("preregistration_commit") or "commit unknown"
        st.error(
            f"**2026 HOLDOUT: SCORED ONCE AND FROZEN.** Committed as {frozen}, after the "
            f"comparisons were pre-registered ({prereg}). No model was changed after this run; "
            "the dashboard layers are the same frozen fits. Model-file hashes were added to the "
            "pre-registration file after the run (marked as a post-run addendum)."
        )
        st.markdown(
            f"Forecast issued after {h['forecast_issued_after']}, label year {h['label_year']} "
            f"(partial year, GBIF snapshot October 2026): {h['n_rows']:,} at-risk cells, "
            f"**{h['n_new_cells']} first sightings**. The low-effort cells (deciles 1-3) had only "
            "9 first sightings, so low-effort holdout numbers carry almost no information."
        )
        rows = []
        for name, m in hold["models"].items():
            v = m["h1"]
            ci, dd = v["pr_auc_ci95"], v.get("pr_auc_minus_adjacency_ci95")
            rows.append(
                {
                    "model": name,
                    "PR-AUC": f"{v['pr_auc']:.3f}",
                    "95% CI": f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                    "ROC-AUC": f"{v['roc_auc']:.3f}",
                    "P@100": f"{v['precision@100']:.3f}",
                    "P@500": f"{v['precision@500']:.3f}",
                    "P@2000": f"{v['precision@2000']:.3f}",
                    "PR-AUC minus adjacency [95% CI]": (
                        "-"
                        if dd is None
                        else f"{v['pr_auc_minus_adjacency']:+.3f} [{dd[0]:+.3f}, {dd[1]:+.3f}]"
                    ),
                    "Spearman with effort": f"{v['spearman_with']['meta_effort_target']:.2f}",
                }
            )
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch", height=35 * len(rows) + 38
        )

    with st.expander("Main test 2022-2025 and validation 2019-2021 (Phase 5)"):
        cols = [
            "model",
            "pr_auc",
            "pr_auc_ci_lo",
            "pr_auc_ci_hi",
            "roc_auc",
            "precision@100",
            "precision@500",
            "precision@2000",
            "low_precision@100",
            "rho_effort",
        ]
        st.dataframe(summary[cols].round(4), hide_index=True, width="stretch")
        val = pd.DataFrame(
            {
                k: {"model": v["model"]} | {m: round(x, 4) for m, x in v["validation"].items()}
                for k, v in labels["layers"].items()
            }
        ).T
        st.markdown("Validation (out-of-fold 2019-2021) for the dashboard layers")
        st.dataframe(val, width="stretch")
        for f in ["effort_decile_metrics.png", "reliability_h1.png"]:
            if (FIG / f).exists():
                st.image(str(FIG / f))


def main() -> None:
    """Page layout: forecast map, backtest and evaluation tabs."""
    st.set_page_config(page_title="Asian hornet spread forecast", layout="wide")
    st.title("Vespa velutina (Asian hornet), Western Europe: where will it be sighted next?")
    missing = [f for f in REQUIRED if not (RES / f).exists()]
    if missing:
        st.error(f"Missing results files in {RES}: {missing}. Run `make dashboard-data` first.")
        st.stop()
    d = load()
    tab_map, tab_bt, tab_eval = st.tabs(["Forecast map", "Backtest 2022-2025", "Evaluation"])
    with tab_map:
        map_tab(d)
    with tab_bt:
        backtest_tab(d)
    with tab_eval:
        eval_tab(d)


main()
