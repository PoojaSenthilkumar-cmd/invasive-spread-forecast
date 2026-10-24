"""Streamlit dashboard: frozen 1-year forecast layers from the end of 2025, and the evaluation.

Run with ``make dashboard`` (after ``make dashboard-data``). Reads only files in ``results/``
and ``reports/figures/``; it never fits, tunes or re-scores a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RES, FIG = ROOT / "results", ROOT / "reports" / "figures"
ORDER = ["C", "A", "B", "ADJ"]
NAMES = {"C": "C (default)", "A": "A", "B": "B", "ADJ": "Adjacency baseline"}
# rank classes shared by all layers (rank 1 = highest forecast)
CLASSES = [
    (100, "top 100", [128, 0, 38]),
    (500, "101-500", [227, 26, 28]),
    (2000, "501-2,000", [253, 141, 60]),
    (10000, "2,001-10,000", [254, 217, 118]),
    (10**9, "rest", [255, 255, 204]),
]
OCC_RGB = [90, 90, 90]
NOTE_A = (
    "**A tracks recording effort.** Its forecasts correlate with Insecta recording effort "
    "(Spearman {a_t:.2f} on 2022-2025, {a_h:.2f} on 2026), more than B ({b_t:.2f} / {b_h:.2f}) "
    "or C ({c_t:.2f} / {c_h:.2f}). Part of its accuracy comes from predicting where people record."
)
NOTE_C = (
    "**C did not beat adjacency on validation** (2019-2021 PR-AUC {c_v:.3f} vs adjacency "
    "{adj_v:.3f}). It beat adjacency on the 2022-2025 test and the 2026 holdout, but it was chosen "
    "as the dashboard default after the test results were seen."
)


@st.cache_data
def load() -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    """Cached read of the results files the dashboard shows."""
    layers = pd.read_parquet(RES / "dashboard_layers.parquet")
    labels = json.loads((RES / "dashboard_labels.json").read_text())
    hold = json.loads((RES / "holdout_metrics.json").read_text())
    summary = pd.read_csv(RES / "summary_h1.csv")
    return layers, labels, hold, summary


def pk_table(lab: dict, ks: list[int]) -> pd.DataFrame:
    """Validation/test precision@k table for one layer."""
    rows = {
        "validation 2019-2021": lab["validation"],
        "test 2022-2025": lab["test"],
    }
    df = pd.DataFrame(
        {
            split: {f"precision@{k}": m[f"precision@{k}"] for k in ks} | {"PR-AUC": m["pr_auc"]}
            for split, m in rows.items()
        }
    ).T
    return df.map(lambda v: f"{v:.3f}")


def layer_label(key: str, labels: dict) -> str:
    """Visible one-line label: what the layer is and its validation/test precision@k."""
    lab, ks = labels["layers"][key], labels["ks"]
    v, t = lab["validation"], lab["test"]
    pk = " · ".join(f"P@{k} {v[f'precision@{k}']:.3f} / {t[f'precision@{k}']:.3f}" for k in ks)
    return f"**Layer {NAMES[key]}: {lab['title']}.** Precision@k, validation / test: {pk}"


def colour(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """At-risk cells with rank-band colours and tooltip fields for one layer."""
    risk = df[df["status"] == "at risk"].copy()
    r = risk[f"rank_{key}"].to_numpy()
    cls = np.searchsorted([c[0] for c in CLASSES], r, side="left")
    rgb = np.array([c[2] for c in CLASSES])[cls]
    risk["rgba"] = [list(x) + [210] for x in rgb]
    risk["band"] = np.array([c[1] for c in CLASSES])[cls]
    risk["prob_txt"] = risk[f"prob_{key}"].map(
        lambda p: f"{p:.4f}" if pd.notna(p) else "n/a (ranking only)"
    )
    risk["rank"] = r
    return risk


def notes(labels: dict) -> tuple[str, str]:
    """The two caveat notes (A tracks effort; C did not beat adjacency on validation)."""
    L = labels["layers"]
    a = NOTE_A.format(
        a_t=L["A"]["test_spearman_effort"],
        a_h=L["A"]["holdout_spearman_effort"],
        b_t=L["B"]["test_spearman_effort"],
        b_h=L["B"]["holdout_spearman_effort"],
        c_t=L["C"]["test_spearman_effort"],
        c_h=L["C"]["holdout_spearman_effort"],
    )
    c = NOTE_C.format(
        c_v=L["C"]["validation"]["pr_auc"],
        adj_v=L["ADJ"]["validation"]["pr_auc"],
    )
    return a, c


def map_tab(layers: pd.DataFrame, labels: dict) -> None:
    """Forecast map with the layer toggle, its label and caveats."""
    key = st.radio(
        "Forecast layer",
        ORDER,
        index=0,
        format_func=lambda k: (
            labels["layers"][k]["title"] if k == "ADJ" else f"{k}: {labels['layers'][k]['title']}"
        ),
        horizontal=False,
    )
    lab = labels["layers"][key]
    st.info(layer_label(key, labels) + f"\n\n{lab['what']}")
    note_a, note_c = notes(labels)
    st.warning(note_a + "\n\n" + note_c)

    c1, c2 = st.columns([1, 3])
    with c1:
        show_rest = st.checkbox("Show cells ranked below 10,000", value=False)
        show_occ = st.checkbox("Show cells occupied by 2025", value=True)
        render = st.radio(
            "Map renderer",
            ["Static image", "Interactive (WebGL)"],
            help="The interactive map needs WebGL in the browser; the static image always works.",
        )
        st.markdown("**Rank in the 2026 forecast**")
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
        st.caption(
            "Forecast issued at the end of 2025 for a first sighting in 2026 (5 km cells). "
            "Probabilities are isotonic-calibrated on 2019-2021, when the base rate was lower, so "
            "they under-state the risk; use the ranks."
        )
    if render == "Static image":
        with c2:
            st.markdown(f"#### Layer {NAMES[key]}: {lab['title']}")
            st.pyplot(
                static_map(layers, key, show_rest, show_occ, f"Layer {NAMES[key]}: {lab['title']}")
            )
        return
    risk = colour(layers, key)
    if not show_rest:
        risk = risk[risk["rank"] <= 10000]
    deck_layers = []
    if show_occ:
        occ = layers[layers["status"] != "at risk"].assign(
            rgba=[OCC_RGB + [160]] * int((layers["status"] != "at risk").sum()),
            band="occupied by 2025",
            prob_txt="-",
            rank="-",
        )
        deck_layers.append(_scatter(occ))
    deck_layers.append(_scatter(risk))
    with c2:
        st.markdown(f"#### Layer {NAMES[key]}: {lab['title']}")
        st.pydeck_chart(
            pdk.Deck(
                layers=deck_layers,
                initial_view_state=pdk.ViewState(latitude=47.0, longitude=2.0, zoom=4.2),
                map_style=None,
                tooltip={"text": "{cell_id} ({country})\n{band} | rank {rank}\nP(2026) {prob_txt}"},
            ),
            height=640,
        )


@st.cache_data
def static_map(layers: pd.DataFrame, key: str, show_rest: bool, show_occ: bool, title: str):
    """Static map image (no WebGL needed) of one layer, same colours as the interactive map."""
    risk = colour(layers, key)
    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    ax.scatter(layers["lon"], layers["lat"], s=0.6, c="#e8e8e8", marker="s", linewidths=0)
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
    ax.set_aspect(1 / np.cos(np.deg2rad(47.0)))
    ax.set_xlim(-11, 15.5)
    ax.set_ylim(35.5, 58.5)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", markerscale=6, fontsize=8, title="rank in 2026 forecast")
    fig.tight_layout()
    return fig


def _scatter(df: pd.DataFrame) -> pdk.Layer:
    cols = ["lon", "lat", "rgba", "cell_id", "country", "band", "prob_txt", "rank"]
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


def eval_tab(labels: dict, hold: dict, summary: pd.DataFrame) -> None:
    """Evaluation tab: the frozen 2026 holdout plus the Phase 5 test/validation tables."""
    h = hold["holdout"]
    frozen = labels.get("holdout_frozen") or "commit unknown"
    prereg = labels.get("preregistration_commit") or "commit unknown"
    st.error(
        f"**2026 HOLDOUT: SCORED ONCE AND FROZEN.** Committed as {frozen}, after the comparisons "
        f"were pre-registered ({prereg}). No model was changed after this run; the dashboard "
        "layers are the same frozen fits."
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
        ci, d = v["pr_auc_ci95"], v.get("pr_auc_minus_adjacency_ci95")
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
                    if d is None
                    else f"{v['pr_auc_minus_adjacency']:+.3f} [{d[0]:+.3f}, {d[1]:+.3f}]"
                ),
                "Spearman with effort": f"{v['spearman_with']['meta_effort_target']:.2f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=35 * len(rows) + 38)

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
    """Page layout: forecast map and evaluation tabs."""
    st.set_page_config(page_title="Asian hornet spread forecast", layout="wide")
    st.title("Vespa velutina (Asian hornet), Western Europe: where will it be sighted next?")
    missing = [
        f
        for f in [
            "dashboard_layers.parquet",
            "dashboard_labels.json",
            "holdout_metrics.json",
            "summary_h1.csv",
        ]
        if not (RES / f).exists()
    ]
    if missing:
        st.error(f"Missing results files: {missing}. Run `make dashboard-data` first.")
        st.stop()
    layers, labels, hold, summary = load()
    tab_map, tab_eval = st.tabs(["Forecast map", "Evaluation"])
    with tab_map:
        map_tab(layers, labels)
    with tab_eval:
        eval_tab(labels, hold, summary)


main()
