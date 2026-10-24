"""Phase 4: spread classifier, static SDM baselines, naive baselines and time-aware calibration.

Every model is expressed as ``fit(cutoff_year) -> score(rows)``, where ``fit`` may only use labels
whose window ends at or before ``cutoff_year``. The same wrapper then gives

* expanding-window out-of-fold (OOF) scores for the validation years (fold ``y`` is fit with
  cutoff ``y - 1`` and scored on rows whose label window starts in ``y``),
* an isotonic calibrator fit on those OOF scores, and
* a final model fit with cutoff = last validation year, scored on the test years.

Default bias handling for the spread model (Phase 2 recommendation, user-approved):
target-group negatives (sampled proportional to Insecta effort in the label window),
inverse-effort weights on positives, population/road covariates neutralised to their training
median at prediction time, and recording-effort features kept as explicit inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

from src.features.build_features import features_path, is_feature
from src.features.covariates_join import covariates_path
from src.preprocess.bias import ENV_FEATURES, _glm, inverse_effort_weights
from src.preprocess.dataset_builder import panel_path

BIAS_COVS = ["log_pop_density", "road_density", "dist_road_km"]
EFFORT_FEATS = [
    "log_effort_t",
    "log_effort_cum",
    "log_effort_mean3",
    "log_effort_hym_t",
    "log_observers_t",
    "log_effort_ring3_t",
]

Scorer = Callable[[pd.DataFrame], np.ndarray]


def load_table(cfg: dict[str, Any]) -> pd.DataFrame:
    """Feature table joined with static cell covariates."""
    df = pd.read_parquet(features_path(cfg))
    cov = pd.read_parquet(covariates_path(cfg))
    cov["log_pop_density"] = np.log1p(cov["pop_density"])
    cov = cov[["cell_id", *ENV_FEATURES, *BIAS_COVS]]
    cov[ENV_FEATURES + BIAS_COVS] = cov[ENV_FEATURES + BIAS_COVS].astype(np.float32)
    return df.merge(cov, on="cell_id", how="left", validate="many_to_one")


def dynamic_features(df: pd.DataFrame) -> list[str]:
    """Spread-process features from the Phase 3 builder (excluding effort)."""
    return [
        c for c in df.columns if is_feature(c) and c not in EFFORT_FEATS + ENV_FEATURES + BIAS_COVS
    ]


def label_window(df: pd.DataFrame, k: int) -> tuple[pd.Series, pd.Series]:
    """First and last year covered by the ``y_h{k}`` label of each row."""
    return df["year"] + 1, df["year"] + k


@dataclass
class Variant:
    """A spread-model configuration (feature set + bias handling)."""

    name: str
    features: list[str]
    negatives: str = "target_group"  # "target_group" | "target_group_strict" | "uniform"
    inverse_effort: bool = True
    neutralise: list[str] = field(default_factory=lambda: list(BIAS_COVS))


def _effort_floor(effort: np.ndarray, q: float) -> float:
    """``q``-quantile of strictly positive effort (>= 1), as in ``inverse_effort_weights``."""
    pos = effort[effort > 0]
    return max(float(np.quantile(pos, q)) if pos.size else 1.0, 1.0)


def sample_training(
    rows: pd.DataFrame, y: str, v: Variant, mcfg: dict[str, Any], rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-window negative sampling (target-group or uniform) and positive weights."""
    parts = []
    for _, g in rows.groupby("window", sort=True):
        pos = g[g[y] == 1]
        neg = g[g[y] == 0]
        n_neg = min(len(neg), max(int(mcfg["min_neg_per_window"]), mcfg["neg_per_pos"] * len(pos)))
        e = neg["meta_effort_target"].to_numpy(dtype=float)
        if v.negatives == "target_group":
            # p ~ effort + floor: recorded cells dominate, but unrecorded cells stay eligible
            # (otherwise "zero effort" alone separates positives from negatives).
            pe = e + _effort_floor(e, mcfg["effort_floor_quantile"])
            idx = rng.choice(len(neg), size=n_neg, replace=False, p=pe / pe.sum())
        elif v.negatives == "target_group_strict":
            # first implementation (kept as an ablation): negatives only from recorded cells
            ok = np.flatnonzero(e > 0)
            n_neg = min(n_neg, len(ok))
            idx = rng.choice(ok, size=n_neg, replace=False, p=e[ok] / e[ok].sum())
        else:
            idx = rng.choice(len(neg), size=n_neg, replace=False)
        parts += [pos, neg.iloc[idx]]
    tr = pd.concat(parts)
    w = np.ones(len(tr))
    is_pos = tr[y].to_numpy() == 1
    if v.inverse_effort and is_pos.any():
        wp = inverse_effort_weights(
            tr.loc[is_pos, "meta_effort_target"].to_numpy(), mcfg["effort_floor_quantile"]
        )
        if v.negatives != "target_group_strict":
            wp = np.minimum(wp, float(mcfg["max_weight"]))  # cap: no single cell dominates
            wp /= wp.mean()
        w[is_pos] = wp
    w[is_pos] *= (~is_pos).sum() / max(w[is_pos].sum(), 1.0)  # balance classes by weight
    return tr, w


def _neutralised(rows: pd.DataFrame, v: Variant, medians: pd.Series) -> pd.DataFrame:
    x = rows[v.features].copy()
    for c in v.neutralise:
        if c in x:
            x[c] = medians[c]
    return x


def lgbm_fitter(
    table: pd.DataFrame, k: int, v: Variant, params: dict, mcfg: dict, seed: int
) -> Callable[[int], Scorer]:
    """``fit(cutoff)`` for a LightGBM spread model on ``y_h{k}`` labels ending <= cutoff."""
    y = f"y_h{k}"

    def fit(cutoff: int) -> Scorer:
        _, end = label_window(table, k)
        rows = table[(end <= cutoff) & table[y].notna()]
        tr, w = sample_training(rows, y, v, mcfg, np.random.default_rng(seed + cutoff))
        medians = tr[v.features].median()
        model = lgb.LGBMClassifier(**params, random_state=seed, verbose=-1)
        model.fit(tr[v.features], tr[y].astype(int), sample_weight=w)

        def score(r: pd.DataFrame) -> np.ndarray:
            return model.predict_proba(_neutralised(r, v, medians))[:, 1]

        score.model = model  # type: ignore[attr-defined]
        return score

    return fit


def static_sdm_fitter(
    cfg: dict[str, Any], table: pd.DataFrame, kind: str, effort: pd.DataFrame
) -> Callable[[int], Scorer]:
    """``fit(cutoff)`` for a pooled presence/background SDM (``kind`` = "lr" | "rf").

    Presences: cells occupied by the end of ``cutoff``. Background: target-group sample of
    the other cells (proportional to Insecta effort up to ``cutoff``); inverse-effort weights on
    presences. Climate + land-cover + elevation only, no temporal dynamics.
    """
    cov = pd.read_parquet(covariates_path(cfg)).set_index("cell_id")
    start = cfg["time"]["start_year"]
    first_year = (
        pd.read_parquet(panel_path(cfg), columns=["cell_id", "first_detection_window"])
        .drop_duplicates("cell_id")
        .set_index("cell_id")["first_detection_window"]
        .reindex(cov.index)
        .astype(float)
        + start
    )
    rng_seed = int(cfg["model"]["seed"])
    mcfg = cfg["model"]

    def fit(cutoff: int) -> Scorer:
        rng = np.random.default_rng(rng_seed + cutoff)
        e = effort[effort["window"] <= cutoff - start].groupby("cell_id")["effort_insecta"].sum()
        e = e.reindex(cov.index).fillna(0).to_numpy()
        occ = (first_year <= cutoff).to_numpy()
        pres = np.flatnonzero(occ)
        pool = np.flatnonzero(~occ & (e > 0))
        bg = pool[
            rng.choice(len(pool), size=int(mcfg["sdm_background_n"]), p=e[pool] / e[pool].sum())
        ]
        X = cov[ENV_FEATURES].to_numpy()
        Xt = np.r_[X[pres], X[bg]]
        yt = np.r_[np.ones(len(pres)), np.zeros(len(bg))]
        wp = inverse_effort_weights(e[pres], mcfg["effort_floor_quantile"])
        wt = np.r_[wp, np.ones(len(bg)) * wp.sum() / len(bg)]
        if kind == "lr":
            m = _glm()
            m.fit(Xt, yt, logisticregression__sample_weight=wt)
        else:
            m = RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=rng_seed
            )
            m.fit(Xt, yt, sample_weight=wt)
        cell_score = pd.Series(m.predict_proba(X)[:, 1], index=cov.index)
        return lambda r: cell_score.reindex(r["cell_id"]).to_numpy()

    return fit


def fixed_scorer(fn: Scorer) -> Callable[[int], Scorer]:
    """Wrap a training-free baseline in the ``fit(cutoff)`` interface."""
    return lambda cutoff: fn


BASELINES: dict[str, Scorer] = {
    # cells next to occupied cells (binary; ties broken randomly when ranking)
    "adjacency (binary)": lambda r: (r["occ_ring1"].to_numpy() > 0).astype(float),
    # adjacency ranked by number of occupied neighbours (1-ring, then 2-ring as tie-break)
    "adjacency (ranked)": lambda r: r["occ_ring1"].to_numpy() * 100.0 + r["occ_ring2"].to_numpy(),
    "distance to front": lambda r: -r["dist_km"].to_numpy(),
    "random": lambda r: np.random.default_rng(0).random(len(r)),
}


def oof_and_test(
    fit: Callable[[int], Scorer],
    table: pd.DataFrame,
    k: int,
    val_years: list[int],
    test_years: list[int],
    last_year: int,
) -> dict[str, Any]:
    """Expanding-window OOF scores on validation years, isotonic calibrator, test scores."""
    y = f"y_h{k}"
    start, end = label_window(table, k)
    val_last = max(val_years)
    oof = []
    for vy in val_years:
        rows = table[(start == vy) & (end <= val_last) & table[y].notna()]
        if len(rows):
            oof.append(pd.DataFrame({"idx": rows.index, "score": fit(vy - 1)(rows)}))
    oof = pd.concat(oof)
    yo = table.loc[oof["idx"], y].to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof["score"], yo)
    test_rows = table[(start >= min(test_years)) & (end <= last_year) & table[y].notna()]
    final = fit(val_last)
    s = final(test_rows)
    return {
        "oof": oof.assign(y=yo, prob=iso.predict(oof["score"])),
        "calibrator": iso,
        "final": final,
        "test": pd.DataFrame({"idx": test_rows.index, "score": s, "prob": iso.predict(s)}),
    }


def tune_lgbm(
    table: pd.DataFrame, v: Variant, cfg: dict[str, Any], k: int = 1
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Optuna search over LightGBM params; objective = mean PR-AUC over expanding-window folds."""
    import optuna

    mcfg, seed = cfg["model"], int(cfg["model"]["seed"])
    y = f"y_h{k}"
    start, end = label_window(table, k)
    folds = []
    for vy in mcfg["val_target_years"]:
        tr_rows = table[(end <= vy - 1) & table[y].notna()]
        tr, w = sample_training(tr_rows, y, v, mcfg, np.random.default_rng(seed + vy))
        va = table[(start == vy) & table[y].notna()]
        folds.append((tr, w, va))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "n_jobs": -1,
        }
        aps, iters = [], []
        for tr, w, va in folds:
            m = lgb.LGBMClassifier(**params, random_state=seed, verbose=-1)
            medians = tr[v.features].median()
            m.fit(
                tr[v.features],
                tr[y].astype(int),
                sample_weight=w,
                eval_set=[(_neutralised(va, v, medians), va[y].astype(int))],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            p = m.predict_proba(_neutralised(va, v, medians), num_iteration=m.best_iteration_)
            aps.append(average_precision_score(va[y], p[:, 1]))
            iters.append(m.best_iteration_)
        trial.set_user_attr("n_estimators", int(np.median(iters)))
        trial.set_user_attr("fold_pr_auc", aps)
        return float(np.mean(aps))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=int(mcfg["n_trials"]))
    best = dict(study.best_params)
    best.update(
        n_estimators=max(50, study.best_trial.user_attrs["n_estimators"]),
        subsample_freq=1,
        n_jobs=-1,
    )
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    return best, trials
