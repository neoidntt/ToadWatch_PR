"""v2 - development tools. Everything here uses the TRAIN + VALIDATION records only
(the "development set"); the test set is never read.

* ``cross_validate_variants``  repeated stratified CV of every model version
* ``cv_grid_search``           per-version tuning by CV (same grid and procedure for each)
* ``choose_combined``          the pre-declared rule that picks early vs stacked fusion
* ``learning_curve``           F1 against training-set size
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from . import config as C
from .modeling import compute_metrics, predict_variant, proba_variant, train_variant


def cross_validate_variants(X: pd.DataFrame, y: pd.Series, params: dict | None = None,
                            variants=None, n_splits: int = C.CV_FOLDS, n_repeats: int = C.CV_REPEATS,
                            seed: int = C.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-fold metrics, out-of-fold probabilities of repeat 0)."""
    variants = variants or C.VARIANTS
    y = y.reset_index(drop=True); X = X.reset_index(drop=True)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows, oof = [], pd.DataFrame({"y": y})
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        repeat = k // n_splits
        for v in variants:
            mc = train_variant(v, X.iloc[tr], y.iloc[tr], params)
            pred = predict_variant(mc, X.iloc[te])
            rows.append({"repeat": repeat, "fold": k % n_splits, "variant": v,
                         **compute_metrics(y.iloc[te], pred)})
            if repeat == 0:
                oof.loc[te, f"p_{v}"] = proba_variant(mc, X.iloc[te])
    return pd.DataFrame(rows), oof


def summarize_cv(folds: pd.DataFrame) -> pd.DataFrame:
    g = folds.groupby("variant")[["accuracy", "precision", "recall", "f1", "false_alarm_rate"]]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    return out.sort_values("f1_mean", ascending=False)


def paired_fold_differences(folds: pd.DataFrame, a: str, b: str, metric: str = "f1") -> dict:
    """metric(a) - metric(b) per CV fold: mean, std and the share of folds where a wins."""
    w = folds.pivot_table(index=["repeat", "fold"], columns="variant", values=metric)
    d = w[a] - w[b]
    return {"mean_diff": float(d.mean()), "std_diff": float(d.std()), "share_a_wins": float((d > 0).mean())}


def cv_grid_search(X: pd.DataFrame, y: pd.Series, variant: str, grid: dict | None = None,
                   n_splits: int = C.CV_FOLDS, seed: int = C.SEED) -> pd.DataFrame:
    """Mean CV F1 of one version for every setting combination (same grid for every version)."""
    grid = grid or C.TUNING_GRID
    keys = list(grid)
    y = y.reset_index(drop=True); X = X.reset_index(drop=True)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        f1s = []
        for tr, te in skf.split(X, y):
            mc = train_variant(variant, X.iloc[tr], y.iloc[tr], params)
            f1s.append(compute_metrics(y.iloc[te], predict_variant(mc, X.iloc[te]))["f1"])
        rows.append({**params, "cv_f1_mean": float(np.mean(f1s)), "cv_f1_std": float(np.std(f1s))})
    return pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)


def best_params(results: pd.DataFrame, grid: dict | None = None) -> dict:
    grid = grid or C.TUNING_GRID
    row = results.iloc[0]
    out = {}
    for k in grid:
        v = row[k]
        out[k] = None if pd.isna(v) else (int(v) if k in ("n_estimators", "max_depth", "min_samples_leaf") else (float(v) if isinstance(v, (float, np.floating)) else v))
    return out


def choose_combined(cv_summary: pd.DataFrame) -> dict:
    """Pre-declared rule: the combined design with the higher mean CV F1 wins; if they differ by
    less than TIE_MARGIN the simpler early-fusion model (one forest on all features) is kept."""
    e = float(cv_summary.loc["combined_early", "f1_mean"])
    s = float(cv_summary.loc["combined_stacked", "f1_mean"])
    choice = "combined_stacked" if s - e > C.TIE_MARGIN else "combined_early"
    return {"choice": choice, "early_cv_f1": e, "stacked_cv_f1": s, "tie_margin": C.TIE_MARGIN}


def learning_curve(X: pd.DataFrame, y: pd.Series, params: dict | None = None,
                   fractions=(0.1, 0.25, 0.5, 0.75, 1.0), n_splits: int = 5,
                   seed: int = C.SEED) -> pd.DataFrame:
    """F1 on the held-out CV fold when training on only a fraction of the training folds."""
    y = y.reset_index(drop=True); X = X.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows = []
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        for frac in fractions:
            sub = np.sort(rng.choice(tr, size=max(int(len(tr) * frac), 20), replace=False))
            for v in C.VARIANTS:
                mc = train_variant(v, X.iloc[sub], y.iloc[sub], params)
                rows.append({"fraction": frac, "n_train": len(sub), "variant": v,
                             "f1": compute_metrics(y.iloc[te], predict_variant(mc, X.iloc[te]))["f1"]})
    return pd.DataFrame(rows).groupby(["variant", "fraction", "n_train"], as_index=False)["f1"].agg(["mean", "std"]).reset_index()
