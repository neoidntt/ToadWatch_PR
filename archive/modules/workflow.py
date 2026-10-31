"""v2 workflow - one function per stage, used by BOTH the notebooks and ``scripts/run_v2.py``.

    prepare_data      steps 1-2  load, clean, deduplicate, split, pair, lock the test set
    run_diagnosis     where does the combined model lose? (CV on train+val only)
    tune_and_freeze   tune every version by CV, pick early vs stacked, freeze the config
    final_test        the one-shot test run
    dev_charts / final_charts   figures into outputs/v2/figures

Select the version BEFORE importing anything from toadwatch:
    os.environ["TOADWATCH_VERSION"] = "v2"
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from .cv import (best_params, choose_combined, cross_validate_variants, cv_grid_search,
                 learning_curve, paired_fold_differences, summarize_cv)
from .features import apply_policy, build_all_features
from .modeling import feature_importance, save_models, train_all
from .pipeline import run_data_preparation
from .splitting import load_split
from .stats import complementarity
from .tuning import freeze_v2_config, load_final_config

POLICIES = {
    "v1 policy\n(style on, no cross)": (False, False),
    "no style features\n(v2 design)": (True, False),
    "no style +\ncross features\n(tested, not used)": (True, True),   # cross only reaches the stacked meta-learner
}


def _need_v2():
    if C.VERSION != "v2":
        raise RuntimeError('Set os.environ["TOADWATCH_VERSION"] = "v2" before importing toadwatch.')


# ------------------------------------------------------------------ data
def prepare_data() -> dict:
    _need_v2()
    return run_data_preparation(save=True)


def dev_data(use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(records, ALL features, labels) for train+val. The test split is never loaded here."""
    recs = pd.concat([load_split("train"), load_split("val")], ignore_index=True)
    cache = C.INTERIM_DIR / "features_all_dev.parquet"
    if use_cache and cache.exists():
        X = pd.read_parquet(cache)
    else:
        X = build_all_features(recs)
        C.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        X.to_parquet(cache)
    return recs, X, recs["label"].astype(int)


def dev_features() -> tuple[pd.DataFrame, pd.Series]:
    """Features under the v2 policy (no style shortcuts, with cross features)."""
    _, X, y = dev_data()
    return apply_policy(X, C.EXCLUDE_STYLE_FEATURES, C.USE_CROSS_FEATURES), y


# ------------------------------------------------------------------ diagnosis
def run_diagnosis() -> dict:
    """CV (train+val only) of every model under three feature policies + complementarity."""
    _need_v2()
    C.ensure_dirs()
    _, X_all, y = dev_data()
    rows, folds_by_policy = [], {}
    for policy, (no_style, cross) in POLICIES.items():
        X = apply_policy(X_all, no_style, cross)
        folds, oof = cross_validate_variants(X, y, None, list(C.VARIANTS), n_repeats=2)
        folds_by_policy[policy] = (folds, oof)
        s = folds.groupby("variant")["f1"].agg(["mean", "std"]).reset_index()
        for _, r in s.iterrows():
            rows.append({"policy": policy, "variant": r["variant"], "f1_mean": r["mean"], "f1_std": r["std"]})
    ablation = pd.DataFrame(rows)
    ablation.to_csv(C.TABLES_DIR / "diagnosis_ablation.csv", index=False)

    folds, oof = folds_by_policy[list(POLICIES)[-1]]
    preds = {v: (oof[f"p_{v}"] >= 0.5).astype(int).values for v in folds["variant"].unique()}
    comp = complementarity(y, preds)
    comp.to_csv(C.TABLES_DIR / "diagnosis_complementarity.csv", header=["records"])
    oof.to_parquet(C.INTERIM_DIR / "oof_diagnosis.parquet")
    return {"ablation": ablation, "folds_by_policy": folds_by_policy, "complementarity": comp, "y": y}


# ------------------------------------------------------------------ tuning
def tune_and_freeze() -> dict:
    """Tune call_only / message_only / combined_early by CV with the same grid and procedure,
    compare early vs stacked fusion by CV, apply the declared choice rule, freeze the config."""
    _need_v2()
    C.ensure_dirs()
    X, y = dev_features()
    per_variant, grids = {}, {}
    for v in ("call_only", "message_only", "combined_early"):
        res = cv_grid_search(X, y, v)
        res.to_csv(C.TABLES_DIR / f"tuning_grid_{v}.csv", index=False)
        grids[v] = res
        per_variant[v] = best_params(res)
    params = {"per_variant": per_variant}
    folds, oof = cross_validate_variants(X, y, params)
    summary = summarize_cv(folds)
    choice = choose_combined(summary)
    folds.to_csv(C.TABLES_DIR / "cv_folds_tuned.csv", index=False)
    summary.to_csv(C.TABLES_DIR / "cv_summary_tuned.csv")
    oof.to_parquet(C.INTERIM_DIR / "oof_tuned.parquet")
    pd.DataFrame([{"a": a, "b": b, **paired_fold_differences(folds, a, b)}
                  for a, b in (("combined_early", "call_only"), ("combined_stacked", "call_only"),
                               ("combined_early", "message_only"), ("combined_stacked", "combined_early"))]
                 ).to_csv(C.TABLES_DIR / "cv_paired_differences.csv", index=False)
    cfg = freeze_v2_config(per_variant, choice, summary)
    models = train_all(X, y, params)          # development models (train+val), for importance charts only
    save_models(models, "tuned_models")
    feature_importance(models).to_csv(C.TABLES_DIR / "feature_importance.csv", index=False)
    return {"config": cfg, "summary": summary, "folds": folds, "oof": oof, "choice": choice,
            "grids": grids, "models": models, "y": y}


def learning_curves() -> pd.DataFrame:
    _need_v2()
    X, y = dev_features()
    lc = learning_curve(X, y, load_final_config()["params"])
    lc.to_csv(C.TABLES_DIR / "learning_curve.csv", index=False)
    return lc


# ------------------------------------------------------------------ final test
def final_test() -> dict:
    _need_v2()
    from .evaluation import load_final_results, run_final_test
    return load_final_results() if C.FINAL_TEST_LOCK.exists() else run_final_test()
