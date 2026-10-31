"""v3 workflow: one function per stage, shared by the notebooks and ``scripts/run_v3.py``.

    prepare        load, clean, cluster into families, bucket, build mixed pairs, lock the test buckets
    audit          ceilings, overlap / near-duplicate leak checks, shortcut baselines, label audit
    tune_and_cv    per-version tuning + family-safe cross-validation of all 7 versions, freeze the settings
    shuffle_check  the same CV with shuffled labels (must collapse to chance)
    final_test     the one-shot test

Select the version BEFORE importing toadwatch:  os.environ["TOADWATCH_VERSION"] = "v3"
"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C
from . import data_v3, protocol_v3 as P
from .cv import best_params
from .modeling import feature_importance, save_models, train_all


def _need_v3():
    if C.VERSION != "v3":
        raise RuntimeError('Set os.environ["TOADWATCH_VERSION"] = "v3" before importing toadwatch.')


def prepare() -> dict:
    _need_v3()
    return data_v3.prepare_data(save=True)


def audit() -> dict:
    _need_v3()
    C.ensure_dirs()
    recs_dev, _, _, folds = P.dev_table()
    recs_all = pd.read_parquet(C.PROCESSED_DIR / "records_all.parquet")
    out = {"ceilings": pd.Series(P.ceilings(recs_dev), name="accuracy"),
           "overlap": P.overlap_checks(recs_all), "near_dup": P.near_duplicate_leak(recs_dev),
           "shortcuts": P.shortcut_baselines(recs_dev, folds)}
    out["suspects"], out["suspect_summary"] = P.label_audit(recs_dev)
    out["ceilings"].to_csv(C.TABLES_DIR / "audit_ceilings.csv", header=True)
    out["overlap"].to_csv(C.TABLES_DIR / "audit_overlap.csv", index=False)
    out["near_dup"].to_csv(C.TABLES_DIR / "audit_near_duplicates.csv", index=False)
    out["shortcuts"].to_csv(C.TABLES_DIR / "audit_shortcut_baselines.csv", index=False)
    out["suspects"].to_csv(C.TABLES_DIR / "label_audit_suspects_to_review.csv", index=False)
    out["suspect_summary"].to_csv(C.TABLES_DIR / "label_audit_by_source.csv", index=False)
    return out


def tune_and_cv() -> dict:
    """Tune the three Random-Forest versions with the same grid, cross-validate all seven, freeze the settings."""
    _need_v3()
    C.ensure_dirs()
    if C.FINAL_TEST_LOCK.exists():
        raise RuntimeError("The v3 final test already ran; settings can no longer change.")
    recs, X, y, folds = P.dev_table()
    per, grids = {}, {}
    for v in ("call_only", "message_only", "combined_early"):
        g = P.cv_grid_search(X, y, folds, recs, v)
        g.to_csv(C.TABLES_DIR / f"tuning_grid_{v}.csv", index=False)
        grids[v], per[v] = g, best_params(g)
        print(v, "best", per[v], "cv_f1", round(g.cv_f1_mean.iloc[0], 4), flush=True)
    params = {"per_variant": per}
    # How much should the style-matched hard records count in training? Chosen by CV on the development folds with the
    # three Random-Forest versions, by balanced accuracy on the hard regime (ties -> the smaller weight), then used for ALL models.
    rows = []
    for w in C.HARD_WEIGHT_GRID:
        f, _ = P.cross_validate(X, y, folds, recs, params, variants=["call_only", "message_only", "combined_early"], hard_weight=w)
        sm = P.summarize(f)
        rows.append({"hard_weight": w, "hard_balanced_accuracy": sm["hard_balanced_accuracy_mean"].mean(), "hard_accuracy": sm["hard_accuracy_mean"].mean(),
                     "public_accuracy": sm["public_accuracy_mean"].mean(), "overall_accuracy": sm["accuracy_mean"].mean()})
        print("hard_weight", w, {k: round(v, 4) for k, v in rows[-1].items()}, flush=True)
    hw = pd.DataFrame(rows)
    hw.to_csv(C.TABLES_DIR / "tuning_hard_weight.csv", index=False)
    best_w = float(hw.sort_values(["hard_balanced_accuracy", "hard_weight"], ascending=[False, True]).iloc[0]["hard_weight"])
    print("chosen hard_weight:", best_w, flush=True)
    folds_df, oof = P.cross_validate(X, y, folds, recs, params, hard_weight=best_w)
    folds_df.to_csv(C.TABLES_DIR / "cv_folds.csv", index=False)      # saved before anything else can fail
    oof.to_parquet(C.INTERIM_DIR / "oof.parquet")
    summary = P.summarize(folds_df)
    summary.to_csv(C.TABLES_DIR / "cv_summary.csv")
    oof.to_parquet(C.INTERIM_DIR / "oof.parquet")
    cfg = {"params": params, "hard_weight": best_w, "selected_by": "mean F1 of family-safe leave-one-fold-out CV on the development buckets (test never used)",
           "ceilings": P.ceilings(recs), "seed": C.SEED}
    C.FINAL_CONFIG_JSON.write_text(json.dumps(cfg, indent=2, default=str))
    from .protocol_v3 import weights_for
    models = train_all(X, y, params, sample_weight=weights_for(recs, best_w))
    save_models(models, "dev_models")
    feature_importance({k: v for k, v in models.items() if k in ("call_only", "message_only", "combined_early")}).to_csv(
        C.TABLES_DIR / "feature_importance.csv", index=False)
    return {"config": cfg, "summary": summary, "folds": folds_df, "oof": oof, "grids": grids, "hard_weight_table": hw}


def shuffle_check(variants=("call_only", "message_only", "combined_early", "combined_text")) -> pd.DataFrame:
    """Train on SHUFFLED labels: honest models fall to the majority-class rate. Anything higher = leakage."""
    _need_v3()
    recs, X, y, folds = P.dev_table()
    cfg = json.loads(C.FINAL_CONFIG_JSON.read_text())
    f, _ = P.cross_validate(X, y, folds, recs, cfg["params"], variants=list(variants), shuffle_labels=True, hard_weight=cfg.get("hard_weight", 1.0))
    out = P.summarize(f)
    out["majority_class_rate"] = max(y.mean(), 1 - y.mean())
    out.to_csv(C.TABLES_DIR / "audit_label_shuffle.csv")
    return out


def final_test() -> dict:
    _need_v3()
    return P.load_final_results() if C.FINAL_TEST_LOCK.exists() else P.run_final_test()


def load_prepared() -> dict:
    """The saved data of an earlier ``prepare`` (used once the test is locked, when re-preparing is refused)."""
    _need_v3()
    return {"records": pd.read_parquet(C.PROCESSED_DIR / "records_all.parquet"),
            "reports": json.loads((C.TABLES_DIR / "v3_data_report.json").read_text())}


def load_cv_results() -> dict:
    """The saved tuning / cross-validation results of an earlier ``tune_and_cv``."""
    _need_v3()
    T = C.TABLES_DIR
    return {"config": json.loads(C.FINAL_CONFIG_JSON.read_text()), "summary": pd.read_csv(T / "cv_summary.csv", index_col="variant"),
            "folds": pd.read_csv(T / "cv_folds.csv"),
            "grids": {v: pd.read_csv(T / f"tuning_grid_{v}.csv") for v in ("call_only", "message_only", "combined_early")},
            "hard_weight_table": pd.read_csv(T / "tuning_hard_weight.csv"), "oof": pd.read_parquet(C.INTERIM_DIR / "oof.parquet")}
