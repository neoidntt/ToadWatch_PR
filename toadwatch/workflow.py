"""One function per stage, shared by the notebooks and ``scripts/run_study.py``.

    prepare       load, clean, families, buckets, connected call+message pairs, lock the test buckets
    audit         ceilings, overlap and near-copy checks, shortcut baselines, label audit
    tune_and_cv   tune the two Random Forests, tune the thresholds, cross-validate the three models, freeze the settings
    shuffle       the same forests on shuffled labels (must fall to chance)
    final_test    the one-shot test
    pr_cases      the three models on real Puerto Rico cases, when data/external/pr_cases.csv has rows
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from . import dataset_builder, protocol as P
from .models import FraudDetectors, decide, save_detectors


def prepare() -> dict:
    return dataset_builder.prepare_data(save=True)


def load_prepared() -> dict:
    """The saved data of an earlier ``prepare`` (used once the test is locked, when re-preparing is refused)."""
    return {"records": pd.read_parquet(C.PROCESSED_DIR / "records_all.parquet"),
            "reports": json.loads((C.TABLES_DIR / "data_report.json").read_text())}


def audit() -> dict:
    C.ensure_dirs()
    recs, X, folds = P.dev_table()
    cfg = json.loads(C.FINAL_CONFIG_JSON.read_text()) if C.FINAL_CONFIG_JSON.exists() else None
    out = {"ceilings": pd.Series(P.ceilings(recs), name="accuracy"), "overlap": P.overlap_checks(pd.read_parquet(C.PROCESSED_DIR / "records_all.parquet")),
           "near_dup": P.near_duplicate_leak(recs), "shortcuts": P.shortcut_baselines(recs, folds)}
    if cfg:                                            # the label audit needs the out-of-fold forest scores
        oof = pd.read_parquet(C.INTERIM_DIR / "oof_scores.parquet")
        out["suspects"], out["suspect_summary"] = P.label_audit(recs, oof)
        out["suspects"].to_csv(C.TABLES_DIR / "label_audit_suspects_to_review.csv", index=False)
        out["suspect_summary"].to_csv(C.TABLES_DIR / "label_audit_by_source.csv", index=False)
    out["ceilings"].to_csv(C.TABLES_DIR / "audit_ceilings.csv", header=True)
    out["overlap"].to_csv(C.TABLES_DIR / "audit_overlap.csv", index=False)
    out["near_dup"].to_csv(C.TABLES_DIR / "audit_near_duplicates.csv", index=False)
    out["shortcuts"].to_csv(C.TABLES_DIR / "audit_shortcut_baselines.csv", index=False)
    return out


def tune_and_cv() -> dict:
    """Tune, cross-validate and freeze. Nothing here reads the test buckets."""
    C.ensure_dirs()
    if C.FINAL_TEST_LOCK.exists():
        raise RuntimeError("The final test already ran; settings can no longer change.")
    recs, X, folds = P.dev_table()
    best, tables = P.tune_forests(X, recs, folds)
    for ch, t in tables.items():
        t.to_csv(C.TABLES_DIR / f"tuning_grid_{ch}.csv", index=False)
        print(ch, "best settings:", best[ch], "| tier-balanced accuracy", round(float(t.iloc[0]["tier_balanced_accuracy"]), 4), flush=True)
    oof = P.oof_scores(X, recs, best, folds)
    oof.to_parquet(C.INTERIM_DIR / "oof_scores.parquet")                          # saved before anything else can fail
    thresholds = P.tune_thresholds(oof, recs)
    ev = P.evaluate(recs, oof, thresholds)
    rows = []
    for f in sorted(set(folds)):
        m = (folds == f)
        for name, pred in ev["preds"].items():
            mm = P.attack_metrics(recs["label"].to_numpy()[m], pred[m])
            rows.append({"model": name, "fold": int(f), "accuracy": mm["accuracy"], "f1": mm["f1"], "balanced_accuracy": mm["balanced_accuracy"]})
    folds_df = pd.DataFrame(rows)
    ev["overall"].to_csv(C.TABLES_DIR / "cv_overall.csv")
    ev["by_tier"].to_csv(C.TABLES_DIR / "cv_by_tier.csv", index=False)
    ev["by_pair_type"].to_csv(C.TABLES_DIR / "cv_by_pair_type.csv", index=False)
    ev["channel"].to_csv(C.TABLES_DIR / "cv_detectors.csv", index=False)
    folds_df.to_csv(C.TABLES_DIR / "cv_folds.csv", index=False)
    cfg = {"models": list(C.MODELS), "params": best, "thresholds": thresholds, "ceilings": P.ceilings(recs), "seed": C.SEED,
           "selected_by": "forest settings: tier-averaged balanced accuracy of each forest on its own text; thresholds: attack-level F1; "
                          "both from out-of-fold scores of the development data (the test buckets were never used)"}
    C.FINAL_CONFIG_JSON.write_text(json.dumps(cfg, indent=2, default=str))
    det = FraudDetectors(best).fit(X, recs["call_label"], recs["msg_label"])        # development fit: feature importances only
    det.importances().to_csv(C.TABLES_DIR / "feature_importance.csv", index=False)
    return {"config": cfg, "overall": ev["overall"], "by_tier": ev["by_tier"], "by_pair_type": ev["by_pair_type"], "channel": ev["channel"],
            "folds": folds_df, "oof": oof, "grids": tables, "model_scores": ev["model_scores"], "recs": recs}


def load_cv_results() -> dict:
    T = C.TABLES_DIR
    return {"config": json.loads(C.FINAL_CONFIG_JSON.read_text()), "overall": pd.read_csv(T / "cv_overall.csv", index_col=0),
            "by_tier": pd.read_csv(T / "cv_by_tier.csv"), "by_pair_type": pd.read_csv(T / "cv_by_pair_type.csv"), "channel": pd.read_csv(T / "cv_detectors.csv"),
            "folds": pd.read_csv(T / "cv_folds.csv"), "oof": pd.read_parquet(C.INTERIM_DIR / "oof_scores.parquet"),
            "grids": {ch: pd.read_csv(T / f"tuning_grid_{ch}.csv") for ch in ("call", "message")}}


def shuffle() -> pd.DataFrame:
    recs, X, folds = P.dev_table()
    cfg = json.loads(C.FINAL_CONFIG_JSON.read_text())
    out = P.shuffle_check(X, recs, folds, cfg["params"])
    out.to_csv(C.TABLES_DIR / "audit_label_shuffle.csv")
    return out


def final_test() -> dict:
    return P.load_final_results() if C.FINAL_TEST_LOCK.exists() else P.run_final_test()


def pr_cases() -> pd.DataFrame | None:
    """The three models on real Puerto Rico cases (data/external/pr_cases.csv: call_transcript, message_text, label)."""
    from .cleaning import first_url
    from .data_loading import load_pr_cases
    from .features import build_features
    import joblib
    cases = load_pr_cases()
    if cases is None or not (C.MODELS_DIR / "final_detectors.joblib").exists():
        return None
    cfg = json.loads(C.FINAL_CONFIG_JSON.read_text())
    rec = pd.DataFrame({"call_text": cases["call_transcript"], "msg_text": cases["message_text"], "msg_url": cases["message_text"].map(first_url)})
    sc = joblib.load(C.MODELS_DIR / "final_detectors.joblib").scores(build_features(rec))
    y = cases["label"].astype(int).to_numpy()
    rows = [{"model": m, **{k: v for k, v in P.attack_metrics(y, p).items() if k in ("n", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp")}}
            for m, p in decide(sc, cfg["thresholds"]).items()]
    out = pd.DataFrame(rows).set_index("model")
    out.to_csv(C.TABLES_DIR / "puerto_rico_cases.csv")
    return out
