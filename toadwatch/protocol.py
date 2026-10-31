"""How accuracy is measured here (three models, one protocol).

* Development data = train + validation buckets, grouped into 4 CV folds. A fold never contains part of a family of
  near-identical scripts that is also in another fold, so cross-validation cannot reward memorised templates.
* Two Random Forests (call, message) are trained on three folds and score the fourth; the three models are read from
  those out-of-fold scores (see models.py). Forest settings and decision thresholds are chosen on out-of-fold results only.
* Sanity checks that MUST come out at chance (label shuffle, source-family shortcut baselines inside the matched tier), ceilings
  (what a perfect one-channel model could reach) and a label audit make leakage and bad labels visible.
* Results are always reported per TIER: `hard` and `matched` records cannot be solved by recognising a dataset, `unmatched`
  records can, so they are optimistic.
* The test buckets are used once (run_final_test). Intervals resample whole call families (cluster bootstrap).
"""
from __future__ import annotations

import datetime as dt
import itertools
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

from . import config as C
from .features import build_features, channel_columns
from .models import FraudDetectors, decide, make_forest, model_scores, save_detectors
from .splitting import load_split, verify_test_untouched
from .stats import mcnemar_exact

PAIR_TYPES = ("FF", "FS", "SF", "SS")


# ------------------------------------------------------------------ development table
def dev_table(use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """(records, features, fold ids) for train + validation. The test split is never loaded here."""
    recs = pd.concat([load_split("train"), load_split("val")], ignore_index=True)
    cache = C.INTERIM_DIR / "features_dev.parquet"
    if use_cache and cache.exists() and len(pd.read_parquet(cache)) == len(recs):
        X = pd.read_parquet(cache)
    else:
        X = build_features(recs)
        C.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        X.to_parquet(cache)
    return recs, X, recs["fold"].to_numpy()


def ceilings(recs: pd.DataFrame) -> dict:
    """Best accuracy of a model that sees ONE channel perfectly (it knows that half's true label).
    Under the either-is-fraud rule a perfect call model still misses every attack whose fraud is only in the message."""
    y = recs["label"].to_numpy()
    return {"majority_class": float(max(y.mean(), 1 - y.mean())),
            "call_oracle": float((recs["call_label"].to_numpy() == y).mean()),
            "message_oracle": float((recs["msg_label"].to_numpy() == y).mean()),
            "combined_oracle": float(((recs["call_label"] | recs["msg_label"]).to_numpy() == y).mean())}


# ------------------------------------------------------------------ out-of-fold scores
def oof_scores(X: pd.DataFrame, recs: pd.DataFrame, params: dict | None, folds: np.ndarray, shuffle_labels: bool = False,
               seed: int = C.SEED) -> pd.DataFrame:
    """Leave-one-fold-out scores of the two forests for every development record (nobody scores a record it trained on)."""
    recs, X = recs.reset_index(drop=True), X.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(index=X.index, columns=["p_call", "p_msg"], dtype=float)
    for f in sorted(set(folds)):
        tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        y_c, y_m = recs["call_label"].to_numpy()[tr], recs["msg_label"].to_numpy()[tr]
        if shuffle_labels:
            y_c, y_m = rng.permutation(y_c), rng.permutation(y_m)
        det = FraudDetectors(params).fit(X.iloc[tr], y_c, y_m)
        out.iloc[te] = det.scores(X.iloc[te]).to_numpy()
    return out


# ------------------------------------------------------------------ metrics
def attack_metrics(y, pred, score=None) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {"n": int(len(y)), "correct": int(tn + tp), "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
         "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
         "balanced_accuracy": balanced_accuracy_score(y, pred), "mcc": matthews_corrcoef(y, pred),
         "false_alarm_rate": fp / (fp + tn) if (fp + tn) else 0.0, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    if score is not None and len(np.unique(y)) == 2:
        m["roc_auc"] = roc_auc_score(y, score)
    return m


def evaluate(recs: pd.DataFrame, scores: pd.DataFrame, thresholds: dict) -> dict:
    """Everything reported for a set of records: the 3 models overall, by tier, by pair type, plus each detector on its own text."""
    recs = recs.reset_index(drop=True)
    scores = scores.reset_index(drop=True)
    y = recs["label"].to_numpy()
    preds, cont = decide(scores, thresholds), model_scores(scores, thresholds)
    overall = pd.DataFrame({m: attack_metrics(y, preds[m], cont[m]) for m in C.MODELS}).T
    tier_rows, pair_rows = [], []
    for m in C.MODELS:
        for tier, idx in recs.groupby("tier").groups.items():
            idx = np.asarray(list(idx))
            if len(np.unique(y[idx])) == 2:
                mm = attack_metrics(y[idx], preds[m][idx])
                tier_rows.append({"model": m, "tier": tier, "n": mm["n"], "accuracy": mm["accuracy"], "f1": mm["f1"],
                                  "balanced_accuracy": mm["balanced_accuracy"], "recall": mm["recall"], "false_alarm_rate": mm["false_alarm_rate"]})
        for pt in PAIR_TYPES:
            idx = np.flatnonzero((recs["pair_type"] == pt).to_numpy())
            if len(idx):
                pair_rows.append({"model": m, "pair_type": pt, "n": len(idx), "accuracy": float((preds[m][idx] == y[idx]).mean())})
    chan_rows = []
    for ch, p, lab, t in (("call detector", "p_call", "call_label", thresholds["call_only"]), ("message detector", "p_msg", "msg_label", thresholds["message_only"])):
        yl, pl = recs[lab].to_numpy(), (scores[p].to_numpy() >= t).astype(int)
        for tier, idx in list(recs.groupby("tier").groups.items()) + [("all", recs.index)]:
            idx = np.asarray(list(idx))
            if len(np.unique(yl[idx])) == 2:
                mm = attack_metrics(yl[idx], pl[idx], scores[p].to_numpy()[idx])
                chan_rows.append({"detector": ch, "tier": tier, "n": mm["n"], "accuracy": mm["accuracy"], "balanced_accuracy": mm["balanced_accuracy"],
                                  "f1": mm["f1"], "precision": mm["precision"], "recall": mm["recall"], "roc_auc": mm.get("roc_auc", np.nan)})
    return {"overall": overall, "by_tier": pd.DataFrame(tier_rows), "by_pair_type": pd.DataFrame(pair_rows),
            "channel": pd.DataFrame(chan_rows), "preds": preds, "model_scores": cont}


# ------------------------------------------------------------------ tuning
def _tier_balanced_accuracy(recs: pd.DataFrame, p: np.ndarray, label_col: str) -> float:
    """Balanced accuracy of one forest on its own text, averaged over the tiers (so the honest tiers count as much as the easy one)."""
    vals = []
    for _, idx in recs.groupby("tier").groups.items():
        idx = np.asarray(list(idx))
        yl = recs[label_col].to_numpy()[idx]
        if len(np.unique(yl)) == 2:
            vals.append(balanced_accuracy_score(yl, (p[idx] >= 0.5).astype(int)))
    return float(np.mean(vals))


def tune_forests(X: pd.DataFrame, recs: pd.DataFrame, folds: np.ndarray, grid: dict | None = None) -> tuple[dict, dict]:
    """Same grid for both forests; each keeps the settings with the best tier-averaged balanced accuracy on out-of-fold data."""
    grid = grid or C.TUNING_GRID
    keys = list(grid)
    recs, X = recs.reset_index(drop=True), X.reset_index(drop=True)
    best, tables = {}, {}
    for ch, label_col in (("call", "call_label"), ("message", "msg_label")):
        cols, y, rows = channel_columns(X.columns, ch), recs[label_col].to_numpy(), []
        for values in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, values))
            p = np.zeros(len(X))
            for f in sorted(set(folds)):
                tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
                rf = make_forest(params).fit(X.iloc[tr][cols], y[tr])
                p[te] = rf.predict_proba(X.iloc[te][cols])[:, list(rf.classes_).index(1)]
            rows.append({**params, "tier_balanced_accuracy": _tier_balanced_accuracy(recs, p, label_col),
                         "overall_accuracy": float(((p >= 0.5).astype(int) == y).mean())})
        t = pd.DataFrame(rows).sort_values("tier_balanced_accuracy", ascending=False).reset_index(drop=True)
        tables[ch] = t
        best[ch] = {k: (None if pd.isna(t.iloc[0][k]) else (int(t.iloc[0][k]) if k in ("n_estimators", "max_depth", "min_samples_leaf") else
                                                              (float(t.iloc[0][k]) if isinstance(t.iloc[0][k], (float, np.floating)) else t.iloc[0][k])))
                    for k in keys}
    return best, tables


def tune_thresholds(scores: pd.DataFrame, recs: pd.DataFrame, grid=None) -> dict:
    """Decision thresholds that maximise each model's attack-level F1 on the out-of-fold scores (ties -> closest to 0.5)."""
    grid = grid or C.THRESHOLD_GRID
    y = recs["label"].to_numpy()
    p_c, p_m = scores["p_call"].to_numpy(), scores["p_msg"].to_numpy()
    f1 = lambda pred: f1_score(y, pred, zero_division=0)
    pick = lambda cands: max(cands, key=lambda c: (round(c[1], 6), -abs(np.mean(c[0]) - 0.5)))[0]
    t_call = pick([(t, f1((p_c >= t).astype(int))) for t in grid])
    t_msg = pick([(t, f1((p_m >= t).astype(int))) for t in grid])
    both = pick([((tc, tm), f1(((p_c >= tc) | (p_m >= tm)).astype(int))) for tc in grid for tm in grid])
    return {"call_only": float(t_call), "message_only": float(t_msg), "combined": [float(both[0]), float(both[1])]}


# ------------------------------------------------------------------ audits (all use the same two forests; no extra model)
def _source_family(s: str) -> str:
    s = re.sub(r"_(0|1)$", "", s)
    return {"sms_ham": "sms_uci", "sms_spam": "sms_uci", "augmented_ham_en_es_es": "sms_uci_es", "augmented_spam_en_es_es": "sms_uci_es",
            "spaphish_legitimate": "spaphish", "spaphish_phishing": "spaphish"}.get(s, s)


def shortcut_baselines(recs: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    """Predict the label from metadata only, never reading a text. In the matched and hard tiers every baseline must sit near
    the majority-class rate (nothing but the CONTENT can give the answer); in the unmatched tier a source gives it away."""
    recs = recs.reset_index(drop=True)
    recs = recs.assign(call_family=recs["call_source"].map(_source_family), msg_family=recs["msg_source"].map(_source_family))
    y = recs["label"].to_numpy()
    sets = {"message type only": ["msg_kind"], "message language only": ["msg_lang"], "tier only": ["tier"],
            "call source family only": ["call_family"], "message source family only": ["msg_family"],
            "call + message source families": ["call_family", "msg_family"]}
    rows = []
    for name, cols in sets.items():
        pred = np.zeros(len(recs), int)
        key = recs[cols].astype(str).agg("|".join, axis=1)
        for f in sorted(set(folds)):
            tr, te = folds != f, folds == f
            rate = pd.Series(y[tr]).groupby(key[tr].to_numpy()).mean()
            pred[te] = (key[te].map(rate).fillna(y[tr].mean()).to_numpy() >= 0.5).astype(int)
        row = {"baseline": name, "accuracy_all": accuracy_score(y, pred)}
        for tier in C.TIERS:
            m = (recs["tier"] == tier).to_numpy()
            row[f"accuracy_{tier}"] = accuracy_score(y[m], pred[m]) if m.sum() else np.nan
        rows.append(row)
    maj = {f"majority_{t}": float(max(y[(recs.tier == t).to_numpy()].mean(), 1 - y[(recs.tier == t).to_numpy()].mean())) if (recs.tier == t).any() else np.nan for t in C.TIERS}
    return pd.DataFrame(rows).assign(**{k: v for k, v in maj.items()})


def overlap_checks(recs_all: pd.DataFrame) -> pd.DataFrame:
    """Ids and families shared between the test buckets and everything else, and between development folds (all must be 0)."""
    t, d = recs_all[recs_all.split == "test"], recs_all[recs_all.split != "test"]
    chk = {"call ids shared test/dev": set(t.call_id) & set(d.call_id), "message ids shared test/dev": set(t.item_id) & set(d.item_id),
           "call families shared test/dev": set(t.call_group) & set(d.call_group), "message families shared test/dev": set(t.msg_group) & set(d.msg_group)}
    for a, b in ((0, 1), (0, 2), (1, 3)):
        da, db = recs_all[recs_all.fold == a], recs_all[recs_all.fold == b]
        chk[f"call families shared fold {a}/{b}"] = set(da.call_group) & set(db.call_group)
        chk[f"message families shared fold {a}/{b}"] = set(da.msg_group) & set(db.msg_group)
    return pd.DataFrame([{"check": k, "shared": len(v), "ok": len(v) == 0} for k, v in chk.items()])


def near_duplicate_leak(recs_dev: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Share of each fold's texts that has a near-copy (cosine > threshold) in the OTHER development folds."""
    rows = []
    for col, name in (("call_text", "call"), ("msg_text", "message")):
        M = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, min_df=2, max_features=60000).fit_transform(recs_dev[col])
        for f in sorted(recs_dev.fold.unique()):
            te, tr = np.flatnonzero(recs_dev.fold == f), np.flatnonzero(recs_dev.fold != f)
            best = np.asarray((M[te] @ M[tr].T).max(axis=1).todense()).ravel()
            rows.append({"channel": name, "fold": int(f), "share_with_near_copy": float((best > threshold).mean())})
    return pd.DataFrame(rows)


def label_audit(recs_dev: pd.DataFrame, scores: pd.DataFrame, top: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Texts whose given label the held-out forest contradicts most confidently - candidates for wrong labels (a list for a human)."""
    recs_dev, scores = recs_dev.reset_index(drop=True), scores.reset_index(drop=True)
    parts = []
    for ch, p, lab, src, txt, idc in (("call", "p_call", "call_label", "call_source", "call_text", "call_id"),
                                      ("message", "p_msg", "msg_label", "msg_source", "msg_text", "item_id")):
        pg = np.where(recs_dev[lab] == 1, scores[p], 1 - scores[p])
        parts.append(pd.DataFrame({"channel": ch, "id": recs_dev[idc], "source": recs_dev[src], "tier": recs_dev["tier"], "given_label": recs_dev[lab],
                                   "prob_of_given_label": pg, "text_start": recs_dev[txt].str.slice(0, 240)}))
    allc = pd.concat(parts, ignore_index=True)
    summary = (allc.assign(contested=allc.prob_of_given_label < 0.1).groupby(["channel", "source"])
               .agg(items=("id", "size"), contested=("contested", "sum")).reset_index())
    summary["contested_share"] = summary["contested"] / summary["items"]
    return allc.sort_values("prob_of_given_label").head(top).reset_index(drop=True), summary.sort_values("contested_share", ascending=False)


def shuffle_check(X, recs, folds, params) -> pd.DataFrame:
    """Train the same forests on SHUFFLED labels: every model must fall to the majority-class rate (anything higher = leakage)."""
    sc = oof_scores(X, recs, params, folds, shuffle_labels=True)
    ev = evaluate(recs, sc, {"call_only": 0.5, "message_only": 0.5, "combined": [0.5, 0.5]})
    out = ev["overall"][["accuracy", "balanced_accuracy", "f1", "roc_auc"]].astype(float)
    out["majority_class_rate"] = ceilings(recs)["majority_class"]
    return out


# ------------------------------------------------------------------ cluster bootstrap (resample whole families)
def cluster_bootstrap(y, pred, groups, metric: str = "accuracy", n: int = C.N_BOOTSTRAP, seed: int = C.SEED) -> tuple[float, float, float]:
    y, pred = np.asarray(y), np.asarray(pred)
    gid, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(gid))]
    fn = {"accuracy": accuracy_score, "f1": f1_score, "balanced_accuracy": balanced_accuracy_score}[metric]
    rng = np.random.default_rng(seed)
    vals = [fn(y[idx], pred[idx]) for idx in (np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))]) for _ in range(n))]
    return float(fn(y, pred)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def cluster_bootstrap_diff(y, pa, pb, groups, n: int = C.N_BOOTSTRAP, seed: int = C.SEED) -> dict:
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    gid, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(gid))]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        d.append(accuracy_score(y[idx], pa[idx]) - accuracy_score(y[idx], pb[idx]))
    d = np.array(d)
    return {"diff": float(accuracy_score(y, pa) - accuracy_score(y, pb)), "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975)),
            "p_value": float(min(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0))}


# ------------------------------------------------------------------ final test
def run_final_test(force: bool = False) -> dict:
    """Train the two forests on all development data with the frozen settings and test ONCE."""
    import json
    if C.FINAL_TEST_LOCK.exists() and not force:
        raise RuntimeError(f"The final test already ran ({C.FINAL_TEST_LOCK.read_text().strip()}). It is allowed once.")
    cfg = json.loads(C.FINAL_CONFIG_JSON.read_text())
    if not verify_test_untouched():
        raise RuntimeError("records_test.parquet changed since the split was made - results would not be valid.")
    recs, X, _ = dev_table()
    det = FraudDetectors(cfg["params"]).fit(X, recs["call_label"], recs["msg_label"])
    test = load_split("test", allow_test=True).reset_index(drop=True)
    sc = det.scores(build_features(test))
    ev = evaluate(test, sc, cfg["thresholds"])
    y, preds = test["label"].to_numpy(), ev["preds"]

    rows = []
    for m in C.MODELS:
        _, a_lo, a_hi = cluster_bootstrap(y, preds[m], test["call_group"], "accuracy")
        _, f_lo, f_hi = cluster_bootstrap(y, preds[m], test["call_group"], "f1")
        r = ev["overall"].loc[m].to_dict()
        bt = binomtest(int(r["correct"]), int(r["n"]), max(y.mean(), 1 - y.mean()), alternative="greater")
        rows.append({"model": m, **r, "acc_ci_low": a_lo, "acc_ci_high": a_hi, "f1_ci_low": f_lo, "f1_ci_high": f_hi, "p_better_than_majority_class": bt.pvalue})
    results = pd.DataFrame(rows).set_index("model")
    tier_rows = []
    for _, r in ev["by_tier"].iterrows():
        msk = (test["tier"] == r["tier"]).to_numpy()
        _, lo, hi = cluster_bootstrap(y[msk], preds[r["model"]][msk], test["call_group"].to_numpy()[msk], "accuracy")
        tier_rows.append({**r.to_dict(), "acc_ci_low": lo, "acc_ci_high": hi})
    by_tier = pd.DataFrame(tier_rows)
    pw = []
    for a, b in (("combined", "call_only"), ("combined", "message_only")):
        d = cluster_bootstrap_diff(y, preds[a], preds[b], test["call_group"])
        mc = mcnemar_exact(y, preds[a], preds[b])
        pw.append({"model_a": a, "model_b": b, "acc_diff_a_minus_b": d["diff"], "ci_low": d["ci_low"], "ci_high": d["ci_high"], "bootstrap_p": d["p_value"],
                   "a_only_right": mc["a_only_right"], "b_only_right": mc["b_only_right"], "mcnemar_p": mc["p_value"]})
    out = {"results": results, "by_tier": by_tier, "by_pair_type": ev["by_pair_type"], "channel": ev["channel"], "pairwise": pd.DataFrame(pw),
           "ceilings": ceilings(test), "config": cfg,
           "predictions": test.assign(**{f"pred_{m}": p for m, p in preds.items()}, **{f"score_{m}": s for m, s in ev["model_scores"].items()},
                                       p_call=sc["p_call"].values, p_msg=sc["p_msg"].values)}
    C.ensure_dirs()
    T = C.TABLES_DIR
    results.to_csv(T / "final_test_results.csv")
    by_tier.to_csv(T / "final_test_by_tier.csv", index=False)
    ev["by_pair_type"].to_csv(T / "final_test_by_pair_type.csv", index=False)
    ev["channel"].to_csv(T / "final_test_detectors.csv", index=False)
    out["pairwise"].to_csv(T / "final_test_pairwise.csv", index=False)
    pd.Series(out["ceilings"]).to_csv(T / "final_test_ceilings.csv", header=["accuracy"])
    out["predictions"].to_parquet(T / "final_test_predictions.parquet", index=False)
    save_detectors(det, "final_detectors")
    C.FINAL_TEST_LOCK.write_text(f"ran at {dt.datetime.now().isoformat(timespec='seconds')}\n")
    return out


def load_final_results() -> dict:
    import json
    if not C.FINAL_TEST_LOCK.exists():
        raise FileNotFoundError("The final test has not run yet.")
    T = C.TABLES_DIR
    return {"results": pd.read_csv(T / "final_test_results.csv", index_col="model"), "by_tier": pd.read_csv(T / "final_test_by_tier.csv"),
            "by_pair_type": pd.read_csv(T / "final_test_by_pair_type.csv"), "channel": pd.read_csv(T / "final_test_detectors.csv"),
            "pairwise": pd.read_csv(T / "final_test_pairwise.csv"), "ceilings": pd.read_csv(T / "final_test_ceilings.csv", index_col=0)["accuracy"].to_dict(),
            "predictions": pd.read_parquet(T / "final_test_predictions.parquet"), "config": json.loads(C.FINAL_CONFIG_JSON.read_text())}
