"""v3 evaluation protocol: how "true accuracy" is measured here.

* Development (train + validation) = buckets 0-7, grouped into 4 CV folds; the folds never split a family of
  near-identical scripts, so cross-validation cannot reward memorised templates.
* Every model version is trained on 3 folds and scored on the 4th (leave-one-fold-out).
* Sanity checks that MUST come out at chance level (label shuffle, shortcut baselines) and a ceiling check
  (a single channel cannot beat the pair-type ceiling) make leakage visible instead of assumed away.
* A label audit lists the items whose given label a held-out model contradicts most confidently, for a human to check.
* The test buckets are used once (final_test); confidence intervals resample whole call families (cluster
  bootstrap), because records of one family are not independent.
"""
from __future__ import annotations

import datetime as dt
import itertools

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline

from . import config as C
from .features import build_features
from .modeling import compute_metrics, make_model, predict_variant, proba_variant, save_models, train_all, train_variant
from .splitting import load_split, verify_test_untouched
from .stats import mcnemar_exact
from .text_stack import TEXT_CALL, TEXT_MSG
from .tuning import load_final_config

PAIR_TYPES = ("FF", "FS", "SF", "SS")


# ------------------------------------------------------------------ data access
def with_text(X: pd.DataFrame, recs: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X[TEXT_CALL], X[TEXT_MSG] = recs["call_text"].values, recs["msg_text"].values
    return X


def dev_table(use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, np.ndarray]:
    """(dev records, features + raw text, labels, fold id) for train + validation. The test split is not loaded."""
    recs = pd.concat([load_split("train"), load_split("val")], ignore_index=True)
    cache = C.INTERIM_DIR / "features_dev.parquet"
    if use_cache and cache.exists() and len(pd.read_parquet(cache)) == len(recs):
        feats = pd.read_parquet(cache)
    else:
        feats = build_features(recs)
        C.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(cache)
    return recs, with_text(feats, recs), recs["label"].astype(int), recs["fold"].to_numpy()


def ceilings(recs: pd.DataFrame) -> dict:
    """Best possible accuracy of a model that sees ONE channel perfectly (it knows that channel's true label)."""
    y = recs["label"].to_numpy()
    return {"majority_class": float(max(y.mean(), 1 - y.mean())),
            "call_oracle": float((recs["call_label"].to_numpy() == y).mean()),
            "message_oracle": float((recs["msg_label"].to_numpy() == y).mean()),
            "both_channels_oracle": float(((recs["call_label"] & recs["msg_label"]).to_numpy() == y).mean())}


# ------------------------------------------------------------------ metrics
def full_metrics(y, pred, proba=None) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    m = compute_metrics(y, pred)
    m["balanced_accuracy"] = balanced_accuracy_score(y, pred)
    m["mcc"] = matthews_corrcoef(y, pred)
    if proba is not None and len(np.unique(y)) == 2:
        m["roc_auc"] = roc_auc_score(y, proba)
    return m


def regime_columns(recs: pd.DataFrame, y, pred) -> dict:
    """Accuracy / F1 separately for the two regimes. 'public' is optimistic (a source's style reveals its class);
    'hard' is style-matched, so only the content can give the answer."""
    y, pred, out = np.asarray(y), np.asarray(pred), {}
    for reg in ("public", "hard"):
        m = (recs["regime"] == reg).to_numpy()
        if m.sum() and len(np.unique(y[m])) == 2:
            out[f"{reg}_accuracy"] = float(accuracy_score(y[m], pred[m]))
            out[f"{reg}_f1"] = float(f1_score(y[m], pred[m], zero_division=0))
            out[f"{reg}_balanced_accuracy"] = float(balanced_accuracy_score(y[m], pred[m]))
            out[f"{reg}_n"] = int(m.sum())
    return out


def accuracy_by(recs: pd.DataFrame, pred, col: str) -> pd.Series:
    ok = pd.Series(np.asarray(pred) == recs["label"].to_numpy(), index=recs.index)
    return ok.groupby(recs[col]).mean()


def weights_for(recs: pd.DataFrame, hard_weight: float) -> np.ndarray:
    """Training weights: records of the style-matched 'hard' regime count ``hard_weight`` times as much.
    Reason: in the public regime a source's writing style reveals its class, so models drift towards
    recognising datasets; the hard regime is where the content has to be read."""
    return np.where(recs["regime"].to_numpy() == "hard", float(hard_weight), 1.0)


# ------------------------------------------------------------------ cross-validation (family-safe folds)
def cross_validate(X: pd.DataFrame, y: pd.Series, folds: np.ndarray, recs: pd.DataFrame, params: dict | None = None,
                   variants=None, shuffle_labels: bool = False, seed: int = C.SEED, hard_weight: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-fold-out. Returns (one row per fold x version, out-of-fold probabilities)."""
    variants = variants or C.VARIANTS
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    recs = recs.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    w_all = weights_for(recs, hard_weight)
    rows, oof = [], pd.DataFrame({"y": y, "fold": folds, "pair_type": recs["pair_type"].values})
    for f in sorted(set(folds)):
        tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        y_tr = pd.Series(rng.permutation(y.iloc[tr].to_numpy()), index=y.index[tr]) if shuffle_labels else y.iloc[tr]
        for v in variants:
            mc = train_variant(v, X.iloc[tr], y_tr, params, sample_weight=w_all[tr])
            pred, proba = predict_variant(mc, X.iloc[te]), proba_variant(mc, X.iloc[te])
            rows.append({"fold": f, "variant": v, **full_metrics(y.iloc[te], pred, proba), **regime_columns(recs.iloc[te], y.iloc[te], pred),
                         **{f"acc_{t}": a for t, a in accuracy_by(recs.iloc[te], pred, "pair_type").items()}})
            oof.loc[te, f"p_{v}"] = proba
        print(f"  fold {f} done", flush=True)
    return pd.DataFrame(rows), oof


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    cols = (["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "public_accuracy", "public_f1", "hard_accuracy", "hard_f1",
             "hard_balanced_accuracy"] + [c for c in folds if c.startswith("acc_")])
    use = [c for c in cols if c in folds]
    out = (folds.groupby("variant")[use].mean().add_suffix("_mean")
           .join(folds.groupby("variant")[["accuracy", "f1"]].std().add_suffix("_std")))
    return out.reindex([v for v in C.VARIANTS if v in out.index])


def cv_grid_search(X, y, folds, recs, variant: str, grid: dict | None = None, hard_weight: float = 1.0) -> pd.DataFrame:
    grid = grid or C.TUNING_GRID
    keys = list(grid)
    y = y.reset_index(drop=True); X = X.reset_index(drop=True)
    w_all = weights_for(recs.reset_index(drop=True), hard_weight)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        f1s = []
        for f in sorted(set(folds)):
            tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            mc = train_variant(variant, X.iloc[tr], y.iloc[tr], params, sample_weight=w_all[tr])
            f1s.append(f1_score(y.iloc[te], predict_variant(mc, X.iloc[te])))
        rows.append({**params, "cv_f1_mean": float(np.mean(f1s)), "cv_f1_std": float(np.std(f1s))})
    return pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ sanity checks that must come out at chance
def shortcut_baselines(recs: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    """Predict the label from metadata only. Message type / language / hard-vs-public flags must stay near the
    majority-class rate; a call SOURCE identifies the call's class, so it is capped by the call ceiling."""
    y = recs["label"].to_numpy()
    rows = []
    sets = {"message type only": ["msg_kind"], "message language only": ["msg_lang"],
            "hard-vs-public flag only": ["regime"],
            "call source only": ["call_source"], "message source only": ["msg_source"],
            "call source + message source": ["call_source", "msg_source"]}
    for name, cols in sets.items():
        pred = np.zeros(len(recs), int)
        key = recs[cols].astype(str).agg("|".join, axis=1)
        for f in sorted(set(folds)):
            tr, te = folds != f, folds == f
            rate = pd.Series(y[tr]).groupby(key[tr].to_numpy()).mean()
            pred[te] = (key[te].map(rate).fillna(y[tr].mean()).to_numpy() >= 0.5).astype(int)
        hard = (recs["regime"] == "hard").to_numpy()
        rows.append({"baseline": name, "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
                     "f1": f1_score(y, pred, zero_division=0),
                     "accuracy_on_hard_regime": accuracy_score(y[hard], pred[hard]), "accuracy_on_public_regime": accuracy_score(y[~hard], pred[~hard])})
    return pd.DataFrame(rows)


def overlap_checks(recs_all: pd.DataFrame) -> pd.DataFrame:
    """Ids and family ids shared between the test buckets and everything else (all must be 0)."""
    t, d = recs_all[recs_all.split == "test"], recs_all[recs_all.split != "test"]
    chk = {"call ids shared test/dev": set(t.call_id) & set(d.call_id), "message ids shared test/dev": set(t.item_id) & set(d.item_id),
           "call families shared test/dev": set(t.call_group) & set(d.call_group), "message families shared test/dev": set(t.msg_group) & set(d.msg_group)}
    for a, b in ((0, 1), (0, 2), (1, 3)):
        da, db = recs_all[recs_all.fold == a], recs_all[recs_all.fold == b]
        chk[f"call families shared fold {a}/{b}"] = set(da.call_group) & set(db.call_group)
        chk[f"message families shared fold {a}/{b}"] = set(da.msg_group) & set(db.msg_group)
    return pd.DataFrame([{"check": k, "shared": len(v), "ok": len(v) == 0} for k, v in chk.items()])


def near_duplicate_leak(recs_dev: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Share of each fold's texts that have a near-copy (cosine > threshold) in the OTHER development folds."""
    rows = []
    for col, name in (("call_text", "call"), ("msg_text", "message")):
        vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, min_df=2, max_features=60000).fit(recs_dev[col])
        M = vec.transform(recs_dev[col])
        for f in sorted(recs_dev.fold.unique()):
            te, tr = np.flatnonzero(recs_dev.fold == f), np.flatnonzero(recs_dev.fold != f)
            best = np.asarray((M[te] @ M[tr].T).max(axis=1).todense()).ravel()
            rows.append({"channel": name, "fold": int(f), f"share_with_near_copy_(cos>{threshold})": float((best > threshold).mean())})
    return pd.DataFrame(rows)


def label_audit(recs_dev: pd.DataFrame, top: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Items whose GIVEN label a held-out text model contradicts most confidently - candidates for wrong labels.

    One text model per channel predicts the channel's own label (call_label / msg_label); each item is scored by a
    model that never saw its fold. Not an automatic verdict: it is a short list for a human to read.
    """
    out = []
    for col, lab, idc, name in (("call_text", "call_label", "call_id", "call"), ("msg_text", "msg_label", "item_id", "message")):
        d = recs_dev.drop_duplicates(idc).reset_index(drop=True)
        p_given = np.zeros(len(d))
        for f in sorted(d.fold.unique()):
            tr, te = d.fold != f, d.fold == f
            vec = "word" if name == "call" else "char_wb"
            model = make_pipeline(TfidfVectorizer(analyzer=vec, ngram_range=(1, 2) if name == "call" else (2, 5), min_df=3, sublinear_tf=True, max_features=40000),
                                  LogisticRegression(C=10, max_iter=2000, class_weight="balanced", solver="liblinear"))
            model.fit(d.loc[tr, col].str.slice(0, 3000), d.loc[tr, lab])
            p = model.predict_proba(d.loc[te, col].str.slice(0, 3000))[:, 1]
            p_given[te.to_numpy()] = np.where(d.loc[te, lab] == 1, p, 1 - p)
        d["channel"], d["given_label"], d["prob_of_given_label"] = name, d[lab], p_given
        d["source"] = d["call_source"] if name == "call" else d["msg_source"]
        d["text_start"] = d[col].str.replace(r"\s+", " ", regex=True).str.slice(0, 240)
        d["id"] = d[idc]
        out.append(d[["channel", "id", "source", "given_label", "prob_of_given_label", "text_start"]])
    allc = pd.concat(out, ignore_index=True)
    summary = (allc.assign(suspect=allc.prob_of_given_label < 0.1).groupby(["channel", "source"])
               .agg(items=("id", "size"), suspects=("suspect", "sum"), mean_prob_of_given_label=("prob_of_given_label", "mean")).reset_index())
    summary["suspect_share"] = summary.suspects / summary["items"]
    return allc.sort_values("prob_of_given_label").head(top).reset_index(drop=True), summary.sort_values("suspect_share", ascending=False)


# ------------------------------------------------------------------ final test
def cluster_bootstrap(y, pred, groups, metric: str = "accuracy", n: int = C.N_BOOTSTRAP, seed: int = C.SEED) -> tuple[float, float, float]:
    """95 % interval that resamples whole FAMILIES (records of one family are not independent)."""
    y, pred = np.asarray(y), np.asarray(pred)
    gid, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(gid))]
    fn = {"accuracy": accuracy_score, "f1": f1_score, "balanced_accuracy": balanced_accuracy_score}[metric]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[k] for k in pick])
        vals.append(fn(y[idx], pred[idx]))
    return float(fn(y, pred)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def cluster_bootstrap_diff(y, pa, pb, groups, metric: str = "accuracy", n: int = C.N_BOOTSTRAP, seed: int = C.SEED) -> dict:
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    gid, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(gid))]
    fn = {"accuracy": accuracy_score, "f1": f1_score}[metric]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        d.append(fn(y[idx], pa[idx]) - fn(y[idx], pb[idx]))
    d = np.array(d)
    return {"diff": float(fn(y, pa) - fn(y, pb)), "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975)),
            "p_value": float(min(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0))}


def run_final_test(force: bool = False) -> dict:
    """Train on all development buckets with the frozen settings and test ONCE."""
    if C.FINAL_TEST_LOCK.exists() and not force:
        raise RuntimeError(f"v3 final test already ran ({C.FINAL_TEST_LOCK.read_text().strip()}). It is allowed once.")
    cfg = load_final_config()
    if not verify_test_untouched():
        raise RuntimeError("records_test.parquet changed since the split was made - results would not be valid.")
    recs_dev, X_dev, y_dev, _ = dev_table()
    test = load_split("test", allow_test=True).reset_index(drop=True)
    X_test = with_text(build_features(test), test)
    y_test = test["label"].astype(int)
    models = train_all(X_dev, y_dev, cfg["params"], sample_weight=weights_for(recs_dev, cfg.get("hard_weight", 1.0)))

    rows, preds, probas = [], {}, {}
    for v, mc in models.items():
        preds[v], probas[v] = predict_variant(mc, X_test), proba_variant(mc, X_test)
        m = full_metrics(y_test, preds[v], probas[v])
        _, a_lo, a_hi = cluster_bootstrap(y_test, preds[v], test["call_group"], "accuracy")
        _, f_lo, f_hi = cluster_bootstrap(y_test, preds[v], test["call_group"], "f1")
        bt = binomtest(int(m["correct"]), int(m["n"]), max(y_test.mean(), 1 - y_test.mean()), alternative="greater")
        reg = regime_columns(test, y_test, preds[v])
        for r_ in ("hard", "public"):
            msk = (test["regime"] == r_).to_numpy()
            _, lo_, hi_ = cluster_bootstrap(y_test[msk], preds[v][msk], test["call_group"][msk], "accuracy")
            reg[f"{r_}_acc_ci_low"], reg[f"{r_}_acc_ci_high"] = lo_, hi_
        rows.append({"variant": v, **m, **reg, "acc_ci_low": a_lo, "acc_ci_high": a_hi, "f1_ci_low": f_lo, "f1_ci_high": f_hi,
                     "p_better_than_majority_class": bt.pvalue,
                     **{f"acc_{t}": a for t, a in accuracy_by(test, preds[v], "pair_type").items()},
                     })
    results = pd.DataFrame(rows).set_index("variant")
    pairs = [("combined_early", "call_only"), ("combined_early", "message_only"), ("combined_stacked", "call_only"),
             ("combined_text", "call_only_text"), ("combined_text", "message_only_text"), ("combined_text", "combined_early")]
    pw = []
    for a, b in pairs:
        if a in preds and b in preds:
            d = cluster_bootstrap_diff(y_test, preds[a], preds[b], test["call_group"], "accuracy")
            mc_ = mcnemar_exact(y_test, preds[a], preds[b])
            pw.append({"model_a": a, "model_b": b, "acc_diff_a_minus_b": d["diff"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                       "bootstrap_p": d["p_value"], "a_only_right": mc_["a_only_right"], "b_only_right": mc_["b_only_right"], "mcnemar_p": mc_["p_value"]})
    out = {"results": results, "pairwise": pd.DataFrame(pw), "ceilings": ceilings(test),
           "predictions": test.assign(**{f"pred_{v}": p for v, p in preds.items()}, **{f"p_{v}": p for v, p in probas.items()}), "config": cfg}
    C.ensure_dirs()
    results.to_csv(C.TABLES_DIR / "final_test_results.csv")
    out["pairwise"].to_csv(C.TABLES_DIR / "final_test_pairwise.csv", index=False)
    pd.Series(out["ceilings"]).to_csv(C.TABLES_DIR / "final_test_ceilings.csv", header=["accuracy"])
    out["predictions"].to_parquet(C.TABLES_DIR / "final_test_predictions.parquet", index=False)
    save_models(models, "final_models")
    C.FINAL_TEST_LOCK.write_text(f"ran at {dt.datetime.now().isoformat(timespec='seconds')}\n")
    return out


def load_final_results() -> dict:
    if not C.FINAL_TEST_LOCK.exists():
        raise FileNotFoundError("The v3 final test has not run yet.")
    T = C.TABLES_DIR
    return {"results": pd.read_csv(T / "final_test_results.csv", index_col="variant"), "pairwise": pd.read_csv(T / "final_test_pairwise.csv"),
            "ceilings": pd.read_csv(T / "final_test_ceilings.csv", index_col=0)["accuracy"].to_dict(),
            "predictions": pd.read_parquet(T / "final_test_predictions.parquet"), "config": load_final_config()}
