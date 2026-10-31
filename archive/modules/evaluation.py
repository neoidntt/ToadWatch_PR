"""Final test: every model version runs on the held-out test set exactly once.

``run_final_test`` refuses to run when
  * the tuning step has not frozen a configuration (``final_config.json``),
  * the test file no longer matches its fingerprint,
  * or it already ran (``FINAL_TEST_DONE.lock``).

v1 trains on the training split; v2 trains on train + validation (both were only used for
development / cross-validation, never the test set).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
from scipy.stats import binomtest

from . import config as C
from .cleaning import first_url
from .data_loading import load_pr_cases
from .features import build_features
from .modeling import compute_metrics, predict_variant, proba_variant, save_models, train_all
from .splitting import load_split, verify_test_untouched
from .stats import bootstrap_ci, complementarity, mcnemar_exact, paired_bootstrap_diff, probability_scores
from .tuning import load_final_config


def binomial_vs_chance(correct: int, n: int, p: float = 0.5) -> dict:
    """One-sided exact binomial test: is accuracy greater than ``p`` (50 % = random guess)?"""
    res = binomtest(correct, n, p, alternative="greater")
    ci = res.proportion_ci(confidence_level=0.95, method="exact")
    return {"p_value": res.pvalue, "ci95_low": ci.low, "ci95_high": ci.high}


def _breakdown(df: pd.DataFrame, by: str, models: dict, X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for value, idx in df.groupby(by).groups.items():
        for variant, mc in models.items():
            m = compute_metrics(df.loc[idx, "label"], predict_variant(mc, X.loc[idx]))
            rows.append({by: value, "variant": variant, "n": m["n"], "accuracy": m["accuracy"],
                         "recall": m["recall"], "false_alarm_rate": m["false_alarm_rate"]})
    return pd.DataFrame(rows)


def _pr_records(pr: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "label": pr["label"].astype(int),
        "call_text": pr["call_transcript"], "msg_text": pr["message_text"],
        "msg_url": pr["message_text"].map(first_url),
    })


def pairwise_tests(y, preds: dict, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Is model A different from model B on the same records? (F1 difference + McNemar)"""
    rows = []
    for a, b in pairs:
        d = paired_bootstrap_diff(y, preds[a], preds[b], "f1")
        m = mcnemar_exact(y, preds[a], preds[b])
        rows.append({"model_a": a, "model_b": b, "f1_diff_a_minus_b": d["diff"],
                     "f1_diff_ci_low": d["ci_low"], "f1_diff_ci_high": d["ci_high"],
                     "bootstrap_p": d["p_value"], "a_only_right": m["a_only_right"],
                     "b_only_right": m["b_only_right"], "mcnemar_p": m["p_value"]})
    return pd.DataFrame(rows)


def run_final_test(force: bool = False) -> dict:
    """Train with the frozen configuration and test once."""
    if C.FINAL_TEST_LOCK.exists() and not force:
        raise RuntimeError(f"Final test already ran ({C.FINAL_TEST_LOCK.read_text().strip()}). "
                           "The schedule allows a single run; pass force=True only if you must repeat it.")
    cfg = load_final_config()
    if not verify_test_untouched():
        raise RuntimeError("records_test.parquet changed since the split was made - results would not be valid.")

    parts = ["train"] if C.VERSION == "v1" else ["train", "val"]
    train = pd.concat([load_split(p) for p in parts], ignore_index=True)
    test = load_split("test", allow_test=True)
    models = train_all(build_features(train), train["label"].astype(int), cfg["params"])
    X_test = build_features(test)
    y_test = test["label"].astype(int)

    rows, preds, probas = [], {}, {}
    for variant, mc in models.items():
        preds[variant] = predict_variant(mc, X_test)
        probas[variant] = proba_variant(mc, X_test)
        m = compute_metrics(y_test, preds[variant])
        _, f1_lo, f1_hi = bootstrap_ci(y_test, preds[variant], "f1")
        _, acc_lo, acc_hi = bootstrap_ci(y_test, preds[variant], "accuracy")
        rows.append({"variant": variant, **m, "f1_ci_low": f1_lo, "f1_ci_high": f1_hi,
                     "acc_ci_low": acc_lo, "acc_ci_high": acc_hi,
                     **probability_scores(y_test, probas[variant]), **binomial_vs_chance(m["correct"], m["n"])})
    results = pd.DataFrame(rows).set_index("variant")

    chosen = cfg.get("combined_choice", "combined")
    pairs = [(chosen, "call_only"), (chosen, "message_only")]
    if "combined_early" in preds and "combined_stacked" in preds:
        pairs.append(("combined_stacked", "combined_early"))
    errors = test.assign(**{f"pred_{v}": p for v, p in preds.items()}, **{f"p_{v}": p for v, p in probas.items()})
    out = {"results": results, "config": cfg, "chosen_combined": chosen,
           "pairwise": pairwise_tests(y_test, preds, pairs),
           "complementarity": complementarity(y_test, preds).to_frame("test_records").T,
           "by_msg_kind": _breakdown(test, "msg_kind", models, X_test),
           "by_call_source": _breakdown(test, "call_source", models, X_test),
           "errors": errors}

    pr = load_pr_cases()
    if pr is not None:
        pr_rec = _pr_records(pr)
        X_pr = build_features(pr_rec)
        pr_rows = []
        for variant, mc in models.items():
            m = compute_metrics(pr_rec["label"], predict_variant(mc, X_pr))
            pr_rows.append({"variant": variant, **m, **binomial_vs_chance(m["correct"], m["n"])})
        out["puerto_rico"] = pd.DataFrame(pr_rows).set_index("variant")

    C.ensure_dirs()
    results.to_csv(C.TABLES_DIR / "final_test_results.csv")
    out["pairwise"].to_csv(C.TABLES_DIR / "final_test_pairwise.csv", index=False)
    out["complementarity"].to_csv(C.TABLES_DIR / "final_test_complementarity.csv")
    out["by_msg_kind"].to_csv(C.TABLES_DIR / "final_test_by_msg_kind.csv", index=False)
    out["by_call_source"].to_csv(C.TABLES_DIR / "final_test_by_call_source.csv", index=False)
    errors.to_parquet(C.TABLES_DIR / "final_test_predictions.parquet", index=False)
    if "puerto_rico" in out:
        out["puerto_rico"].to_csv(C.TABLES_DIR / "final_test_puerto_rico.csv")
    save_models(models, "final_models")
    C.FINAL_TEST_LOCK.write_text(f"ran at {dt.datetime.now().isoformat(timespec='seconds')}\n")
    return out


def load_final_results() -> dict:
    """Re-read the tables written by ``run_final_test`` (the test itself runs only once)."""
    if not C.FINAL_TEST_LOCK.exists():
        raise FileNotFoundError("The final test has not run yet.")
    T = C.TABLES_DIR
    out = {"results": pd.read_csv(T / "final_test_results.csv", index_col="variant"),
           "by_msg_kind": pd.read_csv(T / "final_test_by_msg_kind.csv"),
           "by_call_source": pd.read_csv(T / "final_test_by_call_source.csv"),
           "errors": pd.read_parquet(T / "final_test_predictions.parquet"),
           "config": load_final_config()}
    out["chosen_combined"] = out["config"].get("combined_choice", "combined")
    for key, name in (("pairwise", "final_test_pairwise.csv"), ("complementarity", "final_test_complementarity.csv")):
        if (T / name).exists():
            out[key] = pd.read_csv(T / name, index_col=0 if key == "complementarity" else None)
    if (T / "final_test_puerto_rico.csv").exists():
        out["puerto_rico"] = pd.read_csv(T / "final_test_puerto_rico.csv", index_col="variant")
    return out
