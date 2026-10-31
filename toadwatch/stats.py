"""Uncertainty and significance tools (v2).

A difference of two cases out of 818 is not evidence that one model is better. These
functions put a confidence interval around every score and test paired differences between
two models evaluated on the SAME records (bootstrap and exact McNemar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)

from . import config as C

_METRICS = {"f1": f1_score, "accuracy": accuracy_score,
            "precision": lambda y, p: precision_score(y, p, zero_division=0),
            "recall": lambda y, p: recall_score(y, p, zero_division=0)}


def bootstrap_ci(y, pred, metric: str = "f1", n: int = C.N_BOOTSTRAP, seed: int = C.SEED,
                 level: float = 0.95) -> tuple[float, float, float]:
    """(point estimate, lower, upper) percentile-bootstrap interval."""
    y, pred = np.asarray(y), np.asarray(pred)
    fn = _METRICS[metric]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    vals = np.array([fn(y[i], pred[i]) for i in idx])
    a = (1 - level) / 2
    return float(fn(y, pred)), float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def paired_bootstrap_diff(y, pred_a, pred_b, metric: str = "f1", n: int = C.N_BOOTSTRAP,
                          seed: int = C.SEED) -> dict:
    """metric(A) - metric(B) on the same resampled records; two-sided bootstrap p-value."""
    y, pa, pb = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    fn = _METRICS[metric]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    diffs = np.array([fn(y[i], pa[i]) - fn(y[i], pb[i]) for i in idx])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(fn(y, pa) - fn(y, pb)), "ci_low": float(np.quantile(diffs, 0.025)),
            "ci_high": float(np.quantile(diffs, 0.975)), "p_value": float(min(p, 1.0))}


def mcnemar_exact(y, pred_a, pred_b) -> dict:
    """Exact McNemar test on the records where exactly one model is right."""
    y, pa, pb = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    a_only = int(((pa == y) & (pb != y)).sum())   # A right, B wrong
    b_only = int(((pa != y) & (pb == y)).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else binomtest(min(a_only, b_only), n, 0.5).pvalue
    return {"a_only_right": a_only, "b_only_right": b_only, "p_value": float(p)}


def probability_scores(y, proba) -> dict:
    return {"roc_auc": float(roc_auc_score(y, proba)),
            "avg_precision": float(average_precision_score(y, proba)),
            "brier": float(brier_score_loss(y, proba))}


def complementarity(y, preds: dict[str, np.ndarray], a: str = "call_only", b: str = "message_only") -> pd.Series:
    """How often each channel is right alone. 'only_message_right' is the room a combined
    model has to improve on the call model; if it is ~0, fusion cannot help on this data."""
    y = np.asarray(y)
    ra, rb = np.asarray(preds[a]) == y, np.asarray(preds[b]) == y
    return pd.Series({"both_right": int((ra & rb).sum()), f"only_{a}_right": int((ra & ~rb).sum()),
                      f"only_{b}_right": int((~ra & rb).sum()), "both_wrong": int((~ra & ~rb).sum())})
