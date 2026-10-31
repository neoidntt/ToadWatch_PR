"""v3 figures, drawn from the saved tables (a chart is skipped if its stage has not run yet)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import reporting as R

PT_NAMES = {"FF": "FF\nfraud call +\nfraud message\n(= FRAUD)", "FS": "FS\nfraud call +\nsafe message", "SF": "SF\nsafe call +\nfraud message", "SS": "SS\nsafe call +\nsafe message"}


def _read(name, **kw):
    p = C.TABLES_DIR / name
    return pd.read_csv(p, **kw) if p.exists() else None


def plot_pair_type_accuracy(res: pd.DataFrame, title: str, name: str | None = None, suffix: str = ""):
    """Accuracy per pair type: shows exactly WHERE single-channel models fail (they cannot see the other half)."""
    cols = [f"acc_{t}{suffix}" for t in R.__dict__.get("PT", ("FF", "FS", "SF", "SS"))]
    variants = [v for v in R.LABELS if v in res.index]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.85 / len(variants)
    for i, v in enumerate(variants):
        ax.bar(np.arange(4) + i * w, [res.loc[v, c] for c in cols], w, color=R._color(v), label=R.LABELS[v])
    ax.set_xticks(np.arange(4) + w * (len(variants) - 1) / 2, [PT_NAMES[t] for t in ("FF", "FS", "SF", "SS")], fontsize=8)
    ax.set_ylim(0, 1.05); ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_ylabel("accuracy"); ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    R._save(fig, name)
    return fig


def plot_accuracy_vs_ceilings(res: pd.DataFrame, ceilings: dict, title: str, name: str | None = None):
    """Overall accuracy with the single-channel ceilings: a one-channel model above its line would signal leakage."""
    variants = [v for v in R.LABELS if v in res.index]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.bar(range(len(variants)), [res.loc[v, "accuracy"] if "accuracy" in res else res.loc[v, "accuracy_mean"] for v in variants],
           color=[R._color(v) for v in variants])
    for lab, ls, col in (("call_oracle", "--", "#5b7db1"), ("message_oracle", "-.", "#d08c3c"), ("majority_class", ":", "grey")):
        ax.axhline(ceilings[lab], ls=ls, color=col, label={"call_oracle": "ceiling: a perfect CALL-only model", "message_oracle": "ceiling: a perfect MESSAGE-only model",
                                                           "majority_class": "always say 'not fraud'"}[lab])
    ax.set_xticks(range(len(variants)), [R.LABELS[v] for v in variants], rotation=18, fontsize=8)
    ax.set_ylim(0.4, 1.02); ax.set_ylabel("accuracy"); ax.set_title(title); ax.legend(fontsize=8, loc="lower right")
    R._save(fig, name)
    return fig


def plot_shuffle(shuf: pd.DataFrame, name: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(range(len(shuf)), shuf["accuracy_mean"], yerr=shuf["accuracy_std"], capsize=3, color=[R._color(v) for v in shuf.index])
    ax.axhline(float(shuf["majority_class_rate"].iloc[0]), ls=":", color="grey", label="majority-class rate")
    ax.set_xticks(range(len(shuf)), [R.LABELS[v] for v in shuf.index], fontsize=8, rotation=10); ax.set_ylim(0, 1)
    ax.set_title("Sanity check: trained on SHUFFLED labels (must fall to chance)"); ax.legend(fontsize=8)
    R._save(fig, name)
    return fig


def plot_shortcuts(sc: pd.DataFrame, ceilings: dict, name: str | None = None):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.barh(sc["baseline"], sc["accuracy"], color="#888")
    ax.axvline(ceilings["majority_class"], ls=":", color="grey", label="majority class")
    ax.axvline(ceilings["call_oracle"], ls="--", color="#5b7db1", label="call-only ceiling")
    ax.set_xlim(0.4, 1); ax.invert_yaxis(); ax.set_xlabel("accuracy from metadata alone")
    ax.set_title("Shortcut baselines: metadata alone must not beat the ceilings"); ax.legend(fontsize=8)
    R._save(fig, name)
    return fig


def plot_label_audit(sm: pd.DataFrame, name: str | None = None):
    d = sm[sm["items"] >= 20].sort_values("suspect_share").tail(14)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([f"{c}: {s}" for c, s in zip(d.channel, d.source)], d.suspect_share * 100, color="#c0504d")
    ax.set_xlabel("% of items whose given label a held-out model contradicts (p < 0.1)")
    ax.set_title("Label audit: sources worth a manual look"); ax.tick_params(axis="y", labelsize=7)
    R._save(fig, name)
    return fig


def plot_regimes(res: pd.DataFrame, name: str | None = None):
    """Accuracy in the hard regime (content only, honest) next to the public regime (optimistic), with family-bootstrap intervals."""
    variants = [v for v in R.LABELS if v in res.index]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.38
    for j, (reg, col, lab) in enumerate((("hard", "#c0504d", "hard regime: content only (honest, few families)"), ("public", "#7f9fbf", "public regime: source style reveals the class (optimistic)"))):
        acc = res[f"{reg}_accuracy"].reindex(variants)
        lo, hi = res[f"{reg}_acc_ci_low"].reindex(variants), res[f"{reg}_acc_ci_high"].reindex(variants)
        ax.bar(np.arange(len(variants)) + (j - 0.5) * w, acc, w, yerr=[acc - lo, hi - acc], capsize=2, color=col, label=lab)
    ax.axhline(0.6, ls=":", color="grey")
    ax.set_xticks(range(len(variants)), [R.LABELS[v] for v in variants], rotation=15, fontsize=8)
    ax.set_ylim(0.3, 1.02); ax.set_ylabel("test accuracy"); ax.set_title("True accuracy: where the content has to be read vs where style gives it away")
    ax.legend(fontsize=8, loc="lower right")
    R._save(fig, name)
    return fig


def plot_pairwise_acc(pw: pd.DataFrame, name: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.7 * len(pw)))
    for i, r in pw.reset_index(drop=True).iterrows():
        ax.errorbar(r.acc_diff_a_minus_b, i, xerr=[[r.acc_diff_a_minus_b - r.ci_low], [r.ci_high - r.acc_diff_a_minus_b]], fmt="o", color="#333", capsize=4)
    ax.axvline(0, color="grey", ls=":")
    ax.set_yticks(range(len(pw)), [f"{R.LABELS.get(a, a)}\nminus {R.LABELS.get(b, b)}" for a, b in zip(pw.model_a, pw.model_b)], fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("accuracy difference, 95 % family-bootstrap interval (right of 0 = first model better)")
    R._save(fig, name)
    return fig


def plot_acc_ci(res: pd.DataFrame, name: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.55 * len(res)))
    for i, (v, r) in enumerate(res.iterrows()):
        ax.errorbar(r["accuracy"], i, xerr=[[r["accuracy"] - r["acc_ci_low"]], [r["acc_ci_high"] - r["accuracy"]]], fmt="o", color=R._color(v), capsize=4)
    ax.set_yticks(range(len(res)), [R.LABELS.get(v, v) for v in res.index]); ax.invert_yaxis()
    ax.set_xlabel("TEST accuracy with 95 % family-bootstrap interval"); ax.set_title("True accuracy and its uncertainty")
    R._save(fig, name)
    return fig


def make_all_charts() -> list:
    C.ensure_dirs()
    made = []

    def done(name, fig):
        plt.close(fig)
        made.append(C.FIGURES_DIR / f"{name}.png")

    cv = _read("cv_summary.csv", index_col="variant")
    ceil = _read("audit_ceilings.csv", index_col=0)
    ceilings = ceil["accuracy"].to_dict() if ceil is not None else None
    if cv is not None:
        cv2 = cv.rename(columns={c: c.replace("_mean", "") for c in cv.columns})
        done("v3_01_cv_accuracy_by_pair_type", plot_pair_type_accuracy(cv2, "Cross-validation: accuracy by pair type", "v3_01_cv_accuracy_by_pair_type"))
        if ceilings:
            done("v3_02_cv_accuracy_vs_ceilings", plot_accuracy_vs_ceilings(cv, ceilings, "Cross-validated accuracy vs single-channel ceilings", "v3_02_cv_accuracy_vs_ceilings"))
    if (f := _read("cv_folds.csv")) is not None:
        done("v3_03_cv_accuracy_boxplot", R.plot_cv_scores(f, "accuracy", "v3_03_cv_accuracy_boxplot"))
    if (sh := _read("audit_label_shuffle.csv", index_col="variant")) is not None:
        done("v3_04_shuffle_check", plot_shuffle(sh, "v3_04_shuffle_check"))
    if (sc := _read("audit_shortcut_baselines.csv")) is not None and ceilings:
        done("v3_05_shortcut_baselines", plot_shortcuts(sc, ceilings, "v3_05_shortcut_baselines"))
    if (la := _read("label_audit_by_source.csv")) is not None:
        done("v3_06_label_audit", plot_label_audit(la, "v3_06_label_audit"))
    if (res := _read("final_test_results.csv", index_col="variant")) is not None:
        done("v3_10_test_acc_ci", plot_acc_ci(res, "v3_10_test_acc_ci"))
        done("v3_11_test_pair_type", plot_pair_type_accuracy(res, "FINAL TEST: accuracy by pair type", "v3_11_test_pair_type"))
        tc = _read("final_test_ceilings.csv", index_col=0)
        if tc is not None:
            done("v3_12_test_vs_ceilings", plot_accuracy_vs_ceilings(res, tc["accuracy"].to_dict(), "FINAL TEST accuracy vs single-channel ceilings", "v3_12_test_vs_ceilings"))
        if "hard_accuracy" in res:
            done("v3_17_test_regimes", plot_regimes(res, "v3_17_test_regimes"))
        done("v3_13_test_confusion", R.plot_confusion_matrices(res, "FINAL TEST confusion matrices", "v3_13_test_confusion"))
    if (pw := _read("final_test_pairwise.csv")) is not None:
        done("v3_14_test_pairwise", plot_pairwise_acc(pw, "v3_14_test_pairwise"))
    if (C.TABLES_DIR / "final_test_predictions.parquet").exists():
        e = pd.read_parquet(C.TABLES_DIR / "final_test_predictions.parquet")
        probas = {c[2:]: e[c] for c in e.columns if c.startswith("p_") and c[2:] in R.LABELS}
        done("v3_15_test_roc_pr", R.plot_roc_pr(e["label"], probas, "v3_15_test_roc_pr"))
        done("v3_16_test_prob_hist", R.plot_prob_hist(e["label"], probas, "v3_16_test_prob_hist"))
    return made
