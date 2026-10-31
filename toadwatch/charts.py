"""Figures for the three models, drawn from the tables the workflow saved (a chart is skipped if its stage has not run)."""
from __future__ import annotations

import matplotlib

try:                      # inside Jupyter / VS Code notebooks keep the inline backend so charts show
    get_ipython  # type: ignore[name-defined]  # noqa: B018
except NameError:         # plain scripts: draw to files only (works without a display)
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from . import config as C  # noqa: E402

COLORS = {"call_only": "#5b7db1", "message_only": "#d08c3c", "combined": "#4c9f70"}
NAMES = C.MODEL_NAMES
PT_NAMES = {"FF": "FF\nfraud call +\nfraud message", "FS": "FS\nfraud call +\nsafe-looking message", "SF": "SF\nsafe-looking call +\nfraud message", "SS": "SS\nboth safe\n(= not fraud)"}
TIER_NOTE = {"hard": "hard (content only)", "matched": "matched (content only)", "unmatched": "unmatched (optimistic: source gives class away)"}


def _read(name, **kw):
    p = C.TABLES_DIR / name
    return pd.read_csv(p, **kw) if p.exists() else None


def _save(fig, name):
    if name:
        C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(C.FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_accuracy_vs_ceilings(overall: pd.DataFrame, ceilings: dict, title: str, name: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.bar(range(3), [overall.loc[m, "accuracy"] for m in C.MODELS], color=[COLORS[m] for m in C.MODELS])
    for lab, ls, col, txt in (("call_oracle", "--", "#5b7db1", "ceiling: a perfect CALL-only model"), ("message_oracle", "-.", "#d08c3c", "ceiling: a perfect MESSAGE-only model"),
                              ("majority_class", ":", "grey", "always say the majority class")):
        ax.axhline(ceilings[lab], ls=ls, color=col, label=txt)
    for i, m in enumerate(C.MODELS):
        ax.text(i, overall.loc[m, "accuracy"] + 0.005, f"{overall.loc[m, 'accuracy']:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(3), [NAMES[m] for m in C.MODELS], fontsize=8); ax.set_ylim(0.4, 1.02); ax.set_ylabel("accuracy")
    ax.set_title(title); ax.legend(fontsize=7, loc="lower right")
    _save(fig, name)
    return fig


def plot_pair_type(by_pair: pd.DataFrame, title: str, name: str | None = None):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    w = 0.26
    for i, m in enumerate(C.MODELS):
        d = by_pair[by_pair.model == m].set_index("pair_type").reindex(["FF", "FS", "SF", "SS"])
        ax.bar(np.arange(4) + (i - 1) * w, d["accuracy"], w, color=COLORS[m], label=NAMES[m])
    ax.set_xticks(range(4), [PT_NAMES[t] for t in ("FF", "FS", "SF", "SS")], fontsize=8)
    ax.set_ylim(0, 1.05); ax.axhline(0.5, color="grey", ls=":", lw=1); ax.set_ylabel("accuracy"); ax.set_title(title); ax.legend(fontsize=7, loc="lower right")
    _save(fig, name)
    return fig


def plot_tiers(by_tier: pd.DataFrame, title: str, name: str | None = None):
    tiers = [t for t in C.TIERS if t in set(by_tier.tier)]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    w = 0.26
    for i, m in enumerate(C.MODELS):
        d = by_tier[by_tier.model == m].set_index("tier").reindex(tiers)
        err = [d["accuracy"] - d["acc_ci_low"], d["acc_ci_high"] - d["accuracy"]] if "acc_ci_low" in d else None
        ax.bar(np.arange(len(tiers)) + (i - 1) * w, d["accuracy"], w, yerr=err, capsize=2, color=COLORS[m], label=NAMES[m])
    ax.set_xticks(range(len(tiers)), [TIER_NOTE[t] for t in tiers], fontsize=8)
    ax.set_ylim(0.3, 1.02); ax.axhline(0.5, color="grey", ls=":", lw=1); ax.set_ylabel("accuracy"); ax.set_title(title); ax.legend(fontsize=7, loc="lower right")
    _save(fig, name)
    return fig


def plot_detectors(channel: pd.DataFrame, title: str, name: str | None = None):
    """How well each forest recognises a scam in ITS OWN text, by tier (balanced accuracy)."""
    tiers = [t for t in ("hard", "matched", "unmatched", "all") if t in set(channel.tier)]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.36
    for i, (det, col) in enumerate((("call detector", COLORS["call_only"]), ("message detector", COLORS["message_only"]))):
        d = channel[channel.detector == det].set_index("tier").reindex(tiers)
        ax.bar(np.arange(len(tiers)) + (i - 0.5) * w, d["balanced_accuracy"], w, color=col, label=det)
        for j, v in enumerate(d["balanced_accuracy"]):
            ax.text(j + (i - 0.5) * w, v + 0.005, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(tiers)), tiers); ax.set_ylim(0.4, 1.05); ax.set_ylabel("balanced accuracy on its own text"); ax.set_title(title); ax.legend(fontsize=8)
    _save(fig, name)
    return fig


def plot_cv_boxplot(folds: pd.DataFrame, name: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [folds[folds.model == m]["accuracy"] for m in C.MODELS]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    for b, m in zip(bp["boxes"], C.MODELS):
        b.set_facecolor(COLORS[m])
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i + 1), d, color="black", s=12, zorder=3)
    ax.set_xticks(range(1, 4), [NAMES[m] for m in C.MODELS], fontsize=8); ax.set_ylabel("accuracy per CV fold"); ax.set_title("Cross-validation accuracy per fold (development data)")
    _save(fig, name)
    return fig


def plot_acc_ci(res: pd.DataFrame, name: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    for i, m in enumerate(C.MODELS):
        r = res.loc[m]
        ax.errorbar(r["accuracy"], i, xerr=[[r["accuracy"] - r["acc_ci_low"]], [r["acc_ci_high"] - r["accuracy"]]], fmt="o", color=COLORS[m], capsize=4)
    ax.set_yticks(range(3), [NAMES[m] for m in C.MODELS], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("TEST accuracy with 95 % family-bootstrap interval"); ax.set_title("True accuracy and its uncertainty")
    _save(fig, name)
    return fig


def plot_confusion(res: pd.DataFrame, name: str | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, m in zip(axes, C.MODELS):
        r = res.loc[m]
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["not fraud", "fraud"]); ax.set_yticks([0, 1], ["not fraud", "fraud"]); ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(NAMES[m], fontsize=9)
    fig.suptitle("Final test: confusion matrices"); fig.tight_layout()
    _save(fig, name)
    return fig


def plot_roc_pr(y, scores: dict, name: str | None = None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for m in C.MODELS:
        fpr, tpr, _ = roc_curve(y, scores[m]); pr, rc, _ = precision_recall_curve(y, scores[m])
        a1.plot(fpr, tpr, color=COLORS[m], label=NAMES[m]); a2.plot(rc, pr, color=COLORS[m], label=NAMES[m])
    a1.plot([0, 1], [0, 1], ":", color="grey")
    a1.set(xlabel="false-alarm rate", ylabel="recall", title="ROC curve"); a2.set(xlabel="recall", ylabel="precision", title="Precision-recall curve", ylim=(0.4, 1.02))
    a1.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    _save(fig, name)
    return fig


def plot_pairwise(pw: pd.DataFrame, name: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 2.4))
    for i, r in pw.reset_index(drop=True).iterrows():
        ax.errorbar(r.acc_diff_a_minus_b, i, xerr=[[r.acc_diff_a_minus_b - r.ci_low], [r.ci_high - r.acc_diff_a_minus_b]], fmt="o", color="#333", capsize=4)
    ax.axvline(0, color="grey", ls=":")
    ax.set_yticks(range(len(pw)), [f"{NAMES[a]}\nminus {NAMES[b]}" for a, b in zip(pw.model_a, pw.model_b)], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("accuracy difference with 95 % family-bootstrap interval (right of 0 = combined better)")
    _save(fig, name)
    return fig


def plot_shuffle(shuf: pd.DataFrame, name: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(range(3), [shuf.loc[m, "accuracy"] for m in C.MODELS], color=[COLORS[m] for m in C.MODELS])
    ax.axhline(float(shuf["majority_class_rate"].iloc[0]), ls=":", color="grey", label="majority-class rate")
    ax.set_xticks(range(3), [NAMES[m] for m in C.MODELS], fontsize=8); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title("Sanity check: trained on SHUFFLED labels (must fall to chance)")
    _save(fig, name)
    return fig


def plot_shortcuts(sc: pd.DataFrame, name: str | None = None):
    tiers = [t for t in C.TIERS if f"accuracy_{t}" in sc]
    fig, axes = plt.subplots(1, len(tiers), figsize=(5 * len(tiers), 3.8), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), tiers):
        ax.barh(sc["baseline"], sc[f"accuracy_{t}"], color="#888")
        ax.axvline(float(sc[f"majority_{t}"].iloc[0]), ls=":", color="k", label="majority class")
        ax.set_title(TIER_NOTE[t], fontsize=9); ax.set_xlim(0.3, 1.02); ax.invert_yaxis(); ax.tick_params(axis="y", labelsize=7)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Metadata-only baselines (no text read): high in 'unmatched' = a dataset gives the class away"); fig.tight_layout()
    _save(fig, name)
    return fig


def plot_label_audit(sm: pd.DataFrame, name: str | None = None):
    d = sm[sm["items"] >= 20].sort_values("contested_share").tail(14)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([f"{c}: {s}" for c, s in zip(d.channel, d.source)], d.contested_share * 100, color="#c0504d")
    ax.set_xlabel("% of texts whose given label the held-out forest contradicts (p < 0.1)"); ax.set_title("Label audit: sources worth a manual look"); ax.tick_params(axis="y", labelsize=7)
    _save(fig, name)
    return fig


def plot_importance(imp: pd.DataFrame, name: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, ch, col in zip(axes, ("call", "message"), (COLORS["call_only"], COLORS["message_only"])):
        d = imp[imp.channel == ch].head(12).iloc[::-1]
        ax.barh(d["feature"], d["importance"], color=col); ax.set_title(f"{ch} Random Forest: top features", fontsize=10); ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    _save(fig, name)
    return fig


def plot_threshold_sweep(oof: pd.DataFrame, recs: pd.DataFrame, thresholds: dict, name: str | None = None):
    from sklearn.metrics import f1_score
    y, ts = recs["label"].to_numpy(), np.array(C.THRESHOLD_GRID)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, [f1_score(y, (oof["p_call"].to_numpy() >= t).astype(int)) for t in ts], color=COLORS["call_only"], label=NAMES["call_only"])
    ax.plot(ts, [f1_score(y, (oof["p_msg"].to_numpy() >= t).astype(int)) for t in ts], color=COLORS["message_only"], label=NAMES["message_only"])
    ax.plot(ts, [f1_score(y, ((oof["p_call"].to_numpy() >= t) | (oof["p_msg"].to_numpy() >= t)).astype(int)) for t in ts], color=COLORS["combined"], label="Combined (same threshold on both)")
    ax.set(xlabel="decision threshold", ylabel="attack-level F1 (out-of-fold)", title="F1 versus threshold"); ax.legend(fontsize=7)
    _save(fig, name)
    return fig


def make_all_charts() -> list:
    import json
    C.ensure_dirs()
    made = []

    def done(name, fig):
        plt.close(fig)
        made.append(C.FIGURES_DIR / f"{name}.png")

    cfg = json.loads(C.FINAL_CONFIG_JSON.read_text()) if C.FINAL_CONFIG_JSON.exists() else None
    if (cv := _read("cv_overall.csv", index_col=0)) is not None and cfg:
        done("01_cv_accuracy_vs_ceilings", plot_accuracy_vs_ceilings(cv, cfg["ceilings"], "Cross-validated accuracy vs one-channel ceilings", "01_cv_accuracy_vs_ceilings"))
    if (bp := _read("cv_by_pair_type.csv")) is not None:
        done("02_cv_pair_type", plot_pair_type(bp, "Cross-validation: accuracy by pair type", "02_cv_pair_type"))
    if (bt := _read("cv_by_tier.csv")) is not None:
        done("03_cv_tiers", plot_tiers(bt, "Cross-validation: accuracy by tier", "03_cv_tiers"))
    if (ch := _read("cv_detectors.csv")) is not None:
        done("04_cv_detectors", plot_detectors(ch, "Each Random Forest on its own text (cross-validation)", "04_cv_detectors"))
    if (f := _read("cv_folds.csv")) is not None:
        done("05_cv_boxplot", plot_cv_boxplot(f, "05_cv_boxplot"))
    if (imp := _read("feature_importance.csv")) is not None:
        done("06_feature_importance", plot_importance(imp, "06_feature_importance"))
    if (sh := _read("audit_label_shuffle.csv", index_col=0)) is not None:
        done("07_shuffle_check", plot_shuffle(sh, "07_shuffle_check"))
    if (sc := _read("audit_shortcut_baselines.csv")) is not None:
        done("08_shortcut_baselines", plot_shortcuts(sc, "08_shortcut_baselines"))
    if (la := _read("label_audit_by_source.csv")) is not None:
        done("09_label_audit", plot_label_audit(la, "09_label_audit"))
    if (C.INTERIM_DIR / "oof_scores.parquet").exists() and cfg and (C.PROCESSED_DIR / "records_train.parquet").exists():
        from . import protocol as P
        recs, _, _ = P.dev_table()
        done("10_threshold_sweep", plot_threshold_sweep(pd.read_parquet(C.INTERIM_DIR / "oof_scores.parquet"), recs, cfg["thresholds"], "10_threshold_sweep"))
    if (res := _read("final_test_results.csv", index_col=0)) is not None:
        tc = _read("final_test_ceilings.csv", index_col=0)
        done("11_test_acc_ci", plot_acc_ci(res, "11_test_acc_ci"))
        if tc is not None:
            done("12_test_vs_ceilings", plot_accuracy_vs_ceilings(res, tc["accuracy"].to_dict(), "FINAL TEST accuracy vs one-channel ceilings", "12_test_vs_ceilings"))
        done("13_test_confusion", plot_confusion(res, "13_test_confusion"))
    if (bp := _read("final_test_by_pair_type.csv")) is not None:
        done("14_test_pair_type", plot_pair_type(bp, "FINAL TEST: accuracy by pair type", "14_test_pair_type"))
    if (bt := _read("final_test_by_tier.csv")) is not None:
        done("15_test_tiers", plot_tiers(bt, "FINAL TEST: accuracy by tier (95 % family-bootstrap intervals)", "15_test_tiers"))
    if (dt_ := _read("final_test_detectors.csv")) is not None:
        done("16_test_detectors", plot_detectors(dt_, "FINAL TEST: each Random Forest on its own text", "16_test_detectors"))
    if (pw := _read("final_test_pairwise.csv")) is not None:
        done("17_test_pairwise", plot_pairwise(pw, "17_test_pairwise"))
    if (C.TABLES_DIR / "final_test_predictions.parquet").exists():
        e = pd.read_parquet(C.TABLES_DIR / "final_test_predictions.parquet")
        done("18_test_roc_pr", plot_roc_pr(e["label"], {m: e[f"score_{m}"] for m in C.MODELS}, "18_test_roc_pr"))
    return made
