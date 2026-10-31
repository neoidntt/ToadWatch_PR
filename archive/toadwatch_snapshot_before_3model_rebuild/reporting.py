"""Figures used by the notebooks and the report (saved to outputs/figures)."""
from __future__ import annotations

import matplotlib

try:                      # inside Jupyter / VS Code notebooks keep the inline backend so charts show
    get_ipython  # type: ignore[name-defined]  # noqa: B018
except NameError:         # plain scripts: draw to files only (works without a display)
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

LABELS = {"call_only": "Call only", "message_only": "Message / email / link only", "combined": "Combined"}


def _save(fig, name: str | None):
    if name:
        C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(C.FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_class_balance(df: pd.DataFrame, by: str, title: str, name: str | None = None):
    counts = df.groupby([by, "label"]).size().unstack(fill_value=0).rename(columns={0: "safe", 1: "fraud"})
    ax = counts.plot.barh(stacked=True, figsize=(8, max(3, 0.35 * len(counts))), color=["#4c9f70", "#c0504d"])
    ax.set_title(title)
    ax.set_xlabel("rows")
    _save(ax.figure, name)
    return ax.figure


def plot_metric_bars(table: pd.DataFrame, title: str, name: str | None = None):
    metrics = ["accuracy", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(table)
    for i, (variant, row) in enumerate(table.iterrows()):
        ax.bar(np.arange(len(metrics)) + i * width, [row[m] for m in metrics], width, label=LABELS.get(variant, variant))
    ax.set_xticks(np.arange(len(metrics)) + width * (len(table) - 1) / 2, metrics)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, name)
    return fig


def plot_confusion_matrices(table: pd.DataFrame, title: str, name: str | None = None):
    fig, axes = plt.subplots(1, len(table), figsize=(4 * len(table), 3.6))
    for ax, (variant, row) in zip(np.atleast_1d(axes), table.iterrows()):
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["safe", "fraud"])
        ax.set_yticks([0, 1], ["safe", "fraud"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(LABELS.get(variant, variant), fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, name)
    return fig


def plot_feature_importance(imp: pd.DataFrame, variant: str, top: int = 12, name: str | None = None):
    d = imp[imp["variant"] == variant].head(top).iloc[::-1]
    colors = d["channel"].map({"call": "#5b7db1", "message": "#d08c3c"})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(d["feature"], d["importance"], color=colors)
    ax.set_title(f"Top features - {LABELS.get(variant, variant)} (blue = call, orange = message)")
    _save(fig, name)
    return fig


# ====================================================================== v2 charts
import seaborn as sns  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

PALETTE = {"call_only": "#5b7db1", "message_only": "#d08c3c", "combined": "#4c9f70",
           "combined_early": "#4c9f70", "combined_stacked": "#8e5ea2"}
LABELS.update({"combined_early": "Combined (one forest)", "combined_stacked": "Combined (stacked)",
               "call_only_text": "Call only + text", "message_only_text": "Message only + text", "combined_text": "Combined + text"})
PALETTE.update({"call_only_text": "#3b5b8f", "message_only_text": "#b06a1c", "combined_text": "#2f7d4f"})


def _color(v):
    return PALETTE.get(v, "#888888")


def plot_cv_scores(folds: pd.DataFrame, metric: str = "f1", name: str | None = None):
    """Spread of the score over all CV folds - shows whether differences are bigger than noise."""
    order = [v for v in LABELS if v in set(folds["variant"])]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.boxplot(data=folds, x="variant", y=metric, order=order, palette=[_color(v) for v in order], ax=ax, width=0.5)
    sns.stripplot(data=folds, x="variant", y=metric, order=order, color="black", size=3, alpha=0.5, ax=ax)
    ax.set_xticks(range(len(order)), [LABELS[v] for v in order], rotation=12, fontsize=8)
    ax.set_title(f"{metric.upper()} over repeated cross-validation folds (train+val only)")
    ax.set_xlabel("")
    _save(fig, name)
    return fig


def plot_ablation(abl: pd.DataFrame, name: str | None = None):
    """abl: columns policy, variant, f1_mean, f1_std."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    policies = list(dict.fromkeys(abl["policy"]))
    variants = [v for v in LABELS if v in set(abl["variant"])]
    w = 0.8 / len(variants)
    for i, v in enumerate(variants):
        d = abl[abl["variant"] == v].set_index("policy").reindex(policies)
        ax.bar(np.arange(len(policies)) + i * w, d["f1_mean"], w, yerr=d["f1_std"], capsize=2,
               color=_color(v), label=LABELS[v])
    ax.set_xticks(np.arange(len(policies)) + w * (len(variants) - 1) / 2, policies, fontsize=8)
    lo = max(0.5, float(abl["f1_mean"].min()) - 0.05)
    ax.set_ylim(lo, 1.0)
    ax.set_ylabel("mean CV F1")
    ax.set_title("What each preprocessing / feature choice does to each model")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, name)
    return fig


def plot_complementarity(counts: pd.Series, name: str | None = None):
    """Share of records where each channel is right alone: the room fusion has to add value."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    pct = counts / counts.sum() * 100
    colors = ["#4c9f70", "#5b7db1", "#d08c3c", "#c0504d"]
    ax.barh(["records"], [pct.iloc[0]], color=colors[0], label=pct.index[0])
    left = pct.iloc[0]
    for k in range(1, len(pct)):
        ax.barh(["records"], [pct.iloc[k]], left=left, color=colors[k], label=f"{pct.index[k]} ({int(counts.iloc[k])})")
        left += pct.iloc[k]
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of records")
    ax.set_title("Who is right? (call model vs message model)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2)
    _save(fig, name)
    return fig


def plot_roc_pr(y, probas: dict, name: str | None = None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for v, p in probas.items():
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=_color(v), label=LABELS.get(v, v))
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, color=_color(v), label=LABELS.get(v, v))
    a1.plot([0, 1], [0, 1], ":", color="grey")
    a1.set(xlabel="false-alarm rate", ylabel="recall", title="ROC curve")
    a2.set(xlabel="recall", ylabel="precision", title="Precision-recall curve", ylim=(0.5, 1.02))
    a1.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, name)
    return fig


def plot_prob_hist(y, probas: dict, name: str | None = None):
    """How confident each model is, split by the true class."""
    fig, axes = plt.subplots(1, len(probas), figsize=(3.6 * len(probas), 3.2), sharey=True)
    y = np.asarray(y)
    for ax, (v, p) in zip(np.atleast_1d(axes), probas.items()):
        p = np.asarray(p)
        ax.hist(p[y == 0], bins=20, range=(0, 1), alpha=0.7, color="#4c9f70", label="safe")
        ax.hist(p[y == 1], bins=20, range=(0, 1), alpha=0.7, color="#c0504d", label="fraud")
        ax.set_title(LABELS.get(v, v), fontsize=9)
        ax.set_xlabel("predicted fraud probability")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, name)
    return fig


def plot_threshold_sweep(y, probas: dict, name: str | None = None):
    """F1 as the decision threshold moves (from out-of-fold probabilities; test never used to pick it)."""
    from sklearn.metrics import f1_score
    ts = np.linspace(0.1, 0.9, 33)
    fig, ax = plt.subplots(figsize=(7, 4))
    for v, p in probas.items():
        ax.plot(ts, [f1_score(y, (np.asarray(p) >= t).astype(int)) for t in ts], color=_color(v), label=LABELS.get(v, v))
    ax.axvline(0.5, ls=":", color="grey")
    ax.set(xlabel="decision threshold", ylabel="F1", title="F1 versus decision threshold")
    ax.legend(fontsize=8)
    _save(fig, name)
    return fig


def plot_learning_curve(lc: pd.DataFrame, name: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for v, g in lc.groupby("variant"):
        ax.errorbar(g["n_train"], g["mean"], yerr=g["std"], marker="o", ms=3, capsize=2, color=_color(v), label=LABELS.get(v, v))
    ax.set(xlabel="training records", ylabel="CV F1", title="Learning curves (train+val only)")
    ax.legend(fontsize=8)
    _save(fig, name)
    return fig


def plot_f1_ci(results: pd.DataFrame, name: str | None = None):
    """Test F1 with 95 % bootstrap intervals: overlapping bars = no real difference."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (v, r) in enumerate(results.iterrows()):
        ax.errorbar(r["f1"], i, xerr=[[r["f1"] - r["f1_ci_low"]], [r["f1_ci_high"] - r["f1"]]], fmt="o",
                    color=_color(v), capsize=4)
    ax.set_yticks(range(len(results)), [LABELS.get(v, v) for v in results.index])
    ax.invert_yaxis()
    ax.set_xlabel("test F1 with 95 % bootstrap interval")
    ax.set_title("Is any model really better than another?")
    _save(fig, name)
    return fig


def plot_pairwise(pairwise: pd.DataFrame, name: str | None = None):
    fig, ax = plt.subplots(figsize=(7.5, 0.9 + 0.8 * len(pairwise)))
    for i, r in pairwise.iterrows():
        ax.errorbar(r["f1_diff_a_minus_b"], i, xerr=[[r["f1_diff_a_minus_b"] - r["f1_diff_ci_low"]],
                    [r["f1_diff_ci_high"] - r["f1_diff_a_minus_b"]]], fmt="o", color="#333", capsize=4)
    ax.axvline(0, color="grey", ls=":")
    ax.set_yticks(range(len(pairwise)), [f"{LABELS.get(a, a)}\nminus {LABELS.get(b, b)}" for a, b in zip(pairwise.model_a, pairwise.model_b)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("F1 difference (95 % paired bootstrap interval); right of 0 = first model better")
    _save(fig, name)
    return fig


def plot_kind_heatmap(by_kind: pd.DataFrame, name: str | None = None):
    piv = by_kind.pivot(index="variant", columns="msg_kind", values="accuracy")
    piv = piv.loc[[v for v in LABELS if v in piv.index]]
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="Blues", vmin=0.7, vmax=1.0, cbar=False, ax=ax)
    ax.set_yticklabels([LABELS[v] for v in piv.index], rotation=0, fontsize=8)
    ax.set_title("Accuracy by message type")
    ax.set_xlabel("")
    ax.set_ylabel("")
    _save(fig, name)
    return fig


def plot_channel_importance(imp: pd.DataFrame, variant: str, name: str | None = None):
    """Share of the forest's importance that comes from call, message and cross-channel features."""
    d = imp[imp["variant"] == variant].groupby("channel")["importance"].sum()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(d.index, d.values, color=["#5b7db1" if c == "call" else "#d08c3c" if c == "message" else "#8e5ea2" for c in d.index])
    ax.set_title(f"Importance by channel - {LABELS.get(variant, variant)}", fontsize=10)
    _save(fig, name)
    return fig


def plot_meta_weights(weights: pd.Series, name: str | None = None):
    w = weights.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(w) + 1))
    ax.barh(w.index, w.values, color=["#c0504d" if x < 0 else "#4c9f70" for x in w.values])
    ax.set_title("Stacked model: how the meta-learner weighs each input")
    _save(fig, name)
    return fig
