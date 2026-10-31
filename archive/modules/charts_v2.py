"""Draw every v2 figure from the tables the workflow saved. Each chart is skipped (not an
error) if the stage that produces its data has not run yet, so this can be called any time."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from . import config as C
from . import reporting as R


def _read(name, **kw):
    p = C.TABLES_DIR / name
    return pd.read_csv(p, **kw) if p.exists() else None


def make_all_charts() -> list:
    C.ensure_dirs()
    T = C.TABLES_DIR
    made = []

    def done(fig_name, fig):
        plt.close(fig)
        made.append(C.FIGURES_DIR / f"{fig_name}.png")

    if (abl := _read("diagnosis_ablation.csv")) is not None:
        done("v2_01_ablation", R.plot_ablation(abl, "v2_01_ablation"))
    if (comp := _read("diagnosis_complementarity.csv", index_col=0)) is not None:
        done("v2_02_complementarity_dev", R.plot_complementarity(comp["records"], "v2_02_complementarity_dev"))
    if (folds := _read("cv_folds_tuned.csv")) is not None:
        done("v2_03_cv_f1_boxplot", R.plot_cv_scores(folds, "f1", "v2_03_cv_f1_boxplot"))
        done("v2_04_cv_accuracy_boxplot", R.plot_cv_scores(folds, "accuracy", "v2_04_cv_accuracy_boxplot"))
    if (oof := (pd.read_parquet(C.INTERIM_DIR / "oof_tuned.parquet") if (C.INTERIM_DIR / "oof_tuned.parquet").exists() else None)) is not None:
        probas = {c[2:]: oof[c] for c in oof.columns if c.startswith("p_")}
        done("v2_05_cv_roc_pr", R.plot_roc_pr(oof["y"], probas, "v2_05_cv_roc_pr"))
        done("v2_06_cv_prob_hist", R.plot_prob_hist(oof["y"], probas, "v2_06_cv_prob_hist"))
        done("v2_07_cv_threshold_sweep", R.plot_threshold_sweep(oof["y"], probas, "v2_07_cv_threshold_sweep"))
    if (lc := _read("learning_curve.csv")) is not None:
        done("v2_08_learning_curve", R.plot_learning_curve(lc, "v2_08_learning_curve"))
    if (imp := _read("feature_importance.csv")) is not None:
        for v in ("call_only", "message_only", "combined_early"):
            if v in set(imp["variant"]):
                done(f"v2_09_importance_{v}", R.plot_feature_importance(imp, v, name=f"v2_09_importance_{v}"))
        if "combined_early" in set(imp["variant"]):
            done("v2_10_channel_importance", R.plot_channel_importance(imp, "combined_early", "v2_10_channel_importance"))
    if (res := _read("final_test_results.csv", index_col="variant")) is not None:
        done("v2_11_test_metrics", R.plot_metric_bars(res, "Final test metrics (v2)", "v2_11_test_metrics"))
        done("v2_12_test_confusion", R.plot_confusion_matrices(res, "Final test confusion matrices (v2)", "v2_12_test_confusion"))
        done("v2_13_test_f1_ci", R.plot_f1_ci(res, "v2_13_test_f1_ci"))
    if (pw := _read("final_test_pairwise.csv")) is not None:
        done("v2_14_test_pairwise", R.plot_pairwise(pw, "v2_14_test_pairwise"))
    if (bk := _read("final_test_by_msg_kind.csv")) is not None:
        done("v2_15_test_by_kind", R.plot_kind_heatmap(bk, "v2_15_test_by_kind"))
    if (T / "final_test_predictions.parquet").exists():
        e = pd.read_parquet(T / "final_test_predictions.parquet")
        probas = {c[2:]: e[c] for c in e.columns if c.startswith("p_") and c != "p_value"}
        done("v2_16_test_roc_pr", R.plot_roc_pr(e["label"], probas, "v2_16_test_roc_pr"))
        done("v2_17_test_prob_hist", R.plot_prob_hist(e["label"], probas, "v2_17_test_prob_hist"))
        if (comp := _read("final_test_complementarity.csv", index_col=0)) is not None:
            done("v2_18_test_complementarity", R.plot_complementarity(comp.iloc[0], "v2_18_test_complementarity"))
    return made
