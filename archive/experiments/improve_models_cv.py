"""Can every model get more accurate in a fair way?  (development data only)

Compares, by repeated stratified cross-validation on TRAIN + VALIDATION of the v2 data:
  baseline   call_only / message_only / combined_stacked   (current v2 models, settings tuned on this same data)
  upgraded   the same three versions with an extra TF-IDF text model per channel (text_stack.py)
The test set is never loaded. Because this experiment iterates on the same development data, its numbers are
optimistic estimates; a claim about generalisation needs a fresh test set.

    python experiments/improve_models_cv.py
"""
import os
import sys
from pathlib import Path

os.environ["TOADWATCH_VERSION"] = "v2"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402

from toadwatch import config as C  # noqa: E402
from toadwatch import workflow as W  # noqa: E402
from toadwatch.modeling import compute_metrics, predict_variant, proba_variant, train_variant  # noqa: E402
from toadwatch.text_stack import TextStack  # noqa: E402
from toadwatch.tuning import load_final_config  # noqa: E402


def main(n_splits=5, n_repeats=2):
    recs, X, y = W.dev_data()
    X = X[[c for c in X.columns if c not in C.STYLE_FEATURES]].copy()       # same no-style policy as v2
    Xf = X.copy()                                                            # features only (baselines)
    X["txt_call"], X["txt_msg"] = recs["call_text"].values, recs["msg_text"].values   # + raw text (TextStack)
    params = load_final_config()["params"]
    y = y.reset_index(drop=True); X = X.reset_index(drop=True); Xf = Xf.reset_index(drop=True)
    rf_params = params["per_variant"]["combined_early"]

    models = {
        "call_only (baseline)": lambda tr: train_variant("call_only", Xf.iloc[tr], y.iloc[tr], params),
        "message_only (baseline)": lambda tr: train_variant("message_only", Xf.iloc[tr], y.iloc[tr], params),
        "combined_stacked (baseline)": lambda tr: train_variant("combined_stacked", Xf.iloc[tr], y.iloc[tr], params),
        "call_only + text": lambda tr: TextStack(True, False, rf_params).fit(X.iloc[tr], y.iloc[tr]),
        "message_only + text": lambda tr: TextStack(False, True, rf_params).fit(X.iloc[tr], y.iloc[tr]),
        "combined + text": lambda tr: TextStack(True, True, rf_params).fit(X.iloc[tr], y.iloc[tr]),
    }
    rows, oof = [], pd.DataFrame({"y": y})
    for k, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=C.SEED).split(X, y)):
        for name, fit in models.items():
            m = fit(tr)
            if isinstance(m, tuple):
                pred, proba = predict_variant(m, Xf.iloc[te]), proba_variant(m, Xf.iloc[te])
            else:
                proba = m.predict_proba(X.iloc[te])[:, 1]; pred = (proba >= 0.5).astype(int)
            rows.append({"model": name, "fold": k, **compute_metrics(y.iloc[te], pred)})
            if k < n_splits:
                oof.loc[te, name] = proba
        print(f"fold {k + 1}/{n_splits * n_repeats} done", flush=True)
    folds = pd.DataFrame(rows)
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(C.TABLES_DIR / "experiment_text_models_cv_folds.csv", index=False)   # saved first
    oof.to_parquet(C.INTERIM_DIR / "oof_text_models.parquet")
    metrics = ["accuracy", "precision", "recall", "f1"]
    summary = (folds.groupby("model")[metrics].mean().add_suffix("_mean")
               .join(folds.groupby("model")[["accuracy", "f1"]].std().add_suffix("_std")).reindex(list(models)))
    summary.to_csv(C.TABLES_DIR / "experiment_text_models_cv.csv")
    print(summary.round(4).to_string())

    w = folds.pivot(index="fold", columns="model", values="f1")
    print("\npaired F1 differences over folds (mean, share of folds first model wins):")
    for a, b in (("combined + text", "call_only + text"), ("combined + text", "message_only + text"),
                 ("combined + text", "combined_stacked (baseline)"), ("call_only + text", "call_only (baseline)"),
                 ("message_only + text", "message_only (baseline)")):
        d = w[a] - w[b]
        print(f"  {a:>20s} - {b:<28s} {d.mean():+.4f}   wins {int((d > 0).sum())}/{len(d)}")
    # where does the upgraded combined model still fail?
    pred = (oof["combined + text"] >= 0.5).astype(int)
    by = pd.DataFrame({"kind": recs["msg_kind"].values, "src": recs["call_source"].values, "ok": (pred == oof["y"]).values})
    print("\nupgraded combined, out-of-fold accuracy by message type:", by.groupby("kind").ok.mean().round(3).to_dict())
    print("by call source:", by.groupby("src").ok.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
