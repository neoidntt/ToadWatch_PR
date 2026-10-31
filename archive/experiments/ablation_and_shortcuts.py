"""Scratch experiments on TRAIN/VALIDATION only (never the test set).

Questions
  1. Does the model lean on dataset artefacts? (message type / dialogue style). Source names
     are NOT tested alone: every source file sits in one class folder, so a source name
     encodes the label by construction.
  2. How much does the Tranco-rank stand-in for domain age matter?

    python experiments/ablation_and_shortcuts.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import f1_score  # noqa: E402

from toadwatch import config as C  # noqa: E402
from toadwatch.modeling import make_model  # noqa: E402
from toadwatch.splitting import load_split  # noqa: E402
from toadwatch.tuning import load_final_config  # noqa: E402


def fit_score(cols, tr, va, params):
    m = make_model(params).fit(tr[cols], tr["label"])
    return f1_score(va["label"], m.predict(va[cols]))


if __name__ == "__main__":
    params = load_final_config()["params"]
    tr = pd.read_parquet(C.INTERIM_DIR / "features_train.parquet")
    va = pd.read_parquet(C.INTERIM_DIR / "features_val.parquet")
    rec_tr, rec_va = load_split("train"), load_split("val")
    feats = [c for c in tr.columns if c != "label"]
    rows = []

    # 1a. shortcut: the message type alone must be at chance level (classes are balanced per kind)
    for col in ("msg_kind",):
        both = pd.concat([rec_tr[col], rec_va[col]]).astype("category").cat.codes
        t = pd.DataFrame({"x": both[:len(rec_tr)].values, "label": rec_tr["label"].values})
        v = pd.DataFrame({"x": both[len(rec_tr):].values, "label": rec_va["label"].values})
        rows.append((f"only {col}", fit_score(["x"], t, v, params)))

    # 1b. call model without length / dialogue-style features
    style = ["call_question_marks", "call_n_words", "call_n_chars", "call_avg_word_len", "call_exclamations"]
    call_cols = [c for c in feats if c.startswith("call_")]
    rows.append(("call_only, all features", fit_score(call_cols, tr, va, params)))
    rows.append(("call_only, no style/length", fit_score([c for c in call_cols if c not in style], tr, va, params)))

    # 2. Tranco stand-in for domain age
    msg_cols = [c for c in feats if c.startswith("msg_")]
    rows.append(("message_only, all features", fit_score(msg_cols, tr, va, params)))
    rows.append(("message_only, no Tranco", fit_score([c for c in msg_cols if "tranco" not in c], tr, va, params)))

    out = pd.DataFrame(rows, columns=["experiment", "val_f1"]).round(4)
    out.to_csv(C.TABLES_DIR / "experiment_ablation.csv", index=False)
    print(out.to_string(index=False))
