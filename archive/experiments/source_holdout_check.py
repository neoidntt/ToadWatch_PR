"""Do the text-model gains survive when a whole dataset family is unseen?  (development data only)

A perfect cross-validation score usually means a shortcut (the writing style of each source), not real
detection. Here whole families are held out of training and used only for testing:

  calls     hold out one dialogue family at a time (agent_conversation_labels / labels / scam_dialogue_label;
            each family has both a safe and a fraud file) and train on everything else;
  messages  hold out one message type at a time (sms / email / url) and train on the other two.

Compared: the hand-made-feature Random Forest vs a TF-IDF + logistic-regression text model.
The test set is never loaded.

    python experiments/source_holdout_check.py
"""
import os
import sys
from pathlib import Path

os.environ["TOADWATCH_VERSION"] = "v2"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from toadwatch import config as C  # noqa: E402
from toadwatch import workflow as W  # noqa: E402
from toadwatch.modeling import make_model  # noqa: E402
from toadwatch.text_stack import MAX_CHARS, _tfidf_lr  # noqa: E402
from toadwatch.tuning import load_final_config  # noqa: E402


def main():
    recs, X, y = W.dev_data()
    X = X[[c for c in X.columns if c not in C.STYLE_FEATURES]].reset_index(drop=True)
    recs = recs.reset_index(drop=True); y = y.reset_index(drop=True)
    per = load_final_config()["params"]["per_variant"]
    rows = []

    def run(channel, held_name, test_mask):
        tr, te = ~test_mask, test_mask
        if channel == "call":
            cols = [c for c in X.columns if c.startswith("call_")]
            rf = make_model(per["call_only"]).fit(X.loc[tr, cols], y[tr])
            txt = _tfidf_lr("word").fit(recs.loc[tr, "call_text"].str.slice(0, MAX_CHARS), y[tr])
            p_rf, p_tx = rf.predict(X.loc[te, cols]), txt.predict(recs.loc[te, "call_text"].str.slice(0, MAX_CHARS))
        else:
            cols = [c for c in X.columns if c.startswith("msg_")]
            rf = make_model(per["message_only"]).fit(X.loc[tr, cols], y[tr])
            txt = _tfidf_lr("char").fit(recs.loc[tr, "msg_text"].str.slice(0, MAX_CHARS), y[tr])
            p_rf, p_tx = rf.predict(X.loc[te, cols]), txt.predict(recs.loc[te, "msg_text"].str.slice(0, MAX_CHARS))
        for name, p in (("feature Random Forest", p_rf), ("TF-IDF text model", p_tx)):
            rows.append({"channel": channel, "held-out (unseen) data": held_name, "model": name, "test records": int(te.sum()),
                         "accuracy": accuracy_score(y[te], p), "f1": f1_score(y[te], p)})

    fam = recs.call_source.str.replace(r"_[01]$", "", regex=True)
    for f in ("agent_conversation_labels", "labels", "scam_dialogue_label"):
        run("call", f, (fam == f))
    for k in ("sms", "email", "url"):
        run("message", k, (recs.msg_kind == k))
    out = pd.DataFrame(rows)
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(C.TABLES_DIR / "experiment_source_holdout.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
