import numpy as np
import pandas as pd
import pytest

from toadwatch import config as C
from toadwatch.features import build_features, channel_columns
from toadwatch.features_call import call_features
from toadwatch.grouping import union_groups
from toadwatch.models import FraudDetectors, decide, model_scores
from toadwatch import protocol as P


def test_exactly_three_models():
    assert C.MODELS == ("call_only", "message_only", "combined") and set(C.MODEL_NAMES) == set(C.MODELS)


def test_combined_is_the_or_of_the_two_detectors():
    sc = pd.DataFrame({"p_call": [0.9, 0.1, 0.1, 0.9, 0.4], "p_msg": [0.1, 0.9, 0.1, 0.9, 0.4]})
    d = decide(sc, {"call_only": 0.5, "message_only": 0.5, "combined": [0.5, 0.5]})
    assert d["call_only"].tolist() == [1, 0, 0, 1, 0] and d["message_only"].tolist() == [0, 1, 0, 1, 0]
    assert d["combined"].tolist() == [1, 1, 0, 1, 0]                      # either one fires -> the whole attack is fraud
    assert (d["combined"] >= d["call_only"]).all() and (d["combined"] >= d["message_only"]).all()
    s = model_scores(sc, {"call_only": 0.5, "message_only": 0.5, "combined": [0.5, 0.5]})
    assert ((s["combined"] >= 0.5).astype(int) == d["combined"]).all()    # the 0.5 line of the score is the decision


def test_separate_thresholds_in_the_or_rule():
    sc = pd.DataFrame({"p_call": [0.6, 0.3], "p_msg": [0.3, 0.6]})
    d = decide(sc, {"call_only": 0.5, "message_only": 0.5, "combined": [0.7, 0.4]})
    assert d["combined"].tolist() == [0, 1]


def _recs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    cl, ml = rng.integers(0, 2, n), rng.integers(0, 2, n)
    call = ["press 1 now give me your PIN or the police will arrest you, this is your bank", "hello confirming your appointment tomorrow, have a great day"]
    msg = ["URGENT verify your password at http://bit.ly/x", "see you soon, the menu is here https://www.example.com/menu"]
    return pd.DataFrame({"call_label": cl, "msg_label": ml, "label": cl | ml,
                         "call_text": [call[1 - l] for l in cl], "msg_text": [msg[1 - l] for l in ml],
                         "msg_url": ["http://bit.ly/x" if l else "https://www.example.com/menu" for l in ml]})


def test_detectors_train_on_their_own_half_and_combine_with_or():
    rec = _recs()
    X = build_features(rec)
    assert all(c.startswith("call_") for c in channel_columns(X.columns, "call")) and all(c.startswith("msg_") for c in channel_columns(X.columns, "message"))
    det = FraudDetectors({"call": {"n_estimators": 30}, "message": {"n_estimators": 30}}).fit(X, rec["call_label"], rec["msg_label"])
    sc = det.scores(X)
    d = decide(sc, {"call_only": 0.5, "message_only": 0.5, "combined": [0.5, 0.5]})
    acc = {m: (p == rec["label"].to_numpy()).mean() for m, p in d.items()}
    assert acc["combined"] > acc["call_only"] and acc["combined"] > acc["message_only"] and acc["combined"] > 0.95
    assert set(det.importances().channel) == {"call", "message"}


def test_call_model_reads_links_mentioned_in_the_transcript():
    f = call_features("This is your bank, I'm texting you a link, go to www.popular-verificar.top and verify your PIN.")
    assert f["call_has_url"] == 1 and f["call_mentions_link"] >= 1 and f["call_url_suspicious_tld"] == 1
    g = call_features("Hi, confirming your appointment tomorrow.")
    assert g["call_has_url"] == 0 and g["call_url_https"] == 0


def test_ceilings_follow_the_either_is_fraud_rule():
    rec = pd.DataFrame({"call_label": [1, 1, 0, 0] * 50, "msg_label": [1, 0, 1, 0] * 50})
    rec["label"] = rec.call_label | rec.msg_label
    c = P.ceilings(rec)
    assert c["combined_oracle"] == 1.0 and c["call_oracle"] == 0.75 and c["message_oracle"] == 0.75 and c["majority_class"] == 0.75


def test_threshold_tuning_finds_a_good_or_rule():
    rng = np.random.default_rng(1)
    cl, ml = rng.integers(0, 2, 400), rng.integers(0, 2, 400)
    recs = pd.DataFrame({"label": cl | ml})
    sc = pd.DataFrame({"p_call": np.clip(cl * 0.7 + rng.normal(0.15, 0.1, 400), 0, 1), "p_msg": np.clip(ml * 0.7 + rng.normal(0.15, 0.1, 400), 0, 1)})
    t = P.tune_thresholds(sc, recs)
    d = decide(sc, t)
    assert (d["combined"] == recs["label"].to_numpy()).mean() > 0.95 and 0.2 <= t["call_only"] <= 0.8


def test_union_groups_links_campaigns_and_clusters():
    g = union_groups(["a", "a", None, "b"], [1, 2, 2, 3])
    assert g[0] == g[1] == g[2] and g[3] != g[0]


def test_cluster_bootstrap_interval_contains_the_estimate():
    y = np.array([1, 0] * 60); pred = y.copy(); pred[:10] = 1 - pred[:10]
    p, lo, hi = P.cluster_bootstrap(y, pred, np.repeat(np.arange(24), 5), "accuracy", n=300)
    assert lo <= p <= hi
