import numpy as np
import pandas as pd

from toadwatch.features import build_all_features
from toadwatch.text_stack import TextStack


def _frame(n=120):
    label = np.array([0, 1] * (n // 2))
    calls = ["hello confirming your appointment tomorrow thank you have a great day",
             "this is your bank press 1 now give me your PIN or police will arrest you"]
    msgs = ["see you soon here is the menu https://www.example.com/menu", "URGENT verify your password at http://bit.ly/x"]
    rec = pd.DataFrame({"label": label, "call_text": [calls[l] + f" ref {i}" for i, l in enumerate(label)],
                        "msg_text": [msgs[l] + f" id{i}" for i, l in enumerate(label)],
                        "msg_url": ["https://www.example.com/menu" if l == 0 else "http://bit.ly/x" for l in label]})
    X = build_all_features(rec)
    X["txt_call"], X["txt_msg"] = rec["call_text"].values, rec["msg_text"].values
    return X, rec["label"]


def test_text_stack_variants_fit_and_predict():
    X, y = _frame()
    for use_call, use_msg, n_bases in ((True, False, 2), (False, True, 2), (True, True, 4)):
        m = TextStack(use_call, use_msg, {"n_estimators": 20}, n_splits=3).fit(X.iloc[:90], y.iloc[:90])
        proba = m.predict_proba(X.iloc[90:])
        assert proba.shape == (30, 2) and np.allclose(proba.sum(axis=1), 1)
        assert len(m.meta_weights()) == n_bases
        assert (m.predict(X.iloc[90:]) == y.iloc[90:]).mean() > 0.9


def test_text_stack_ignores_the_other_channel():
    X, y = _frame()
    m = TextStack(True, False, {"n_estimators": 20}, n_splits=3).fit(X, y)
    assert set(m.names_) == {"rf_call", "text_call"}
