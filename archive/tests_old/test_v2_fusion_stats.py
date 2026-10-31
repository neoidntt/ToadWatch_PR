import numpy as np
import pandas as pd
import pytest

from toadwatch import config as C
from toadwatch.cv import choose_combined, cross_validate_variants
from toadwatch.features import apply_policy, build_all_features, columns_for
from toadwatch.features_cross import cross_feature_frame
from toadwatch.fusion import StackedFusion
from toadwatch.modeling import resolve_params, train_variant
from toadwatch.stats import bootstrap_ci, complementarity, mcnemar_exact, paired_bootstrap_diff


def _records(n=120, seed=0):
    rng = np.random.default_rng(seed)
    label = np.array([0, 1] * (n // 2))
    calls = ["press 1 now give me your PIN or police arrest you, this is your bank",
             "hello confirming your appointment tomorrow, have a great day"]
    msgs = ["URGENT verify your account at http://bit.ly/x password",
            "see you soon, here is the menu https://www.example.com/menu"]
    return pd.DataFrame({
        "label": label,
        "call_text": [calls[l] if rng.random() > 0.1 else calls[1 - l] for l in label],
        "msg_text": [msgs[l] if rng.random() > 0.3 else msgs[1 - l] for l in label],
        "msg_url": ["http://bit.ly/x" if l else "https://www.example.com/menu" for l in label]})


def test_cross_features_exist_only_in_the_full_table_and_policy_drops_them():
    X = build_all_features(_records(10))
    assert any(c.startswith("cross_") for c in X.columns)
    no_cross = apply_policy(X, exclude_style=False, use_cross=False)
    assert not any(c.startswith("cross_") for c in no_cross.columns)
    no_style = apply_policy(X, exclude_style=True, use_cross=True)
    assert not set(C.STYLE_FEATURES) & set(no_style.columns)
    # single-channel models never see cross features
    assert not any(c.startswith("cross_") for c in columns_for("call_only", X.columns) + columns_for("message_only", X.columns))
    assert not any(c.startswith("cross_") for c in columns_for("combined_early", X.columns))   # plain concatenation
    assert any(c.startswith("cross_") for c in columns_for("combined_stacked", X.columns))


def test_cross_features_detect_shared_brand_and_agreement():
    rec = pd.DataFrame({"call_text": ["This is Banco Popular. Urgent: press 1, give me your PIN."],
                        "msg_text": ["Banco Popular URGENT: verify your password http://popular-login.xyz"],
                        "msg_url": ["http://popular-login.xyz"]})
    X = build_all_features(rec)
    assert X["cross_shared_brand"].iloc[0] >= 1
    assert X["cross_both_urgent"].iloc[0] == 1 and X["cross_both_ask_credentials"].iloc[0] == 1


def test_stacked_fusion_trains_meta_on_out_of_fold_only_and_predicts():
    rec = _records(160)
    X, y = apply_policy(build_all_features(rec), True, True), rec["label"]
    model = StackedFusion(call_params={"n_estimators": 30}, msg_params={"n_estimators": 30}, n_splits=3).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2) and np.allclose(proba.sum(axis=1), 1)
    assert set(model.predict(X)) <= {0, 1}
    assert "call_forest" in model.meta_weights().index


def test_stacked_fit_does_not_look_at_new_data():
    """Predicting must not change the model: fit once, predict twice on different data."""
    rec = _records(120)
    X, y = apply_policy(build_all_features(rec), True, True), rec["label"]
    model = StackedFusion(call_params={"n_estimators": 20}, msg_params={"n_estimators": 20}, n_splits=3).fit(X.iloc[:80], y.iloc[:80])
    a = model.predict_proba(X.iloc[80:100])
    model.predict_proba(X.iloc[100:])
    assert np.allclose(a, model.predict_proba(X.iloc[80:100]))


def test_resolve_params_per_variant():
    p = {"per_variant": {"call_only": {"max_depth": 3}, "message_only": {"max_depth": 5}, "combined_early": {"max_depth": 7}}}
    assert resolve_params("call_only", p) == {"max_depth": 3}
    assert resolve_params("combined_early", p) == {"max_depth": 7}
    assert resolve_params("combined_stacked", p) == {"call": {"max_depth": 3}, "msg": {"max_depth": 5}}
    assert resolve_params("call_only", {"max_depth": 9}) == {"max_depth": 9}   # v1 flat dict still works


def test_cross_validation_never_mixes_train_and_test_rows():
    rec = _records(100)
    X, y = apply_policy(build_all_features(rec), True, True), rec["label"]
    folds, oof = cross_validate_variants(X, y, {"n_estimators": 20}, ["call_only", "combined_early"], n_splits=3, n_repeats=1)
    assert len(folds) == 3 * 2
    assert oof["p_call_only"].notna().all()          # every row is predicted exactly once, out of fold


def test_choose_combined_prefers_simple_model_on_a_tie():
    s = pd.DataFrame({"f1_mean": [0.90, 0.9005]}, index=["combined_early", "combined_stacked"])
    assert choose_combined(s)["choice"] == "combined_early"
    s2 = pd.DataFrame({"f1_mean": [0.90, 0.95]}, index=["combined_early", "combined_stacked"])
    assert choose_combined(s2)["choice"] == "combined_stacked"


def test_bootstrap_and_paired_tests():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    good = np.where(rng.random(400) < 0.95, y, 1 - y)
    bad = np.where(rng.random(400) < 0.70, y, 1 - y)
    point, lo, hi = bootstrap_ci(y, good, "f1", n=300)
    assert lo < point < hi
    assert paired_bootstrap_diff(y, good, bad, n=300)["p_value"] < 0.01
    assert mcnemar_exact(y, good, bad)["p_value"] < 0.01
    same = mcnemar_exact(y, good, good)
    assert same["p_value"] == 1.0 and same["a_only_right"] == 0


def test_complementarity_counts_sum_to_n():
    y = np.array([1, 1, 0, 0])
    c = complementarity(y, {"call_only": np.array([1, 0, 0, 1]), "message_only": np.array([1, 1, 1, 0])})
    assert c.sum() == 4 and c["both_right"] == 1 and c["only_message_only_right"] == 2
