import pandas as pd
import pytest

from toadwatch import config as C
from toadwatch import evaluation
from toadwatch.features import build_features, columns_for
from toadwatch.features_call import call_features
from toadwatch.features_message import lookalike, url_features
from toadwatch.modeling import compute_metrics, train_all


def test_call_features_detect_scam_patterns():
    f = call_features("This is the IRS. Press 1 now or police will arrest you. Give me your PIN and "
                      "social security number, pay with gift cards.")
    assert f["call_press_one"] == 1 and f["call_pin_request"] >= 1
    assert f["call_claims_government"] >= 1 and f["call_payment_methods"] >= 1
    safe = call_features("Hi, confirming your appointment tomorrow. Have a great day.")
    assert safe["call_pin_request"] == 0 and safe["call_press_one"] == 0


def test_spanish_patterns():
    f = call_features("Presione 1 ahora, su cuenta fue suspendida. Necesito su contraseña.")
    assert f["call_press_one"] == 1 and f["call_pin_request"] == 1 and f["call_urgent_words"] >= 1


def test_url_features():
    f = url_features("http://bit.ly/abc")
    assert f["msg_is_shortener"] == 1 and f["msg_https"] == 0
    assert url_features("https://paypa1.com/login")["msg_lookalike_typo"] == 1
    assert url_features("https://paypal.com/login")["msg_lookalike_typo"] == 0
    assert url_features("http://192.168.0.1/x")["msg_has_ip"] == 1
    assert all(v == 0 for v in url_features("").values())      # no link -> all zeros
    url_features("WWW.SINOTOM.COM）上发布了贵方")                 # malformed host must not crash


def test_lookalike_scores_brand_similarity():
    sim, typo, _ = lookalike("amaz0n.com")
    assert sim > 0.8 and typo == 1


def test_variants_use_their_own_columns():
    cols = ["call_a", "call_b", "msg_a"]
    assert columns_for("call_only", cols) == ["call_a", "call_b"]
    assert columns_for("message_only", cols) == ["msg_a"]
    assert columns_for("combined", cols) == cols
    with pytest.raises(ValueError):
        columns_for("nope", cols)


def test_train_all_and_metrics(calls_df, msgs_df):
    rec = pd.DataFrame({"label": [0, 1] * 30,
                        "call_text": ["hello about your delivery", "press 1 now give me your PIN"] * 30,
                        "msg_text": ["see you soon", "click http://bit.ly/x to verify"] * 30,
                        "msg_url": ["", "http://bit.ly/x"] * 30})
    X, y = build_features(rec), rec["label"]
    models = train_all(X, y, {"n_estimators": 20})
    assert set(models) == set(C.VARIANTS)
    m = compute_metrics(y, models["combined"][0].predict(X[models["combined"][1]]))
    assert m["accuracy"] == 1.0 and m["tp"] + m["tn"] == 60


def test_binomial_test_against_chance():
    strong = evaluation.binomial_vs_chance(90, 100)
    weak = evaluation.binomial_vs_chance(52, 100)
    assert strong["p_value"] < 0.001 < 0.05 < weak["p_value"]


def test_final_test_refuses_without_frozen_config_and_second_run(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "FINAL_CONFIG_JSON", tmp_path / "missing.json")
    monkeypatch.setattr(C, "FINAL_TEST_LOCK", tmp_path / "lock")
    with pytest.raises(FileNotFoundError):
        evaluation.run_final_test()
    (tmp_path / "lock").write_text("ran")
    with pytest.raises(RuntimeError, match="already ran"):
        evaluation.run_final_test()
