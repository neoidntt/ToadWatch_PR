import pandas as pd

from toadwatch.report_builder import _pval, _sci, verdict


def _d(c_lo, c_hi, m_lo=0.05, m_hi=0.1, chosen="combined_stacked"):
    rows = [{"model_a": chosen, "model_b": "call_only", "f1_diff_a_minus_b": (c_lo + c_hi) / 2,
             "f1_diff_ci_low": c_lo, "f1_diff_ci_high": c_hi, "bootstrap_p": 0.01, "a_only_right": 9, "b_only_right": 2, "mcnemar_p": 0.03},
            {"model_a": chosen, "model_b": "message_only", "f1_diff_a_minus_b": (m_lo + m_hi) / 2,
             "f1_diff_ci_low": m_lo, "f1_diff_ci_high": m_hi, "bootstrap_p": 0.0, "a_only_right": 90, "b_only_right": 5, "mcnemar_p": 0.0}]
    return {"chosen": chosen, "pair": pd.DataFrame(rows)}


def test_verdict_supported_when_interval_above_zero():
    assert verdict(_d(0.003, 0.026))["headline"] == "supported"


def test_verdict_partly_supported_when_not_distinguishable():
    v = verdict(_d(-0.002, 0.012))
    assert v["headline"] == "partly supported" and v["call"]["state"] == "indistinguishable"


def test_verdict_not_supported_when_significantly_worse():
    assert verdict(_d(-0.03, -0.004))["headline"] == "not supported"


def test_p_value_formatting():
    assert _pval(0.0001) == "< 0.001" and _pval(0.0347) == "0.035"
    assert _sci(0.0) == "< 1e-300" and _sci(0.2) == "0.200"
