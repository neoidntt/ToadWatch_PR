import numpy as np
import pandas as pd

from toadwatch import config as C
from toadwatch import data_loading
from toadwatch.dataset_builder import build_records
from toadwatch.hard_data import generate_hard_set


def _pools():
    rows_c, rows_m = [], []
    for i in range(2400):
        lab = int(i % 5 < 2)
        tier = ["unmatched", "matched", "hard"][i % 3]
        rows_c.append({"call_id": f"c{i}", "text": f"call {i}", "label": lab, "source": "s", "lang": "en", "group": f"cg{i}", "is_hard": tier == "hard", "tier": tier})
        k = ["sms", "email", "url"][i % 3] if tier == "unmatched" else ["sms", "email"][i % 2]
        rows_m.append({"item_id": f"m{i}", "kind": k, "text": f"msg {i}", "label": lab, "source": "s", "lang": "en", "group": f"mg{i}", "is_hard": tier == "hard", "tier": tier, "url": ""})
    return pd.DataFrame(rows_c), pd.DataFrame(rows_m)


def test_records_use_the_either_part_is_fraud_rule_and_never_reuse_items():
    calls, msgs = _pools()
    rec = build_records(calls, msgs, seed=1)
    assert set(rec.pair_type) == {"FF", "FS", "SF", "SS"} and set(rec.tier) == {"hard", "matched", "unmatched"}
    assert (rec["label"] == (rec.call_label | rec.msg_label)).all()
    assert (rec.groupby("pair_type").label.first().to_dict() == {"FF": 1, "FS": 1, "SF": 1, "SS": 0})
    assert rec.call_id.is_unique and rec.item_id.is_unique
    assert rec.groupby("call_group").bucket.nunique().max() == 1
    test, dev = rec[rec.split == "test"], rec[rec.split != "test"]
    assert not set(test.call_group) & set(dev.call_group) and not set(test.msg_group) & set(dev.msg_group)
    assert abs(rec.label.mean() - (1 - C.PAIR_MIX["SS"])) < 0.04


def test_tiers_are_never_mixed_inside_a_record():
    calls, msgs = _pools()
    rec = build_records(calls, msgs, seed=2)
    ct = calls.set_index("call_id").tier
    mt = msgs.set_index("item_id").tier
    assert (rec.call_id.map(ct) == rec.tier).all() and (rec.item_id.map(mt) == rec.tier).all()


def test_hard_set_is_balanced_and_style_matched():
    c, m = generate_hard_set(seed=5)
    assert c.label.value_counts().nunique() == 1
    assert (m.groupby(["kind", "label"]).size().unstack()[0] == m.groupby(["kind", "label"]).size().unstack()[1]).all()
    assert not c.source.str.contains("fraud|safe").any() and not m.source.str.contains("fraud|safe").any()
    sms = m[m.kind == "sms"].assign(link=lambda d: d.text.str.contains("http"))
    assert 0.25 < sms[sms.label == 1].link.mean() < 0.8 and 0.25 < sms[sms.label == 0].link.mean() < 0.9


def test_new_sms_campaign_file_is_normalised_deduplicated_and_capped(tmp_path, monkeypatch):
    rows = []
    for camp in range(30):
        for k in range(6):                                              # 6 near-identical messages per campaign, irregular spacing
            rows.append({"message": f"  Su   paquete \n no  llego {camp}  pague  en  http://x.example/{camp}{k}  ", "language_code": "es", "campaign_id": float(camp)})
    rows += [{"message": "same text", "language_code": "en", "campaign_id": np.nan}] * 3 + [{"message": "other unrelated text here", "language_code": "en", "campaign_id": np.nan}]
    f = tmp_path / "sms.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    monkeypatch.setattr(C, "SMS_CAMPAIGNS_CSV", f)
    out = data_loading.load_sms_campaigns(seed=1)
    assert (out["label"] == 1).all() and (out["kind"] == "sms").all()
    assert not out["text"].str.contains(r"\s{2,}|\n").any()                                 # one space between words
    assert out["text"].is_unique and out.dropna(subset=["group"]).groupby("group").size().max() <= C.SMS_CAMPAIGN_MAX_PER_CAMPAIGN
    assert out["group"].dropna().str.startswith("camp:").all()
