import numpy as np
import pandas as pd

from toadwatch import config as C
from toadwatch.data_v3 import build_records
from toadwatch.grouping import assign_buckets, group_texts
from toadwatch.hard_data import generate_hard_set
from toadwatch.protocol_v3 import ceilings, cluster_bootstrap, shortcut_baselines, weights_for


def test_grouping_joins_near_copies_and_flags_cross_label_conflicts():
    base = "press 1 now to talk to an agent about your amazon order, we need your card number today"
    texts = pd.Series([base, base + " thanks", "hello the clinic confirms your appointment tomorrow afternoon", "a completely different message about lunch plans on friday"])
    g, drop, rep = group_texts(texts, pd.Series([1, 1, 0, 0]), threshold=0.5)
    assert g[0] == g[1] and g[0] != g[2] and not drop.any()
    _, drop2, _ = group_texts(texts, pd.Series([1, 0, 0, 0]), threshold=0.5, conflict_threshold=0.5)
    assert drop2[:2].all() and not drop2[2:].any()          # the same text can't be both classes


def test_assign_buckets_never_splits_a_group():
    rng = np.random.default_rng(0)
    groups = pd.Series([f"g{i % 37}" for i in range(400)])
    labels = pd.Series([i % 2 for i in range(400)])
    b = assign_buckets(groups, labels, 10, rng, strata=groups.map(lambda g: str(int(g[1:]) % 2)))
    assert b.groupby(groups).nunique().max() == 1 and set(b) <= set(range(10))


def _pools():
    rows_c, rows_m = [], []
    for i in range(1500):
        lab = int(i % 5 < 3)
        rows_c.append({"call_id": f"c{i}", "text": f"call {i}", "label": lab, "source": "s", "lang": "en", "group": f"cg{i}", "is_hard": i % 6 == 0})
        k = ["sms", "email", "url"][i % 3]
        rows_m.append({"item_id": f"m{i}", "kind": k, "text": f"msg {i}", "label": lab, "source": "s", "lang": "en", "group": f"mg{i}", "is_hard": i % 6 == 0, "url": ""})
    m = pd.DataFrame(rows_m)
    m.loc[(m.is_hard) & (m.kind == "url"), "kind"] = "sms"
    return pd.DataFrame(rows_c), m


def test_build_records_applies_toad_rule_and_never_reuses_items():
    calls, msgs = _pools()
    rec = build_records(calls, msgs, seed=1)
    assert set(rec.pair_type) == {"FF", "FS", "SF", "SS"}
    assert (rec["label"] == (rec.pair_type == "FF").astype(int)).all()                          # fraud only if BOTH are fraud
    assert (rec["label"] == (rec.call_label & rec.msg_label)).all()
    assert rec.call_id.is_unique and rec.item_id.is_unique
    assert rec.groupby("call_group").bucket.nunique().max() == 1
    assert set(rec.regime) == {"public", "hard"}
    assert not (rec[rec.regime == "hard"].call_is_hard == False).any()
    test, dev = rec[rec.split == "test"], rec[rec.split != "test"]
    assert not set(test.call_group) & set(dev.call_group) and not set(test.msg_group) & set(dev.msg_group)
    assert abs(rec.groupby("regime").label.mean().loc["public"] - C.PAIR_MIX["FF"]) < 0.03


def test_ceilings_and_shortcut_baselines():
    calls, msgs = _pools()
    rec = build_records(calls, msgs, seed=2)
    c = ceilings(rec)
    assert c["both_channels_oracle"] == 1.0 and 0.8 < c["call_oracle"] < 0.9 and 0.55 < c["majority_class"] < 0.65
    sb = shortcut_baselines(rec.assign(fold=rec.bucket // 2).query("split != 'test'"), rec.assign(fold=rec.bucket // 2).query("split != 'test'")["fold"].to_numpy())
    assert abs(sb.set_index("baseline").loc["message type only", "accuracy"] - c["majority_class"]) < 0.06


def test_weights_and_cluster_bootstrap():
    rec = pd.DataFrame({"regime": ["hard", "public", "hard"]})
    assert weights_for(rec, 4).tolist() == [4.0, 1.0, 4.0]
    y = np.array([1, 0] * 60); pred = y.copy(); pred[:10] = 1 - pred[:10]
    groups = np.repeat(np.arange(24), 5)
    p, lo, hi = cluster_bootstrap(y, pred, groups, "accuracy", n=300)
    assert lo <= p <= hi and 0.8 < p < 1


def test_hard_set_is_balanced_and_style_matched():
    c, m = generate_hard_set(seed=5)
    assert c.label.value_counts().nunique() == 1
    assert (m.groupby(["kind", "label"]).size().unstack()[0] == m.groupby(["kind", "label"]).size().unstack()[1]).all()   # safe == fraud inside every message type
    assert not c.text.duplicated().any() and not m.text.duplicated().any()
    assert not c.source.str.contains("fraud|safe").any() and not m.source.str.contains("fraud|safe").any()     # class-neutral source names
    sms = m[m.kind == "sms"].assign(link=lambda d: d.text.str.contains("http"))
    assert 0.25 < sms[sms.label == 1].link.mean() < 0.8 and 0.25 < sms[sms.label == 0].link.mean() < 0.9       # a link doesn't give the class away
    assert c.group.str.startswith("hard:call").all() and c.groupby("group").label.nunique().max() == 1
