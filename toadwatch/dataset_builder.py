"""Build the data: public files + my hard set -> clean -> families -> buckets -> connected call + message pairs.

Each RECORD is one TOAD attack = a call transcript and a message / email / link that belong together.
Label rule: the attack is FRAUD if EITHER part is fraud (by definition both parts are then part of a fraud attack, even
when one of them looks harmless on its own). Pair types:  FF fraud+fraud, FS fraud call + safe-looking message,
SF safe-looking call + fraud message, SS both safe (= not fraud).

Every record also keeps the label of each half (``call_label`` / ``msg_label``): the two Random Forests are trained on those.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .cleaning import clean_calls, clean_messages, extract_urls, host_of
from .data_loading import load_calls, load_messages
from .grouping import assign_buckets, group_texts, union_groups
from .hard_data import generate_hard_set
from .splitting import save_records

KINDS_BY_TIER = {"hard": ("sms", "email"), "matched": ("sms", "email"), "unmatched": ("sms", "email", "url")}


def _base_domain(url: str) -> str:
    labels = host_of(url).split(".")
    return ".".join(labels[-2:]) if len(labels) >= 2 else host_of(url)


def _tier(df: pd.DataFrame, matched_sources) -> pd.Series:
    return pd.Series(np.where(df["is_hard"], "hard", np.where(df["source"].isin(matched_sources), "matched", "unmatched")), index=df.index)


def build_pools() -> dict:
    """Clean the public data, add the hard set; every text gets a family id (``group``) and a ``tier``."""
    calls, calls_rep = clean_calls(load_calls())
    msgs, msgs_rep = clean_messages(load_messages())
    reports = {"calls": calls_rep, "messages": msgs_rep}

    g, drop, rep = group_texts(calls["text"], calls["label"])
    calls = calls.assign(group=[f"pub:c{x}" for x in g])[~drop].reset_index(drop=True)
    reports["call_families"] = rep

    parts, msg_reps = [], {}
    for kind, d in msgs.groupby("kind"):
        d = d.reset_index(drop=True)
        if kind == "url":                                       # same registered domain = same family
            d["group"] = "pub:u:" + d["url"].map(_base_domain)
            msg_reps[kind] = {"texts": len(d), "families": int(d["group"].nunique())}
        else:
            g, drop, rep = group_texts(d["text"], d["label"])
            # a scam campaign (new SMS file) and a text-similarity cluster must not put near-copies in different buckets
            d["group"] = [f"pub:{kind}{x}" for x in union_groups(g, d["group"].to_numpy())]
            d = d[~drop].reset_index(drop=True)
            msg_reps[kind] = rep
        parts.append(d)
    msgs = pd.concat(parts, ignore_index=True)
    reports["message_families"] = msg_reps

    hard_calls, hard_msgs = generate_hard_set()
    hard_msgs = hard_msgs.assign(url=hard_msgs["text"].map(lambda t: next(iter(extract_urls(t)), "")))
    calls["is_hard"], msgs["is_hard"] = False, False
    hard_calls["is_hard"], hard_msgs["is_hard"] = True, True
    calls = pd.concat([calls, hard_calls.drop_duplicates("text")], ignore_index=True)
    msgs = pd.concat([msgs, hard_msgs.drop_duplicates("text")], ignore_index=True)
    calls["tier"] = _tier(calls, C.MATCHED_CALL_SOURCES)
    msgs["tier"] = _tier(msgs, C.MATCHED_MSG_SOURCES)
    return {"calls": calls, "msgs": msgs, "reports": reports}


def _take(pool: pd.DataFrame, n: int, rng: np.random.Generator, kinds=None) -> pd.DataFrame:
    """Draw ``n`` distinct rows, spread evenly over the message kinds when ``kinds`` is given."""
    pool = pool.sample(frac=1, random_state=int(rng.integers(1 << 31)))
    if kinds is None:
        return pool.head(n)
    per = int(np.ceil(n / len(kinds)))
    out = pd.concat([pool[pool["kind"] == k].head(per) for k in kinds])
    if len(out) < n:
        out = pd.concat([out, pool.drop(out.index, errors="ignore").head(n - len(out))])
    return out.sample(frac=1, random_state=int(rng.integers(1 << 31))).head(n)


def build_records(calls: pd.DataFrame, msgs: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """Bucket the families, then pair calls with messages inside every (tier, bucket) with the OR label rule."""
    rng = np.random.default_rng(seed)
    calls, msgs = calls.reset_index(drop=True).copy(), msgs.reset_index(drop=True).copy()

    def stratum(d: pd.DataFrame, extra: pd.Series | None = None) -> pd.Series:
        # a family is placed ONCE: its stratum comes from the family as a whole, never from each row separately
        fam_label = d.groupby("group")["label"].agg(lambda x: int(x.mean() >= 0.5))
        fam_tier = d.groupby("group")["tier"].agg(lambda x: x.iloc[0])
        s_ = d["group"].map(fam_label).astype(str) + "|" + d["group"].map(fam_tier).astype(str)
        return s_ if extra is None else extra + "|" + s_

    calls["bucket"] = assign_buckets(calls["group"], calls["label"], C.N_BUCKETS, rng, strata=stratum(calls))
    msgs["bucket"] = assign_buckets(msgs["group"], msgs["label"], C.N_BUCKETS, rng, strata=stratum(msgs, msgs["kind"]))
    assert calls.groupby("group")["bucket"].nunique().max() == 1 and msgs.groupby("group")["bucket"].nunique().max() == 1

    mix = C.PAIR_MIX
    p_fraud_call, p_fraud_msg = mix["FF"] + mix["FS"], mix["FF"] + mix["SF"]
    rows = []
    for tier in C.TIERS:
        ct, mt = calls[calls.tier == tier], msgs[msgs.tier == tier]
        for b in range(C.N_BUCKETS):
            cb, mb = ct[ct.bucket == b], mt[mt.bucket == b]
            fc, sc, fm, sm = cb[cb.label == 1], cb[cb.label == 0], mb[mb.label == 1], mb[mb.label == 0]
            n_total = int(min(len(fc) / p_fraud_call, len(sc) / (1 - p_fraud_call), len(fm) / p_fraud_msg, len(sm) / (1 - p_fraud_msg)))
            if n_total < 4:
                continue
            n = {k: int(round(v * n_total)) for k, v in mix.items()}
            calls_f, calls_s = _take(fc, n["FF"] + n["FS"], rng), _take(sc, n["SF"] + n["SS"], rng)
            kinds = KINDS_BY_TIER[tier]
            msgs_f, msgs_s = _take(fm, n["FF"] + n["SF"], rng, kinds), _take(sm, n["FS"] + n["SS"], rng, kinds)
            plan = {"FF": (calls_f.iloc[:n["FF"]], msgs_f.iloc[:n["FF"]]), "FS": (calls_f.iloc[n["FF"]:], msgs_s.iloc[:n["FS"]]),
                    "SF": (calls_s.iloc[:n["SF"]], msgs_f.iloc[n["FF"]:]), "SS": (calls_s.iloc[n["SF"]:], msgs_s.iloc[n["FS"]:])}
            for ptype, (c, m) in plan.items():
                k = min(len(c), len(m))
                c, m = c.head(k), m.head(k)
                rows.append(pd.DataFrame({
                    "tier": tier, "bucket": b, "pair_type": ptype,
                    "call_label": c["label"].values, "msg_label": m["label"].values,
                    "call_id": c["call_id"].values, "call_text": c["text"].values, "call_source": c["source"].values, "call_group": c["group"].values,
                    "item_id": m["item_id"].values, "msg_kind": m["kind"].values, "msg_text": m["text"].values, "msg_url": m["url"].fillna("").values,
                    "msg_source": m["source"].values, "msg_lang": m["lang"].values, "msg_group": m["group"].values}))
    rec = pd.concat(rows, ignore_index=True)
    rec["label"] = (rec["call_label"] | rec["msg_label"]).astype(int)            # THE RULE: either part is fraud -> the attack is fraud
    rec["split"] = np.where(rec.bucket.isin(C.TEST_BUCKETS), "test", np.where(rec.bucket.isin(C.VAL_BUCKETS), "val", "train"))
    rec["fold"] = np.where(rec.split == "test", -1, rec.bucket // 2)
    rec = rec.sample(frac=1, random_state=seed).sort_values(["bucket"], kind="stable").reset_index(drop=True)
    rec.insert(0, "record_id", [f"rec_{i:05d}" for i in range(len(rec))])
    return rec


def prepare_data(save: bool = True) -> dict:
    """The whole data pipeline; with ``save`` it also writes the split files and the CSV folders."""
    from .dataset_access import export_dataset_folders
    if save and C.FINAL_TEST_LOCK.exists():
        raise RuntimeError("The final test already ran: re-splitting would invalidate its results.")
    C.ensure_dirs()
    pools = build_pools()
    rec = build_records(pools["calls"], pools["msgs"])
    out = {**pools, "records": rec}
    if save:
        out["manifest"] = save_records(rec)
        rec.to_parquet(C.PROCESSED_DIR / "records_all.parquet", index=False)
        (C.TABLES_DIR / "data_report.json").write_text(json.dumps(pools["reports"], indent=2, default=str))
        export_dataset_folders()
    return out
