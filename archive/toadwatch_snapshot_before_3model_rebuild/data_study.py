"""v3 data builder: public data + hard set -> families -> buckets -> mixed pairs under the TOAD rule.

Label rule (from the proposal): a case is FRAUD only if the call AND the message are both fraud.
Pair types: FF (fraud call + fraud message) = fraud;  FS, SF, SS = not fraud.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .cleaning import clean_calls, clean_messages, host_of
from .dataset_access import export_dataset_folders
from .data_loading import load_calls, load_messages
from .grouping import assign_buckets, group_texts
from .hard_data import generate_hard_set
from .splitting import save_records

KINDS = ("sms", "email", "url")


def _base_domain(url: str) -> str:
    labels = host_of(url).split(".")
    return ".".join(labels[-2:]) if len(labels) >= 2 else host_of(url)


def build_pools(save_reports: bool = True) -> dict:
    """Clean the public data, add the hard set, return calls / messages with a ``group`` column."""
    calls_raw, msgs_raw = load_calls(), load_messages()
    calls, calls_rep = clean_calls(calls_raw)
    msgs, msgs_rep = clean_messages(msgs_raw)

    hard_calls, hard_msgs = generate_hard_set()
    hard_msgs = hard_msgs.copy()
    hard_msgs["url"] = hard_msgs["text"].map(lambda t: next(iter(__import__("toadwatch.cleaning", fromlist=["x"]).extract_urls(t)), ""))
    hard_calls = hard_calls.drop_duplicates("text")
    hard_msgs = hard_msgs.drop_duplicates("text")

    reports = {"calls": calls_rep, "messages": msgs_rep}
    # --- families of near-identical texts (public data only; hard data already has one group per skeleton)
    g, drop, rep = group_texts(calls["text"], calls["label"])
    calls = calls.assign(group=[f"pub:c{x}" for x in g])[~drop].reset_index(drop=True)
    reports["call_groups"] = rep
    parts, msg_reps = [], {}
    for kind, d in msgs.groupby("kind"):
        d = d.reset_index(drop=True)
        if kind == "url":                                   # same registered domain = same family
            d["group"] = "pub:u:" + d["url"].map(_base_domain)
            msg_reps[kind] = {"texts": len(d), "groups": int(d["group"].nunique())}
        else:
            g, drop, rep = group_texts(d["text"], d["label"])
            d["group"] = [f"pub:{kind}{x}" for x in g]
            d = d[~drop].reset_index(drop=True)
            msg_reps[kind] = rep
        parts.append(d)
    msgs = pd.concat(parts, ignore_index=True)
    reports["message_groups"] = msg_reps

    calls["is_hard"] = False
    msgs["is_hard"] = False
    hard_calls["is_hard"] = True
    hard_msgs["is_hard"] = True
    calls = pd.concat([calls, hard_calls], ignore_index=True)
    msgs = pd.concat([msgs, hard_msgs], ignore_index=True)
    return {"calls": calls, "msgs": msgs, "reports": reports}


def _take(pool: pd.DataFrame, n: int, rng: np.random.Generator, kinds=None) -> pd.DataFrame:
    """Draw ``n`` distinct rows, evenly across the message kinds when ``kinds`` is given."""
    pool = pool.sample(frac=1, random_state=int(rng.integers(1 << 31)))
    if kinds is None:
        return pool.head(n)
    per = int(np.ceil(n / len(kinds)))
    out = pd.concat([pool[pool["kind"] == k].head(per) for k in kinds])
    if len(out) < n:
        out = pd.concat([out, pool.drop(out.index, errors="ignore").head(n - len(out))])
    return out.sample(frac=1, random_state=int(rng.integers(1 << 31))).head(n)


def build_records(calls: pd.DataFrame, msgs: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """Bucket the families, then pair inside every bucket with the TOAD label rule and PAIR_MIX shares."""
    rng = np.random.default_rng(seed)
    calls = calls.reset_index(drop=True).copy()
    msgs = msgs.reset_index(drop=True).copy()
    # A family is placed ONCE, even if it holds both classes: its stratum comes from the family's majority label
    # (never from each row), otherwise the same family could be bucketed twice and land on both sides of a split.
    def stratum(d: pd.DataFrame, extra: pd.Series | None = None) -> pd.Series:
        fam_label = d.groupby("group")["label"].agg(lambda x: int(x.mean() >= 0.5))
        fam_hard = d.groupby("group")["is_hard"].max()
        s_ = d["group"].map(fam_label).astype(str) + "|" + d["group"].map(fam_hard).astype(str)
        return s_ if extra is None else extra + "|" + s_
    calls["bucket"] = assign_buckets(calls["group"], calls["label"], C.N_BUCKETS, rng, strata=stratum(calls))
    msgs["bucket"] = assign_buckets(msgs["group"], msgs["label"], C.N_BUCKETS, rng, strata=stratum(msgs, msgs["kind"]))
    assert calls.groupby("group")["bucket"].nunique().max() == 1 and msgs.groupby("group")["bucket"].nunique().max() == 1
    mix = C.PAIR_MIX
    p_fraud_call = mix["FF"] + mix["FS"]               # share of records whose call is fraud
    rows = []
    # Two regimes, built separately and never mixed inside a record:
    #   "public": public data. Every source file holds ONE class, so a model can score high by recognising the
    #             dataset's writing style - an optimistic regime.
    #   "hard":   my style-matched set. Both classes are written alike, so only the content can give the answer.
    for regime, is_hard, kinds in (("public", False, KINDS), ("hard", True, ("sms", "email"))):
        cr, mr = calls[calls.is_hard == is_hard], msgs[msgs.is_hard == is_hard]
        for b in range(C.N_BUCKETS):
            cb, mb = cr[cr.bucket == b], mr[mr.bucket == b]
            fc, sc = cb[cb.label == 1], cb[cb.label == 0]
            fm, sm = mb[mb.label == 1], mb[mb.label == 0]
            n_total = int(min(len(fc) / p_fraud_call, len(sc) / (1 - p_fraud_call),
                              len(fm) / (mix["FF"] + mix["SF"]), len(sm) / (mix["FS"] + mix["SS"])))
            n = {k: int(round(v * n_total)) for k, v in mix.items()}
            calls_f = _take(fc, n["FF"] + n["FS"], rng)
            calls_s = _take(sc, n["SF"] + n["SS"], rng)
            msgs_f = _take(fm, n["FF"] + n["SF"], rng, kinds)
            msgs_s = _take(sm, n["FS"] + n["SS"], rng, kinds)
            # fraud calls -> [FF | FS], safe calls -> [SF | SS]; fraud messages -> [FF | SF], safe messages -> [FS | SS]
            plan = {"FF": (calls_f.iloc[:n["FF"]], msgs_f.iloc[:n["FF"]]), "FS": (calls_f.iloc[n["FF"]:], msgs_s.iloc[:n["FS"]]),
                    "SF": (calls_s.iloc[:n["SF"]], msgs_f.iloc[n["FF"]:]), "SS": (calls_s.iloc[n["SF"]:], msgs_s.iloc[n["FS"]:])}
            for ptype, (c, m) in plan.items():
                m = m.head(len(c))
                c = c.head(len(m))
                rows.append(pd.DataFrame({
                    "regime": regime, "bucket": b, "pair_type": ptype, "label": int(ptype == "FF"),
                    "call_id": c["call_id"].values, "call_text": c["text"].values, "call_source": c["source"].values,
                    "call_group": c["group"].values, "call_label": c["label"].values, "call_is_hard": c["is_hard"].values,
                    "item_id": m["item_id"].values, "msg_kind": m["kind"].values, "msg_text": m["text"].values,
                    "msg_url": m["url"].fillna("").values, "msg_source": m["source"].values, "msg_lang": m["lang"].values,
                    "msg_group": m["group"].values, "msg_label": m["label"].values, "msg_is_hard": m["is_hard"].values}))
    rec = pd.concat(rows, ignore_index=True)
    rec["split"] = np.where(rec.bucket.isin(C.TEST_BUCKETS), "test", np.where(rec.bucket.isin(C.VAL_BUCKETS), "val", "train"))
    rec["fold"] = np.where(rec.split == "test", -1, rec.bucket // 2)
    rec = rec.sample(frac=1, random_state=seed).sort_values(["bucket"], kind="stable").reset_index(drop=True)
    rec.insert(0, "record_id", [f"rec_{i:05d}" for i in range(len(rec))])
    return rec


def prepare_data(save: bool = True) -> dict:
    """Run the whole v3 data pipeline; with ``save`` also writes the split files and the CSV folders."""
    if save and C.FINAL_TEST_LOCK.exists():
        raise RuntimeError("The v3 final test already ran: re-splitting would invalidate its results.")
    C.ensure_dirs()
    pools = build_pools()
    rec = build_records(pools["calls"], pools["msgs"])
    out = {**pools, "records": rec}
    if save:
        out["manifest"] = save_records(rec)
        rec.to_parquet(C.PROCESSED_DIR / "records_all.parquet", index=False)
        export_dataset_folders()
        import json
        (C.TABLES_DIR / "v3_data_report.json").write_text(json.dumps(pools["reports"], indent=2, default=str))
    return out
