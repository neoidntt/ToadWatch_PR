"""Step 1 decision - how a call is paired with a message, email or link.

A TOAD case is "a call followed by a message-side item". No public dataset contains real
call+message pairs, so the pairs are built here, reproducibly, with these rules:

1. A call is only paired with a message-side item of the SAME label (fraud call + fraud
   message, safe call + safe message). This is the TOAD pattern the proposal targets.
2. Pairing happens INSIDE each split, so nothing crosses train / val / test.
3. Every call is used exactly once; message items are drawn evenly from SMS, email and URL
   (as far as the pool allows), so the message type is not correlated with the label.
4. The draw uses the fixed project seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

KINDS = ("sms", "email", "url")


def balance_calls(calls: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """Equal number of safe and fraud calls: keep every labelled safe dialogue and fill the
    gap with an equal share of each big safe corpus (banking / telecom)."""
    n_fraud = int((calls["label"] == 1).sum())
    safe = calls[calls["label"] == 0]
    is_corpus = safe["source"].isin(["banking_300k", "telecom_200k", "telecom_corpus_supplimental"])
    fixed, corpus = safe[~is_corpus], safe[is_corpus]
    need = max(n_fraud - len(fixed), 0)
    if need and len(corpus):
        per_source = int(np.ceil(need / corpus["source"].nunique()))
        corpus = pd.concat([g.sample(min(len(g), per_source), random_state=seed)
                            for _, g in corpus.groupby("source")])
        corpus = corpus.sample(frac=1, random_state=seed).head(need)
        safe = pd.concat([fixed, corpus])
    else:
        safe = fixed.sample(n_fraud, random_state=seed) if len(fixed) > n_fraud else fixed
    fraud = calls[calls["label"] == 1]
    n = min(len(fraud), len(safe))
    return pd.concat([fraud.sample(n, random_state=seed), safe.sample(n, random_state=seed)],
                     ignore_index=True)


def balance_messages(msgs: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """Within every message type keep the same number of safe and fraud items, so the type
    carries no information about the label."""
    parts = []
    for kind, g in msgs.groupby("kind"):
        n = int(g.groupby("label").size().min())
        parts.extend(x.sample(n, random_state=seed) for _, x in g.groupby("label"))
    return pd.concat(parts, ignore_index=True)


def _draw_messages(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Pick ``n`` distinct items, evenly across the kinds, topping up from what is left."""
    pool = pool.sample(frac=1, random_state=int(rng.integers(1 << 31)))
    take = []
    per_kind = int(np.ceil(n / len(KINDS)))
    for kind in KINDS:
        take.append(pool[pool["kind"] == kind].head(per_kind))
    chosen = pd.concat(take)
    if len(chosen) < n:
        rest = pool.drop(chosen.index)
        chosen = pd.concat([chosen, rest.head(n - len(chosen))])
    chosen = chosen.sample(frac=1, random_state=int(rng.integers(1 << 31)))
    if len(chosen) < n:
        raise ValueError(f"message pool too small: need {n}, have {len(chosen)}")
    return chosen.head(n)


def build_records(calls: pd.DataFrame, msgs: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """Pair calls with messages inside each (split, label). Both inputs need a ``split`` column."""
    rng = np.random.default_rng(seed)
    out = []
    for split in ("train", "val", "test"):
        for label in (0, 1):
            c = calls[(calls["split"] == split) & (calls["label"] == label)].sample(frac=1, random_state=seed)
            m_pool = msgs[(msgs["split"] == split) & (msgs["label"] == label)]
            m = _draw_messages(m_pool, len(c), rng)
            out.append(pd.DataFrame({
                "split": split,
                "label": label,
                "call_id": c["call_id"].values,
                "call_text": c["text"].values,
                "call_source": c["source"].values,
                "item_id": m["item_id"].values,
                "msg_kind": m["kind"].values,
                "msg_text": m["text"].values,
                "msg_url": m["url"].values,
                "msg_source": m["source"].values,
                "msg_lang": m["lang"].values,
            }))
    rec = pd.concat(out, ignore_index=True)
    rec.insert(0, "record_id", [f"rec_{i:05d}" for i in range(len(rec))])
    return rec
