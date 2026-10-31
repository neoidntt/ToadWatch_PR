"""Step 1 - read every raw source into two unified tables.

calls : call_id, text, label, source, lang
msgs  : item_id, kind (sms | email | url), text, label, source, lang

Label convention (taken from the FOLDER, never from the files' own label columns, because
e.g. the URL csv files use ``ClassLabel`` with 1.0 = legitimate): 0 = safe, 1 = fraud.

Large files are streamed in chunks and sampled uniformly (reservoir sampling), so the
whole project runs on a laptop with little free RAM.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from . import config as C

CALL_COLS = ["call_id", "text", "label", "source", "lang"]
MSG_COLS = ["item_id", "kind", "text", "label", "source", "lang", "group"]


# ------------------------------------------------------------------ sampling helpers
def reservoir_sample(chunks: Iterable[pd.DataFrame], k: int, seed: int) -> pd.DataFrame:
    """Uniform random sample of ``k`` rows from a stream of DataFrames (bounded memory)."""
    rng = np.random.default_rng(seed)
    kept: pd.DataFrame | None = None
    for chunk in chunks:
        chunk = chunk.copy()
        chunk["_key"] = rng.random(len(chunk))
        kept = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
        if len(kept) > k:
            kept = kept.nsmallest(k, "_key")
    if kept is None:
        return pd.DataFrame()
    return kept.drop(columns="_key").reset_index(drop=True)


def iter_json_array(path: Path, read_bytes: int = 1 << 22) -> Iterator[dict]:
    """Stream the objects of a huge top-level JSON array without loading the file."""
    decoder = json.JSONDecoder()
    buf, pos = "", 0
    with open(path, "r", encoding="utf-8") as fh:
        eof = False
        started = False
        while True:
            if not eof and len(buf) - pos < read_bytes // 4:
                data = fh.read(read_bytes)
                if data:
                    buf = buf[pos:] + data
                    pos = 0
                else:
                    eof = True
            while pos < len(buf) and buf[pos] in " \r\n\t,":
                pos += 1
            if not started:
                if pos < len(buf) and buf[pos] == "[":
                    pos += 1
                    started = True
                    continue
                if eof:
                    return
            if pos >= len(buf):
                if eof:
                    return
                continue
            if buf[pos] == "]":
                return
            try:
                obj, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    return
                data = fh.read(read_bytes)
                if not data:
                    eof = True
                buf = buf[pos:] + data
                pos = 0
                continue
            pos = end
            yield obj


def _json_chunks(path: Path, size: int = 50_000) -> Iterator[pd.DataFrame]:
    batch: list[dict] = []
    for obj in iter_json_array(path):
        batch.append(obj)
        if len(batch) >= size:
            yield pd.DataFrame(batch)
            batch = []
    if batch:
        yield pd.DataFrame(batch)


# ------------------------------------------------------------------ calls
def _small_dialogue_file(path: Path, label: int, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({
        "call_id": [f"{source}_{i}" for i in range(len(df))],
        "text": df["dialogue"],
        "label": label,
        "source": source,
        "lang": "en",
    })


def load_labeled_calls() -> pd.DataFrame:
    """Small labelled dialogue files (800 rows each) in both class folders."""
    parts = []
    for folder, label in ((C.CALL_SAFE_DIR, 0), (C.CALL_DANGEROUS_DIR, 1)):
        for name in ("agent_conversation_labels", "labels", "scam_dialogue_label"):
            parts.append(_small_dialogue_file(folder / f"{name}_{label}.csv", label, f"{name}_{label}"))
    yt = pd.read_csv(C.CALL_DANGEROUS_DIR / "youtube_scam_conversations.csv")
    parts.append(pd.DataFrame({
        "call_id": [f"youtube_{i}" for i in range(len(yt))],
        "text": yt["dialogue"], "label": 1, "source": "youtube_scam", "lang": "en",
    }))
    return pd.concat(parts, ignore_index=True)


def load_nc_state_calls() -> pd.DataFrame:
    """FTC-confirmed robocall transcripts (NC State Robocall Audio Dataset). All fraud."""
    meta = pd.read_csv(C.NC_STATE_DIR / "metadata.csv")
    return pd.DataFrame({
        "call_id": [f"ncstate_{i}_{Path(f).stem}" for i, f in enumerate(meta["file_name"])],
        "text": meta["transcript"],
        "label": 1,
        "source": "ncstate_robocall",
        "lang": meta["language"].fillna("en"),
    })


def load_corpus_calls(path: Path, n: int, seed: int, chunksize: int = 500_000) -> pd.DataFrame:
    """Sample ``n`` whole conversations from a turn-per-row corpus and join their turns."""
    ids: set[str] = set()
    for ch in pd.read_csv(path, usecols=["conversation_id"], chunksize=chunksize):
        ids.update(ch["conversation_id"].unique())
    rng = np.random.default_rng(seed)
    ordered = sorted(ids)
    chosen = set(rng.choice(ordered, size=min(n, len(ordered)), replace=False))

    rows = []
    for ch in pd.read_csv(path, usecols=["conversation_id", "date_time", "text"], chunksize=chunksize):
        rows.append(ch[ch["conversation_id"].isin(chosen)])
    turns = pd.concat(rows, ignore_index=True).sort_values(["conversation_id", "date_time"], kind="stable")
    convo = turns.groupby("conversation_id", sort=False)["text"].apply(lambda s: " ".join(map(str, s)))
    source = path.stem
    return pd.DataFrame({
        "call_id": [f"{source}_{cid[:12]}" for cid in convo.index],
        "text": convo.values, "label": 0, "source": source, "lang": "en",
    })


def load_calls(corpus_n: int | None = None, seed: int = C.SEED) -> pd.DataFrame:
    """All call transcripts (raw, uncleaned)."""
    corpus_n = corpus_n or int(C.CORPUS_CALL_POOL * C.OVERSAMPLE)
    parts = [load_labeled_calls(), load_nc_state_calls()]
    for i, name in enumerate(("banking_300k.csv", "telecom_200k.csv", "telecom_corpus_supplimental.csv")):
        parts.append(load_corpus_calls(C.CALL_SAFE_DIR / name, corpus_n, seed + i))
    return pd.concat(parts, ignore_index=True)[CALL_COLS]


# ------------------------------------------------------------------ messages
def _msg_frame(df: pd.DataFrame, kind: str, label: int, source: str, lang: str = "en") -> pd.DataFrame:
    out = pd.DataFrame({"text": df["text"].values})
    out["kind"] = kind
    out["label"] = label
    out["source"] = source
    out["lang"] = lang
    out["item_id"] = [f"{source}_{i}" for i in range(len(out))]
    out["group"] = None                      # filled later by clustering (or by campaign id for the campaign file)
    return out[MSG_COLS]


def load_sms_campaigns(seed: int = C.SEED) -> pd.DataFrame:
    """60k real phishing SMS (English + Spanish), fraud only, grouped in scam campaigns.

    The file repeats each campaign many times with tiny changes, so: whitespace is normalised (its irregular spacing
    would otherwise reveal the file), exact repeats are dropped, and at most ``SMS_CAMPAIGN_MAX_PER_CAMPAIGN`` messages per
    campaign (and ``SMS_CAMPAIGN_CAP`` overall) are kept. The campaign id becomes the family id, so a whole campaign
    always lands in one split.
    """
    if not C.SMS_CAMPAIGNS_CSV.exists():
        return pd.DataFrame(columns=MSG_COLS)
    df = pd.read_csv(C.SMS_CAMPAIGNS_CSV, usecols=["message", "language_code", "campaign_id"])
    df["message"] = df["message"].astype(str).str.split().str.join(" ")
    df = df[df["message"].str.len() >= C.MIN_TEXT_CHARS].drop_duplicates("message").sample(frac=1, random_state=seed)
    has = df["campaign_id"].notna()
    keep = pd.concat([df[has].groupby("campaign_id").head(C.SMS_CAMPAIGN_MAX_PER_CAMPAIGN), df[~has]])
    keep = keep.sample(n=min(len(keep), C.SMS_CAMPAIGN_CAP), random_state=seed).reset_index(drop=True)
    out = pd.DataFrame({"text": keep["message"].values})
    out["kind"], out["label"], out["source"] = "sms", 1, "sms_phishing_campaigns"
    out["lang"] = keep["language_code"].fillna("en").values
    out["item_id"] = [f"sms_campaign_{i}" for i in range(len(out))]
    out["group"] = [f"camp:{int(c)}" if pd.notna(c) else None for c in keep["campaign_id"]]
    return out[MSG_COLS]


def load_sms() -> pd.DataFrame:
    """SMS spam/ham. The `augmented_*` files repeat the English text, so only the Spanish
    translation column is new information."""
    parts = []
    for folder, name, label in ((C.MSG_SAFE_DIR, "sms_ham", 0), (C.MSG_DANGEROUS_DIR, "sms_spam", 1)):
        parts.append(_msg_frame(pd.read_csv(folder / f"{name}.csv"), "sms", label, name))
    for folder, name, label in ((C.MSG_SAFE_DIR, "augmented_ham_en_es", 0),
                                (C.MSG_DANGEROUS_DIR, "augmented_spam_en_es", 1)):
        df = pd.read_csv(folder / f"{name}.csv").rename(columns={"text_es": "es"})
        parts.append(_msg_frame(df.assign(text=df["es"]), "sms", label, name + "_es", lang="es"))
    parts.append(load_sms_campaigns())
    return pd.concat(parts, ignore_index=True)


def _email_chunks(path: Path, chunksize: int = 50_000) -> Iterator[pd.DataFrame]:
    for ch in pd.read_csv(path, usecols=["subject", "body", "source"], chunksize=chunksize):
        subj = ch["subject"].fillna("").astype(str)
        body = ch["body"].fillna("").astype(str).str.slice(0, C.MAX_BODY_CHARS)
        yield pd.DataFrame({"text": (subj + "\n" + body).str.strip(), "orig": ch["source"].astype(str)})


def load_emails(cap_per_class: int | None = None, seed: int = C.SEED) -> pd.DataFrame:
    cap = cap_per_class or int(C.EMAIL_CAP_PER_CLASS * C.OVERSAMPLE)
    parts = []
    for i, (folder, name, label) in enumerate(((C.MSG_SAFE_DIR, "legitimate_emails", 0),
                                               (C.MSG_DANGEROUS_DIR, "spam_scam_emails", 1))):
        sample = reservoir_sample(_email_chunks(folder / f"{name}.csv"), cap, seed + i)
        df = _msg_frame(sample, "email", label, name)
        df["source"] = name + "_" + sample["orig"].values
        parts.append(df)
    # Spanish emails (SpaPhish): small, load fully.
    for folder, name, label in ((C.MSG_SAFE_DIR, "spaphish_legitimate_0", 0),
                                (C.MSG_DANGEROUS_DIR, "spaphish_phishing_1", 1)):
        raw = pd.read_csv(folder / f"{name}.csv", usecols=["subject", "body"])
        text = (raw["subject"].fillna("").astype(str) + "\n"
                + raw["body"].fillna("").astype(str).str.slice(0, C.MAX_BODY_CHARS)).str.strip()
        parts.append(_msg_frame(pd.DataFrame({"text": text}), "email", label, name, lang="es"))
    return pd.concat(parts, ignore_index=True)


def load_urls(cap_per_class: int | None = None, seed: int = C.SEED) -> pd.DataFrame:
    """URL lists: csv files with pre-computed columns plus the two big JSON files."""
    cap = cap_per_class or int(C.URL_CAP_PER_CLASS * C.OVERSAMPLE)
    parts = []
    for i, (folder, csv_name, json_name, label) in enumerate((
            (C.MSG_SAFE_DIR, "legitimate_urls", "legitimate", 0),
            (C.MSG_DANGEROUS_DIR, "scam_spam_urls", "spam", 1))):
        csv_df = pd.read_csv(folder / f"{csv_name}.csv", usecols=["URL"]).rename(columns={"URL": "text"})
        parts.append(_msg_frame(csv_df, "url", label, csv_name))
        js = reservoir_sample(_json_chunks(folder / f"{json_name}.json"), cap, seed + 10 + i)
        parts.append(_msg_frame(js[["text"]], "url", label, f"{json_name}_json"))
    return pd.concat(parts, ignore_index=True)


def load_messages(seed: int = C.SEED) -> pd.DataFrame:
    """SMS + email + URL items (raw, uncleaned)."""
    return pd.concat([load_sms(), load_emails(seed=seed), load_urls(seed=seed)], ignore_index=True)[MSG_COLS]


# ------------------------------------------------------------------ Puerto Rico cases
PR_COLUMNS = ["case_id", "call_transcript", "message_text", "label", "source", "notes"]


def load_pr_cases() -> pd.DataFrame | None:
    """Puerto Rico-specific secondary test set (DACO alerts, faculty, documented cases).

    Returns None until the user fills ``data/external/pr_cases.csv`` with real rows.
    """
    if not C.PR_CASES_CSV.exists():
        return None
    df = pd.read_csv(C.PR_CASES_CSV)
    missing = set(PR_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{C.PR_CASES_CSV} is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["call_transcript", "message_text", "label"]).reset_index(drop=True)
    return df if len(df) else None
