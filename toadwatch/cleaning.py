"""Step 2 - clean the data and remove duplicates.

Rules (each one is counted in the returned report so the notebook can show it):

calls
  * keep only English, non-empty transcripts of at least MIN_TRANSCRIPT_CHARS characters
    (NC State has 54 Chinese transcripts; the model's keyword features are EN/ES);
  * strip speaker tags (``Suspect:``, ``caller:``, ``agent:`` ...) because several tags exist
    only in one class and would let the model "cheat";
  * drop exact duplicates (after normalising case/punctuation/whitespace);
  * drop texts that appear with BOTH labels (contradictory ground truth);
  * drop NEAR-duplicates (TF-IDF cosine > NEAR_DUP_THRESHOLD): robocall scripts are replayed
    with tiny wording changes, and a copy in train plus a copy in test would inflate accuracy.

messages
  * same text-level duplicate rules;
  * URLs are de-duplicated by host name, and a host that appears in both classes is dropped,
    so the same domain can never land in the training set and the test set (proposal,
    "The Instrument").
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config as C

SPEAKER_RE = re.compile(r"\b(?:suspect|innocent|caller|receiver|agent|customer)\s*:\s*", re.IGNORECASE)
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"'\)\]\[]+", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^0-9a-záéíóúñü]+")


def strip_speakers(text: str) -> str:
    return re.sub(r"\s+", " ", SPEAKER_RE.sub(" ", text)).strip()


def norm_key(text: str) -> str:
    """Case/punctuation/whitespace-insensitive key used to detect duplicates."""
    return _NON_ALNUM.sub(" ", str(text).lower()).strip()


def extract_urls(text: str) -> list[str]:
    return [u.rstrip(".,;:!?") for u in URL_RE.findall(str(text))]


def first_url(text: str) -> str:
    urls = extract_urls(text)
    return urls[0] if urls else ""


def host_of(url: str) -> str:
    """Lower-case host name without ``www.`` and port; '' if it cannot be parsed."""
    url = str(url).strip()
    if "://" not in url:
        url = "//" + url
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return ""
    host = host.lower().strip(".")
    return host[4:] if host.startswith("www.") else host


def drop_duplicates_and_conflicts(df: pd.DataFrame, key_col: str) -> tuple[pd.DataFrame, dict]:
    """Remove repeated keys; keys seen with both labels are removed entirely."""
    n0 = len(df)
    labels_per_key = df.groupby(key_col)["label"].nunique()
    conflicting = set(labels_per_key[labels_per_key > 1].index)
    out = df[~df[key_col].isin(conflicting)]
    n_conflict = n0 - len(out)
    out = out.drop_duplicates(subset=key_col, keep="first")
    return out, {"conflicting_label_rows": n_conflict, "duplicate_rows": len(df) - n_conflict - len(out)}


def drop_near_duplicates(df: pd.DataFrame, text_col: str = "text",
                         threshold: float = C.NEAR_DUP_THRESHOLD, block: int = 1000) -> pd.DataFrame:
    """Greedy near-duplicate removal: keep a row only if no EARLIER kept row is more similar
    than ``threshold`` (TF-IDF cosine). Row order decides which copy survives."""
    df = df.reset_index(drop=True)
    if len(df) < 2:
        return df
    X = TfidfVectorizer(sublinear_tf=True, dtype=np.float32).fit_transform(df[text_col])
    keep = np.ones(len(df), dtype=bool)
    for start in range(0, len(df), block):
        sims = (X[start:start + block] @ X.T).toarray()
        for r in range(sims.shape[0]):
            i = start + r
            if keep[i] and i > 0:
                earlier = sims[r, :i]
                if (earlier[keep[:i]] > threshold).any():
                    keep[i] = False
    return df[keep].reset_index(drop=True)


def near_duplicate_share(texts_a: pd.Series, texts_b: pd.Series,
                         threshold: float = C.NEAR_DUP_THRESHOLD) -> float:
    """Share of ``texts_b`` that has a near-identical text (cosine > threshold) in ``texts_a``.
    Used on train vs validation to prove the split has no script copied across it."""
    vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32).fit(pd.concat([texts_a, texts_b]))
    best = (vec.transform(texts_b) @ vec.transform(texts_a).T).max(axis=1).toarray().ravel()
    return float((best > threshold).mean())


# ------------------------------------------------------------------ calls
def clean_calls(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {"rows_raw": len(df)}
    out = df.dropna(subset=["text"]).copy()
    out["text"] = out["text"].astype(str).map(strip_speakers)
    n = len(out)
    out = out[(out["lang"] == "en") & (out["text"].str.len() >= C.MIN_TRANSCRIPT_CHARS)]
    report["unusable_rows"] = n - len(out) + (len(df) - n)
    out["_key"] = out["text"].map(norm_key)
    out, dup = drop_duplicates_and_conflicts(out, "_key")
    report.update(dup)
    out = out.drop(columns="_key")
    n = len(out)
    out = drop_near_duplicates(out)
    report["near_duplicate_rows"] = n - len(out)
    report["rows_clean"] = len(out)
    report["class_counts"] = out["label"].value_counts().sort_index().to_dict()
    return out, report


# ------------------------------------------------------------------ messages
def clean_messages(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {"rows_raw": len(df)}
    out = df.dropna(subset=["text"]).copy()
    out["text"] = out["text"].astype(str).str.split().str.join(" ")      # one space between words, everywhere (spacing must not reveal a source)
    out = out[out["text"].str.len() >= C.MIN_TEXT_CHARS]
    report["unusable_rows"] = len(df) - len(out)

    out["url"] = out["text"].where(out["kind"] == "url", out["text"].map(first_url))
    out["url"] = out["url"].fillna("")

    # URL items: one row per host, hosts that appear in both classes disappear.
    is_url = out["kind"] == "url"
    urls = out[is_url].copy()
    urls["_key"] = urls["url"].map(host_of)
    urls = urls[urls["_key"] != ""]
    urls, url_dup = drop_duplicates_and_conflicts(urls, "_key")
    report["url_duplicate_hosts"] = url_dup

    other = out[~is_url].copy()
    other["_key"] = other["text"].map(norm_key)
    other = other[other["_key"] != ""]
    other, text_dup = drop_duplicates_and_conflicts(other, "_key")
    report["text_duplicates"] = text_dup

    cleaned = pd.concat([other, urls], ignore_index=True).drop(columns="_key")
    report["rows_clean"] = len(cleaned)
    report["counts_by_kind_label"] = {f"{k}|{l}": int(v) for (k, l), v in
                                      cleaned.groupby(["kind", "label"]).size().items()}
    return cleaned, report
