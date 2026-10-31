"""Step 3 - features computed from a message, email or link.

The same function is used whatever the source (SMS, email or URL list), as the proposal
asks: wording features from the text plus link features from the first URL found in it
(shorteners, HTTPS, look-alike domain names, domain reputation). Names start with ``msg_``.
"""
from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from functools import lru_cache
from urllib.parse import parse_qsl, urlsplit

import numpy as np
import pandas as pd

from . import config as C
from .cleaning import extract_urls, host_of

PREFIX = "msg_"

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
              "cutt.ly", "shorturl.at", "tiny.cc", "bit.do", "t.ly", "rb.gy", "lnkd.in", "s.id",
              "v.gd", "tr.im", "shorte.st", "bl.ink", "short.io", "soo.gd", "clck.ru", "qr.ae"}
SUSPICIOUS_TLDS = {"xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "link", "work", "support",
                   "club", "cn", "ru", "buzz", "icu", "cyou", "monster", "zip", "mov", "rest",
                   "fit", "surf", "cam", "sbs", "online", "site", "shop", "vip", "pw", "cc"}
# Brands people in Puerto Rico / the US are impersonated as (second-level label, no dots).
BRANDS = ["paypal", "amazon", "apple", "microsoft", "google", "netflix", "facebook", "instagram",
          "whatsapp", "ebay", "walmart", "chase", "wellsfargo", "bankofamerica", "citibank",
          "capitalone", "usps", "fedex", "dhl", "irs", "bancopopular", "firstbank", "orientalbank",
          "autoexpreso", "athmovil", "evertec", "zelle", "venmo", "cashapp", "dropbox", "docusign",
          "linkedin", "steam", "coinbase", "binance", "claropr", "libertypr", "daco"]

TEXT_PATTERNS = {
    "urgent": r"\b(urgent|immediately|asap|now|expires?|expired|suspended|limited time|last chance|"
              r"act now|final notice|urgente|inmediatamente|expira|suspendid[ao])\b",
    "credentials": r"\b(password|passcode|pin|login|log in|sign in|verify|confirm|update your|"
                   r"credentials|ssn|social security|contraseña|verifique|confirme|inicie sesión)\b",
    "money": r"\b(free|win|won|winner|prize|cash|reward|refund|payment|bank|account|invoice|"
             r"\$\d|usd|credit|loan|gratis|premio|reembolso|pago|cuenta|banco)\b",
    "click": r"\b(click|tap|press|open the link|follow the link|haga clic|presione|abra el enlace)\b",
    "toll_debt": r"\b(toll|autoexpreso|unpaid|overdue|outstanding balance|peaje|deuda|vencid[ao])\b",
    "delivery": r"\b(package|parcel|delivery|usps|fedex|dhl|tracking|paquete|entrega|envío)\b",
}
_TEXT_RX = {k: re.compile(v, re.IGNORECASE) for k, v in TEXT_PATTERNS.items()}
_IP_RX = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


# ------------------------------------------------------------------ Tranco (domain reputation)
@lru_cache(maxsize=1)
def _tranco() -> dict[str, int]:
    if not C.TRANCO_CSV.exists():
        return {}
    df = pd.read_csv(C.TRANCO_CSV, header=None, names=["rank", "domain"])
    return dict(zip(df["domain"], df["rank"]))


def tranco_rank(host: str) -> int:
    """Rank of the host (or of its parent domains) in the Tranco top-1M; 0 if absent."""
    ranks = _tranco()
    labels = host.split(".")
    for i in range(len(labels) - 1):
        r = ranks.get(".".join(labels[i:]))
        if r:
            return int(r)
    return 0


# ------------------------------------------------------------------ helpers
def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = pd.Series(list(s)).value_counts(normalize=True)
    return float(-(counts * np.log2(counts)).sum())


def _sld(host: str) -> str:
    labels = host.split(".")
    return labels[-2] if len(labels) >= 2 else host


def lookalike(host: str) -> tuple[float, int, int]:
    """(best similarity of the host's name to a known brand, brand typo-squat flag,
    brand-name-inside-an-unofficial-host flag)."""
    sld = _sld(host).replace("-", "")
    best, impersonates = 0.0, 0
    for brand in BRANDS:
        sim = SequenceMatcher(None, sld, brand).ratio()
        if sld != brand:
            best = max(best, sim)
        if brand in host.replace("-", "").replace(".", "") and sld != brand and len(brand) >= 5:
            impersonates = 1
    return best, int(0.8 <= best < 1.0), impersonates


def url_features(url: str) -> dict[str, float]:
    f: dict[str, float] = {}
    if not url:
        keys = ["has_url", "url_length", "host_length", "n_dots", "host_hyphens", "host_digits",
                "subdomains", "has_ip", "https", "has_at", "is_shortener", "suspicious_tld",
                "path_length", "query_params", "url_entropy", "url_digit_ratio",
                "lookalike_sim", "lookalike_typo", "brand_in_host", "in_tranco", "tranco_log_rank"]
        return {PREFIX + k: 0.0 for k in keys}
    raw = url.strip()
    try:
        parsed = urlsplit(raw if "://" in raw else "//" + raw)
    except ValueError:  # malformed host (e.g. stray full-width characters)
        parsed = urlsplit("")
    host = host_of(raw)
    labels = host.split(".")
    sim, typo, impersonates = lookalike(host)
    rank = tranco_rank(host) if C.USE_TRANCO_RANK else 0
    f["has_url"] = 1
    f["url_length"] = len(raw)
    f["host_length"] = len(host)
    f["n_dots"] = raw.count(".")
    f["host_hyphens"] = host.count("-")
    f["host_digits"] = sum(ch.isdigit() for ch in host)
    f["subdomains"] = max(len(labels) - 2, 0)
    f["has_ip"] = int(bool(_IP_RX.match(host)))
    f["https"] = int(raw.lower().startswith("https://"))
    f["has_at"] = int("@" in raw)
    f["is_shortener"] = int(host in SHORTENERS)
    f["suspicious_tld"] = int(labels[-1] in SUSPICIOUS_TLDS)
    f["path_length"] = len(parsed.path)
    f["query_params"] = len(parse_qsl(parsed.query, keep_blank_values=True))
    f["url_entropy"] = _entropy(raw)
    f["url_digit_ratio"] = sum(ch.isdigit() for ch in raw) / max(len(raw), 1)
    f["lookalike_sim"] = sim
    f["lookalike_typo"] = typo
    f["brand_in_host"] = impersonates
    f["in_tranco"] = int(rank > 0)
    f["tranco_log_rank"] = math.log10(rank) if rank > 0 else 7.0  # 7.0 = "not in the top 1M"
    return {PREFIX + k: float(v) for k, v in f.items()}


def text_features(text: str) -> dict[str, float]:
    text = str(text)
    n = max(len(text), 1)
    f = {PREFIX + "n_chars": len(text),
         PREFIX + "n_words": len(text.split()),
         PREFIX + "digit_ratio": sum(ch.isdigit() for ch in text) / n,
         PREFIX + "upper_ratio": sum(ch.isupper() for ch in text) / n,
         PREFIX + "exclamations": text.count("!"),
         PREFIX + "question_marks": text.count("?"),
         PREFIX + "n_urls": len(extract_urls(text))}
    for name, rx in _TEXT_RX.items():
        f[PREFIX + "kw_" + name] = len(rx.findall(text))
    return f


def message_features(text: str, url: str) -> dict[str, float]:
    return {**text_features(text), **url_features(url)}


def message_feature_frame(texts: pd.Series, urls: pd.Series) -> pd.DataFrame:
    rows = [message_features(t, u) for t, u in zip(texts, urls.fillna(""))]
    df = pd.DataFrame(rows, index=texts.index)
    if C.USE_DOMAIN_AGE:
        from .domain_age import domain_ages
        hosts = urls.fillna("").map(host_of)
        ages = domain_ages(hosts.tolist())
        df[PREFIX + "domain_age_days"] = hosts.map(ages)
    return df
