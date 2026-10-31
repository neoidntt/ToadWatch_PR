"""Optional domain-age lookup (schedule, step 3: "usar una muestra para la edad del dominio si
las búsquedas son muy lentas").

WHOIS lookups need network access and are slow, so this is OFF by default
(``config.USE_DOMAIN_AGE = False``). When enabled, only a random sample of at most
``config.DOMAIN_AGE_SAMPLE`` domains is queried; the rest get NaN, and every result is cached
on disk so a domain is never looked up twice. The offline stand-in used by default is the
Tranco popularity rank (``features_message.tranco_rank``).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from . import config as C

CACHE = C.INTERIM_DIR / "domain_age_cache.json"


def _load_cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def lookup_age_days(domain: str) -> float:
    """Age of the domain in days, or NaN if WHOIS is unavailable or has no creation date."""
    try:
        import whois  # python-whois, optional dependency
    except ImportError:
        return float("nan")
    try:
        created = whois.whois(domain).creation_date
        created = created[0] if isinstance(created, list) else created
        if created is None:
            return float("nan")
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - created).days
    except Exception:  # network error, rate limit, unknown TLD ...
        return float("nan")


def domain_ages(domains: list[str], seed: int = C.SEED) -> dict[str, float]:
    """Map domain -> age in days for a reproducible sample of ``domains`` (others NaN)."""
    cache = _load_cache()
    unique = sorted(set(d for d in domains if d))
    rng = np.random.default_rng(seed)
    sample = set(rng.choice(unique, size=min(C.DOMAIN_AGE_SAMPLE, len(unique)), replace=False)) if unique else set()
    for d in sample:
        if d not in cache:
            cache[d] = lookup_age_days(d)
    C.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache))
    return {d: cache.get(d, float("nan")) for d in unique}
