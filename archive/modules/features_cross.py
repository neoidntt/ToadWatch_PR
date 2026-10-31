"""v2 - features that only exist when BOTH channels are seen (the point of "multichannel").

A TOAD scam is a call that sets up a story plus a message that finishes it, so the two parts
usually point at the same brand, both push urgency, and the call often announces the message
("I'm sending you a text with a link"). None of this is visible to a single-channel model.
Every value is computed from the two texts / their own features only - never from the label.
"""
from __future__ import annotations

import re

import pandas as pd

PREFIX = "cross_"

BRAND_WORDS = {
    "paypal": r"paypal", "amazon": r"amazon", "apple": r"apple|icloud", "microsoft": r"microsoft|windows",
    "google": r"google", "netflix": r"netflix", "facebook": r"facebook|instagram|meta\b",
    "whatsapp": r"whatsapp", "walmart": r"walmart", "chase": r"\bchase\b", "wells": r"wells ?fargo",
    "bofa": r"bank of america", "citi": r"citi ?bank", "capone": r"capital one", "usps": r"\busps\b|postal",
    "fedex": r"fedex", "dhl": r"\bdhl\b", "irs": r"\birs\b|internal revenue", "ssa": r"social security",
    "medicare": r"medicare|medicaid", "ftc": r"\bftc\b|federal trade", "popular": r"banco popular",
    "firstbank": r"firstbank|first bank", "oriental": r"oriental", "autoexpreso": r"autoexpreso",
    "ath": r"ath m[óo]vil", "zelle": r"zelle", "venmo": r"venmo", "bank": r"\bbank\b|\bbanco\b",
}
_BRAND_RX = {k: re.compile(v, re.IGNORECASE) for k, v in BRAND_WORDS.items()}
_FOLLOWUP_RX = re.compile(r"\b(text|sms|message|email|e-mail|link|website|click|mensaje|enlace|correo)\b", re.I)
_CALLBACK_RX = re.compile(r"\b(call|phone|number|callback|call back|llame|llamada|tel[ée]fono)\b", re.I)

CALL_SIGNALS = ["pin_request", "ssn_request", "card_account_number", "press_one", "urgent_words",
                "threat_legal", "stay_on_line", "claims_bank", "claims_government", "payment_methods",
                "account_problem", "remote_access"]
MSG_SIGNALS = ["kw_urgent", "kw_credentials", "kw_money", "kw_click", "kw_toll_debt", "kw_delivery",
               "is_shortener", "suspicious_tld", "lookalike_typo", "brand_in_host", "has_ip", "has_at"]


def _brands(text: str) -> set[str]:
    return {k for k, rx in _BRAND_RX.items() if rx.search(text)}


def cross_feature_frame(calls_feats: pd.DataFrame, msgs_feats: pd.DataFrame,
                        call_texts: pd.Series, msg_texts: pd.Series) -> pd.DataFrame:
    """One row per record, aligned with the (already computed) call / message feature frames."""
    call_score = sum((calls_feats["call_" + n] > 0).astype(int) for n in CALL_SIGNALS)
    msg_score = sum((msgs_feats["msg_" + n] > 0).astype(int) for n in MSG_SIGNALS)
    rows = []
    for ct, mt in zip(call_texts, msg_texts):
        cb, mb = _brands(str(ct)), _brands(str(mt))
        rows.append({
            PREFIX + "shared_brand": len(cb & mb),
            PREFIX + "call_announces_message": int(bool(_FOLLOWUP_RX.search(str(ct)))),
            PREFIX + "message_asks_callback": int(bool(_CALLBACK_RX.search(str(mt)))),
        })
    out = pd.DataFrame(rows, index=calls_feats.index)
    out[PREFIX + "call_signal_count"] = call_score.values
    out[PREFIX + "msg_signal_count"] = msg_score.values
    out[PREFIX + "both_urgent"] = ((calls_feats["call_urgent_words"] > 0) & (msgs_feats["msg_kw_urgent"] > 0)).astype(int).values
    out[PREFIX + "both_ask_credentials"] = ((calls_feats["call_secret_asks"] > 0) & (msgs_feats["msg_kw_credentials"] > 0)).astype(int).values
    out[PREFIX + "secret_ask_plus_link"] = ((calls_feats["call_secret_asks"] > 0) & (msgs_feats["msg_has_url"] > 0)).astype(int).values
    out[PREFIX + "both_suspicious"] = ((call_score >= 2) & (msg_score >= 2)).astype(int).values
    out[PREFIX + "signal_product"] = (call_score * msg_score).values
    out[PREFIX + "signal_gap"] = (call_score - msg_score).abs().values
    return out
