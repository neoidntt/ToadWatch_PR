"""Step 3 - features computed from a call transcript.

Keyword / phrase patterns from the proposal ("requests for a PIN, urgent language, or claims
of being a bank or government office"), in English and Spanish. Every feature name starts
with ``call_`` so the three model versions can pick their columns by prefix.
"""
from __future__ import annotations

import re

import pandas as pd

PREFIX = "call_"

# name -> regex (matched on lower-case text)
PATTERNS: dict[str, str] = {
    # asks for secrets / identity
    "pin_request": r"\b(pin|passcode|password|one[- ]time (?:code|password)|verification code|security code|"
                   r"cvv|contraseña|clave|código de (?:verificación|seguridad))\b",
    "ssn_request": r"\b(social security|ssn|social insurance|seguro social)\b",
    "card_account_number": r"\b(card number|credit card|debit card|account number|routing number|"
                           r"número de (?:tarjeta|cuenta))\b",
    "verify_confirm": r"\b(verify|confirm your|validate|verificar|confirmar su)\b",
    # pressure
    "press_one": r"\b(press (?:1|one|2|two|9|nine)|presione (?:1|uno|2|dos)|oprima (?:1|uno))\b",
    "urgent_words": r"\b(urgent|urgently|immediately|right away|right now|as soon as possible|last chance|"
                    r"final notice|expires?|expiring|suspended|suspend|deactivated|locked|"
                    r"urgente|inmediatamente|ahora mismo|último aviso|suspendida?)\b",
    "threat_legal": r"\b(arrest|warrant|lawsuit|legal action|police|prosecut\w*|jail|fine|penalty|"
                    r"arresto|demanda|policía|multa)\b",
    "stay_on_line": r"\b(stay on the line|do not hang up|don't hang up|no cuelgue|permanezca en la línea)\b",
    "dont_tell": r"\b(don't tell|do not tell|keep (?:this|it) (?:confidential|secret)|no le diga)\b",
    # claims of being an institution
    "claims_bank": r"\b(bank|banco|banking|credit union|visa|mastercard|fraud (?:department|alert)|"
                   r"banco popular|firstbank|oriental|ath m[óo]vil|zelle|paypal)\b",
    "claims_government": r"\b(irs|internal revenue|social security administration|medicare|medicaid|"
                         r"federal trade commission|ftc|fbi|department of|government|hacienda|"
                         r"daco|autoexpreso|treasury|court|officer|agent badge)\b",
    "claims_company": r"\b(microsoft|windows|apple|amazon|google|netflix|geek squad|norton|mcafee|"
                      r"tech(?:nical)? support|computer)\b",
    # money / payment methods
    "payment_methods": r"\b(gift cards?|wire transfers?|western union|moneygram|bitcoins?|crypto|cash app|"
                       r"tarjetas? de regalo|giro)\b",
    "money_words": r"\b(refund|reimburse|prize|winner|won|reward|lottery|owed|debt|payment|deposit|"
                   r"reembolso|premio|ganador|deuda|pago)\b",
    "account_problem": r"\b(problem with your|issue with your|unauthorized|suspicious activity|"
                       r"compromised|fraudulent|hacked|virus|problema con su|actividad sospechosa)\b",
    "remote_access": r"\b(remote access|teamviewer|anydesk|download|install|log ?in to your computer)\b",
    # safe-looking call vocabulary (helps the forest separate normal calls)
    "appointment_delivery": r"\b(appointment|delivery|deliver\w*|package|reschedule|schedule|"
                            r"cita|entrega|paquete)\b",
    "polite_service": r"\b(thank you for calling|how can i (?:help|assist)|have a (?:great|nice) day|"
                      r"is there anything else|gracias por llamar)\b",
}
_COMPILED = {name: re.compile(rx, re.IGNORECASE) for name, rx in PATTERNS.items()}


def call_features(text: str) -> dict[str, float]:
    """Feature dict for one transcript."""
    text = str(text)
    words = text.split()
    n_words = max(len(words), 1)
    feats: dict[str, float] = {}
    for name, rx in _COMPILED.items():
        feats[PREFIX + name] = len(rx.findall(text))
    feats[PREFIX + "n_words"] = len(words)
    feats[PREFIX + "n_chars"] = len(text)
    feats[PREFIX + "digit_ratio"] = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    feats[PREFIX + "question_marks"] = text.count("?")
    feats[PREFIX + "exclamations"] = text.count("!")
    feats[PREFIX + "avg_word_len"] = sum(len(w) for w in words) / n_words
    feats[PREFIX + "urgency_density"] = (feats[PREFIX + "urgent_words"]
                                         + feats[PREFIX + "threat_legal"]) / n_words * 100
    feats[PREFIX + "secret_asks"] = (feats[PREFIX + "pin_request"] + feats[PREFIX + "ssn_request"]
                                     + feats[PREFIX + "card_account_number"])
    return feats


def call_feature_frame(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([call_features(t) for t in texts], index=texts.index)
