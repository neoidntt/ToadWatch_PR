"""Demo helper (schedule step 7): score ONE call transcript + ONE message/email/link."""
from __future__ import annotations

import pandas as pd

from . import config as C
from .cleaning import first_url, strip_speakers
from .features import build_features
from .modeling import load_models


def load_demo_models() -> dict:
    """The final models if the test already ran, otherwise the tuned ones (train split only)."""
    name = "final_models" if (C.MODELS_DIR / "final_models.joblib").exists() else "tuned_models"
    return load_models(name)


def explain_case(call_text: str, message_text: str, models: dict | None = None) -> pd.DataFrame:
    """Probability of fraud from each version for a single call + message pair."""
    models = models or load_demo_models()
    rec = pd.DataFrame({"call_text": [strip_speakers(call_text)], "msg_text": [message_text],
                        "msg_url": [first_url(message_text) or (message_text if " " not in message_text.strip()
                                                                 and "." in message_text else "")]})
    X = build_features(rec)
    rows = []
    for variant, (model, cols) in models.items():
        p = float(model.predict_proba(X[cols])[0, list(model.classes_).index(1)])
        rows.append({"version": variant, "fraud_probability": round(p, 3),
                     "verdict": "FRAUD" if p >= 0.5 else "looks safe"})
    return pd.DataFrame(rows)
