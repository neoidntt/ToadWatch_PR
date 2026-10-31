"""Features of the two channels. Names start with ``call_`` or ``msg_`` so each Random Forest picks only its own columns."""
from __future__ import annotations

import pandas as pd

from . import config as C
from .features_call import PREFIX as CALL_PREFIX, call_feature_frame
from .features_message import PREFIX as MSG_PREFIX, message_feature_frame


def build_features(records: pd.DataFrame) -> pd.DataFrame:
    """One row per record: call_* columns (transcript, incl. links it mentions) + msg_* columns (SMS / email / link).
    Raw length / punctuation counts are removed (they identify the corpus, not the scam)."""
    calls = call_feature_frame(records["call_text"])
    msgs = message_feature_frame(records["msg_text"], records["msg_url"])
    X = pd.concat([calls, msgs], axis=1)
    return X.drop(columns=[c for c in C.STYLE_FEATURES if c in X.columns])


def channel_columns(columns, channel: str) -> list[str]:
    """Columns a detector may see: 'call' or 'message'."""
    prefix = {"call": CALL_PREFIX, "message": MSG_PREFIX}[channel]
    return [c for c in columns if c.startswith(prefix)]
