"""Turn paired records into the feature table used by the model versions.

Feature policy (v2): the full table is built once, then ``apply_policy`` drops the raw
length / punctuation-count "style" columns (they separate the datasets, not the scams) and
keeps or drops the cross-channel columns. v1 keeps its original policy (style on, no cross).
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from .features_call import PREFIX as CALL_PREFIX, call_feature_frame
from .features_cross import PREFIX as CROSS_PREFIX, cross_feature_frame
from .features_message import PREFIX as MSG_PREFIX, message_feature_frame

# Early fusion = plain concatenation of the two feature sets (the original design).
# Stacked fusion additionally gets the cross-channel features, but only inside its small
# meta-learner, so they cannot dilute the forests.
_CONCAT = (CALL_PREFIX, MSG_PREFIX)
MODEL_PREFIX = {"call_only": (CALL_PREFIX,), "message_only": (MSG_PREFIX,),
                "combined": _CONCAT, "combined_early": _CONCAT,
                "combined_stacked": (CALL_PREFIX, MSG_PREFIX, CROSS_PREFIX),
                # v3: same feature columns plus the raw text (prefix txt_), read by text_stack.TextStack
                "call_only_text": (CALL_PREFIX, "txt_call"), "message_only_text": (MSG_PREFIX, "txt_msg"),
                "combined_text": (CALL_PREFIX, MSG_PREFIX, "txt_")}


def build_all_features(records: pd.DataFrame) -> pd.DataFrame:
    """Every feature (call_*, msg_*, cross_*), before any policy is applied."""
    calls = call_feature_frame(records["call_text"])
    msgs = message_feature_frame(records["msg_text"], records["msg_url"])
    cross = cross_feature_frame(calls, msgs, records["call_text"], records["msg_text"])
    return pd.concat([calls, msgs, cross], axis=1)


def apply_policy(X: pd.DataFrame, exclude_style: bool, use_cross: bool) -> pd.DataFrame:
    drop = [c for c in X.columns
            if (exclude_style and c in C.STYLE_FEATURES) or (not use_cross and c.startswith(CROSS_PREFIX))]
    return X.drop(columns=drop)


def build_features(records: pd.DataFrame) -> pd.DataFrame:
    """Feature table under the active version's policy (see config)."""
    return apply_policy(build_all_features(records), C.EXCLUDE_STYLE_FEATURES, C.USE_CROSS_FEATURES)


def columns_for(variant: str, columns) -> list[str]:
    """Columns a model version is allowed to see."""
    if variant not in MODEL_PREFIX:
        raise ValueError(f"unknown variant {variant!r}; expected one of {tuple(MODEL_PREFIX)}")
    return [c for c in columns if c.startswith(MODEL_PREFIX[variant])]


def feature_table(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return build_features(records), records["label"].astype(int)
