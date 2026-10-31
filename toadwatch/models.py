"""THE THREE MODELS.

Two Random Forests are trained - one per channel - and three models are read from them:

    call_only     fraud  <=>  p_call >= t_call             (the call forest alone)
    message_only  fraud  <=>  p_msg  >= t_msg              (the message forest alone)
    combined      fraud  <=>  p_call >= t_c  OR  p_msg >= t_m

The combined rule is the TOAD definition: the call and the message belong to ONE attack, so if either part is detected
as fraud the whole attack is fraud - including the part that looked harmless on its own. It is not a third classifier:
it uses the same two forests.

Each forest is trained on the label of ITS OWN text (is this transcript a scam? is this message a scam?), never on the
other half's label.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from . import config as C
from .features import channel_columns


def make_forest(params: dict | None = None) -> RandomForestClassifier:
    return RandomForestClassifier(**{**C.BASE_RF_PARAMS, **(params or {})})


class FraudDetectors:
    """The call Random Forest and the message Random Forest."""

    def __init__(self, params: dict | None = None):
        self.params = params or {}          # {"call": {...}, "message": {...}} Random-Forest settings

    def fit(self, X: pd.DataFrame, y_call, y_msg, sample_weight=None) -> "FraudDetectors":
        self.cols_ = {ch: channel_columns(X.columns, ch) for ch in ("call", "message")}
        self.forests_ = {ch: make_forest(self.params.get(ch)).fit(X[self.cols_[ch]], y, sample_weight=sample_weight)
                         for ch, y in (("call", np.asarray(y_call)), ("message", np.asarray(y_msg)))}
        return self

    def scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """Probability that each channel's text is a scam."""
        out = {}
        for ch, name in (("call", "p_call"), ("message", "p_msg")):
            rf = self.forests_[ch]
            out[name] = rf.predict_proba(X[self.cols_[ch]])[:, list(rf.classes_).index(1)]
        return pd.DataFrame(out, index=X.index)

    def importances(self) -> pd.DataFrame:
        rows = [{"channel": ch, "feature": f, "importance": v}
                for ch, rf in self.forests_.items() for f, v in zip(self.cols_[ch], rf.feature_importances_)]
        return pd.DataFrame(rows).sort_values(["channel", "importance"], ascending=[True, False]).reset_index(drop=True)


def decide(scores: pd.DataFrame, thresholds: dict) -> dict[str, np.ndarray]:
    """The three models' yes/no decisions (1 = fraud). thresholds = {"call_only": t, "message_only": t, "combined": [t_call, t_msg]}."""
    t_c, t_m = thresholds["combined"]
    p_call, p_msg = scores["p_call"].to_numpy(), scores["p_msg"].to_numpy()
    return {"call_only": (p_call >= thresholds["call_only"]).astype(int),
            "message_only": (p_msg >= thresholds["message_only"]).astype(int),
            "combined": ((p_call >= t_c) | (p_msg >= t_m)).astype(int)}


def model_scores(scores: pd.DataFrame, thresholds: dict) -> dict[str, np.ndarray]:
    """A 0-1 score per model whose 0.5 line is exactly that model's decision (for ROC / PR curves)."""
    t_c, t_m = thresholds["combined"]
    f = lambda p, t: np.clip(np.asarray(p) / (2 * t), 0, 1)
    return {"call_only": f(scores["p_call"], thresholds["call_only"]), "message_only": f(scores["p_msg"], thresholds["message_only"]),
            "combined": np.maximum(f(scores["p_call"], t_c), f(scores["p_msg"], t_m))}


def save_detectors(det: FraudDetectors, name: str = "detectors") -> None:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(det, C.MODELS_DIR / f"{name}.joblib")
