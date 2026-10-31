"""v2 - late fusion (stacking) of the two channels.

Why: when 40+ call features and 30+ message features are concatenated into one forest, the
weaker channel's noisy columns compete for splits and can dilute the strong channel. Stacking
lets each channel keep its own forest and learns, with a small logistic regression, how much
to trust each one.

No leakage: the meta-learner is trained ONLY on out-of-fold probabilities of the training
data (every probability it sees comes from a forest that never saw that row). At prediction
time the forests trained on the full training set are used. Test data never enters ``fit``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import config as C


def _logit(p: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class StackedFusion(BaseEstimator, ClassifierMixin):
    """Call forest + message forest -> logistic-regression meta-learner (+ cross features)."""

    def __init__(self, call_params=None, msg_params=None, meta_C=1.0, n_splits=5, seed=C.SEED):
        self.call_params = call_params
        self.msg_params = msg_params
        self.meta_C = meta_C
        self.n_splits = n_splits
        self.seed = seed

    # -- helpers
    def _rf(self, params):
        return RandomForestClassifier(**{**C.BASE_RF_PARAMS, **(params or {})})

    def _split_cols(self, X: pd.DataFrame):
        self.call_cols_ = [c for c in X.columns if c.startswith("call_")]
        self.msg_cols_ = [c for c in X.columns if c.startswith("msg_")]
        self.cross_cols_ = [c for c in X.columns if c.startswith("cross_")]

    def _meta_matrix(self, p_call, p_msg, X: pd.DataFrame) -> np.ndarray:
        parts = [_logit(p_call)[:, None], _logit(p_msg)[:, None]]
        if self.cross_cols_:
            parts.append(X[self.cross_cols_].to_numpy(dtype=float))
        return np.hstack(parts)

    # -- sklearn API
    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        y = np.asarray(y)
        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, float)
        self._split_cols(X)
        self.classes_ = np.unique(y)
        oof = np.zeros((len(X), 2))
        skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
        for tr, va in skf.split(X, y):
            for j, (cols, params) in enumerate(((self.call_cols_, self.call_params), (self.msg_cols_, self.msg_params))):
                rf = self._rf(params).fit(X.iloc[tr][cols], y[tr], sample_weight=w[tr])
                oof[va, j] = rf.predict_proba(X.iloc[va][cols])[:, 1]
        self.call_model_ = self._rf(self.call_params).fit(X[self.call_cols_], y, sample_weight=w)
        self.msg_model_ = self._rf(self.msg_params).fit(X[self.msg_cols_], y, sample_weight=w)
        Z = self._meta_matrix(oof[:, 0], oof[:, 1], X)
        self.scaler_ = StandardScaler().fit(Z)
        self.meta_ = LogisticRegression(C=self.meta_C, class_weight="balanced", max_iter=2000,
                                        random_state=self.seed).fit(self.scaler_.transform(Z), y, sample_weight=w)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_call = self.call_model_.predict_proba(X[self.call_cols_])[:, 1]
        p_msg = self.msg_model_.predict_proba(X[self.msg_cols_])[:, 1]
        return self.meta_.predict_proba(self.scaler_.transform(self._meta_matrix(p_call, p_msg, X)))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def meta_weights(self) -> pd.Series:
        """Standardised meta-learner weights: how much it trusts each channel / cross feature."""
        names = ["call_forest", "message_forest"] + list(self.cross_cols_)
        return pd.Series(self.meta_.coef_[0], index=names, name="weight")
