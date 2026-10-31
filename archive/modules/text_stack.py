"""Experimental: add a word-level text model to every channel (same upgrade for all 3 versions).

Each channel gets two base learners: the existing Random Forest on hand-made features, and a
TF-IDF + logistic-regression model on the raw text (call transcript: word 1-2 grams; message /
email / link: character 2-5 grams, which also read URLs). A small logistic meta-learner is
trained ONLY on out-of-fold probabilities (see fusion.py for why that avoids leakage).

    TextStack(use_call=True,  use_msg=False)   upgraded "call only"
    TextStack(use_call=False, use_msg=True)    upgraded "message / email / link only"
    TextStack(use_call=True,  use_msg=True)    upgraded "combined"

X must hold the feature columns (call_*, msg_*) AND the raw ``call_text`` / ``msg_text`` columns.
Caveat to remember: text models can pick up dataset-specific wording (names, template phrases), so
a higher score here is not by itself proof of better real-world detection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .fusion import _logit

MAX_CHARS = 3000
TEXT_CALL, TEXT_MSG = "txt_call", "txt_msg"      # raw-text columns (prefix txt_ so feature selectors never pick them up)


def _tfidf_lr(kind: str):
    vec = (TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=40000)
           if kind == "word" else
           TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=60000))
    return make_pipeline(vec, LogisticRegression(C=10, max_iter=3000, class_weight="balanced", solver="liblinear"))


class TextStack(BaseEstimator, ClassifierMixin):
    def __init__(self, use_call=True, use_msg=True, rf_params=None, n_splits=5, seed=C.SEED, meta_C=1.0,
                 call_rf_params=None, msg_rf_params=None, call_col=TEXT_CALL, msg_col=TEXT_MSG):
        self.use_call = use_call
        self.use_msg = use_msg
        self.rf_params = rf_params
        self.call_rf_params = call_rf_params
        self.msg_rf_params = msg_rf_params
        self.call_col = call_col
        self.msg_col = msg_col
        self.n_splits = n_splits
        self.seed = seed
        self.meta_C = meta_C

    def _specs(self, X):
        def rf(params):
            return RandomForestClassifier(**{**C.BASE_RF_PARAMS, **(params or self.rf_params or {})})
        specs = {}
        if self.use_call:
            specs["rf_call"] = ("feat", [c for c in X.columns if c.startswith("call_") and c != self.call_col], rf(self.call_rf_params))
            specs["text_call"] = ("text", self.call_col, _tfidf_lr("word"))
        if self.use_msg:
            specs["rf_msg"] = ("feat", [c for c in X.columns if c.startswith("msg_") and c != self.msg_col], rf(self.msg_rf_params))
            specs["text_msg"] = ("text", self.msg_col, _tfidf_lr("char"))
        return specs

    @staticmethod
    def _fit_one(model, data, y, w):
        if hasattr(model, "steps"):                       # pipeline: the weight goes to the final classifier
            return model.fit(data, y, **{f"{model.steps[-1][0]}__sample_weight": w})
        return model.fit(data, y, sample_weight=w)

    @staticmethod
    def _data(kind, col, X):
        return X[col] if kind == "feat" else X[col].astype(str).str.slice(0, MAX_CHARS)

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        y = np.asarray(y)
        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, float)
        self.classes_ = np.unique(y)
        specs = self._specs(X)
        self.names_ = list(specs)
        oof = np.zeros((len(X), len(specs)))
        for tr, va in StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed).split(X, y):
            for j, (kind, col, model) in enumerate(specs.values()):
                m = self._fit_one(clone(model), self._data(kind, col, X.iloc[tr]), y[tr], w[tr])
                oof[va, j] = m.predict_proba(self._data(kind, col, X.iloc[va]))[:, 1]
        self.fitted_ = {n: (kind, col, self._fit_one(clone(model), self._data(kind, col, X), y, w)) for n, (kind, col, model) in specs.items()}
        self.scaler_ = StandardScaler().fit(_logit(oof))
        self.meta_ = LogisticRegression(C=self.meta_C, class_weight="balanced", max_iter=2000,
                                        random_state=self.seed).fit(self.scaler_.transform(_logit(oof)), y, sample_weight=w)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        P = np.column_stack([m.predict_proba(self._data(kind, col, X))[:, 1] for kind, col, m in self.fitted_.values()])
        return self.meta_.predict_proba(self.scaler_.transform(_logit(P)))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def meta_weights(self) -> pd.Series:
        return pd.Series(self.meta_.coef_[0], index=self.names_, name="weight")
