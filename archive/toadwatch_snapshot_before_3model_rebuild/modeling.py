"""Steps 3-4 - the Random Forest and its three versions (call-only, message-only, combined)."""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from . import config as C
from .features import columns_for
from .fusion import StackedFusion
from .text_stack import TextStack


def make_model(params: dict | None = None) -> RandomForestClassifier:
    """Same algorithm and training process for every version (control variable)."""
    return RandomForestClassifier(**{**C.BASE_RF_PARAMS, **(params or {})})


def resolve_params(variant: str, params: dict | None) -> dict | None:
    """``params`` is either one flat dict shared by every version (v1), or
    ``{"per_variant": {variant: dict}}`` (v2: every version tuned with the same procedure)."""
    if params and "per_variant" in params:
        per = params["per_variant"]
        if variant in ("combined_stacked", "combined_text"):   # each channel keeps the settings tuned for it
            return {"call": per.get("call_only"), "msg": per.get("message_only")}
        if variant == "call_only_text":
            return {"call": per.get("call_only"), "msg": None}
        if variant == "message_only_text":
            return {"call": None, "msg": per.get("message_only")}
        return per.get(variant)
    return params


def train_variant(variant: str, X: pd.DataFrame, y: pd.Series, params: dict | None = None, sample_weight=None):
    """Fit one version on its own columns. Returns (model, columns)."""
    cols = columns_for(variant, X.columns)
    p = resolve_params(variant, params)
    if variant in ("call_only_text", "message_only_text", "combined_text"):
        model = TextStack(use_call=variant != "message_only_text", use_msg=variant != "call_only_text",
                          call_rf_params=(p or {}).get("call"), msg_rf_params=(p or {}).get("msg")).fit(X[cols], y, sample_weight=sample_weight)
    elif variant == "combined_stacked":
        model = StackedFusion(call_params=(p or {}).get("call"), msg_params=(p or {}).get("msg")).fit(X[cols], y, sample_weight=sample_weight)
    else:
        model = make_model(p).fit(X[cols], y, sample_weight=sample_weight)
    return model, cols


def train_all(X: pd.DataFrame, y: pd.Series, params: dict | None = None, sample_weight=None) -> dict:
    return {v: train_variant(v, X, y, params, sample_weight) for v in C.VARIANTS}


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy, precision, recall, F1 and the confusion-matrix counts (fraud = positive)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "correct": int(tn + tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def predict_variant(model_cols: tuple, X: pd.DataFrame):
    model, cols = model_cols
    return model.predict(X[cols])


def proba_variant(model_cols: tuple, X: pd.DataFrame):
    """Probability of fraud (class 1) for every row."""
    model, cols = model_cols
    return model.predict_proba(X[cols])[:, list(model.classes_).index(1)]


def score_all(models: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Metrics table (one row per version) on the given split."""
    rows = []
    for variant, mc in models.items():
        rows.append({"variant": variant, **compute_metrics(y, predict_variant(mc, X))})
    return pd.DataFrame(rows).set_index("variant")


def feature_importance(models: dict) -> pd.DataFrame:
    """Gini importance of every feature, per version (step 4: which features matter most)."""
    parts = []
    for variant, (model, cols) in models.items():
        if isinstance(model, TextStack):       # importances of its forests (text models have none)
            for key, sub_cols in (("rf_call", None), ("rf_msg", None)):
                if key in model.fitted_:
                    _, cols_k, rf_k = model.fitted_[key]
                    t = pd.DataFrame({"feature": cols_k, "importance": rf_k.feature_importances_ / max(sum(k.startswith("rf_") for k in model.fitted_), 1)})
                    t["variant"] = variant
                    parts.append(t)
            continue
        if isinstance(model, StackedFusion):   # importances of its two forests, side by side
            for sub, sub_cols in ((model.call_model_, model.call_cols_), (model.msg_model_, model.msg_cols_)):
                t = pd.DataFrame({"feature": sub_cols, "importance": sub.feature_importances_ / 2})
                t["variant"] = variant
                parts.append(t)
            continue
        s = pd.Series(model.feature_importances_, index=cols, name="importance").reset_index()
        s.columns = ["feature", "importance"]
        s["variant"] = variant
        parts.append(s)
    out = pd.concat(parts, ignore_index=True)
    out["channel"] = out["feature"].str.split("_").str[0].map({"call": "call", "msg": "message", "cross": "cross"})
    return out.sort_values(["variant", "importance"], ascending=[True, False]).reset_index(drop=True)


def save_models(models: dict, name: str = "models") -> None:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, C.MODELS_DIR / f"{name}.joblib")


def load_models(name: str = "models") -> dict:
    return joblib.load(C.MODELS_DIR / f"{name}.joblib")
