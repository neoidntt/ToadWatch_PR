"""Step 4 - adjust the Random Forest settings using ONLY the validation set.

One shared configuration is chosen for the three versions (same algorithm and training
process = control variable): the one with the best mean validation F1 across versions.
The result is frozen in ``outputs/final_config.json`` BEFORE the test set is touched.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from . import config as C
from .modeling import compute_metrics, train_all, predict_variant

PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 12, 24],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", 0.3],
}


def grid_search(X_train, y_train, X_val, y_val, grid: dict | None = None) -> pd.DataFrame:
    """Train every combination on train, score every version on validation."""
    grid = grid or PARAM_GRID
    keys = list(grid)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        models = train_all(X_train, y_train, params)
        row = {**params}
        for variant, mc in models.items():
            m = compute_metrics(y_val, predict_variant(mc, X_val))
            row[f"{variant}_f1"] = m["f1"]
            row[f"{variant}_accuracy"] = m["accuracy"]
        row["mean_f1"] = float(np.mean([row[f"{v}_f1"] for v in C.VARIANTS]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_f1", ascending=False).reset_index(drop=True)


def freeze_final_config(results: pd.DataFrame) -> dict:
    """Write the best configuration to disk; refuses to overwrite after the final test ran."""
    if C.FINAL_TEST_LOCK.exists():
        raise RuntimeError("The final test already ran; the configuration can no longer change.")
    best = results.iloc[0]
    params = {k: (None if pd.isna(best[k]) else best[k]) for k in PARAM_GRID}
    params["n_estimators"] = int(params["n_estimators"])
    params["min_samples_leaf"] = int(params["min_samples_leaf"])
    if params["max_depth"] is not None:
        params["max_depth"] = int(params["max_depth"])
    if isinstance(params["max_features"], float):
        params["max_features"] = float(params["max_features"])
    cfg = {"params": params, "selected_by": "mean validation F1 over the three versions",
           "mean_val_f1": float(best["mean_f1"]), "seed": C.SEED}
    C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    C.FINAL_CONFIG_JSON.write_text(json.dumps(cfg, indent=2))
    return cfg


def load_final_config() -> dict:
    if not C.FINAL_CONFIG_JSON.exists():
        raise FileNotFoundError("Step 4 is not finished: run the tuning and freeze the configuration first.")
    return json.loads(C.FINAL_CONFIG_JSON.read_text())


def freeze_v2_config(per_variant: dict, choice: dict, cv_summary: pd.DataFrame) -> dict:
    """v2: write per-version settings and the pre-declared early/stacked choice to disk."""
    if C.FINAL_TEST_LOCK.exists():
        raise RuntimeError("The final test already ran; the configuration can no longer change.")
    cfg = {"params": {"per_variant": per_variant}, "combined_choice": choice["choice"],
           "choice_detail": choice,
           "selected_by": "mean F1 of repeated stratified CV on train+val (test set never used)",
           "cv_summary": cv_summary.round(5).reset_index().to_dict("records"), "seed": C.SEED}
    C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    C.FINAL_CONFIG_JSON.write_text(json.dumps(cfg, indent=2, default=str))
    return cfg
