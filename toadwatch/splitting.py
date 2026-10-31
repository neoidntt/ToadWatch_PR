"""Step 2 - 70/10/20 split with a fixed seed, and a test set that stays untouched.

Calls and message-side items are split INDEPENDENTLY (stratified), and only afterwards
paired inside each split (see ``pairing.py``). That way a call or a message that is in the
test set can never show up in training, even inside a different pair.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C

SPLITS = ("train", "val", "test")


def _strata(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    key = df[cols].astype(str).agg("|".join, axis=1)
    counts = key.map(key.value_counts())
    # strata with fewer than 3 members cannot be stratified twice: fall back to the label
    return key.where(counts >= 3, "rare|" + df["label"].astype(str))


def assign_splits(df: pd.DataFrame, strata_cols: list[str], seed: int = C.SEED) -> pd.DataFrame:
    """Add a ``split`` column (train 70 / val 10 / test 20), stratified on ``strata_cols``."""
    r = C.SPLIT_RATIOS
    assert abs(sum(r.values()) - 1.0) < 1e-9
    df = df.reset_index(drop=True).copy()
    strata = _strata(df, strata_cols)
    idx_rest, idx_test = train_test_split(df.index, test_size=r["test"], stratify=strata, random_state=seed)
    val_share = r["val"] / (r["train"] + r["val"])
    idx_train, idx_val = train_test_split(idx_rest, test_size=val_share, stratify=strata[idx_rest],
                                          random_state=seed)
    df["split"] = ""
    df.loc[idx_train, "split"] = "train"
    df.loc[idx_val, "split"] = "val"
    df.loc[idx_test, "split"] = "test"
    return df


# ------------------------------------------------------------------ persistence
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_records(records: pd.DataFrame) -> dict:
    """Write one file per split. The test file is made read-only and fingerprinted."""
    C.ensure_dirs()
    manifest: dict = {"seed": C.SEED, "ratios": C.SPLIT_RATIOS, "files": {}}
    for name in SPLITS:
        path = C.PROCESSED_DIR / f"records_{name}.parquet"
        if path.exists():
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        part = records[records["split"] == name].reset_index(drop=True)
        part.to_parquet(path, index=False)
        manifest["files"][name] = {"rows": len(part),
                                   "fraud": int((part["label"] == 1).sum()),
                                   "safe": int((part["label"] == 0).sum()),
                                   "sha256": _sha256(path)}
        if name == "test":
            os.chmod(path, stat.S_IRUSR)  # read-only: "guardar aparte ... sin tocar"
    C.SPLIT_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_split(name: str, allow_test: bool = False) -> pd.DataFrame:
    """Load a split. The test split is refused unless the caller explicitly allows it."""
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}")
    if name == "test" and not allow_test:
        raise PermissionError("The test set is locked until Step 5 (final testing). "
                              "Use toadwatch.protocol.run_final_test().")
    return pd.read_parquet(C.PROCESSED_DIR / f"records_{name}.parquet")


def verify_test_untouched() -> bool:
    """True if the test file still matches the fingerprint written when it was created."""
    manifest = json.loads(C.SPLIT_MANIFEST_JSON.read_text())
    path = C.PROCESSED_DIR / "records_test.parquet"
    return _sha256(path) == manifest["files"]["test"]["sha256"]
