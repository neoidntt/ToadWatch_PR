"""Browse the data directly from code (and from plain CSV files on disk).

    from toadwatch import dataset_access as DA
    DA.raw_inventory()                 # every raw file you provided: folder, size, role, class
    train = DA.load_split("train")     # paired records, as a DataFrame  (version="v2" by default)
    DA.split_summary("v2")             # sizes / class balance / message types per split
    DA.export_dataset_folders("v2")    # writes ToadWatch_PR/dataset/v2/{train,validation,test}/*.csv

Folder layout written by ``export_dataset_folders`` (open the CSVs in VS Code / Excel):

    dataset/<version>/
        README.md
        train/        training data       (fit the models)
        validation/   fitting/validation  (tune settings; v2 also cross-validates on train+validation)
        test/         final testing data  (locked: read-only, used once for the final result)
            paired_records.csv   one row per TOAD case: call + message/email/link + label
            calls.csv            the call transcripts only
            messages.csv         the SMS / email / URL items only
            README.md            counts and column guide for that set

The loaders read the files the pipeline already wrote; they never recompute or change a split.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pandas as pd

from . import config as C

SEEDS = {"v1": 42, "v2": 2027, "v3": 2028}
SPLIT_FOLDERS = {"train": "train", "val": "validation", "test": "test"}
SPLIT_ROLE = {"train": "training data (fit the models)",
              "val": "fitting / validation data (choose settings, cross-validate)",
              "test": "final testing data (locked, used once for the final result)"}
DATASET_DIR = C.PROJECT_ROOT / "dataset"


def _processed_dir(version: str) -> Path:
    base = C.PROJECT_ROOT / "data" / "processed"
    return base if version == "v1" else base / version


def _check_version(version: str) -> str:
    if version not in ("v1", "v2", "v3"):
        raise ValueError("version must be 'v1', 'v2' or 'v3'")
    return version


# ------------------------------------------------------------------ raw files you provided
RAW_ROLES = {
    "Call Safe data": ("calls", "safe"), "Call Dangerous data": ("calls", "fraud"),
    "SMS, Email, URL Safe": ("messages / emails / links", "safe"),
    "SMS, Email, URL Dangerous": ("messages / emails / links", "fraud"),
}


def raw_inventory() -> pd.DataFrame:
    """One row per raw file in the four data folders (path, size, what it is, its class)."""
    rows = []
    for folder, (role, label) in RAW_ROLES.items():
        root = C.DATA_ROOT / folder
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".csv", ".json", ".zip") and "phishing_1" not in p.parts:
                rows.append({"folder": folder, "file": str(p.relative_to(root)), "size_mb": round(p.stat().st_size / 1e6, 2),
                             "role": role, "class": label, "path": str(p)})
    nc = C.NC_STATE_DIR / "metadata.csv"
    if nc.exists():
        rows.append({"folder": "Call Dangerous data", "file": "robocall-audio-dataset-ver-0.1/metadata.csv",
                     "size_mb": round(nc.stat().st_size / 1e6, 2), "role": "calls (FTC robocall transcripts)", "class": "fraud", "path": str(nc)})
    return pd.DataFrame(rows)


def read_raw(file: str, folder: str, nrows: int | None = 1000) -> pd.DataFrame:
    """Peek into one raw CSV (first ``nrows`` rows; the biggest files are over 1 GB)."""
    return pd.read_csv(C.DATA_ROOT / folder / file, nrows=nrows)


# ------------------------------------------------------------------ the prepared splits
def load_split(name: str, version: str = "v2") -> pd.DataFrame:
    """Paired records of one split ('train', 'val' or 'test') for a version."""
    _check_version(version)
    if name not in SPLIT_FOLDERS:
        raise ValueError(f"split must be one of {tuple(SPLIT_FOLDERS)}")
    path = _processed_dir(version) / f"records_{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the data-preparation notebook / script for {version} first.")
    return pd.read_parquet(path)


def load_all(version: str = "v2") -> pd.DataFrame:
    return pd.concat([load_split(s, version) for s in SPLIT_FOLDERS], ignore_index=True)


def split_summary(version: str = "v2") -> pd.DataFrame:
    """Sizes and balance of every split: records, fraud, safe, and counts per message type."""
    rows = []
    for s in SPLIT_FOLDERS:
        d = load_split(s, version)
        row = {"split": SPLIT_FOLDERS[s], "role": SPLIT_ROLE[s], "records": len(d), "fraud": int((d.label == 1).sum()),
               "safe": int((d.label == 0).sum())}
        row.update({f"msg_{k}": int(v) for k, v in d["msg_kind"].value_counts().sort_index().items()})
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


def calls_table(records: pd.DataFrame) -> pd.DataFrame:
    return records[["record_id", "call_id", "call_text", "label", "call_source"]].rename(
        columns={"call_text": "text", "call_source": "source"})


def messages_table(records: pd.DataFrame) -> pd.DataFrame:
    return records[["record_id", "item_id", "msg_kind", "msg_text", "msg_url", "label", "msg_source", "msg_lang"]].rename(
        columns={"msg_kind": "kind", "msg_text": "text", "msg_url": "url", "msg_source": "source", "msg_lang": "lang"})


def find(text: str, version: str = "v2", split: str | None = None, where: str = "both") -> pd.DataFrame:
    """Case-insensitive search of the call and/or message text across the splits."""
    d = load_split(split, version).assign(split=split) if split else pd.concat(
        [load_split(s, version).assign(split=SPLIT_FOLDERS[s]) for s in SPLIT_FOLDERS], ignore_index=True)
    hit = pd.Series(False, index=d.index)
    if where in ("both", "call"):
        hit |= d["call_text"].str.contains(text, case=False, regex=False, na=False)
    if where in ("both", "message"):
        hit |= d["msg_text"].str.contains(text, case=False, regex=False, na=False)
    return d[hit]


# ------------------------------------------------------------------ export to subfolders
def _write(df: pd.DataFrame, path: Path, readonly: bool) -> None:
    if path.exists():
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    df.to_csv(path, index=False)
    if readonly:
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _readme(version: str, split: str, d: pd.DataFrame) -> str:
    kinds = ", ".join(f"{k} {v}" for k, v in d["msg_kind"].value_counts().sort_index().items())
    note = ("\n**LOCKED.** These files are read-only and the split is fingerprinted. Looking is fine, but never use this set to "
            "choose features or settings - it is used once for the final result.\n" if split == "test" else "")
    return f"""# {SPLIT_FOLDERS[split]} ({version})

{SPLIT_ROLE[split].capitalize()}.{note}
* records: **{len(d)}** (fraud {int((d.label == 1).sum())}, safe {int((d.label == 0).sum())}); message types: {kinds}
* seed {SEEDS[version]}; label `1` = fraud, `0` = safe (taken from the source folder)

| file | what it is |
|---|---|
| `paired_records.csv` | one row per TOAD case: `call_text` + `msg_text` / `msg_url` + `label`, with sources |
| `calls.csv` | the call transcripts only (`record_id`, `call_id`, `text`, `label`, `source`) |
| `messages.csv` | the SMS / email / URL items only (`kind` = sms / email / url, `lang` = en / es) |

A call and a message are paired only inside the same split and with the same label, so no call or message appears in two splits.
"""


def export_dataset_folders(version: str | None = None) -> Path:
    """Write ``dataset/<version>/{train,validation,test}/`` with plain CSV files. Safe to re-run."""
    version = _check_version(version or C.VERSION)
    root = DATASET_DIR / version
    root.mkdir(parents=True, exist_ok=True)
    for s, folder in SPLIT_FOLDERS.items():
        d = load_split(s, version)
        out = root / folder
        out.mkdir(exist_ok=True)
        ro = s == "test"
        _write(d, out / "paired_records.csv", ro)
        _write(calls_table(d), out / "calls.csv", ro)
        _write(messages_table(d), out / "messages.csv", ro)
        _write_text(out / "README.md", _readme(version, s, d), ro)
    summary = split_summary(version)
    _write_text(root / "README.md", f"""# Dataset {version}

Prepared data, split 70 / 10 / 20, one subfolder per set:

* `train/` - training data
* `validation/` - fitting / validation data
* `test/` - final testing data (locked, read-only)

{_md_table(summary[['records', 'fraud', 'safe']])}

Regenerated automatically by the data-preparation step; open it from code with `toadwatch.dataset_access`
(see `notebooks/v2_00_explore_dataset.ipynb`). The raw files you provided stay in the original four folders next to
`ToadWatch_PR/`; `dataset_access.raw_inventory()` lists them.
""", False)
    return root


def _md_table(df: pd.DataFrame) -> str:
    head = "| set | " + " | ".join(df.columns) + " |\n|---|" + "---|" * len(df.columns)
    return head + "".join(f"\n| {i} | " + " | ".join(str(v) for v in r) + " |" for i, r in df.iterrows())


def _write_text(path: Path, text: str, readonly: bool) -> None:
    if path.exists():
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    path.write_text(text)
    if readonly:
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


# ------------------------------------------------------------------ chart
def plot_split_composition(version: str = "v2"):
    """Class balance, message types and text lengths for train / validation / test."""
    import matplotlib.pyplot as plt
    data = {SPLIT_FOLDERS[s]: load_split(s, version) for s in SPLIT_FOLDERS}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    names = list(data)
    for j, (lab, col) in enumerate(((0, "#4c9f70"), (1, "#c0504d"))):
        ax[0].bar([i + (j - .5) * .38 for i in range(3)], [(d.label == lab).sum() for d in data.values()], .38,
                  color=col, label=["safe", "fraud"][j])
    ax[0].set_xticks(range(3), names); ax[0].set_title("Records per set and class"); ax[0].legend()
    bottoms = [0, 0, 0]
    for kind, col in (("sms", "#5b7db1"), ("email", "#d08c3c"), ("url", "#8e5ea2")):
        vals = [(d.msg_kind == kind).sum() for d in data.values()]
        ax[1].bar(names, vals, bottom=bottoms, color=col, label=kind)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax[1].set_title("Message type per set"); ax[1].legend()
    pos = 0
    ticks, labels = [], []
    for n, d in data.items():
        for lab, series, col in (("call", d.call_text.str.len(), "#5b7db1"), ("msg", d.msg_text.str.len(), "#d08c3c")):
            bp = ax[2].boxplot(series.clip(upper=6000), positions=[pos], widths=.6, patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor(col)
            ticks.append(pos); labels.append(f"{n}\n{lab}"); pos += 1
        pos += .6
    ax[2].set_xticks(ticks, labels, fontsize=7); ax[2].set_title("Text length (characters)")
    fig.tight_layout()
    return fig
