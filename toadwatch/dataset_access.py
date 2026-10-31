"""Browse the data directly from code (and from plain CSV files on disk).

    from toadwatch import dataset_access as DA
    DA.raw_inventory()                 # every raw file you provided: folder, size, role, class
    train = DA.load_split("train")     # connected call + message records, as a DataFrame
    DA.split_summary()                 # sizes / class balance / message types per split
    DA.export_dataset_folders()        # writes ToadWatch_PR/dataset/{train,validation,test}/*.csv

Folder layout written by ``export_dataset_folders`` (open the CSVs in VS Code / Excel):

    dataset/
        README.md
        train/        training data
        validation/   fitting / validation data (used to tune the settings)
        test/         final testing data (locked: read-only, used once for the final result)
            paired_records.csv   one row per TOAD attack: call + message/email/link + labels
            calls.csv            the call transcripts only
            messages.csv         the SMS / email / URL items only
            README.md            counts and column guide for that set
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pandas as pd

from . import config as C

SPLIT_FOLDERS = {"train": "train", "val": "validation", "test": "test"}
SPLIT_ROLE = {"train": "training data (fit the models)", "val": "fitting / validation data (tune the settings)",
              "test": "final testing data (locked, used once for the final result)"}
RAW_ROLES = {"Call Safe data": ("calls", "safe"), "Call Dangerous data": ("calls", "fraud"),
             "SMS, Email, URL Safe": ("messages / emails / links", "safe"), "SMS, Email, URL Dangerous": ("messages / emails / links", "fraud")}


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
    return pd.DataFrame(rows)


def read_raw(file: str, folder: str, nrows: int | None = 1000) -> pd.DataFrame:
    """Peek into one raw CSV (first ``nrows`` rows; the biggest files are over 1 GB)."""
    return pd.read_csv(C.DATA_ROOT / folder / file, nrows=nrows)


def load_split(name: str) -> pd.DataFrame:
    """Records of one split ('train', 'val' or 'test')."""
    if name not in SPLIT_FOLDERS:
        raise ValueError(f"split must be one of {tuple(SPLIT_FOLDERS)}")
    path = C.PROCESSED_DIR / f"records_{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the data-preparation step first.")
    return pd.read_parquet(path)


def load_all() -> pd.DataFrame:
    return pd.concat([load_split(s) for s in SPLIT_FOLDERS], ignore_index=True)


def split_summary() -> pd.DataFrame:
    """Sizes and balance of every split: records, fraud, safe, and counts per message type."""
    rows = []
    for s in SPLIT_FOLDERS:
        d = load_split(s)
        row = {"split": SPLIT_FOLDERS[s], "role": SPLIT_ROLE[s], "records": len(d), "fraud": int((d.label == 1).sum()), "safe": int((d.label == 0).sum())}
        row.update({f"msg_{k}": int(v) for k, v in d["msg_kind"].value_counts().sort_index().items()})
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


def calls_table(records: pd.DataFrame) -> pd.DataFrame:
    return records[["record_id", "call_id", "call_text", "call_label", "call_source", "tier"]].rename(
        columns={"call_text": "text", "call_label": "label", "call_source": "source"})


def messages_table(records: pd.DataFrame) -> pd.DataFrame:
    return records[["record_id", "item_id", "msg_kind", "msg_text", "msg_url", "msg_label", "msg_source", "msg_lang", "tier"]].rename(
        columns={"msg_kind": "kind", "msg_text": "text", "msg_url": "url", "msg_label": "label", "msg_source": "source", "msg_lang": "lang"})


def find(text: str, split: str | None = None, where: str = "both") -> pd.DataFrame:
    """Case-insensitive search of the call and/or message text across the splits."""
    d = (load_split(split).assign(split=split) if split else
         pd.concat([load_split(s).assign(split=SPLIT_FOLDERS[s]) for s in SPLIT_FOLDERS], ignore_index=True))
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


def _write_text(path: Path, text: str, readonly: bool) -> None:
    if path.exists():
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    path.write_text(text)
    if readonly:
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _md_table(df: pd.DataFrame) -> str:
    head = "| set | " + " | ".join(df.columns) + " |\n|---|" + "---|" * len(df.columns)
    return head + "".join(f"\n| {i} | " + " | ".join(str(v) for v in r) + " |" for i, r in df.iterrows())


def _readme(split: str, d: pd.DataFrame) -> str:
    kinds = ", ".join(f"{k} {v}" for k, v in d["msg_kind"].value_counts().sort_index().items())
    tiers = ", ".join(f"{k} {v}" for k, v in d["tier"].value_counts().items())
    note = ("\n**LOCKED.** These files are read-only and the split is fingerprinted. Looking is fine, but never use this set to "
            "choose features or settings - it is used once for the final result.\n" if split == "test" else "")
    return f"""# {SPLIT_FOLDERS[split]}

{SPLIT_ROLE[split].capitalize()}.{note}
* records: **{len(d)}** (fraud {int((d.label == 1).sum())}, safe {int((d.label == 0).sum())}); message types: {kinds}; tiers: {tiers}
* `label` = 1 when the ATTACK is fraud, i.e. when the call OR the message is fraud (`call_label` / `msg_label` are the labels of each half)

| file | what it is |
|---|---|
| `paired_records.csv` | one row per attack: `call_text` + `msg_text` / `msg_url`, `label`, `call_label`, `msg_label`, `pair_type` (FF, FS, SF, SS), `tier` |
| `calls.csv` | the call transcripts only |
| `messages.csv` | the SMS / email / URL items only (`kind` = sms / email / url, `lang` = en / es) |

A call and a message are paired only inside the same split, so no call or message appears in two splits; near-identical scripts
(a "family") are never split between sets.
"""


def export_dataset_folders() -> Path:
    """Write ``dataset/{train,validation,test}/`` with plain CSV files. Safe to re-run."""
    root = C.DATASET_DIR
    root.mkdir(parents=True, exist_ok=True)
    for s, folder in SPLIT_FOLDERS.items():
        d = load_split(s)
        out = root / folder
        out.mkdir(exist_ok=True)
        ro = s == "test"
        _write(d, out / "paired_records.csv", ro)
        _write(calls_table(d), out / "calls.csv", ro)
        _write(messages_table(d), out / "messages.csv", ro)
        _write_text(out / "README.md", _readme(s, d), ro)
    _write_text(root / "README.md", f"""# Dataset

Prepared data, one subfolder per set:

* `train/` - training data
* `validation/` - fitting / validation data
* `test/` - final testing data (locked, read-only)

{_md_table(split_summary()[['records', 'fraud', 'safe']])}

Rebuilt automatically whenever the data is prepared; open it from code with `toadwatch.dataset_access`.
The raw files you provided stay in the original four folders next to `ToadWatch_PR/`.
""", False)
    return root


# ------------------------------------------------------------------ chart
def plot_split_composition():
    """Class balance, message types and text lengths for train / validation / test."""
    import matplotlib.pyplot as plt
    data = {SPLIT_FOLDERS[s]: load_split(s) for s in SPLIT_FOLDERS}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    names = list(data)
    for j, (lab, col) in enumerate(((0, "#4c9f70"), (1, "#c0504d"))):
        ax[0].bar([i + (j - .5) * .38 for i in range(3)], [(d.label == lab).sum() for d in data.values()], .38, color=col, label=["safe", "fraud"][j])
    ax[0].set_xticks(range(3), names); ax[0].set_title("Records per set and class"); ax[0].legend()
    bottoms = [0, 0, 0]
    for kind, col in (("sms", "#5b7db1"), ("email", "#d08c3c"), ("url", "#8e5ea2")):
        vals = [(d.msg_kind == kind).sum() for d in data.values()]
        ax[1].bar(names, vals, bottom=bottoms, color=col, label=kind)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax[1].set_title("Message type per set"); ax[1].legend()
    pos, ticks, labels = 0, [], []
    for n, d in data.items():
        for lab, series, col in (("call", d.call_text.str.len(), "#5b7db1"), ("msg", d.msg_text.str.len(), "#d08c3c")):
            bp = ax[2].boxplot(series.clip(upper=6000), positions=[pos], widths=.6, patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor(col)
            ticks.append(pos); labels.append(f"{n}\n{lab}"); pos += 1
        pos += .6
    ax[2].set_xticks(ticks, labels, fontsize=7); ax[2].set_title("Text length (characters)")
    fig.tight_layout()
    return fig
