"""Steps 1-2 end to end: load -> audit -> clean/deduplicate -> split -> pair -> save.

The notebooks and ``scripts/`` call these functions; nothing here touches the test split
except to write it once.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C
from .cleaning import clean_calls, clean_messages
from .data_loading import load_calls, load_messages
from .pairing import balance_calls, balance_messages, build_records
from .dataset_access import export_dataset_folders
from .splitting import assign_splits, save_records


def audit_table(df: pd.DataFrame, text_col: str, source_col: str = "source") -> pd.DataFrame:
    """Step 1: per source - rows, label counts, missing values, exact duplicates, length."""
    rows = []
    for source, g in df.groupby(source_col):
        txt = g[text_col]
        rows.append({
            "source": source,
            "rows": len(g),
            "safe": int((g["label"] == 0).sum()),
            "fraud": int((g["label"] == 1).sum()),
            "missing_text": int(txt.isna().sum()),
            "exact_duplicates": int(txt.duplicated().sum()),
            "median_chars": float(txt.astype(str).str.len().median()),
        })
    return pd.DataFrame(rows).sort_values("source").reset_index(drop=True)


def run_data_preparation(save: bool = True) -> dict:
    """Run Steps 1-2 and return every intermediate table (for the notebooks)."""
    if save and C.FINAL_TEST_LOCK.exists():
        raise RuntimeError("The final test already ran: re-splitting would invalidate its results.")
    C.ensure_dirs()
    calls_raw = load_calls()
    msgs_raw = load_messages()
    audit = {"calls": audit_table(calls_raw, "text"), "messages": audit_table(msgs_raw, "text")}

    calls, calls_report = clean_calls(calls_raw)
    msgs, msgs_report = clean_messages(msgs_raw)

    calls = balance_calls(calls)
    msgs = balance_messages(msgs)
    calls = assign_splits(calls, ["source", "label"])
    msgs = assign_splits(msgs, ["kind", "label"])
    records = build_records(calls, msgs)

    out = {"calls_raw": calls_raw, "msgs_raw": msgs_raw, "audit": audit,
           "calls_report": calls_report, "msgs_report": msgs_report,
           "calls": calls, "msgs": msgs, "records": records}
    if save:
        out["manifest"] = save_records(records)
        out["dataset_dir"] = export_dataset_folders()   # dataset/<version>/{train,validation,test}/*.csv
        audit["calls"].to_csv(C.TABLES_DIR / "audit_calls.csv", index=False)
        audit["messages"].to_csv(C.TABLES_DIR / "audit_messages.csv", index=False)
        (C.TABLES_DIR / "cleaning_report.json").write_text(
            json.dumps({"calls": calls_report, "messages": msgs_report}, indent=2, default=str))
    return out
