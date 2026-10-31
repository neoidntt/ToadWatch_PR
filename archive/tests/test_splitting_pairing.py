import pandas as pd
import pytest

from toadwatch import config as C
from toadwatch import splitting
from toadwatch.pairing import balance_calls, build_records
from toadwatch.splitting import assign_splits


def test_split_sizes_and_reproducibility(calls_df):
    a = assign_splits(calls_df, ["source", "label"])
    b = assign_splits(calls_df, ["source", "label"])
    assert (a["split"] == b["split"]).all()                      # fixed seed
    frac = a["split"].value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.70, abs=0.03)
    assert frac["val"] == pytest.approx(0.10, abs=0.03)
    assert frac["test"] == pytest.approx(0.20, abs=0.03)
    assert a.groupby("split")["label"].mean().between(0.45, 0.55).all()   # stratified


def test_pairing_keeps_labels_and_splits_apart(calls_df, msgs_df):
    calls = assign_splits(calls_df, ["source", "label"])
    msgs = assign_splits(msgs_df, ["kind", "label"])
    rec = build_records(calls, msgs)
    assert len(rec) == len(calls)
    assert rec["call_id"].is_unique and rec["item_id"].is_unique
    call_split = calls.set_index("call_id")["split"]
    msg_split = msgs.set_index("item_id")["split"]
    msg_label = msgs.set_index("item_id")["label"]
    assert (rec["call_id"].map(call_split) == rec["split"]).all()
    assert (rec["item_id"].map(msg_split) == rec["split"]).all()   # no leakage across splits
    assert (rec["item_id"].map(msg_label) == rec["label"]).all()   # fraud call <-> fraud message


def test_balance_calls_equalises_classes(calls_df):
    skew = pd.concat([calls_df[calls_df.label == 1], calls_df[calls_df.label == 0].head(20)])
    out = balance_calls(skew)
    assert out["label"].value_counts().nunique() == 1


def test_test_split_is_locked(tmp_path, monkeypatch, calls_df, msgs_df):
    monkeypatch.setattr(C, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(C, "SPLIT_MANIFEST_JSON", tmp_path / "manifest.json")
    rec = build_records(assign_splits(calls_df, ["source", "label"]), assign_splits(msgs_df, ["kind", "label"]))
    splitting.save_records(rec)
    with pytest.raises(PermissionError):
        splitting.load_split("test")
    assert len(splitting.load_split("test", allow_test=True)) > 0
    assert splitting.verify_test_untouched()
    assert not (tmp_path / "records_test.parquet").stat().st_mode & 0o222   # read-only on disk
