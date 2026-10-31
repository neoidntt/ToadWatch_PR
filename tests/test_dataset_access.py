import numpy as np
import pandas as pd
import pytest

from toadwatch import config as C
from toadwatch import dataset_access as DA
from toadwatch.dataset_builder import build_records


def _pools():
    rows_c, rows_m = [], []
    for i in range(1800):
        lab = int(i % 5 < 2)
        tier = ["unmatched", "matched", "hard"][i % 3]
        rows_c.append({"call_id": f"c{i}", "text": f"press 1 call {i}", "label": lab, "source": "s", "lang": "en", "group": f"cg{i}", "is_hard": tier == "hard", "tier": tier})
        k = ["sms", "email", "url"][i % 3] if tier == "unmatched" else ["sms", "email"][i % 2]
        rows_m.append({"item_id": f"m{i}", "kind": k, "text": f"msg {i}", "label": lab, "source": "s", "lang": "en", "group": f"mg{i}", "is_hard": tier == "hard", "tier": tier, "url": ""})
    return pd.DataFrame(rows_c), pd.DataFrame(rows_m)


@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    rec = build_records(*_pools(), seed=3)
    proc = tmp_path / "processed"
    proc.mkdir()
    for s in ("train", "val", "test"):
        rec[rec.split == s].reset_index(drop=True).to_parquet(proc / f"records_{s}.parquet")
    monkeypatch.setattr(C, "PROCESSED_DIR", proc)
    monkeypatch.setattr(C, "DATASET_DIR", tmp_path / "dataset")
    return rec


def test_summary_and_loading(fake_data):
    s = DA.split_summary()
    assert list(s.index) == ["train", "validation", "test"] and s["records"].sum() == len(fake_data)
    assert (s["fraud"] + s["safe"] == s["records"]).all()
    assert len(DA.load_split("val")) == s.loc["validation", "records"]
    with pytest.raises(ValueError):
        DA.load_split("nope")


def test_export_creates_one_folder_per_set_and_locks_test(fake_data):
    root = DA.export_dataset_folders()
    for folder in ("train", "validation", "test"):
        for f in ("paired_records.csv", "calls.csv", "messages.csv", "README.md"):
            assert (root / folder / f).exists()
    assert len(pd.read_csv(root / "test" / "calls.csv")) == len(DA.load_split("test"))
    assert not (root / "test" / "paired_records.csv").stat().st_mode & 0o222          # read-only
    DA.export_dataset_folders()                                                          # re-export over locked files works
    sets = {f: pd.read_csv(root / f / "paired_records.csv") for f in ("train", "validation", "test")}
    assert not set(sets["train"].call_id) & set(sets["test"].call_id)
    assert not set(sets["validation"].item_id) & set(sets["test"].item_id)
    assert {"label", "call_label", "msg_label", "pair_type", "tier"} <= set(sets["train"].columns)


def test_find_searches_text(fake_data):
    hits = DA.find("press 1")
    assert len(hits) > 0 and set(hits["split"]) <= {"train", "validation", "test"}
