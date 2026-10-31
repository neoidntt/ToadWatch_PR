import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def calls_df():
    rows = []
    for i in range(60):
        rows.append({"call_id": f"s{i}", "text": f"Suspect: hello number {i} about your delivery appointment today",
                     "label": 0, "source": "a" if i % 2 else "b", "lang": "en"})
        rows.append({"call_id": f"f{i}", "text": f"Caller: urgent {i} press 1 now give me your PIN or police arrest",
                     "label": 1, "source": "a" if i % 2 else "b", "lang": "en"})
    return pd.DataFrame(rows)


@pytest.fixture
def msgs_df():
    rows = []
    for i in range(90):
        kind = ["sms", "email", "url"][i % 3]
        for label in (0, 1):
            host = f"site{i}-{label}.com"
            text = f"https://{host}/x" if kind == "url" else f"message {i} {label} see https://{host}/a"
            rows.append({"item_id": f"{kind}{i}_{label}", "kind": kind, "text": text,
                         "url": f"https://{host}/x" if kind == "url" else f"https://{host}/a",
                         "label": label, "source": kind, "lang": "en"})
    return pd.DataFrame(rows)
