"""Schedule steps 1-2: audit, clean, remove duplicates, split 70/10/20, lock the test set.

    python scripts/step1_2_prepare_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toadwatch.pipeline import run_data_preparation  # noqa: E402

if __name__ == "__main__":
    out = run_data_preparation()
    print("calls cleaning :", out["calls_report"])
    print("message cleaning:", out["msgs_report"])
    print("split manifest :", out["manifest"])
