"""Schedule step 5 (8-14 Nov): run the three versions on the test set ONCE.

    python scripts/step5_final_test.py --yes-run-the-test-once

Needs Step 4 to have frozen ``outputs/final_config.json``. Refuses to run a second time.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toadwatch.evaluation import run_final_test  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes-run-the-test-once", action="store_true",
                    help="confirm that the final configuration is frozen and the test set may be used")
    if not ap.parse_args().yes_run_the_test_once:
        sys.exit("Add --yes-run-the-test-once to confirm. The test set can only be used once.")
    out = run_final_test()
    print(out["results"][["n", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp",
                          "p_value"]].round(4))
