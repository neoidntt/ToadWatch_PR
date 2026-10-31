"""Run the v2 (fair fusion) study from the command line, in order.

    python scripts/run_v2.py prepare     # 1. load, clean, deduplicate, split, pair   (~2 min)
    python scripts/run_v2.py diagnose    # 2. why does combined lag? CV on train+val   (~3 min)
    python scripts/run_v2.py tune        # 3. tune, pick early/stacked, freeze config  (~5 min)
    python scripts/run_v2.py final       # 4. ONE-SHOT test (refused a second time)
    python scripts/run_v2.py charts      # 5. all figures into outputs/v2/figures
    python scripts/run_v2.py report      # 6. Word + PDF report and PowerPoint + PDF from the latest run
    python scripts/run_v2.py all         # 1-3 and 5 (never the final test unless --final)
    python scripts/run_v2.py all --final # everything, including the one-shot test and the documents
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["TOADWATCH_VERSION"] = "v2"          # must happen before importing toadwatch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toadwatch import config as C  # noqa: E402
from toadwatch import workflow as W  # noqa: E402
from toadwatch import charts_v2, report_builder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "diagnose", "tune", "final", "charts", "report", "all"])
    ap.add_argument("--final", action="store_true", help="with 'all': also run the one-shot test")
    a = ap.parse_args()
    run = lambda s: a.stage in (s, "all")
    if run("prepare"):
        out = W.prepare_data(); print(out["manifest"])
    if run("diagnose"):
        d = W.run_diagnosis(); print(d["ablation"].round(4).to_string(index=False)); print(d["complementarity"])
    if run("tune"):
        t = W.tune_and_freeze(); print(t["summary"].round(4)); print("combined model chosen:", t["choice"])
    if a.stage == "final" or (a.stage == "all" and a.final):
        r = W.final_test(); print(r["results"][["accuracy", "precision", "recall", "f1", "f1_ci_low", "f1_ci_high"]].round(4))
        print(r["pairwise"].round(4).to_string(index=False))
    if run("charts"):
        if not (C.TABLES_DIR / "learning_curve.csv").exists() and (C.OUTPUTS_DIR / "final_config.json").exists():
            W.learning_curves()                      # ~3 min, train+val only
        print("figures:", [p.name for p in charts_v2.make_all_charts()])
    if run("report") or a.stage == "final":
        try:
            report_builder.build_all()
        except FileNotFoundError as e:
            print("report skipped:", e)


if __name__ == "__main__":
    main()
