"""Run the v3 "true accuracy" study from the command line, in order.

    python scripts/run_v3.py prepare   # 1. clean, families, buckets, mixed pairs, lock test   (~5 min)
    python scripts/run_v3.py audit     # 2. leakage + label audits                              (~3 min)
    python scripts/run_v3.py tune      # 3. tune + family-safe CV of all 7 models + freeze      (~25 min)
    python scripts/run_v3.py shuffle   # 4. shuffled-label sanity check (must fall to chance)   (~10 min)
    python scripts/run_v3.py final     # 5. ONE-SHOT test (refused a second time)               (~3 min)
    python scripts/run_v3.py charts    # 6. all figures into outputs/v3/figures
    python scripts/run_v3.py report    # 7. Word + PDF report and PowerPoint + PDF from the latest run
    python scripts/run_v3.py all       # 1-4 and 6-7 (the final test only with --final)
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["TOADWATCH_VERSION"] = "v3"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toadwatch import charts_v3, report_builder_v3, workflow_v3 as W  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "audit", "tune", "shuffle", "final", "charts", "report", "all"])
    ap.add_argument("--final", action="store_true", help="with 'all': also run the one-shot test")
    a = ap.parse_args()
    run = lambda s: a.stage in (s, "all")
    if run("prepare"):
        print(W.prepare()["manifest"])
    if run("audit"):
        d = W.audit(); print(d["ceilings"]); print(d["overlap"].to_string(index=False)); print(d["shortcuts"].round(3).to_string(index=False))
    if run("tune"):
        t = W.tune_and_cv(); print(t["summary"].round(4).to_string())
    if run("shuffle"):
        print(W.shuffle_check().round(3).to_string())
    if a.stage == "final" or (a.stage == "all" and a.final):
        r = W.final_test(); print(r["results"][["accuracy", "balanced_accuracy", "f1", "acc_ci_low", "acc_ci_high"]].round(4).to_string())
    if run("charts"):
        print([p.name for p in charts_v3.make_all_charts()])
    if run("report") or a.stage == "final":
        try:
            report_builder_v3.build_all()
        except FileNotFoundError as e:
            print("report skipped:", e)


if __name__ == "__main__":
    main()
