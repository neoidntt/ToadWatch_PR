"""Run the study (three models) from the command line, in order.

    python scripts/run_study.py prepare   # 1. clean, families, buckets, connected call+message pairs, lock the test   (~4 min)
    python scripts/run_study.py tune      # 2. tune the two Random Forests + thresholds, cross-validate the 3 models  (~5 min)
    python scripts/run_study.py audit     # 3. leakage / label audits (needs step 2 for the label audit)             (~1 min)
    python scripts/run_study.py shuffle   # 4. shuffled-label sanity check (must fall to chance)                       (~2 min)
    python scripts/run_study.py final     # 5. ONE-SHOT test (refused a second time) + the report files                (~1 min)
    python scripts/run_study.py charts    # 6. all figures into outputs/figures
    python scripts/run_study.py report    # 7. Word + PDF report and PowerPoint + PDF from the latest run
    python scripts/run_study.py all       # 1-4, 6, 7 (the final test only with --final)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toadwatch import charts, report_builder, workflow as W  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "tune", "audit", "shuffle", "final", "charts", "report", "all"])
    ap.add_argument("--final", action="store_true", help="with 'all': also run the one-shot test")
    a = ap.parse_args()
    run = lambda s: a.stage in (s, "all")
    if run("prepare"):
        print(W.prepare()["manifest"])
    if run("tune"):
        t = W.tune_and_cv(); print(t["overall"][["accuracy", "f1", "balanced_accuracy"]].astype(float).round(4)); print(t["config"]["thresholds"])
    if run("audit"):
        d = W.audit(); print(d["ceilings"]); print(d["overlap"].to_string(index=False)); print(d["shortcuts"].round(3).to_string(index=False))
    if run("shuffle"):
        print(W.shuffle().round(3))
    if a.stage == "final" or (a.stage == "all" and a.final):
        r = W.final_test(); print(r["results"][["accuracy", "f1", "balanced_accuracy", "acc_ci_low", "acc_ci_high"]].astype(float).round(4))
    if run("charts"):
        print([p.name for p in charts.make_all_charts()])
    if run("report") or a.stage == "final":
        try:
            report_builder.build_all()
        except FileNotFoundError as e:
            print("report skipped:", e)


if __name__ == "__main__":
    main()
