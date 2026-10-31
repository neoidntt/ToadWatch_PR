"""Write the notebooks in notebooks/ (run: python scripts/make_notebooks.py)."""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = "import sys, pathlib\nROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()\nsys.path.insert(0, str(ROOT))\nfrom toadwatch import config as C, workflow as W, charts, report_builder\nimport pandas as pd\nfrom IPython.display import Image, display\npd.set_option('display.width', 160)"
SHOW = "def show(*names):\n    for n in names:\n        display(Image(filename=str(C.FIGURES_DIR / n)))"


def nb(name, cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(n, ROOT / "notebooks" / name)


REBUILD = ("md", "Rebuilds the Word report, PDF report, PowerPoint and PDF slides from the run you just made (`report/`).")
nb("00_START_HERE.ipynb", [
    ("md", "# ToadWatch_PR - start here\n\nExactly **three models**, built from two Random Forests:\n\n1. **call_only** - the call transcript (it may mention URLs)\n2. **message_only** - SMS / email / URL\n3. **combined** - the attack is **fraud if either the call or the message is flagged** (the two parts are connected, so one fraudulent part makes the whole TOAD attack fraud).\n\n"
           "Run in order, each with *Run All*:\n\n| # | Notebook | Time |\n|---|---|---|\n| 1 | `01_prepare_data` | ~3 min |\n| 2 | `02_audit_data` | ~1 min |\n| 3 | `03_tune_and_cross_validate` | ~5 min |\n| 4 | `04_shuffle_check` | ~2 min |\n| 5 | `05_final_test` (one shot) | ~1 min |\n| 6 | `06_charts_and_report` | ~1 min |\n\nThe final test can run only once; later runs just reload the saved result."),
    ("code", SETUP + "\nprint('ok', C.MODELS)")])
nb("01_prepare_data.ipynb", [
    ("md", "# 1. Prepare data\nClean, remove duplicates, group near-copies into families, split by family, pair calls with messages, lock the test set. Browse the splits in `dataset/train`, `dataset/validation`, `dataset/test`."),
    ("code", SETUP),
    ("code", "if C.FINAL_TEST_LOCK.exists():\n    d = W.load_prepared(); print('final test already ran - loaded the saved data')\nelse:\n    d = W.prepare()\nrec = d['records']\nrec.groupby(['split', 'tier', 'label']).size().unstack()"),
    ("code", "rec.groupby(['split', 'pair_type']).size().unstack()"),
    ("code", "rec[['record_id', 'split', 'tier', 'pair_type', 'call_label', 'msg_label', 'label', 'call_text', 'msg_text']].head(10)")])
nb("02_audit_data.ipynb", [
    ("md", "# 2. Audit\nCeilings, overlap between splits, near-copies, shortcut baselines (a model that only knows the source)."),
    ("code", SETUP),
    ("code", "a = W.audit(); a['ceilings']"),
    ("code", "a['overlap']"),
    ("code", "a['shortcuts'].round(3)"),
    ("md", "The `unmatched` tier comes from class-pure source files, so source identity gives the answer away; the `hard` and `matched` tiers are the honest numbers.")])
nb("03_tune_and_cross_validate.ipynb", [
    ("md", "# 3. Tune and cross-validate\nTunes both forests and the decision thresholds on out-of-fold scores of the development data only, then freezes `outputs/final_config.json`."),
    ("code", SETUP + "\n" + SHOW),
    ("code", "r = W.load_cv_results() if C.FINAL_TEST_LOCK.exists() else W.tune_and_cv()\nr['overall'][['accuracy', 'f1', 'balanced_accuracy']].round(4)"),
    ("code", "r['config']['thresholds'], r['config']['params']"),
    ("code", "r['by_tier'].round(3)"),
    ("code", "r['by_pair_type'].round(3)"),
    ("code", "charts.make_all_charts(); show('01_cv_accuracy_vs_ceilings.png', '03_cv_tiers.png', '05_cv_boxplot.png', '10_threshold_sweep.png')")])
nb("04_shuffle_check.ipynb", [
    ("md", "# 4. Label-shuffle check\nWith shuffled labels every model must fall to about 50 %. If not, something leaks."),
    ("code", SETUP),
    ("code", "W.shuffle().round(3)")])
nb("05_final_test.ipynb", [
    ("md", "# 5. Final test (one shot)\nThe first run scores the locked test set; any later run reloads that result."),
    ("code", SETUP),
    ("code", "t = W.final_test()\nt['results'][['accuracy', 'f1', 'balanced_accuracy', 'acc_ci_low', 'acc_ci_high']].astype(float).round(4)"),
    ("code", "t['by_tier'].round(3)"),
    ("code", "t['pairwise']"),
    REBUILD, ("code", "report_builder.build_all()")])
nb("06_charts_and_report.ipynb", [
    ("md", "# 6. Charts and report"),
    ("code", SETUP + "\n" + SHOW),
    ("code", "names = [p.name for p in charts.make_all_charts()]; names"),
    ("code", "show(*[n for n in names if n[:2] in ('11', '12', '13', '14', '15', '17', '18')])"),
    REBUILD, ("code", "report_builder.build_all()")])
