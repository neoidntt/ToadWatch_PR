# ToadWatch_PR

Single-channel vs multichannel machine learning for detecting **TOAD** (Telephone-Oriented Attack Delivery)
scams in Puerto Rico. William Díaz - Fase 11-1.

## The three models (and only three)
Two Random Forests are trained, one on call transcripts (which may mention URLs) and one on SMS / email / URL text.

| Model | Input | Decision |
|---|---|---|
| `call_only` | call transcript | call forest >= its threshold |
| `message_only` | SMS / email / URL | message forest >= its threshold |
| `combined` | both | **fraud if either part is flagged** (call and message are connected, so one fraudulent part makes the whole attack fraud) |

Label rule: an attack is fraud if its call OR its message is fraud (pair types FF, FS, SF, SS).

## Honest evaluation
Near-identical texts form *families* that stay in one split; thresholds and forest settings are chosen on
out-of-fold development data only; the test set (20 %) is read-only, fingerprinted and scored once.
Results are shown per tier: `hard` (written for this project), `matched` (both classes from the same corpus) and
`unmatched` (class-pure source files: optimistic, a source-only baseline scores ~0.99 there). Audits: ceilings,
overlap, shortcut baselines, label audit, label-shuffle check (falls to chance).

## Run it in VS Code
Open the folder `ToadWatch_PR`, select the Python kernel (`pip install -r requirements.txt`), then run the notebooks
in `notebooks/` in order, each with *Run All*: `00_START_HERE`, `01_prepare_data`, `02_audit_data`,
`03_tune_and_cross_validate`, `04_shuffle_check`, `05_final_test`, `06_charts_and_report`.
Terminal alternative: `python scripts/run_study.py all` then `python scripts/run_study.py final`.
The last cells rebuild `report/` (Word + PDF report, PowerPoint + PDF slides) from the run you just made.

## Layout
`toadwatch/` modules - `scripts/` command line and notebook generator - `tests/` pytest - `dataset/train|validation|test`
CSV views - `outputs/` tables, figures, config - `report/` generated documents - `evidence/` Step-3 class evidence -
`archive/` earlier versions (kept, not used).
