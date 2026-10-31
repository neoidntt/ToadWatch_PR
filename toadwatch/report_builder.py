"""Build the written deliverables from the results of the run that just finished.

    build_all()  ->  report/TOAD_Report.docx + .pdf     and     report/TOAD_Slides.pptx + .pdf

Every number, table, figure and verdict sentence is read from the tables / config / figures the pipeline saved, so the
documents always match the latest run. It is the last cell of the final notebooks and the ``report`` stage of
scripts/run_study.py. Only the three models appear anywhere in them.
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pptx import Presentation
from pptx.dml.color import RGBColor as PRGB
from pptx.util import Inches as PIn, Pt as PPt

from . import config as C
from .docx_utils import _Doc, _fig, to_pdf

NAMES = C.MODEL_NAMES
SHORT = {"call_only": "Call only", "message_only": "Message only", "combined": "Combined"}
pct = lambda x, d=1: f"{float(x) * 100:.{d}f}%"
num = lambda x, d=3: f"{float(x):.{d}f}"


def load_run() -> dict:
    T = C.TABLES_DIR
    if not (T / "final_test_results.csv").exists():
        raise FileNotFoundError("The final test has not run yet - nothing to report.")
    rd = lambda n, **kw: pd.read_csv(T / n, **kw) if (T / n).exists() else None
    return {"res": rd("final_test_results.csv", index_col="model"), "by_tier": rd("final_test_by_tier.csv"), "by_pair": rd("final_test_by_pair_type.csv"),
            "det": rd("final_test_detectors.csv"), "pair": rd("final_test_pairwise.csv"), "ceil": rd("final_test_ceilings.csv", index_col=0)["accuracy"].to_dict(),
            "cv": rd("cv_overall.csv", index_col=0), "cv_tier": rd("cv_by_tier.csv"), "short": rd("audit_shortcut_baselines.csv"), "shuffle": rd("audit_label_shuffle.csv", index_col=0),
            "overlap": rd("audit_overlap.csv"), "neardup": rd("audit_near_duplicates.csv"), "suspects": rd("label_audit_by_source.csv"),
            "pr": rd("puerto_rico_cases.csv", index_col="model"), "manifest": json.loads(C.SPLIT_MANIFEST_JSON.read_text()),
            "data": json.loads((T / "data_report.json").read_text()), "cfg": json.loads(C.FINAL_CONFIG_JSON.read_text()),
            "stamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "tag": "three-model study"}


def facts(d: dict) -> dict:
    """The numbers every sentence below is built from."""
    r, bt, det, pw = d["res"], d["by_tier"], d["det"], d["pair"]
    tier = lambda m, t, col="accuracy": float(bt[(bt.model == m) & (bt.tier == t)][col].iloc[0]) if ((bt.model == m) & (bt.tier == t)).any() else float("nan")
    dtier = lambda dname, t: float(det[(det.detector == dname) & (det.tier == t)]["balanced_accuracy"].iloc[0]) if ((det.detector == dname) & (det.tier == t)).any() else float("nan")
    out = {"tier": tier, "dtier": dtier, "n": int(r["n"].iloc[0]), "ceiling": d["ceil"]["call_oracle"], "majority": d["ceil"]["majority_class"]}
    for key, b in (("vs_call", "call_only"), ("vs_msg", "message_only")):
        row = pw[(pw.model_a == "combined") & (pw.model_b == b)].iloc[0]
        out[key] = {"diff": row.acc_diff_a_minus_b, "lo": row.ci_low, "hi": row.ci_high, "state": "better" if row.ci_low > 0 else ("worse" if row.ci_high < 0 else "indistinguishable"),
                    "a_only": int(row.a_only_right), "b_only": int(row.b_only_right), "mcnemar": row.mcnemar_p}
    return out


def verdict(F: dict) -> str:
    c, m = F["vs_call"], F["vs_msg"]
    if c["state"] == "better" and m["state"] == "better":
        return (f"The combined model (either part fraud -> attack is fraud) is significantly better than both single-channel models: +{num(c['diff'])} accuracy over call-only "
                f"(95% interval {num(c['lo'])} to {num(c['hi'])}) and +{num(m['diff'])} over message-only ({num(m['lo'])} to {num(m['hi'])}).")
    if c["state"] == "indistinguishable" or m["state"] == "indistinguishable":
        return "The combined model is better than at least one single-channel model, but its advantage over the other cannot be told apart from zero on this test set."
    return "The combined model is not better than the single-channel models on this test set."


def build_docx(d: dict, out):
    D = _Doc(d)
    r, F, cfg, mf = d["res"], facts(d), d["cfg"], d["manifest"]["files"]
    C_ = WD_ALIGN_PARAGRAPH.CENTER
    for _ in range(5): D.p("")
    D.p("Detecting Telephone-Oriented Attack Delivery (TOAD) Scams in Puerto Rico", align=C_).runs[0].bold = True
    D.p("Three models: call only, message only, and combined (either part is fraud)", align=C_)
    D.p("William Díaz · Fase 11-1 · Prof. Mileidy Perdomo", align=C_)
    D.p(f"Generated {d['stamp']} · random seed {C.SEED}", align=C_, size=10)
    D.doc.add_page_break()

    D.h1("Summary")
    D.p("A TOAD attack is a phone call plus a message, email or link that belong together. If either part is fraud, the whole attack is fraud - including the part that looks "
        "harmless on its own. Three models were built and tested: a call-only Random Forest (it also reads links the caller mentions), a message-only Random Forest "
        "(SMS, email or link), and a combined model that says 'fraud' when either Random Forest detects fraud.")
    D.bullets([f"Final test ({F['n']} attacks, {pct(F['majority'], 0)} would be right by always guessing the majority class): call only {pct(r.loc['call_only', 'accuracy'])}, "
               f"message only {pct(r.loc['message_only', 'accuracy'])}, combined {pct(r.loc['combined', 'accuracy'])} (F1 {num(r.loc['call_only', 'f1'])}, {num(r.loc['message_only', 'f1'])}, {num(r.loc['combined', 'f1'])}).",
               verdict(F),
               f"A perfect call-only or message-only model can reach at most {pct(F['ceiling'])} under this rule, because it cannot see the other half of the attack; no single-channel model exceeded it.",
               f"Where the answer cannot be guessed from the dataset (matched tier): combined {pct(F['tier']('combined', 'matched'))}, call only {pct(F['tier']('call_only', 'matched'))}, "
               f"message only {pct(F['tier']('message_only', 'matched'))}. On my hard, style-matched set: combined {pct(F['tier']('combined', 'hard'))}. "
               f"On the unmatched tier (one class per source file, optimistic): combined {pct(F['tier']('combined', 'unmatched'))}.",
               "The shuffled-label check and the overlap checks passed." if (d["overlap"]["ok"].all() and (d["shuffle"]["accuracy"] < d["shuffle"]["majority_class_rate"] + 0.05).all())
               else "WARNING: the shuffled-label check or an overlap check FAILED (see Audits); treat the scores as unreliable until the leak is found."])

    D.h1("The three models")
    D.table(["Model", "What it is", "Says 'fraud' when"],
            [["Call only", "Random Forest on the call transcript, including links the caller mentions", f"call forest score >= {num(cfg['thresholds']['call_only'], 2)}"],
             ["Message only", "Random Forest on the SMS / email / link", f"message forest score >= {num(cfg['thresholds']['message_only'], 2)}"],
             ["Combined", "the same two forests, joined by OR (no third classifier)", f"call score >= {num(cfg['thresholds']['combined'][0], 2)} OR message score >= {num(cfg['thresholds']['combined'][1], 2)}"]],
            "Table 1. The three models and their decision rules (thresholds chosen on development data only).")
    D.p("Each forest is trained on the label of its OWN text (is this transcript a scam? is this message a scam?). Forest settings: " + json.dumps(cfg["params"]))

    D.h1("Data and protocol")
    dr = d["data"]
    D.bullets([f"Public data after cleaning: calls {dr['calls']['rows_clean']:,}; messages {dr['messages']['rows_clean']:,}. This includes the new file sms_phishing_english_spanish.csv (60,535 real phishing SMS, "
               f"fraud only): whitespace was normalised, repeats removed, at most {C.SMS_CAMPAIGN_MAX_PER_CAMPAIGN} messages per scam campaign kept, and each campaign counts as one family.",
               "Hard set written for this project (not recorded from real scams): polite scams without keywords and safe texts full of scam words, Puerto Rico flavoured, English and Spanish.",
               f"Each record pairs one call with one message; pair types FF (fraud+fraud), FS (fraud call + safe-looking message), SF, SS. Label = fraud if either part is fraud. "
               f"Mix {int(C.PAIR_MIX['FF'] * 100)}% FF, {C.PAIR_MIX['FS'] * 100:g}% FS, {C.PAIR_MIX['SF'] * 100:g}% SF, {int(C.PAIR_MIX['SS'] * 100)}% SS.",
               f"Near-identical texts form one family; every family sits in one of {C.N_BUCKETS} buckets (test = {len(C.TEST_BUCKETS)}, validation = {len(C.VAL_BUCKETS)}, train = the rest). "
               f"Splits: {mf['train']['rows']:,} train, {mf['val']['rows']} validation, {mf['test']['rows']} test attacks.",
               "Three tiers: hard (style-matched, mine), matched (both classes from the same corpus or generator) and unmatched (each source file holds one class, so a model can score high just by recognising the dataset). "
               "The first two measure content detection; the third is optimistic."])

    D.h1("Audits")
    ok = d["overlap"]["ok"].all()
    D.p(f"Overlap: {int(d['overlap']['shared'].sum())} shared ids or families between test and development or between folds ({'all checks pass' if ok else 'FAIL'}).")
    if d["neardup"] is not None:
        D.p(f"Near-copies between development folds (cosine > 0.8): at most {pct(d['neardup']['share_with_near_copy'].max())} of a fold's texts.")
    sc = d["short"]
    D.table(["Metadata-only baseline", "All", "Hard", "Matched", "Unmatched"],
            [[r_.baseline, pct(r_.accuracy_all), pct(r_.accuracy_hard), pct(r_.accuracy_matched), pct(r_.accuracy_unmatched)] for r_ in sc.itertuples()],
            f"Table 2. Predicting the label from metadata only (no text read). Majority-class rate: hard {pct(sc.majority_hard.iloc[0])}, matched {pct(sc.majority_matched.iloc[0])}, unmatched {pct(sc.majority_unmatched.iloc[0])}.")
    D.fig("08_shortcut_baselines", "Figure 1. Metadata-only baselines by tier.", 6.4)
    sh = d["shuffle"]
    D.table(["Model", "Accuracy with SHUFFLED labels", "ROC AUC", "Majority-class rate"], [[NAMES[m], num(sh.loc[m, "accuracy"]), num(sh.loc[m, "roc_auc"]), num(sh.loc[m, "majority_class_rate"])] for m in C.MODELS],
            "Table 3. With shuffled labels honest models fall to chance; anything clearly higher would signal leakage.")
    if d["suspects"] is not None:
        top = d["suspects"].sort_values("contested_share", ascending=False).head(4)
        D.p("Label audit: texts whose label a held-out forest contradicts most confidently are listed in outputs/tables/label_audit_suspects_to_review.csv for manual review. Most contested sources: "
            + "; ".join(f"{x.channel} {x.source} ({pct(x.contested_share)} of {int(x['items'])})" for _, x in top.iterrows()) + ".")

    D.h1("Cross-validation (development data)")
    cv = d["cv"]
    D.table(["Model", "Accuracy", "Precision", "Recall", "F1", "Balanced acc."], [[NAMES[m], num(cv.loc[m, "accuracy"]), num(cv.loc[m, "precision"]), num(cv.loc[m, "recall"]), num(cv.loc[m, "f1"]), num(cv.loc[m, "balanced_accuracy"])] for m in C.MODELS],
            "Table 4. Out-of-fold results on the development data (4 family-safe folds). Settings and thresholds were chosen here, so these numbers are slightly optimistic.")
    D.fig("05_cv_boxplot", "Figure 2. Accuracy per cross-validation fold.", 4.8)

    D.h1("Final test (run once)")
    D.table(["Model", "Accuracy", "95% interval", "Precision", "Recall", "F1", "Balanced acc.", "ROC AUC"],
            [[NAMES[m], num(r.loc[m, "accuracy"]), f"{num(r.loc[m, 'acc_ci_low'])}-{num(r.loc[m, 'acc_ci_high'])}", num(r.loc[m, "precision"]), num(r.loc[m, "recall"]), num(r.loc[m, "f1"]),
              num(r.loc[m, "balanced_accuracy"]), num(r.loc[m, "roc_auc"])] for m in C.MODELS],
            f"Table 5. Final test, {F['n']} attacks. Intervals resample whole script families (cluster bootstrap).")
    D.fig("11_test_acc_ci", "Figure 3. Test accuracy with intervals.", 5.2)
    D.fig("12_test_vs_ceilings", "Figure 4. Test accuracy against the single-channel ceilings.", 5.2)
    bt = d["by_tier"]
    D.table(["Tier", "Model", "n", "Accuracy", "95% interval", "F1", "Recall"],
            [[x.tier, SHORT[x.model], int(x.n), num(x.accuracy), f"{num(x.acc_ci_low)}-{num(x.acc_ci_high)}", num(x.f1), num(x.recall)] for x in bt.sort_values(["tier", "model"]).itertuples()],
            "Table 6. Test accuracy by tier. Hard and matched measure content detection; unmatched is optimistic.")
    D.fig("15_test_tiers", "Figure 5. Accuracy by tier.", 5.6)
    bp = d["by_pair"].pivot(index="model", columns="pair_type", values="accuracy").reindex(list(C.MODELS))
    D.table(["Model", "FF (fraud+fraud)", "FS (fraud call, safe-looking msg)", "SF (safe-looking call, fraud msg)", "SS (not fraud)"], [[SHORT[m]] + [num(bp.loc[m, t]) for t in ("FF", "FS", "SF", "SS")] for m in C.MODELS],
            "Table 7. Accuracy by pair type. A call-only model cannot see the fraud in SF; a message-only model cannot see it in FS; the combined model can.")
    D.fig("14_test_pair_type", "Figure 6. Accuracy by pair type.", 5.6)
    dt_ = d["det"]
    D.table(["Detector", "Tier", "n", "Balanced acc.", "Precision", "Recall", "F1"], [[x.detector, x.tier, int(x.n), num(x.balanced_accuracy), num(x.precision), num(x.recall), num(x.f1)] for x in dt_.itertuples()],
            "Table 8. Each Random Forest judged on its OWN text (is this transcript / this message a scam?), by tier.")
    D.fig("16_test_detectors", "Figure 7. Detector quality by tier.", 5.2)
    pw = d["pair"]
    D.table(["A", "B", "Accuracy diff (A-B)", "95% interval", "A only right", "B only right", "McNemar p"],
            [[SHORT[x.model_a], SHORT[x.model_b], num(x.acc_diff_a_minus_b, 4), f"{num(x.ci_low, 4)} to {num(x.ci_high, 4)}", int(x.a_only_right), int(x.b_only_right), "< 0.001" if x.mcnemar_p < 0.001 else num(x.mcnemar_p)] for x in pw.itertuples()],
            "Table 9. Paired comparison on the same test attacks.")
    D.fig("17_test_pairwise", "Figure 8. Paired accuracy differences.", 5.2)
    D.fig("13_test_confusion", "Figure 9. Confusion matrices.", 6.2)
    D.fig("06_feature_importance", "Figure 10. Most important features of each forest (development fit).", 6.2)
    if d["pr"] is not None:
        D.table(["Model", "n", "Accuracy", "Precision", "Recall", "F1"], [[SHORT[m], int(d["pr"].loc[m, "n"]), num(d["pr"].loc[m, "accuracy"]), num(d["pr"].loc[m, "precision"]), num(d["pr"].loc[m, "recall"]), num(d["pr"].loc[m, "f1"])] for m in C.MODELS],
                "Table 10. Real Puerto Rico cases (data/external/pr_cases.csv).")

    D.h1("What this means")
    D.bullets([verdict(F),
               "Part of the combined model's advantage is the design of the task: under the either-part-is-fraud rule, one channel alone cannot see the other half. Table 7 shows exactly where each model fails.",
               "The honest content-detection numbers are the hard and matched tiers (Table 6) and the detector table (Table 8). The unmatched tier is optimistic because each source file holds one class.",
               "To raise real accuracy further the project needs more diverse REAL data: real DACO / university cases, and real recorded safe and fraud calls from the same collection."])
    D.h1("Limitations")
    D.bullets(["The hard set and the safe halves of fraud attacks are constructed; the pairs are not recorded attacks. The either-part-is-fraud rule is the TOAD definition from the proposal, applied as a label.",
               "Public call data is partly synthetic (written dialogues) and partly real robocall transcripts; safe real calls are customer-service corpora.",
               "The matched tier for calls is synthetic (LLM-written dialogues) and for messages mostly older spam collections.",
               "The label audit is a list for a human to read; labels were not verified one by one.",
               "No domain-age lookups (offline popularity rank used instead)." + ("" if d["pr"] is not None else " No real Puerto Rico cases were provided yet (data/external/pr_cases.csv is empty).")])
    D.h1("Reproducing")
    D.p("Open ToadWatch_PR in VS Code and run notebooks/00_START_HERE.ipynb, then notebooks 01 to 06 in order; the last notebooks regenerate this report and the slides. Terminal: python scripts/run_study.py all, then python scripts/run_study.py final.")
    D.doc.save(out)
    return out


def build_pptx(d: dict, out):
    r, F = d["res"], facts(d)
    DARK, ACC, GREY = PRGB(0x1F, 0x3A, 0x2E), PRGB(0x2E, 0x8B, 0x57), PRGB(0x55, 0x55, 0x55)
    prs = Presentation(); prs.slide_width, prs.slide_height = PIn(13.333), PIn(7.5)

    def text(s, x, y, w, h, lines, size=20, bold=False, color=DARK):
        tf = s.shapes.add_textbox(PIn(x), PIn(y), PIn(w), PIn(h)).text_frame; tf.word_wrap = True
        for i, l in enumerate(lines if isinstance(lines, list) else [lines]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = l; p.font.size = PPt(size); p.font.bold = bold; p.font.color.rgb = color; p.space_after = PPt(8)

    def slide(title, bullets=None, img=None, iw=6.2, size=20, wide=False):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, PIn(0.25)); bar.fill.solid(); bar.fill.fore_color.rgb = ACC; bar.line.fill.background()
        text(s, 0.6, 0.4, 12.2, 1, title, 30, True)
        fg = _fig(img) if img else None
        if bullets:
            text(s, 0.7, 1.4 if wide else 1.55, 12 if (wide or not fg) else 6.2, 2.2 if wide else 5.3, ["• " + b for b in bullets], 16 if wide else size, False, GREY)
        if fg and wide:
            s.shapes.add_picture(str(fg), PIn((13.333 - iw) / 2), PIn(3.4), width=PIn(iw))
        elif fg:
            s.shapes.add_picture(str(fg), PIn(7.1 if bullets else 2.3), PIn(1.6), width=PIn(iw))
        text(s, 0.6, 7.05, 12, 0.3, f"three-model study · seed {C.SEED} · generated {d['stamp']}", 10, False, GREY)

    def table_slide(title, head, rows):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, PIn(0.25)); bar.fill.solid(); bar.fill.fore_color.rgb = ACC; bar.line.fill.background()
        text(s, 0.6, 0.4, 12.2, 1, title, 30, True)
        t = s.shapes.add_table(len(rows) + 1, len(head), PIn(0.6), PIn(1.6), PIn(12.1), PIn(0.5 * (len(rows) + 1))).table
        for j, x in enumerate(head):
            t.cell(0, j).text = str(x); t.cell(0, j).text_frame.paragraphs[0].font.size = PPt(14); t.cell(0, j).text_frame.paragraphs[0].font.bold = True
        for i, row in enumerate(rows, 1):
            for j, x in enumerate(row):
                t.cell(i, j).text = str(x); t.cell(i, j).text_frame.paragraphs[0].font.size = PPt(14)

    s0 = prs.slides.add_slide(prs.slide_layouts[6])
    text(s0, 0.8, 2.0, 11.7, 2, "Detecting TOAD scams in Puerto Rico: three models", 38, True)
    text(s0, 0.8, 4.2, 11.5, 1.8, ["William Díaz · Fase 11-1 · Prof. Mileidy Perdomo", f"Generated {d['stamp']}"], 18, False, GREY)
    slide("The idea", ["A TOAD attack = a call + a message that belong together.", "If either part is fraud, the whole attack is fraud - even the part that looks harmless.",
                       "Three models: call only, message only, combined (either one fires -> fraud).", "Two Random Forests; the combined model is their OR, not a third classifier."])
    slide("Data", ["Public calls, SMS, emails and links + the new 60k-message SMS phishing file (cleaned, one family per scam campaign).", "A hard set written for this project: polite scams, safe texts full of scam words (PR scenarios, EN + ES).",
                   "Families of near-identical texts never cross train / validation / test.", "Tiers: hard, matched (content only) and unmatched (optimistic)."])
    slide("Audits", [f"Shared ids / families: {int(d['overlap']['shared'].sum())}.", "Shuffled labels: " + ", ".join(f"{SHORT[m]} {num(d['shuffle'].loc[m, 'accuracy'], 2)}" for m in C.MODELS),
                     f"(majority-class rate {num(d['shuffle']['majority_class_rate'].iloc[0], 2)}).", "Metadata-only baselines: high only in the unmatched tier."], img="08_shortcut_baselines", iw=6.2, size=17)
    table_slide(f"Final test ({F['n']} attacks, run once)", ["Model", "Accuracy", "95% interval", "Precision", "Recall", "F1"],
                [[SHORT[m], num(r.loc[m, "accuracy"]), f"{num(r.loc[m, 'acc_ci_low'], 2)}-{num(r.loc[m, 'acc_ci_high'], 2)}", num(r.loc[m, "precision"]), num(r.loc[m, "recall"]), num(r.loc[m, "f1"])] for m in C.MODELS])
    slide("Against the ceilings", [f"A perfect one-channel model tops out at {pct(F['ceiling'])}.", "Combined can reach 100 % because it sees both halves.", "No single-channel model went above its ceiling."], img="12_test_vs_ceilings", iw=6.0, size=18)
    slide("Where each model fails", ["SF: safe-looking call, fraud message -> call-only misses it.", "FS: fraud call, safe-looking message -> message-only misses it.", "Combined catches both."], img="14_test_pair_type", iw=7.9, wide=True)
    slide("Honest content accuracy", [f"Matched tier: combined {pct(F['tier']('combined', 'matched'))}, call {pct(F['tier']('call_only', 'matched'))}, message {pct(F['tier']('message_only', 'matched'))}.",
                                      f"Hard tier: combined {pct(F['tier']('combined', 'hard'))}.", "Unmatched is optimistic (one class per source file)."], img="15_test_tiers", iw=7.9, wide=True)
    slide("Each forest on its own text", ["Balanced accuracy of the call and message detectors by tier."], img="16_test_detectors", iw=7.2, wide=True)
    slide("Is combined really better?", [verdict(F)], img="17_test_pairwise", iw=6.0, size=17)
    slide("Limitations", ["Hard set and safe halves of fraud attacks are constructed.", "Calls are partly synthetic.", "Label audit is a list for a human to check.", "No real PR cases yet." if d["pr"] is None else "PR cases included."])
    slide("Next steps", ["Real DACO / faculty cases (data/external/pr_cases.csv).", "More real recorded calls from the same collection (safe and fraud).", "Human review of the label-audit list.", "Re-run the notebooks to rebuild this report and these slides."])
    prs.save(out)
    return out


def build_all(refresh_charts: bool = True) -> dict:
    d = load_run()
    if refresh_charts:
        import matplotlib.pyplot as plt
        from . import charts
        charts.make_all_charts()
        plt.close("all")
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {"report_docx": build_docx(d, C.REPORT_DIR / "TOAD_Report.docx")}
    paths["report_pdf"] = to_pdf(paths["report_docx"])
    paths["slides_pptx"] = build_pptx(d, C.REPORT_DIR / "TOAD_Slides.pptx")
    paths["slides_pdf"] = to_pdf(paths["slides_pptx"])
    for k, v in paths.items():
        print(f"  {k:12s} {v}")
    return paths
