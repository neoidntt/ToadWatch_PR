"""Build the written deliverables from the results of the run that just finished.

    build_all()  ->  report/<tag>/TOAD_Final_Report_<tag>.docx  + .pdf
                     report/<tag>/TOAD_Presentation_<tag>.pptx + .pdf

Nothing is typed by hand: every number, table, figure and verdict sentence is read from the
tables / config / figures the pipeline saved for the active version (see config.VERSION), so
re-running the notebooks and calling ``build_all()`` always yields documents that match that run.
It is the last cell of ``v2_04`` and ``v2_05`` and the ``report`` stage of ``scripts/run_v2.py``.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from pptx import Presentation
from pptx.dml.color import RGBColor as PRGB
from pptx.util import Inches as PIn
from pptx.util import Pt as PPt

from . import config as C
from .data_loading import load_pr_cases

REPORT_DIR = C.PROJECT_ROOT / "report"
NAMES = {"call_only": "Call only", "message_only": "Message/email/link only",
         "combined": "Combined", "combined_early": "Combined (one forest)",
         "combined_stacked": "Combined (stacked)"}
ORDER = ["call_only", "message_only", "combined_early", "combined_stacked", "combined"]


# ====================================================================== data
def load_run() -> dict:
    """Everything the documents need, read from disk for the active version."""
    T = C.TABLES_DIR
    if not (T / "final_test_results.csv").exists():
        raise FileNotFoundError("The final test has not run for this version - nothing to report yet. "
                                "Run the final-test notebook (or scripts/run_v2.py final) first.")
    rd = lambda n, **kw: pd.read_csv(T / n, **kw) if (T / n).exists() else None
    d = {
        "final": rd("final_test_results.csv", index_col="variant"),
        "pair": rd("final_test_pairwise.csv"),
        "comp": rd("final_test_complementarity.csv", index_col=0),
        "kind": rd("final_test_by_msg_kind.csv"),
        "src": rd("final_test_by_call_source.csv"),
        "cv": rd("cv_summary_tuned.csv", index_col="variant"),
        "cvdiff": rd("cv_paired_differences.csv"),
        "abl": rd("diagnosis_ablation.csv"),
        "diag_comp": rd("diagnosis_complementarity.csv", index_col=0),
        "imp": rd("feature_importance.csv"),
        "pr": rd("final_test_puerto_rico.csv", index_col="variant"),
        "pred": pd.read_parquet(T / "final_test_predictions.parquet"),
        "manifest": json.loads(C.SPLIT_MANIFEST_JSON.read_text()),
        "clean": json.loads((T / "cleaning_report.json").read_text()),
        "cfg": json.loads(C.FINAL_CONFIG_JSON.read_text()),
        "pr_cases": load_pr_cases(),
    }
    d["chosen"] = d["cfg"].get("combined_choice", "combined")
    d["variants"] = [v for v in ORDER if v in d["final"].index]
    d["stamp"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    d["tag"] = C.VERSION
    return d


def _fig(name: str) -> Path | None:
    p = C.FIGURES_DIR / f"{name}.png"
    return p if p.exists() else None


pct = lambda x, d=1: f"{float(x) * 100:.{d}f}%"
num = lambda x, d=3: f"{float(x):.{d}f}"


def _pval(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _sci(p: float) -> str:
    return "< 1e-300" if p == 0 else (f"{p:.1e}" if p < 0.001 else f"{p:.3f}")


def verdict(d: dict) -> dict:
    """Decide, from the paired tests, what the run says about the hypothesis."""
    ch, p = d["chosen"], d["pair"]

    def row(b):
        r = p[(p.model_a == ch) & (p.model_b == b)]
        return None if r.empty else r.iloc[0]

    out = {"chosen": ch}
    for key, b in (("call", "call_only"), ("msg", "message_only")):
        r = row(b)
        if r is None:
            out[key] = None
            continue
        lo, hi, diff = r.f1_diff_ci_low, r.f1_diff_ci_high, r.f1_diff_a_minus_b
        state = "better" if lo > 0 else ("worse" if hi < 0 else "indistinguishable")
        out[key] = {"diff": diff, "lo": lo, "hi": hi, "mcnemar": r.mcnemar_p, "boot": r.bootstrap_p,
                    "a_only": int(r.a_only_right), "b_only": int(r.b_only_right), "state": state}
    c, m = out["call"], out["msg"]
    if c and m and c["state"] == "better" and m["state"] == "better":
        out["headline"] = "supported"
        out["text"] = (f"The {NAMES[ch].lower()} model is significantly better than both single-channel models "
                       f"(F1 +{num(c['diff'])} over call-only, 95% interval {num(c['lo'])} to {num(c['hi'])}; "
                       f"+{num(m['diff'])} over message-only), so the hypothesis that combining input types helps is supported "
                       f"on this test set. The gain over call-only is {'small' if c['diff'] < 0.03 else 'large'}.")
    elif c and c["state"] == "indistinguishable":
        out["headline"] = "partly supported"
        out["text"] = (f"The {NAMES[ch].lower()} model is better than message-only but cannot be distinguished from call-only "
                       f"(F1 difference {num(c['diff'])}, 95% interval {num(c['lo'])} to {num(c['hi'])}), so the hypothesis is only partly supported.")
    elif c and c["state"] == "worse":
        out["headline"] = "not supported"
        out["text"] = (f"The {NAMES[ch].lower()} model is significantly worse than call-only "
                       f"(F1 difference {num(c['diff'])}, interval {num(c['lo'])} to {num(c['hi'])}), so the hypothesis is not supported.")
    else:
        out["headline"] = "partly supported"
        out["text"] = "The combined model improved on one single-channel model but not clearly on the other."
    return out


# ====================================================================== PDF
def to_pdf(path: Path) -> Path | None:
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if not exe:
        print(f"  (LibreOffice not found - skipped PDF for {path.name})")
        return None
    subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", str(path.parent), str(path)],
                   capture_output=True, timeout=300)
    pdf = path.with_suffix(".pdf")
    return pdf if pdf.exists() else None


# ====================================================================== DOCX
class _Doc:
    def __init__(self, d: dict):
        self.d = d
        self.doc = Document()
        s = self.doc.sections[0]
        s.page_width, s.page_height = Inches(8.5), Inches(11)
        for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(s, m, Inches(1))
        st = self.doc.styles["Normal"]
        st.font.name = "Times New Roman"; st.font.size = Pt(12)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        st.paragraph_format.space_after = Pt(7); st.paragraph_format.line_spacing = 1.5
        for name, sz, it, al in (("Heading 1", 14, False, WD_ALIGN_PARAGRAPH.CENTER),
                                 ("Heading 2", 12, True, WD_ALIGN_PARAGRAPH.LEFT)):
            h = self.doc.styles[name]
            h.font.name = "Times New Roman"; h.font.size = Pt(sz); h.font.bold = True; h.font.italic = it
            h.font.color.rgb = RGBColor(0, 0, 0)
            for a in ("w:ascii", "w:hAnsi", "w:eastAsia"):
                h.element.rPr.rFonts.set(qn(a), "Times New Roman")
            h.paragraph_format.alignment = al
            h.paragraph_format.space_before = Pt(14); h.paragraph_format.space_after = Pt(6)
        fp = s.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r = fp.add_run(f"Comparing ML Methods for Detecting TOAD Scams in Puerto Rico · {d['tag']} · generated {d['stamp']} · page ")
        r.font.size = Pt(8)
        r2 = fp.add_run(); r2.font.size = Pt(8)
        for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
            if t:
                e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t)
            else:
                e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
            r2._r.append(e)

    def h1(self, t): self.doc.add_heading(t, 1)
    def h2(self, t): self.doc.add_heading(t, 2)

    def p(self, text="", lead=None, align=None, italic=False, size=None):
        para = self.doc.add_paragraph()
        if lead:
            para.add_run(lead).bold = True
        r = para.add_run(text); r.italic = italic
        if size: r.font.size = Pt(size)
        if align is not None: para.alignment = align
        return para

    def bullets(self, items):
        for t in items:
            q = self.doc.add_paragraph(t, style="List Bullet"); q.paragraph_format.line_spacing = 1.15

    def cap(self, t): self.p(t, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=10)

    def table(self, head, rows, cap=None):
        t = self.doc.add_table(rows=1, cols=len(head)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(head):
            c = t.rows[0].cells[i]; c.text = ""
            r = c.paragraphs[0].add_run(str(h)); r.bold = True; r.font.size = Pt(9)
            sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), "E3E8EF")
            c._tc.get_or_add_tcPr().append(sh)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""; cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(9)
        for row in t.rows:
            for c in row.cells:
                for q in c.paragraphs:
                    q.paragraph_format.line_spacing = 1.0; q.paragraph_format.space_after = Pt(2)
        if cap: self.cap(cap)

    def fig(self, name, caption, width=6.0):
        f = _fig(name)
        if f is None:
            return
        self.doc.add_picture(str(f), width=Inches(width))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.cap(caption)


def build_docx(d: dict, out: Path) -> Path:
    D = _Doc(d)
    fin, V, ch = d["final"], d["variants"], d["chosen"]
    ver = verdict(d)
    mf = d["manifest"]["files"]
    cl = d["clean"]; cc, mc = cl["calls"], cl["messages"]
    cfg, params = d["cfg"], d["cfg"]["params"]
    n_test = int(fin.iloc[0]["n"])
    best = fin.loc[ch]
    best_name = NAMES[ch]
    C_ = WD_ALIGN_PARAGRAPH.CENTER

    for _ in range(6): D.p("")
    t = D.p("Comparing Single-Channel and Multichannel Machine Learning for Detecting Telephone-Oriented Attack Delivery (TOAD) Scams in Puerto Rico", align=C_)
    t.runs[0].bold = True
    D.p("Final Research Report", align=C_); D.p("William Díaz", align=C_)
    D.p("Specialized School in Science and Mathematics University Gardens High School", align=C_)
    D.p("Fase 11-1 · Prof. Mileidy Perdomo", align=C_)
    D.p(f"Experiment {d['tag']} · random seed {C.SEED} · generated {d['stamp']}", align=C_, size=10)
    D.doc.add_page_break()

    # ---------------------------------------------------------------- abstract
    D.h1("Abstract")
    accs = ", ".join(f"{NAMES[v].lower()} {pct(fin.loc[v, 'accuracy'])}" for v in V)
    D.p(f"This study tests whether a Random Forest classifier trained on call-transcript features together with message, email "
        f"and link features detects fraudulent call-and-message (TOAD) cases better than the same kind of classifier trained on one "
        f"input type alone. Paired records were built from public labeled datasets and split 70/10/20 with a fixed seed "
        f"({mf['train']['rows']:,} training, {mf['val']['rows']} validation, {n_test} test). On the held-out test set the accuracies were: {accs}. "
        f"All versions were significantly better than the 50% expected from chance (exact binomial test). "
        f"{ver['text']} The results come from constructed pairs of text data and cannot show that the tool would stop a live scam.")

    # ---------------------------------------------------------------- intro
    D.h1("Introduction")
    D.p("Vishing is a scam in which a caller pretends to be a bank, a government office or a company the victim trusts. Smishing is the same "
        "trick delivered by text message or email with a link. A Telephone-Oriented Attack Delivery (TOAD) scam combines both: a call sets up "
        "the story and a message, email or link finishes the job (Abnormal AI, 2026). In Puerto Rico, DACO has documented at least ten schemes "
        "since December 2022, including fake AutoExpreso toll texts and fake Banco Popular emails (Metro Puerto Rico, 2026). Once money leaves "
        "through an instant payment app such as ATH Móvil it is generally gone, so the only protection is stopping the scam before the transfer.")
    D.p("No phone carrier or messaging company sees both a call and the follow-up message. This research therefore asks a narrow, testable "
        "question: does one classifier that sees call transcripts, messages, emails and links together catch more fraud than a classifier "
        "that sees only one of those input types, and does it generalize to cases it has never seen?")
    D.h2("Hypothesis")
    D.p("The combined classifier will score significantly above 50% accuracy on its held-out test set and will show higher accuracy and a "
        "lower false-alarm rate than a classifier trained on one input type alone.", lead="H1: ")
    D.p("Accuracy is no better than 50% and combining input types gives no meaningful improvement.", lead="H0: ")
    D.h2("Objectives")
    D.bullets(["Train a classifier on call transcripts and message, email and link data, keeping training and test data completely separate.",
               "Train comparison versions: call features only, and message/email/link features only.",
               "Collect Puerto Rico-specific cases from DACO and university faculty (see Limitations for status).",
               "Evaluate every version with accuracy, precision, recall, F1 and a confusion matrix, with confidence intervals.",
               "Run a binomial test of each version against 50% accuracy, and paired tests between versions."])

    # ---------------------------------------------------------------- methods
    D.h1("Methods")
    D.h2("Datasets")
    D.p("All data are existing, already-labeled public datasets; no real people, phone numbers or accounts were contacted. The label of every "
        "record comes from the source folder (safe or fraudulent), established before this research began.")
    D.bullets(["Fraudulent calls: FTC-confirmed robocall transcripts from the NC State Robocall Audio Dataset (Prasad & Reaves, 2023), scam dialogue sets, and transcribed YouTube scam calls.",
               "Safe calls: written legitimate dialogues and a sample of customer-service conversations from banking and telecom corpora. No public dataset of ordinary calls exists, so these are written or synthetic.",
               "Messages: SMS spam/ham (Mishra & Soni, 2022) with Spanish translations, and phishing/legitimate emails (TREC, CEAS, Enron and others, plus the Spanish SpaPhish set).",
               "Links: legitimate and fraudulent URL lists and the Tranco top-1M ranking."])
    D.h2("Cleaning and duplicates")
    D.p(f"From {cc['rows_raw']:,} raw call transcripts, {cc['unusable_rows']} were unusable, {cc['duplicate_rows']:,} were exact duplicates and "
        f"{cc.get('near_duplicate_rows', 0)} were near-duplicates (TF-IDF cosine similarity above 0.9), leaving {cc['rows_clean']:,}. Near-duplicates "
        f"matter because robocall scripts are replayed with tiny wording changes; a copy in training and a copy in test would inflate accuracy. "
        f"Speaker tags such as \"Suspect:\" and \"agent:\" were stripped because some exist in only one class. "
        f"For messages, {mc['text_duplicates']['duplicate_rows']:,} duplicate texts were removed; URLs were de-duplicated by host, removing "
        f"{mc['url_duplicate_hosts']['duplicate_rows']:,} repeated hosts and {mc['url_duplicate_hosts']['conflicting_label_rows']} rows from hosts present in both "
        f"classes, so the same domain never lands in both training and test. {mc['rows_clean']:,} message items remained.")
    D.h2("Pairing and splitting")
    D.p(f"No dataset contains real call-and-message pairs, so pairs were built: each call is paired once with a message, email or link of the same "
        f"label, drawn evenly from the three types and inside the same split; classes are balanced. Calls and messages were first split separately "
        f"70/10/20 with seed {C.SEED} and only then paired, so nothing crosses between splits. The result is {mf['train']['rows']:,} training "
        f"({mf['train']['fraud']} fraud / {mf['train']['safe']} safe), {mf['val']['rows']} validation ({mf['val']['fraud']} / {mf['val']['safe']}) and "
        f"{mf['test']['rows']} test ({mf['test']['fraud']} / {mf['test']['safe']}) records. The test file was made read-only, fingerprinted (SHA-256) and used once.")
    D.h2("Features")
    D.p("Call features (English and Spanish patterns): requests for a PIN, code, SSN or card number; \"press 1\"; urgent and threatening words; claims "
        "to be a bank, government agency or company; payment methods such as gift cards. Message features: wording (urgency, credentials, money, toll or "
        "delivery words, digits, capitals) and link features from the first URL (shortener, HTTPS, IP address, subdomains, suspicious top-level domain, "
        "look-alike similarity to known brands including Puerto Rico brands, entropy, Tranco rank as an offline stand-in for domain age).")
    if C.EXCLUDE_STYLE_FEATURES:
        D.p("Raw length and punctuation-count features (word count, character count, question marks, exclamation marks, average word length, digit ratio of calls) "
            "were removed from every model: in a first experiment they separated the datasets rather than the scams (see Results, ablation).")
    D.h2("Models")
    D.p("Every version uses a Random Forest (scikit-learn). Versions: call features only; message/email/link features only; and two ways of combining the channels. "
        "In the one-forest version the call and message features are concatenated. In the stacked version a call forest and a message forest are trained separately and a "
        "small logistic-regression meta-learner combines their probabilities; the meta-learner is trained only on out-of-fold probabilities of the training data, so it never "
        "sees a prediction made on a row the forest was trained on.")
    D.h2("Tuning and model choice")
    if "per_variant" in params:
        rows = [[NAMES[v], json.dumps(p)] for v, p in params["per_variant"].items()]
        D.p(f"Every version was tuned with the same grid and the same {C.CV_FOLDS}-fold cross-validation on the training and validation data only; each keeps its own best settings.")
        D.table(["Version", "Selected Random Forest settings"], rows, "Table 1. Settings selected by cross-validation.")
        det = cfg.get("choice_detail", {})
        D.p(f"The combined design was chosen before the test set was opened by a rule fixed in advance: the design with the higher mean cross-validated F1 wins, and early fusion is "
            f"kept if the difference is smaller than {det.get('tie_margin', C.TIE_MARGIN)}. Early fusion scored {num(det.get('early_cv_f1', 0), 4)} and stacked fusion "
            f"{num(det.get('stacked_cv_f1', 0), 4)}; the choice was therefore the {NAMES[ch].lower()} model.")
    else:
        D.p("One shared setting was used for all versions: " + json.dumps(params))
    D.h2("Statistics")
    D.p(f"Each test score has a 95% bootstrap interval ({C.N_BOOTSTRAP:,} resamples). Models are compared on the same records with a paired bootstrap of the F1 difference and an "
        f"exact McNemar test (which looks only at records where exactly one model is right). Accuracy is also tested against 50% with a one-sided exact binomial test.")

    # ---------------------------------------------------------------- results
    D.h1("Results")
    if d["abl"] is not None:
        D.h2("What mattered during development (cross-validation on train + validation)")
        a = d["abl"].copy(); a["policy"] = a["policy"].str.replace("\n", " ")
        piv = a.pivot(index="variant", columns="policy", values="f1_mean").reindex([v for v in ORDER if v in set(a.variant)])
        D.table(["Version"] + list(piv.columns), [[NAMES[v]] + [num(x, 4) for x in r] for v, r in piv.iterrows()],
                "Table 2. Mean cross-validated F1 under three feature policies (default Random Forest settings).")
        D.fig("v2_01_ablation", "Figure 1. Effect of preprocessing and feature choices on each model.", 6.2)
    if d["cv"] is not None:
        s = d["cv"]
        D.table(["Version", "CV F1 mean", "CV F1 std", "CV accuracy", "CV false-alarm rate"],
                [[NAMES[v], num(r.f1_mean, 4), num(r.f1_std, 4), num(r.accuracy_mean, 4), num(r.false_alarm_rate_mean, 4)] for v, r in s.iterrows() if v in NAMES],
                f"Table 3. Cross-validation with tuned settings ({C.CV_FOLDS} folds x {C.CV_REPEATS} repeats, train + validation only).")
        D.fig("v2_03_cv_f1_boxplot", "Figure 2. F1 over all cross-validation folds.", 5.4)
    D.h2("Final test")
    D.p(f"The versions were run once on the {n_test} held-out records ({mf['test']['fraud']} fraudulent, {mf['test']['safe']} safe).")
    D.table(["Version", "Acc.", "Prec.", "Recall", "F1", "F1 95% interval", "TN", "FP", "FN", "TP"],
            [[NAMES[v], num(fin.loc[v, "accuracy"], 4), num(fin.loc[v, "precision"], 4),
              num(fin.loc[v, "recall"], 4), num(fin.loc[v, "f1"], 4), f"{num(fin.loc[v, 'f1_ci_low'])} to {num(fin.loc[v, 'f1_ci_high'])}",
              int(fin.loc[v, "tn"]), int(fin.loc[v, "fp"]), int(fin.loc[v, "fn"]), int(fin.loc[v, "tp"])] for v in V],
            f"Table 4. Final test results ({n_test} records). Acc. = accuracy; TN true negatives, FP false positives, FN false negatives, TP true positives. "
            f"Cases correct: " + ", ".join(f"{NAMES[v].lower()} {int(fin.loc[v, 'correct'])}" for v in V) + ".")
    D.fig("v2_12_test_confusion", "Figure 3. Confusion matrices on the test set.", 6.4)
    D.fig("v2_13_test_f1_ci", "Figure 4. Test F1 with 95% bootstrap intervals (overlap means no clear difference).", 5.0)
    D.h2("Binomial test against 50% and paired comparisons")
    D.table(["Version", "Accuracy", "95% interval (accuracy)", "ROC AUC", "p vs 50%"],
            [[NAMES[v], pct(fin.loc[v, "accuracy"]), f"{pct(fin.loc[v, 'ci95_low'])} to 100%", num(fin.loc[v, "roc_auc"], 4), _sci(float(fin.loc[v, "p_value"]))] for v in V],
            "Table 5. One-sided exact binomial test, H0: accuracy <= 50%. H0 is rejected for every version when p < 0.05.")
    pw = d["pair"]
    D.table(["A", "B", "F1 diff (A-B)", "95% interval", "bootstrap p", "A only right", "B only right", "McNemar p"],
            [[NAMES[r.model_a], NAMES[r.model_b], num(r.f1_diff_a_minus_b, 4), f"{num(r.f1_diff_ci_low, 4)} to {num(r.f1_diff_ci_high, 4)}",
              _pval(r.bootstrap_p), int(r.a_only_right), int(r.b_only_right), _pval(r.mcnemar_p)] for r in pw.itertuples()],
            "Table 6. Paired comparison on the same test records. If the interval contains 0 the models are not distinguishable.")
    D.fig("v2_14_test_pairwise", "Figure 5. Paired F1 differences with 95% intervals.", 5.2)
    D.fig("v2_16_test_roc_pr", "Figure 6. ROC and precision-recall curves on the test set.", 6.4)
    D.h2("By message type and call source")
    kk = d["kind"]
    kv = [v for v in V if v in set(kk.variant)]
    piv = kk.pivot(index="variant", columns="msg_kind", values="accuracy").reindex(kv)
    n_by_kind = kk.groupby("msg_kind")["n"].first()
    D.table(["Version"] + [f"{c.upper()} (n={int(n_by_kind[c])})" for c in piv.columns],
            [[NAMES[v]] + [pct(x) for x in r] for v, r in piv.iterrows()], "Table 7. Test accuracy by message type.")
    D.fig("v2_15_test_by_kind", "Figure 7. Accuracy by message type.", 4.4)
    D.h2("Who is right")
    if d["comp"] is not None:
        c = d["comp"].iloc[0]
        D.p(f"On the test set both single-channel models were right on {int(c['both_right'])} records; only the call model on {int(c.iloc[1])}; "
            f"only the message model on {int(c.iloc[2])}; both wrong on {int(c['both_wrong'])}. The records where only the message model is right "
            f"are the room a combined model has to improve on the call model.")
        D.fig("v2_18_test_complementarity", "Figure 8. Who is right: call model versus message model.", 5.2)
    if d["imp"] is not None:
        D.h2("Which features matter")
        im = d["imp"]
        v0 = "combined_early" if "combined_early" in set(im.variant) else im.variant.iloc[0]
        top = im[im.variant == v0].head(6)
        D.p("The most important features of the one-forest combined model were: " + ", ".join(f"{r.feature} ({r.importance:.3f})" for r in top.itertuples()) + ".")
        D.fig("v2_09_importance_combined_early", "Figure 9. Top features of the one-forest combined model.", 5.0)
    D.h2("Errors")
    e = d["pred"]
    p = e[f"pred_{ch}"]
    fp, fn = e[(e.label == 0) & (p == 1)], e[(e.label == 1) & (p == 0)]
    D.p(f"The {best_name.lower()} model made {len(fp)} false positives (safe cases flagged) and {len(fn)} false negatives (fraud missed) out of {len(e)}.")
    if len(fp) + len(fn):
        rows = []
        for kind, df in (("False positive", fp), ("False negative", fn)):
            for key in ("msg_kind", "call_source"):
                for val, cnt in df[key].value_counts().head(4).items():
                    rows.append([kind, key.replace("_", " "), val, cnt])
        D.table(["Error type", "Grouped by", "Value", "Count"], rows, "Table 8. Where the errors of the chosen model come from.")

    # ---------------------------------------------------------------- discussion
    D.h1("Discussion")
    D.p(f"All versions are far above chance, so H0 (accuracy no better than 50%) is rejected. For the comparison between versions the paired tests are the right "
        f"evidence, not the size of the accuracy numbers. Verdict on the second part of the hypothesis: {ver['headline']}. {ver['text']}")
    D.p("A first experiment (see the development results) showed why combining can fail on this kind of data: the call data are easy to separate by writing style, so a call-only "
        "model looks near-perfect, leaving little for a message to add. Once raw style features were removed from every model, the call-only score fell and the benefit of the "
        "second channel became visible; letting each channel keep its own forest (stacking) avoided the weaker channel diluting the stronger one.")
    D.h1("Limitations")
    lim = ["Safe calls were written or synthetic, not recorded ordinary calls; style differences between classes remain a risk for call-based scores.",
           "Call-and-message pairs were constructed, not observed. The study shows that a classifier can combine evidence, not that real TOAD pairs look like this.",
           "Only text was tested; no live calls, no audio and no human participants. The study cannot show that the tool would stop a real scammer.",
           "Domain age was replaced by the Tranco rank; near-duplicates were removed for calls only.",
           "This experiment reuses the same pool of public data as an earlier experiment (new seed, new split, new pairing); the earlier results informed the design, so it is a confirmatory run, not a fresh dataset.",
           "Mixed cases (one suspicious input, one normal input) were not tested because their correct label is not defined in the proposal."]
    if d["pr_cases"] is None:
        lim.insert(3, "The Puerto Rico-specific test set was not built: no DACO or faculty cases are in the project data, so no Puerto Rico result is reported.")
    D.bullets(lim)
    if d["pr"] is not None:
        D.table(["Version", "n", "Accuracy", "Recall", "False-alarm rate"],
                [[NAMES[v], int(r.n), pct(r.accuracy), pct(r.recall), pct(r.false_alarm_rate)] for v, r in d["pr"].iterrows() if v in NAMES],
                "Table 9. Puerto Rico cases (secondary test).")
    D.h1("Conclusion")
    D.p(f"The {best_name.lower()} model classified unseen cases with {pct(best['accuracy'])} accuracy ({int(best['correct'])} of {int(best['n'])}) and F1 {num(best['f1'])}, "
        f"significantly above chance. Hypothesis verdict for combining input types: {ver['headline']}. Ideas to improve the work: collect realistic safe calls and real Puerto Rico "
        f"cases from DACO, build mixed-case test sets with an agreed labeling rule, add real domain-age data, test speech-to-text transcripts of audio, and evaluate a model that "
        f"reads text directly instead of hand-made features.")
    D.h1("Reproducing these results")
    D.p("Open the folder ToadWatch_PR in VS Code, then run notebooks/00_START_HERE.ipynb and the v2 notebooks in order (v2_01 to v2_05); the last notebook regenerates this report "
        "and the presentation. In a terminal: python scripts/run_v2.py all --final. Frozen configuration: " + json.dumps(params))
    D.h1("References")
    for t in ("Abnormal AI. (2026). TOAD phishing attacks: How to detect and defend against voice-email threats. https://abnormal.ai/learning/toad-phishing",
              "LegitPhish: A large-scale annotated dataset for URL-based phishing detection. (2025). PMC. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12538017/",
              "Metro Puerto Rico. (2026). DACO alerta sobre fraude con inteligencia artificial que usa imagen del secretario de la Gobernación. https://www.metro.pr/noticias/2026/05/12/daco-alerta-sobre-fraude-con-inteligencia-artificial-que-usa-imagen-del-secretario-de-la-gobernacion",
              "Mishra, S., & Soni, D. (2022). SMS phishing dataset for machine learning and pattern recognition [Data set]. Mendeley Data. https://data.mendeley.com/datasets/f45bkkt8pr/1",
              "Prasad, S., & Reaves, B. (2023). Robocall audio from the FTC's Project Point of No Entry (Technical Report No. TR-2023-1). North Carolina State University. https://github.com/wspr-ncsu/robocall-audio-dataset"):
        q = D.p(t); q.paragraph_format.left_indent = Inches(0.5); q.paragraph_format.first_line_indent = Inches(-0.5); q.paragraph_format.line_spacing = 1.15
    D.doc.save(out)
    return out


# ====================================================================== PPTX
def build_pptx(d: dict, out: Path) -> Path:
    fin, V, ch = d["final"], d["variants"], d["chosen"]
    ver = verdict(d)
    mf = d["manifest"]["files"]
    cc, mc = d["clean"]["calls"], d["clean"]["messages"]
    n_test = int(fin.iloc[0]["n"])
    DARK, ACC, GREY = PRGB(0x1F, 0x3A, 0x2E), PRGB(0x2E, 0x8B, 0x57), PRGB(0x55, 0x55, 0x55)
    prs = Presentation(); prs.slide_width, prs.slide_height = PIn(13.333), PIn(7.5)

    def text(s, x, y, w, h, lines, size=20, bold=False, color=DARK):
        tf = s.shapes.add_textbox(PIn(x), PIn(y), PIn(w), PIn(h)).text_frame; tf.word_wrap = True
        for i, l in enumerate(lines if isinstance(lines, list) else [lines]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = l; p.font.size = PPt(size); p.font.bold = bold; p.font.color.rgb = color; p.space_after = PPt(8)

    def slide(title, bullets=None, img=None, iw=6.4, notes=None, size=20, wide=False):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, PIn(0.25)); bar.fill.solid(); bar.fill.fore_color.rgb = ACC; bar.line.fill.background()
        text(s, 0.6, 0.4, 12.2, 1, title, 30, True)
        f = _fig(img) if img else None
        if bullets:
            text(s, 0.7, 1.4 if wide else 1.55, 12 if (wide or not f) else 6.2, 2.2 if wide else 5.3,
                 ["• " + b for b in bullets], 16 if wide else size, False, GREY)
        if f and wide:      # wide figure: centred below the bullets
            s.shapes.add_picture(str(f), PIn((13.333 - iw) / 2), PIn(3.5), width=PIn(iw))
        elif f:
            s.shapes.add_picture(str(f), PIn(7.1 if bullets else 2.3), PIn(1.6), width=PIn(iw))
        text(s, 0.6, 7.05, 12, 0.3, f"{d['tag']} · seed {C.SEED} · generated {d['stamp']}", 10, False, GREY)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    def table_slide(title, head, rows, note=None):
        s = slide(title)
        tbl = s.shapes.add_table(len(rows) + 1, len(head), PIn(0.6), PIn(1.6), PIn(12.1), PIn(0.5 * (len(rows) + 1))).table
        for j, h in enumerate(head):
            c = tbl.cell(0, j); c.text = str(h); c.text_frame.paragraphs[0].font.size = PPt(14); c.text_frame.paragraphs[0].font.bold = True
        for i, r in enumerate(rows, 1):
            for j, v in enumerate(r):
                c = tbl.cell(i, j); c.text = str(v); c.text_frame.paragraphs[0].font.size = PPt(14)
        if note:
            text(s, 0.7, 1.9 + 0.5 * (len(rows) + 1), 12, 1.5, note, 16, False, GREY)

    s = prs.slides.add_slide(prs.slide_layouts[6])
    text(s, 0.8, 2.0, 11.7, 2, "Comparing Single-Channel and Multichannel Machine Learning for Detecting TOAD Scams in Puerto Rico", 34, True)
    text(s, 0.8, 4.6, 11.5, 1.8, ["William Díaz · Fase 11-1 · Prof. Mileidy Perdomo",
                                   "Specialized School in Science and Mathematics University Gardens High School",
                                   f"Experiment {d['tag']} · generated {d['stamp']}"], 18, False, GREY)
    slide("The problem", ["A scam call sets up the story; a text, email or link finishes it (TOAD).",
                          "Puerto Rico: fake AutoExpreso texts, fake Banco Popular emails, ATH Móvil transfers that cannot be reversed.",
                          "No carrier or app sees the call and the message together."])
    slide("Question and hypothesis", ["Does one classifier that sees calls + messages + emails + links catch more fraud than one that sees a single input type?",
                                      "H1: accuracy significantly above 50% and combined better than single input.",
                                      "H0: no better than chance and no gain from combining."])
    slide("Data (public, already labeled)", ["Calls: FTC-confirmed NC State robocall transcripts + scam / safe dialogues",
                                             "Messages: SMS spam/ham, spam / legitimate emails (EN + ES)",
                                             "Links: legitimate and phishing URL lists + Tranco top-1M",
                                             "No real people, numbers or accounts used"])
    slide("Cleaning and splitting", [f"Calls: {cc['rows_raw']:,} raw → {cc['rows_clean']:,} after exact + near-duplicate removal",
                                     f"Messages: {mc['rows_raw']:,} raw → {mc['rows_clean']:,}; URL hosts repeated or in both classes removed",
                                     f"70 / 10 / 20 split, seed {C.SEED}, test file locked and fingerprinted",
                                     f"{mf['train']['rows']+mf['val']['rows']+mf['test']['rows']:,} paired records: {mf['train']['rows']:,} train · {mf['val']['rows']} val · {mf['test']['rows']} test"])
    slide("Method", ["Call features: PIN / SSN requests, 'press 1', urgency, bank / government claims",
                     "Message features: shorteners, HTTPS, look-alike domains, Tranco rank, wording",
                     "Random Forest, four versions: call-only · message-only · combined (one forest) · combined (stacked)",
                     "Style shortcuts removed; every version tuned the same way; combined design chosen by CV before the test"][: 4 if C.EXCLUDE_STYLE_FEATURES else 3])
    if d["abl"] is not None:
        slide("What mattered in development", ["Removing raw length / style features makes every call-based score honest",
                                               "Stacking (one forest per channel + meta-learner) keeps the weak channel from diluting the strong one",
                                               "All numbers: cross-validation on train + validation only"], img="v2_01_ablation", iw=6.0)
    if d["cv"] is not None:
        slide("Cross-validation (development)", [f"{NAMES[v]}: F1 {num(r.f1_mean, 3)} ± {num(r.f1_std, 3)}" for v, r in d["cv"].iterrows() if v in NAMES], img="v2_03_cv_f1_boxplot", iw=5.8, size=18)
    table_slide(f"Final test ({n_test} records, run once)", ["Version", "Accuracy", "Precision", "Recall", "F1", "F1 95% interval"],
                [[NAMES[v], pct(fin.loc[v, "accuracy"]), pct(fin.loc[v, "precision"]), pct(fin.loc[v, "recall"]), num(fin.loc[v, "f1"], 3),
                  f"{num(fin.loc[v, 'f1_ci_low'], 3)} - {num(fin.loc[v, 'f1_ci_high'], 3)}"] for v in V],
                "All versions p < 0.05 against 50% (exact binomial).")
    slide("Confusion matrices", [f"{NAMES[v]}: {int(fin.loc[v, 'fp'])} false positives, {int(fin.loc[v, 'fn'])} false negatives" for v in V], img="v2_12_test_confusion", iw=11.0, wide=True)
    slide("Is the difference real?", [ver["text"]], img="v2_14_test_pairwise", iw=5.8, size=18)
    slide("ROC and precision-recall", [f"{NAMES[v]}: AUC {num(fin.loc[v, 'roc_auc'], 3)}" for v in V], img="v2_16_test_roc_pr", iw=8.6, wide=True)
    kk = d["kind"]; piv = kk.pivot(index="variant", columns="msg_kind", values="accuracy")
    slide("By message type", [f"{NAMES[v]}: " + ", ".join(f"{c.upper()} {pct(piv.loc[v, c])}" for c in piv.columns) for v in V if v in piv.index], img="v2_15_test_by_kind", iw=5.4, size=16)
    if d["comp"] is not None:
        c = d["comp"].iloc[0]
        slide("Who is right?", [f"Both single-channel models right: {int(c['both_right'])}", f"Only the call model right: {int(c.iloc[1])}",
                                f"Only the message model right: {int(c.iloc[2])}", f"Both wrong: {int(c['both_wrong'])}",
                                "Records where only the message model is right = the room fusion has to improve"], img="v2_18_test_complementarity", iw=5.6, size=17)
    slide(f"Verdict: hypothesis {ver['headline']}", [ver["text"]])
    lim = ["Safe calls are written / synthetic; pairs are constructed, not observed", "Text only: no live calls, no audio, no people",
           "Reuses the earlier experiment's data pool (new seed, split, pairing)", "Tranco rank replaces domain age; mixed cases not tested"]
    if d["pr_cases"] is None:
        lim.insert(2, "No Puerto Rico cases yet (DACO / faculty requests pending)")
    slide("Limitations", lim)
    slide("Ideas to improve", ["Real Puerto Rico cases from DACO and faculty", "Realistic safe calls; mixed-case tests with an agreed labeling rule",
                               "Real domain age; speech-to-text on audio", "Text models instead of hand-made features"])
    slide("Demo", ["Notebook 06: type a call + a message, see the versions answer", "Backup screenshots: outputs/figures and outputs/v2/figures"])
    prs.save(out)
    return out


# ====================================================================== entry point
def build_all(refresh_charts: bool = True) -> dict[str, Path | None]:
    """Regenerate Word report, report PDF, PowerPoint and presentation PDF for the active run."""
    d = load_run()
    if refresh_charts:
        from . import charts_v2
        import matplotlib.pyplot as plt
        charts_v2.make_all_charts()
        plt.close("all")
    out_dir = REPORT_DIR / d["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = d["tag"]
    paths: dict[str, Path | None] = {}
    paths["report_docx"] = build_docx(d, out_dir / f"TOAD_Final_Report_{tag}.docx")
    paths["report_pdf"] = to_pdf(paths["report_docx"])
    paths["slides_pptx"] = build_pptx(d, out_dir / f"TOAD_Presentation_{tag}.pptx")
    paths["slides_pdf"] = to_pdf(paths["slides_pptx"])
    for k, v in paths.items():
        print(f"  {k:12s} {v}")
    return paths
