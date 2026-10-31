"""Evidence for schedule step 3 (18-31 Oct): features + first models, PRELIMINARY results only.

    python scripts/make_step3_evidence.py

Re-computes everything from the project code once and writes the same document in two languages:
    evidence/Step3_Evidence_PRELIMINARY.docx / .pdf        (English)
    evidence/Step3_Evidencia_PRELIMINAR_ES.docx / .pdf     (Español)

Uses the TRAINING and VALIDATION sets only. The test set is never loaded, and nothing about
tuning or the final result is included.
"""
import datetime as dt
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402

from toadwatch import config as C  # noqa: E402
from toadwatch.features import build_features, columns_for  # noqa: E402
from toadwatch.features_call import call_features  # noqa: E402
from toadwatch.features_message import url_features  # noqa: E402
from toadwatch.modeling import feature_importance, score_all, train_all  # noqa: E402
from toadwatch.splitting import load_split  # noqa: E402

OUT_DIR = C.PROJECT_ROOT / "evidence"
FILE_STEM = {"en": "Step3_Evidence_PRELIMINARY", "es": "Step3_Evidencia_PRELIMINAR_ES"}
COLORS = {"call_only": "#5b7db1", "message_only": "#d08c3c", "combined": "#4c9f70"}

# feature group -> (features, English name, Spanish name, English text, Spanish text)
CALL_GROUPS = [
    (["pin_request", "ssn_request", "card_account_number", "verify_confirm", "secret_asks"],
     "Asks for secrets", "Pide datos secretos",
     "PIN / code / password, SSN, card or account number, 'verify / confirm' wording",
     "PIN / código / contraseña, seguro social (SSN), número de tarjeta o cuenta, palabras como 'verificar / confirmar'"),
    (["press_one", "urgent_words", "threat_legal", "stay_on_line", "dont_tell", "urgency_density"],
     "Pressure / urgency", "Presión / urgencia",
     "'press 1', urgent words (palabras urgentes), legal threats, 'do not hang up', secrecy",
     "'presione 1', palabras urgentes, amenazas legales, 'no cuelgue', pedir secreto"),
    (["claims_bank", "claims_government", "claims_company"],
     "Claims to be an institution", "Dice ser una institución",
     "says it is a bank (banco), a government office (gobierno) or a company",
     "dice ser un banco, una oficina del gobierno o una compañía"),
    (["payment_methods", "money_words", "account_problem", "remote_access"],
     "Money / payment / account problems", "Dinero / pagos / problemas de cuenta",
     "gift cards / wire / crypto, refund / prize words, 'problem with your account', remote access",
     "tarjetas de regalo / giros / cripto, palabras de reembolso o premio, 'problema con su cuenta', acceso remoto"),
    (["appointment_delivery", "polite_service"],
     "Normal-call vocabulary", "Vocabulario de llamadas normales",
     "appointment / delivery words, polite customer-service phrases",
     "palabras de citas y entregas, frases corteses de servicio al cliente"),
    (["n_words", "n_chars", "digit_ratio", "question_marks", "exclamations", "avg_word_len"],
     "Length / punctuation counts", "Largo / conteo de puntuación",
     "words, characters, digit share, '?' and '!' counts, average word length",
     "palabras, caracteres, proporción de dígitos, conteo de '?' y '!', largo promedio de palabra"),
]
MSG_GROUPS = [
    (["kw_urgent", "kw_credentials", "kw_money", "kw_click", "kw_toll_debt", "kw_delivery"],
     "Wording (text of the SMS / email)", "Redacción (texto del SMS / correo)",
     "urgent, credentials, money, 'click', toll / debt and delivery words (EN + ES)",
     "palabras de urgencia, credenciales, dinero, 'clic', peaje / deuda y entregas (inglés + español)"),
    (["n_chars", "n_words", "digit_ratio", "upper_ratio", "exclamations", "question_marks", "n_urls"],
     "Shape of the text", "Forma del texto",
     "characters, words, digit and capital-letter share, '!' and '?' counts, number of links",
     "caracteres, palabras, proporción de dígitos y mayúsculas, conteo de '!' y '?', cantidad de enlaces"),
    (["has_url", "url_length", "host_length", "n_dots", "path_length", "query_params", "url_entropy", "url_digit_ratio"],
     "Link basics", "Aspectos básicos del enlace",
     "length of the link / host / path, dots, query parameters, randomness (entropy), digit share",
     "largo del enlace / host / ruta, puntos, parámetros de consulta, aleatoriedad (entropía), proporción de dígitos"),
    (["host_hyphens", "host_digits", "subdomains", "has_ip", "has_at"],
     "Host structure", "Estructura del host",
     "hyphens and digits in the host, number of subdomains, IP address instead of a name, '@' in the link",
     "guiones y dígitos en el host, cantidad de subdominios, dirección IP en vez de nombre, '@' en el enlace"),
    (["https"], "HTTPS presence", "Presencia de HTTPS",
     "1 if the link starts with https://", "1 si el enlace empieza con https://"),
    (["is_shortener"], "URL shortener (acortadores)", "Acortador de URL",
     "1 if the host is bit.ly, tinyurl.com, t.co ...", "1 si el host es bit.ly, tinyurl.com, t.co ..."),
    (["suspicious_tld"], "Risky top-level domain", "Terminación de dominio riesgosa",
     "1 if the ending is on the risky list (.xyz, .top, .tk ...)", "1 si la terminación está en la lista riesgosa (.xyz, .top, .tk ...)"),
    (["lookalike_sim", "lookalike_typo", "brand_in_host"],
     "Look-alike domain (dominios parecidos)", "Dominio parecido",
     "similarity of the name to brands (paypal, amazon, bancopopular ...); typo-squat flag; brand inside an unofficial host",
     "parecido del nombre a marcas (paypal, amazon, bancopopular ...); alerta de dominio con errores de escritura; marca dentro de un host no oficial"),
    (["in_tranco", "tranco_log_rank"],
     "Domain reputation (stand-in for domain age)", "Reputación del dominio (sustituto de la edad del dominio)",
     "in the Tranco top-1M list? and its log rank (real domain age needs slow WHOIS lookups)",
     "¿está en la lista Tranco top-1M? y su rango en log (la edad real del dominio requiere búsquedas WHOIS lentas)"),
]
VARIANT_NAME = {
    "en": {"call_only": "Call only (solo llamadas)", "message_only": "Message / email / link only (solo mensajes/correos/enlaces)",
           "combined": "Combined (combinado)"},
    "es": {"call_only": "Solo llamadas", "message_only": "Solo mensajes / correos / enlaces", "combined": "Combinado"},
}
VARIANT_SHORT = {
    "en": {"call_only": "Call only", "message_only": "Message / email / link only", "combined": "Combined"},
    "es": {"call_only": "Solo llamadas", "message_only": "Solo mensajes/correos/enlaces", "combined": "Combinado"},
}


# ------------------------------------------------------------------ docx helpers
def shade(cell, fill):
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(sh)


class Doc:
    def __init__(self, footer: str):
        self.d = Document()
        s = self.d.sections[0]
        s.page_width, s.page_height = Inches(8.5), Inches(11)
        s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(0.8)
        st = self.d.styles["Normal"]
        st.font.name = "Calibri"; st.font.size = Pt(10.5)
        st.paragraph_format.space_after = Pt(5); st.paragraph_format.line_spacing = 1.1
        for name, sz in (("Heading 1", 14), ("Heading 2", 12)):
            h = self.d.styles[name]; h.font.name = "Calibri"; h.font.size = Pt(sz); h.font.bold = True
            h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x2E)
            h.paragraph_format.space_before = Pt(12); h.paragraph_format.space_after = Pt(4)
        fp = s.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = fp.add_run(footer); r.font.size = Pt(8)
        r2 = fp.add_run(); r2.font.size = Pt(8)
        for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
            if t:
                e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t)
            else:
                e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
            r2._r.append(e)

    def h1(self, t): self.d.add_heading(t, 1)
    def h2(self, t): self.d.add_heading(t, 2)

    def p(self, text, bold=False, italic=False, size=None, color=None, align=None):
        para = self.d.add_paragraph()
        r = para.add_run(text); r.bold = bold; r.italic = italic
        if size: r.font.size = Pt(size)
        if color: r.font.color.rgb = RGBColor(*color)
        if align is not None: para.alignment = align
        return para

    def bullets(self, items):
        for t in items:
            q = self.d.add_paragraph(t, style="List Bullet"); q.paragraph_format.space_after = Pt(2)

    def mono(self, text):
        para = self.d.add_paragraph()
        r = para.add_run(text); r.font.name = "Consolas"; r.font.size = Pt(8.5)
        para.paragraph_format.left_indent = Inches(0.2)

    def banner(self, text):
        t = self.d.add_table(rows=1, cols=1); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        c = t.rows[0].cells[0]; c.text = ""; shade(c, "FFF4CC")
        r = c.paragraphs[0].add_run(text); r.bold = True; r.font.size = Pt(10.5)
        self.d.add_paragraph()

    def table(self, head, rows, widths, size=9, caption=None):
        t = self.d.add_table(rows=1, cols=len(head)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for i, h in enumerate(head):
            c = t.rows[0].cells[i]; c.text = ""; shade(c, "E3E8EF")
            run = c.paragraphs[0].add_run(str(h)); run.bold = True; run.font.size = Pt(size)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""; cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(size)
        for i, col in enumerate(t.columns):
            col.width = Inches(widths[i])
        for row in t.rows:
            for i, c in enumerate(row.cells):
                c.width = Inches(widths[i])
                for q in c.paragraphs:
                    q.paragraph_format.space_after = Pt(1); q.paragraph_format.line_spacing = 1.0
        if caption:
            self.p(caption, italic=True, size=8.5, align=WD_ALIGN_PARAGRAPH.CENTER)
        else:
            self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def image(self, path, width):
        self.d.add_picture(str(path), width=Inches(width))
        self.d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def short(text, n=230):
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[:n].rstrip() + " ..."


def nonzero(feats: dict, limit=9):
    return [(k, v) for k, v in feats.items() if v not in (0, 0.0)][:limit]


# ------------------------------------------------------------------ charts (localized)
def make_charts(lang, scores, imp, tmp: Path):
    es = lang == "es"
    metrics = [("accuracy", "Exactitud" if es else "Accuracy"), ("precision", "Precisión" if es else "Precision"),
               ("recall", "Sensibilidad (recall)" if es else "Recall"), ("f1", "F1")]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / len(scores)
    for i, v in enumerate(C.VARIANTS):
        ax.bar(np.arange(4) + i * w, [scores.loc[v, m] for m, _ in metrics], w, color=COLORS[v], label=VARIANT_SHORT[lang][v])
    ax.set_xticks(np.arange(4) + w, [n for _, n in metrics]); ax.set_ylim(0.5, 1.0); ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_title("Métricas en validación - primeros modelos (configuración por defecto)" if es
                 else "Validation metrics - first models (default settings)")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(tmp / f"metrics_{lang}.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    cls = ["seguro", "fraude"] if es else ["safe", "fraud"]
    for ax, v in zip(axes, C.VARIANTS):
        r = scores.loc[v]
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha="center", va="center", color="white" if val > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], cls); ax.set_yticks([0, 1], cls)
        ax.set_xlabel("predicho" if es else "predicted"); ax.set_ylabel("real" if es else "true")
        ax.set_title(VARIANT_SHORT[lang][v], fontsize=10)
    fig.suptitle("Matrices de confusión en validación - primeros modelos" if es else "Validation confusion matrices - first models")
    fig.tight_layout(); fig.savefig(tmp / f"confusion_{lang}.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, v in zip(axes, C.VARIANTS):
        d = imp[imp.variant == v].head(7).iloc[::-1]
        ax.barh(d.feature, d.importance, color=d.channel.map({"call": "#5b7db1", "message": "#d08c3c"}))
        ax.set_title(VARIANT_SHORT[lang][v], fontsize=10); ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout(); fig.savefig(tmp / f"importance_{lang}.png", dpi=170, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ document
def build_document(lang, data, tmp: Path) -> Path:
    es = lang == "es"
    L = (lambda en, sp: sp) if es else (lambda en, sp: en)
    train, val, scores = data["train"], data["val"], data["scores"]
    X_tr, X_va, n_feat = data["X_tr"], data["X_va"], data["n_feat"]
    stamp, both = data["stamp"], data["both"]
    names, short_names = VARIANT_NAME[lang], VARIANT_SHORT[lang]
    n_call, n_msg = len(data["call_cols"]), len(data["msg_cols"])
    nf, ns = int((train.label == 1).sum()), int((train.label == 0).sum())
    vf, vs = int((val.label == 1).sum()), int((val.label == 0).sum())

    D = Doc(L("PRELIMINARY (alpha) evidence · schedule step 3 · validation set only · page ",
              "Evidencia PRELIMINAR (alfa) · paso 3 del cronograma · solo conjunto de validación · página "))
    D.p(L("EVIDENCE - Schedule step 3", "EVIDENCIA - Paso 3 del cronograma"), bold=True, size=20, color=(0x1F, 0x3A, 0x2E))
    D.p("Crear las características y los primeros modelos (18 al 31 de oct.)", bold=True, size=13)
    D.p(L("Comparing Single-Channel and Multichannel Machine Learning for Detecting TOAD Scams in Puerto Rico - William Díaz, Fase 11-1",
          "Comparación entre el aprendizaje automático de un solo canal y de varios canales para detectar estafas TOAD (entrega de ataques por teléfono) en Puerto Rico - William Díaz, Fase 11-1"), size=10)
    D.p(L(f"Generated {stamp} by running scripts/make_step3_evidence.py (project code, seed {C.SEED}).",
          f"Generado el {stamp} ejecutando scripts/make_step3_evidence.py (código del proyecto, semilla {C.SEED})."), italic=True, size=9)
    D.banner(L("PRELIMINARY (ALPHA) RESULTS. These are the very first models with default settings, scored on the validation set only. "
               "They are not tuned and they are not final results; the test set was not opened or used for anything in this document.",
               "RESULTADOS PRELIMINARES (ALFA). Estos son los primeros modelos, con la configuración por defecto, evaluados solo en el conjunto de "
               "validación. No están ajustados y no son resultados finales; el conjunto de prueba no se abrió ni se usó para nada en este documento."))

    D.h1(L("1. The four tasks of this step", "1. Las cuatro tareas de este paso"))
    D.table([L("Schedule task (from the cronograma)", "Tarea del cronograma"), L("Status", "Estado"),
             L("Where in the code", "Dónde está en el código"), L("Section", "Sección")],
            [["Crear las características de las llamadas (solicitudes de PIN, palabras urgentes, decir que son del banco o del gobierno)",
              L("Done", "Hecho"), "toadwatch/features_call.py", "2, 4"],
             ["Crear las características de los mensajes, correos y enlaces (acortadores, edad del dominio, HTTPS, dominios parecidos)",
              L("Done, except real domain age (see note)", "Hecho, excepto la edad real del dominio (ver nota)"),
              "toadwatch/features_message.py, domain_age.py", "3, 4"],
             ["Entrenar el Random Forest de tres maneras: solo llamadas, solo mensajes/correos/enlaces y combinado",
              L("Done", "Hecho"), "toadwatch/modeling.py", "5"],
             ["Anotar la exactitud, la precisión, la sensibilidad (recall) y el F1 en el conjunto de validación",
              L("Done", "Hecho"), "toadwatch/modeling.py (compute_metrics)", "6"]],
            [3.4, 1.15, 1.7, 0.65])
    D.p(L("Domain age - honest note. Real domain age needs WHOIS lookups over the network, which are slow. The schedule allows using a sample; "
          "toadwatch/domain_age.py implements a sampled, cached lookup (switch config.USE_DOMAIN_AGE, sample size config.DOMAIN_AGE_SAMPLE = "
          f"{C.DOMAIN_AGE_SAMPLE}), but it is OFF (USE_DOMAIN_AGE = {C.USE_DOMAIN_AGE}) and was NOT run for these results. In its place the models use the "
          "Tranco top-1M rank (msg_in_tranco, msg_tranco_log_rank), an offline stand-in for domain reputation.",
          "Edad del dominio - nota honesta. La edad real del dominio requiere búsquedas WHOIS por la red, que son lentas. El cronograma permite usar una "
          "muestra; toadwatch/domain_age.py implementa una búsqueda con muestra y caché (se activa con config.USE_DOMAIN_AGE; tamaño de la muestra "
          f"config.DOMAIN_AGE_SAMPLE = {C.DOMAIN_AGE_SAMPLE}), pero está DESACTIVADA (USE_DOMAIN_AGE = {C.USE_DOMAIN_AGE}) y NO se ejecutó para estos "
          "resultados. En su lugar los modelos usan el rango Tranco top-1M (msg_in_tranco, msg_tranco_log_rank), un sustituto sin conexión de la reputación del dominio."),
        size=9.5)

    head3 = [L("Group", "Grupo"), L("Features", "Características"), L("What it detects", "Qué detecta")]
    D.h1(L(f"2. Call features ({n_call})", f"2. Características de las llamadas ({n_call})"))
    D.table(head3, [[L(g_en, g_es), ", ".join("call_" + f for f in fs), L(t_en, t_es)] for fs, g_en, g_es, t_en, t_es in CALL_GROUPS],
            [1.5, 2.9, 2.5], size=8.5)
    D.h1(L(f"3. Message / email / link features ({n_msg})", f"3. Características de mensajes, correos y enlaces ({n_msg})"))
    D.table(head3, [[L(g_en, g_es), ", ".join("msg_" + f for f in fs), L(t_en, t_es)] for fs, g_en, g_es, t_en, t_es in MSG_GROUPS],
            [1.5, 2.9, 2.5], size=8.5)

    D.h1(L("4. Worked examples (real records from the training / validation sets)",
           "4. Ejemplos resueltos (registros reales de los conjuntos de entrenamiento y validación)"))
    D.p(L("The features below were computed by the project functions on real records; only the non-zero values are listed.",
          "Las características de abajo las calcularon las funciones del proyecto sobre registros reales; solo se listan los valores distintos de cero. "
          "Los textos de las llamadas están en inglés, como en los datos originales."), size=9.5)
    fraud_call = both[(both.label == 1) & both.call_text.str.contains("social security|PIN|card number", case=False)].iloc[0]
    safe_call = both[both.label == 0].iloc[0]
    for title, row in ((L("Fraud call", "Llamada fraudulenta"), fraud_call), (L("Safe call", "Llamada segura"), safe_call)):
        D.p(f"{title} ({row.call_source}):", bold=True, size=10)
        D.p("\"" + short(row.call_text) + "\"", italic=True, size=9)
        D.table([L("Feature", "Característica"), L("Value", "Valor")],
                [[k, f"{v:g}"] for k, v in nonzero(call_features(row.call_text))], [3.2, 1.2], size=8.5)

    urls = both[both.msg_url.str.len() > 0].copy()
    uf = pd.DataFrame([url_features(u) for u in urls.msg_url], index=urls.index)

    def pick(mask, label_value):
        hit = urls[(urls.label == label_value) & mask.reindex(urls.index).fillna(False)]
        return None if hit.empty else hit.iloc[0]

    real = L(" - real record", " - registro real")
    examples = [(L("Brand name inside an unofficial host (fraud)", "Marca dentro de un host no oficial (fraude)"), pick(uf.msg_brand_in_host == 1, 1)),
                (L("Link without HTTPS (fraud)", "Enlace sin HTTPS (fraude)"), pick((uf.msg_https == 0) & (uf.msg_in_tranco == 0), 1)),
                (L("Safe link with HTTPS", "Enlace seguro con HTTPS"), pick((uf.msg_https == 1) & (uf.msg_in_tranco == 1), 0))]
    show = ["msg_is_shortener", "msg_https", "msg_lookalike_sim", "msg_lookalike_typo", "msg_brand_in_host", "msg_in_tranco"]
    fmt = lambda f, k: f"{f[k]:.2f}" if k == "msg_lookalike_sim" else f"{f[k]:g}"
    rows = []
    for title, r in examples:
        if r is not None:
            rows.append([title + real, short(r.msg_url.strip(), 52)] + [fmt(url_features(r.msg_url), k) for k in show])
    demo = L(" (typed-in demo)", " (demo escrita a mano)")
    for title, u in ((L("URL shortener", "Acortador de URL"), "http://bit.ly/3xYzAbC"),
                     (L("Look-alike domain", "Dominio parecido"), "http://paypa1.com/login"),
                     (L("Brand in unofficial host", "Marca en host no oficial"), "http://bancopopular-secure-login.xyz/verify")):
        rows.append([title + demo, u] + [fmt(url_features(u), k) for k in show])
    D.table([L("Example", "Ejemplo"), L("Link", "Enlace"), L("short-ener", "acortador"), "https", L("look-alike sim", "similitud"),
             L("typo-squat", "parecido (typo)"), L("brand in host", "marca en host"), L("in Tranco", "en Tranco")], rows,
            [1.75, 2.1, 0.5, 0.45, 0.6, 0.5, 0.5, 0.5], size=7.5,
            caption=L("Link features computed by the project code. 'Real record' rows come from the training / validation sets; "
                      "'typed-in demo' rows are made-up links, used only because no shortener or typo-squat link happens to be in those records.",
                      "Características de enlaces calculadas por el código del proyecto. Las filas 'registro real' vienen de los conjuntos de entrenamiento / "
                      "validación; las filas 'demo escrita a mano' son enlaces inventados, usados solo porque en esos registros no hay ningún acortador ni "
                      "dominio con errores de escritura."))

    D.h1(L("5. The first models: Random Forest trained three ways", "5. Los primeros modelos: Random Forest entrenado de tres maneras"))
    D.p(L("Same algorithm and the same settings for all three versions (control variable of the proposal); only the columns each version may see change. "
          "Default settings, no tuning yet (tuning is schedule step 4).",
          "Mismo algoritmo y misma configuración para las tres versiones (variable de control de la propuesta); solo cambian las columnas que puede ver "
          "cada versión. Configuración por defecto, todavía sin ajustar (el ajuste es el paso 4 del cronograma)."), size=9.5)
    cols_seen = {"call_only": "call_*", "message_only": "msg_*", "combined": "call_* + msg_*"}
    D.table([L("Version", "Versión"), L("Columns it sees", "Columnas que ve"), L("Features", "Características"), L("Trained on", "Entrenado con")],
            [[names[v], cols_seen[v], n_feat[v], L(f"{len(train)} training records", f"{len(train)} registros de entrenamiento")] for v in C.VARIANTS],
            [3.1, 1.4, 0.8, 1.6])
    D.mono("RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=1, max_features='sqrt',\n"
           f"                       class_weight='balanced', random_state={C.SEED})")
    D.p(L(f"Data: {len(train)} training records ({nf} fraud / {ns} safe) and {len(val)} validation records ({vf} fraud / {vs} safe). Each record is a call "
          "transcript paired with one message, email or link of the same class (see the pairing rules in the project README).",
          f"Datos: {len(train)} registros de entrenamiento ({nf} fraude / {ns} seguros) y {len(val)} de validación ({vf} fraude / {vs} seguros). Cada registro es "
          "la transcripción de una llamada emparejada con un mensaje, correo o enlace de la misma clase (ver las reglas de emparejamiento en el README del proyecto)."),
        size=9.5)

    D.h1(L("6. Results on the validation set (preliminary)", "6. Resultados en el conjunto de validación (preliminares)"))
    D.table([L("Version", "Versión"), L("Accuracy (exactitud)", "Exactitud"), L("Precision (precisión)", "Precisión"),
             L("Recall (sensibilidad)", "Sensibilidad (recall)"), "F1", L("TN", "VN"), "FP", "FN", L("TP", "VP")],
            [[short_names[v], f"{scores.loc[v, 'accuracy']:.4f}", f"{scores.loc[v, 'precision']:.4f}", f"{scores.loc[v, 'recall']:.4f}",
              f"{scores.loc[v, 'f1']:.4f}", int(scores.loc[v, "tn"]), int(scores.loc[v, "fp"]), int(scores.loc[v, "fn"]), int(scores.loc[v, "tp"])] for v in C.VARIANTS],
            [1.7, 0.85, 0.85, 0.85, 0.6, 0.4, 0.4, 0.4, 0.4], size=8.5,
            caption=L(f"Validation set, {len(val)} records. TN true negatives, FP false positives, FN false negatives, TP true positives (fraud = positive).",
                      f"Conjunto de validación, {len(val)} registros. VN verdaderos negativos, FP falsos positivos, FN falsos negativos, VP verdaderos positivos (fraude = positivo)."))
    D.image(tmp / f"metrics_{lang}.png", 5.2)
    D.image(tmp / f"confusion_{lang}.png", 6.6)
    D.p(L("Most important features of each first model (Random Forest importance; blue = call, orange = message):",
          "Características más importantes de cada primer modelo (importancia del Random Forest; azul = llamada, anaranjado = mensaje):"), size=9.5)
    D.image(tmp / f"importance_{lang}.png", 6.9)

    D.h2(L("How to read this (it is an early snapshot)", "Cómo leer esto (es una primera foto)"))
    D.bullets(L([
        "These numbers come from default settings and one train / validation split. They are a first look, not conclusions.",
        "In this first run the combined model is not above the call-only model. That is recorded as found; it is not a verdict on the hypothesis.",
        "Part of the call signal comes from length / punctuation counts (call_n_words, call_question_marks ...), which can reflect how the datasets were written rather than the scam itself; this is checked in later steps.",
        "Domain age was not computed (see the note in section 1); the Tranco rank stands in for it.",
        "Next, per the schedule: step 4 tunes the settings on the validation set and ranks the features; step 5 is the one-time final test.",
    ], [
        "Estos números vienen de la configuración por defecto y de una sola división de entrenamiento / validación. Son una primera mirada, no conclusiones.",
        "En esta primera corrida el modelo combinado no supera al de solo llamadas. Se anota tal como salió; no es un veredicto sobre la hipótesis.",
        "Parte de la señal de las llamadas viene de conteos de largo y puntuación (call_n_words, call_question_marks ...), que pueden reflejar cómo se escribieron los datos y no la estafa en sí; esto se revisa en pasos posteriores.",
        "No se calculó la edad del dominio (ver la nota de la sección 1); el rango Tranco la sustituye.",
        "Lo siguiente, según el cronograma: el paso 4 ajusta la configuración con el conjunto de validación y ordena las características; el paso 5 es la prueba final, que se hace una sola vez.",
    ]))

    D.h1(L("7. How to reproduce", "7. Cómo reproducirlo"))
    D.mono("python scripts/make_step3_evidence.py        # " + L("this document", "este documento") + "\n"
           "python scripts/step3_features_models.py       # " + L("the schedule script itself", "el script del cronograma") + "\n"
           "notebooks/03_caracteristicas_primeros_modelos.ipynb")
    D.p(L(f"Run log: {stamp} · seed {C.SEED} · train {X_tr.shape} · validation {X_va.shape} · test set NOT loaded.",
          f"Registro de la corrida: {stamp} · semilla {C.SEED} · entrenamiento {X_tr.shape} · validación {X_va.shape} · conjunto de prueba NO cargado."),
        size=8.5, italic=True)

    path = tmp / f"{FILE_STEM[lang]}.docx"
    D.d.save(path)
    return path


def main():
    C.ensure_dirs()
    train, val = load_split("train"), load_split("val")          # the test set is never loaded
    X_tr, y_tr = build_features(train), train["label"].astype(int)
    X_va, y_va = build_features(val), val["label"].astype(int)
    call_cols = {c[5:] for c in X_tr.columns if c.startswith("call_")}
    msg_cols = {c[4:] for c in X_tr.columns if c.startswith("msg_")}
    # the documented feature groups must cover exactly the columns the models use
    assert call_cols == {f for g in CALL_GROUPS for f in g[0]}, call_cols ^ {f for g in CALL_GROUPS for f in g[0]}
    assert msg_cols == {f for g in MSG_GROUPS for f in g[0]}, msg_cols ^ {f for g in MSG_GROUPS for f in g[0]}

    models = train_all(X_tr, y_tr)
    scores = score_all(models, X_va, y_va)
    imp = feature_importance(models)
    data = {"train": train, "val": val, "scores": scores, "X_tr": X_tr, "X_va": X_va, "call_cols": call_cols, "msg_cols": msg_cols,
            "n_feat": {v: len(columns_for(v, X_tr.columns)) for v in C.VARIANTS},
            "stamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "both": pd.concat([train, val], ignore_index=True)}

    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if not exe:
        sys.exit("LibreOffice (soffice) is needed to make the PDF.")
    OUT_DIR.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    for lang in ("en", "es"):
        make_charts(lang, scores, imp, tmp)
        docx_path = build_document(lang, data, tmp)
        subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", str(tmp), str(docx_path)], capture_output=True, timeout=300)
        for ext in ("docx", "pdf"):
            shutil.copy(tmp / f"{FILE_STEM[lang]}.{ext}", OUT_DIR / f"{FILE_STEM[lang]}.{ext}")
        print("wrote", OUT_DIR / f"{FILE_STEM[lang]}.docx")
    print(scores[["n", "accuracy", "precision", "recall", "f1"]].round(4))


if __name__ == "__main__":
    main()
