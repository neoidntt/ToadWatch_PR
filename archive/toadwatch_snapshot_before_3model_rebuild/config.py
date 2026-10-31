"""Central settings for ToadWatch_PR.

Everything that the proposal fixes (seed, 70/10/20 split, Random Forest) lives here so
that every module and notebook uses the same values.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- experiment version
# "v1" = the original study (frozen, final test already used).
# "v2" = the fair-fusion follow-up: new seed / new split / new pairing, cross-channel
#        features, no length/style shortcuts, per-version tuning by cross-validation,
#        early- and late-fusion combined models. Select it BEFORE importing toadwatch:
#        os.environ["TOADWATCH_VERSION"] = "v2"
# "v3" = the strict "true accuracy" study: harder data (mixed pairs under the proposal's TOAD
#        rule + a hard, style-matched set), whole script families kept in one split, leakage and
#        label audits, text-augmented models. See toadwatch/workflow_v3.py.
VERSION = os.environ.get("TOADWATCH_VERSION", "v1")
if VERSION not in ("v1", "v2", "v3"):
    raise ValueError("TOADWATCH_VERSION must be 'v1', 'v2' or 'v3'")
_SUB = "" if VERSION == "v1" else VERSION

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent  # the `Fase` folder that holds the raw data folders

CALL_SAFE_DIR = DATA_ROOT / "Call Safe data"
CALL_DANGEROUS_DIR = DATA_ROOT / "Call Dangerous data"
MSG_SAFE_DIR = DATA_ROOT / "SMS, Email, URL Safe"
MSG_DANGEROUS_DIR = DATA_ROOT / "SMS, Email, URL Dangerous"
NC_STATE_DIR = CALL_DANGEROUS_DIR / "robocall-audio-dataset-ver-0.1"
TRANCO_CSV = MSG_SAFE_DIR / "tranco_W3ZV9-1m.csv" / "top-1m.csv"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / _SUB
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / _SUB
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / _SUB
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"

# Puerto Rico cases collected from DACO / university faculty (secondary test set).
PR_CASES_CSV = EXTERNAL_DIR / "pr_cases.csv"

# ---------------------------------------------------------------- reproducibility
SEED = {"v1": 42, "v2": 2027, "v3": 2028}[VERSION]   # every version draws its own split, sample and pairing
SPLIT_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}

# ---------------------------------------------------------------- sampling caps
# The raw files hold millions of rows; the study only needs a few thousand per class.
# Caps are per label (0 = safe, 1 = fraud) and per source type, applied by uniform
# reservoir sampling with SEED so the sample is reproducible.
CORPUS_CALL_POOL = 600           # conversations drawn from EACH big safe-call corpus
EMAIL_CAP_PER_CLASS = 6000       # rows read from the big email csv files, per class
URL_CAP_PER_CLASS = 12000        # rows read from the JSON / csv URL files, per class
OVERSAMPLE = 1.5                 # read extra rows because cleaning removes duplicates
MAX_BODY_CHARS = 5000            # emails are truncated to this many characters

MIN_TRANSCRIPT_CHARS = 20        # shorter NC State transcripts are unusable
MIN_TEXT_CHARS = 3
NEAR_DUP_THRESHOLD = 0.90        # TF-IDF cosine above which two calls count as the same script

# ---------------------------------------------------------------- features
USE_TRANCO_RANK = True           # offline stand-in for domain age (see README)
USE_DOMAIN_AGE = False           # WHOIS lookups; off by default (slow, needs network)
DOMAIN_AGE_SAMPLE = 500          # if enabled, only this many domains are looked up

# ---------------------------------------------------------------- model
# Same algorithm and training process for the three versions (control variable).
BASE_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": SEED,
}
# v2 compares two ways of combining the channels (see fusion.py); the one with the better
# cross-validated F1 on train+val becomes "the" combined model, chosen BEFORE the test.
VARIANTS = {
    "v1": ("call_only", "message_only", "combined"),
    "v2": ("call_only", "message_only", "combined_early", "combined_stacked"),
    # v3: the Random-Forest family of the proposal, plus the same models with an added word-level
    # text model per channel (text_stack.py). Every upgrade is applied to all three versions.
    "v3": ("call_only", "message_only", "combined_early", "combined_stacked",
           "call_only_text", "message_only_text", "combined_text"),
}[VERSION]

# ---------------------------------------------------------------- v2: fair fusion
EXCLUDE_STYLE_FEATURES = VERSION != "v1"   # drop raw length / punctuation-count shortcuts
USE_CROSS_FEATURES = False                 # tested in v2 diagnosis: no CV gain, so off (flip to retry)
STYLE_FEATURES = (
    "call_n_words", "call_n_chars", "call_avg_word_len", "call_question_marks",
    "call_exclamations", "call_digit_ratio",
    "msg_n_chars", "msg_n_words", "msg_question_marks", "msg_exclamations",
)
CV_FOLDS = 5
CV_REPEATS = 3                 # repeated stratified CV on train+val (the test set is never used)
N_BOOTSTRAP = 2000             # bootstrap resamples for confidence intervals
TIE_MARGIN = 0.002             # CV-F1 difference below which early fusion (simpler) is kept
TUNING_GRID = {
    "n_estimators": [300],
    "max_depth": [None, 12],
    "min_samples_leaf": [1, 3],
    "max_features": ["sqrt", 0.5],
}

# ---------------------------------------------------------------- v3: strict protocol
# Calls and messages are clustered into "families" of near-identical scripts (TF-IDF cosine);
# a whole family always lands in ONE bucket, so no template is shared between train and test.
GROUP_SIM_THRESHOLD = 0.55
N_BUCKETS = 10                   # 10 buckets of ~10 %: test = 2 (20 %), validation = 1 (10 %), train = 7 (70 %)
TEST_BUCKETS = (8, 9)
VAL_BUCKETS = (7,)
DEV_FOLDS = 4                    # development buckets 0-7 are grouped into 4 CV folds (bucket // 2)
# TOAD label rule from the proposal: a case is fraud ONLY if the call AND the message are both fraud.
# Share of each pair type among the records (FF = fraud call + fraud message, FS = fraud call + safe message ...).
PAIR_MIX = {"FF": 0.40, "FS": 0.15, "SF": 0.15, "SS": 0.30}
HARD_PER_SKELETON = {"call": 28, "sms": 30, "email": 22}   # how many variants of each hand-written script to draw
HARD_WEIGHT_GRID = (1, 4, 10, 30)   # training weight of the style-matched 'hard' records, chosen by development CV
V3_CORPUS_CALL_POOL = 1200
V3_EMAIL_CAP_PER_CLASS = 4000
V3_URL_CAP_PER_CLASS = 4000
if VERSION == "v3":
    CORPUS_CALL_POOL = V3_CORPUS_CALL_POOL
    EMAIL_CAP_PER_CLASS = V3_EMAIL_CAP_PER_CLASS
    URL_CAP_PER_CLASS = V3_URL_CAP_PER_CLASS

FINAL_CONFIG_JSON = OUTPUTS_DIR / "final_config.json"
FINAL_TEST_LOCK = OUTPUTS_DIR / "FINAL_TEST_DONE.lock"
SPLIT_MANIFEST_JSON = PROCESSED_DIR / "split_manifest.json"


def ensure_dirs() -> None:
    for d in (INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
