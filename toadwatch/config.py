"""Central settings for ToadWatch_PR (one study, three models).

THE THREE MODELS (nothing else is trained or reported):
    call_only     Random Forest on the call transcript (it also reads links the caller mentions)
    message_only  Random Forest on the SMS / email / link
    combined      "either one is fraud -> the whole attack is fraud": the OR of the two Random Forests above
                  (same two trained forests, no third estimator)

TOAD label rule (the proposal's definition): the call and the message of a case belong to ONE attack. If either part is
fraud, the whole attack is fraud - including the part that looks harmless on its own.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent  # the `Fase` folder that holds the raw data folders

CALL_SAFE_DIR = DATA_ROOT / "Call Safe data"
CALL_DANGEROUS_DIR = DATA_ROOT / "Call Dangerous data"
MSG_SAFE_DIR = DATA_ROOT / "SMS, Email, URL Safe"
MSG_DANGEROUS_DIR = DATA_ROOT / "SMS, Email, URL Dangerous"
NC_STATE_DIR = CALL_DANGEROUS_DIR / "robocall-audio-dataset-ver-0.1"
TRANCO_CSV = MSG_SAFE_DIR / "tranco_W3ZV9-1m.csv" / "top-1m.csv"
SMS_CAMPAIGNS_CSV = MSG_DANGEROUS_DIR / "sms_phishing_english_spanish.csv"     # 60k real phishing SMS (fraud only)

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORT_DIR = PROJECT_ROOT / "report"

PR_CASES_CSV = EXTERNAL_DIR / "pr_cases.csv"      # real Puerto Rico cases (DACO / faculty), when you have them

FINAL_CONFIG_JSON = OUTPUTS_DIR / "final_config.json"
FINAL_TEST_LOCK = OUTPUTS_DIR / "FINAL_TEST_DONE.lock"
SPLIT_MANIFEST_JSON = PROCESSED_DIR / "split_manifest.json"

# ---------------------------------------------------------------- the three models
MODELS = ("call_only", "message_only", "combined")
MODEL_NAMES = {"call_only": "Call only", "message_only": "Message / email / link only", "combined": "Combined (either is fraud -> fraud)"}

# ---------------------------------------------------------------- reproducibility
SEED = 2029
SPLIT_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}

# ---------------------------------------------------------------- how much data is read (raw files hold millions of rows)
CORPUS_CALL_POOL = 1200          # conversations drawn from EACH big safe-call corpus
EMAIL_CAP_PER_CLASS = 4000       # rows read from the big email csv files, per class
URL_CAP_PER_CLASS = 4000         # rows read from the URL files, per class
OVERSAMPLE = 1.5                 # read extra rows because cleaning removes duplicates
MAX_BODY_CHARS = 5000            # emails are truncated to this many characters
SMS_CAMPAIGN_MAX_PER_CAMPAIGN = 2     # the new SMS file repeats each scam campaign many times: keep at most this many per campaign
SMS_CAMPAIGN_CAP = 6000               # ... and at most this many in total

MIN_TRANSCRIPT_CHARS = 20
MIN_TEXT_CHARS = 3
NEAR_DUP_THRESHOLD = 0.90        # TF-IDF cosine above which two calls count as the same script

# ---------------------------------------------------------------- families, buckets and pairing
# Near-identical texts (cosine > GROUP_SIM_THRESHOLD) form one "family"; a family always sits in ONE bucket, so a
# template can never be in train and test at the same time.
GROUP_SIM_THRESHOLD = 0.55
N_BUCKETS = 10                   # test = 2 buckets (20 %), validation = 1 (10 %), train = 7 (70 %)
TEST_BUCKETS = (8, 9)
VAL_BUCKETS = (7,)
DEV_FOLDS = 4                    # development buckets 0-7 are grouped into 4 cross-validation folds (bucket // 2)
# Pair types (call + message of ONE attack): FF = fraud call + fraud message, FS = fraud call + safe-looking message,
# SF = safe-looking call + fraud message, SS = both safe. Rule: fraud if EITHER part is fraud.
PAIR_MIX = {"FF": 0.25, "FS": 0.125, "SF": 0.125, "SS": 0.50}
HARD_PER_SKELETON = {"call": 28, "sms": 30, "email": 22}

# Tiers: how much a record can be trusted to measure CONTENT detection (see dataset_builder.py)
#   hard      my style-matched set (both classes written alike)
#   matched   public collections where BOTH classes come from the same corpus / generator (style cannot give the class away)
#   unmatched public files that hold ONE class each (a model can score high by recognising the dataset itself)
TIERS = ("hard", "matched", "unmatched")
MATCHED_CALL_SOURCES = ("agent_conversation_labels_0", "agent_conversation_labels_1", "labels_0", "labels_1",
                        "scam_dialogue_label_0", "scam_dialogue_label_1")
MATCHED_MSG_SOURCES = ("sms_ham", "sms_spam", "augmented_ham_en_es_es", "augmented_spam_en_es_es",
                       "spaphish_legitimate_0", "spaphish_phishing_1")

# ---------------------------------------------------------------- features
USE_TRANCO_RANK = True           # offline stand-in for domain age (see README)
USE_DOMAIN_AGE = False           # WHOIS lookups; off by default (slow, needs network)
DOMAIN_AGE_SAMPLE = 500
# raw length / punctuation counts mostly reveal which corpus a text is from, so no model sees them
STYLE_FEATURES = (
    "call_n_words", "call_n_chars", "call_avg_word_len", "call_question_marks", "call_exclamations", "call_digit_ratio",
    "msg_n_chars", "msg_n_words", "msg_question_marks", "msg_exclamations",
)

# ---------------------------------------------------------------- Random Forest (same algorithm and grid for both detectors)
BASE_RF_PARAMS = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt",
                  "class_weight": "balanced", "n_jobs": -1, "random_state": SEED}
TUNING_GRID = {"n_estimators": [300], "max_depth": [None, 12, 24], "min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 0.3]}
THRESHOLD_GRID = tuple(round(0.20 + 0.05 * i, 2) for i in range(13))        # 0.20 ... 0.80
N_BOOTSTRAP = 2000


def ensure_dirs() -> None:
    for d in (INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
