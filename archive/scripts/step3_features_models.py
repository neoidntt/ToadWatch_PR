"""Schedule step 3: build call + message features, train the three Random Forests and
report accuracy / precision / recall / F1 on the VALIDATION set (the test set is not read).

    python scripts/step3_features_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toadwatch import config as C  # noqa: E402
from toadwatch.features import build_features  # noqa: E402
from toadwatch.modeling import save_models, score_all, train_all  # noqa: E402
from toadwatch.splitting import load_split  # noqa: E402

if __name__ == "__main__":
    C.ensure_dirs()
    train, val = load_split("train"), load_split("val")
    X_train, y_train = build_features(train), train["label"].astype(int)
    X_val, y_val = build_features(val), val["label"].astype(int)
    X_train.assign(label=y_train).to_parquet(C.INTERIM_DIR / "features_train.parquet")
    X_val.assign(label=y_val).to_parquet(C.INTERIM_DIR / "features_val.parquet")

    models = train_all(X_train, y_train)
    table = score_all(models, X_val, y_val)
    table.to_csv(C.TABLES_DIR / "step3_validation_metrics.csv")
    save_models(models, "step3_models")
    print(table[["n", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]].round(4))
