"""Schedule step 4: tune on the validation set, rank features, freeze the final configuration.

    python scripts/step4_tune.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toadwatch import config as C  # noqa: E402
from toadwatch.modeling import feature_importance, save_models, train_all  # noqa: E402
from toadwatch.tuning import freeze_final_config, grid_search  # noqa: E402

if __name__ == "__main__":
    C.ensure_dirs()
    tr = pd.read_parquet(C.INTERIM_DIR / "features_train.parquet")
    va = pd.read_parquet(C.INTERIM_DIR / "features_val.parquet")
    X_tr, y_tr = tr.drop(columns="label"), tr["label"]
    X_va, y_va = va.drop(columns="label"), va["label"]

    results = grid_search(X_tr, y_tr, X_va, y_va)
    results.to_csv(C.TABLES_DIR / "step4_grid_results.csv", index=False)
    cfg = freeze_final_config(results)
    print("final configuration:", cfg)

    models = train_all(X_tr, y_tr, cfg["params"])
    save_models(models, "tuned_models")
    imp = feature_importance(models)
    imp.to_csv(C.TABLES_DIR / "step4_feature_importance.csv", index=False)
    print(imp.groupby("variant").head(8).to_string(index=False))
