"""
Train on only the K most important features (ranking: feature_importance.csv, mean |SHAP|
of the best model on 129 selected + 12 forecast features) and compare with all 141.

Each K is scored twice with the same folds and seeds as fe_lab.py:
  tuned   the tuned xgbl params as they are
  fair    colsample_bytree scaled so each tree still samples ~27 features (as with 141)
and the change in overall and clean-payer AUC is reported. Writes cache/top<K>_features.json
for every K so a winner can be trained in full:
  python train_gbm.py xgbl --repeats 5 --feats top60 --extra forecast --tag xgbl_top60

Caveat: the ranking was computed on all train rows, so small K is slightly favoured.

Usage:  python topk.py [K ...]        (default: 20 30 45 60 80 100 120)
"""

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd

from features import CACHE, HERE, TARGET
from fe_lab import cv_score
from train_gbm import build_matrix, load_params

warnings.filterwarnings("ignore")


def main():
    ks = [int(k) for k in sys.argv[1:]] or [20, 30, 45, 60, 80, 100, 120]
    df, cols = build_matrix("selected", ["forecast"])
    _, pool = build_matrix("all", ["forecast"])  # every loadable column: anything not kept is dropped
    rank = pd.read_csv(os.path.join(HERE, "feature_importance.csv"))["feature"].tolist()
    rank = [c for c in rank if c in cols]

    tr = df[df["is_train"] == 1].reset_index(drop=True)
    y = tr[TARGET].values.astype(int)
    clean = (tr["n_miss"] == 0).values
    params = dict(load_params("xgbl", "selected"), learning_rate=0.05, n_estimators=5000)
    per_tree = params["colsample_bytree"] * len(cols)

    base_o, base_w = cv_score(tr[cols], y, clean, params)
    print(f"all {len(cols):3d}            overall {base_o.mean():.5f} {np.round(base_o, 5)}  "
          f"clean {base_w.mean():.5f}", flush=True)
    rows = [("all", len(cols), "tuned", base_o.mean(), 0.0, 0.0, base_w.mean(), 0.0)]
    for k in ks:
        top = rank[:k]
        with open(os.path.join(CACHE, f"top{k}_features.json"), "w") as f:
            json.dump({"keep": top, "pool": pool}, f, indent=1)
        for mode in ("tuned", "fair"):
            p = params if mode == "tuned" else dict(params, colsample_bytree=min(1.0, per_tree / k))
            o, w = cv_score(tr[top], y, clean, p)
            d = o - base_o
            rows.append((f"top{k}", k, mode, o.mean(), d.mean(), (d > 0).mean(), w.mean(),
                         (w - base_w).mean()))
            print(f"top{k:<3d} {mode:5s}       overall {o.mean():.5f} {np.round(o, 5)}  "
                  f"clean {w.mean():.5f}  (delta {d.mean():+.5f})", flush=True)

    res = pd.DataFrame(rows, columns=["set", "n_feats", "params", "overall_auc", "delta_overall",
                                      "share_repeats_up", "clean_auc", "delta_clean"])
    print("\n" + res.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    res.to_csv(os.path.join(CACHE, "topk_results.csv"), index=False)


if __name__ == "__main__":
    main()
