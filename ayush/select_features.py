"""
Null-importance feature selection (GPU XGBoost).

Fits the model once on the real target and N times on shuffled targets. A feature is
kept when its real gain beats the 75th percentile of its gain under shuffled labels,
i.e. it carries more signal than the model can find in noise.

Usage:
  python select_features.py [n_null]
      base features only -> cache/selected_features.json (used by --feats selected)
  python select_features.py [n_null] --extra forecast dev app2 ... --out sel_v2
      select from base + extra sets -> cache/sel_v2_features.json
      (train with: --feats sel_v2 --extra forecast dev app2 ...)
"""

import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from features import TARGET, CACHE
from train_gbm import build_matrix

warnings.filterwarnings("ignore")
PARAMS = dict(objective="binary:logistic", learning_rate=0.05, n_estimators=400,
              grow_policy="lossguide", max_depth=0, max_leaves=31, min_child_weight=50,
              subsample=0.8, colsample_bytree=0.5, reg_lambda=5, tree_method="hist",
              device="cuda")


def gains(X, y, seed):
    m = xgb.XGBClassifier(**PARAMS, random_state=seed).fit(X, y, verbose=False)
    g = m.get_booster().get_score(importance_type="total_gain")
    return pd.Series(g).reindex(X.columns).fillna(0.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n_null", type=int, nargs="?", default=30)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", default="selected")
    args = ap.parse_args()
    if args.extra and args.out == "selected":
        raise SystemExit("with --extra, give a new --out name so selected_features.json is kept")

    df, cols = build_matrix("all", args.extra)
    tr = df[df.is_train == 1].reset_index(drop=True)
    X, y = tr[cols], tr[TARGET].values.astype(int)

    real = pd.concat([gains(X, y, s) for s in range(3)], axis=1).mean(1)
    rng = np.random.RandomState(0)
    null = []
    for i in range(args.n_null):
        null.append(gains(X, rng.permutation(y), 100 + i))
        print(f"null run {i + 1}/{args.n_null}", flush=True)
    null = pd.concat(null, axis=1)

    p75 = null.quantile(0.75, axis=1)
    score = np.log1p(real) - np.log1p(p75)
    keep = score[score > 0].sort_values(ascending=False).index.tolist()
    report = pd.DataFrame({"real_gain": real, "null_p75": p75, "score": score}).sort_values("score")
    report.to_csv(os.path.join(CACHE, f"null_importance_{args.out}.csv"))
    out = os.path.join(CACHE, f"{args.out}_features.json")
    with open(out, "w") as f:
        # plain list for the base-only selection (backward compatible), else keep + pool
        json.dump(keep if not args.extra else {"keep": keep, "pool": cols}, f, indent=1)
    print(f"kept {len(keep)}/{len(cols)} features -> {out}")
    import pickle
    fam_of = {}
    for e in args.extra:
        with open(os.path.join(CACHE, f"{e}_features.pkl"), "rb") as f:
            fam_of.update({c: e for c in pickle.load(f).columns if c != "ID"})
    kept = set(keep)
    for fam in ["base"] + args.extra:
        members = [c for c in cols if fam_of.get(c, "base") == fam]
        print(f"  {fam:9s} kept {sum(c in kept for c in members)}/{len(members)}")
