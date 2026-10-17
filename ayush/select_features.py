"""
Null-importance feature selection (GPU XGBoost).

Fits the model once on the real target and N times on shuffled targets. A feature is
kept when its real gain beats the 75th percentile of its gain under shuffled labels,
i.e. it carries more signal than the model can find in noise.

Usage:  python select_features.py [n_null]
Writes: cache/selected_features.json  (used by train_gbm.py --feats selected)
"""

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from features import TARGET, CACHE, feature_columns, get_features

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
    n_null = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    df, _ = get_features()
    tr = df[df.is_train == 1].reset_index(drop=True)
    cols = feature_columns(df)
    X, y = tr[cols], tr[TARGET].values.astype(int)

    real = pd.concat([gains(X, y, s) for s in range(3)], axis=1).mean(1)
    rng = np.random.RandomState(0)
    null = []
    for i in range(n_null):
        null.append(gains(X, rng.permutation(y), 100 + i))
        print(f"null run {i + 1}/{n_null}", flush=True)
    null = pd.concat(null, axis=1)

    p75 = null.quantile(0.75, axis=1)
    score = np.log1p(real) - np.log1p(p75)
    keep = score[score > 0].sort_values(ascending=False).index.tolist()
    report = pd.DataFrame({"real_gain": real, "null_p75": p75, "score": score}).sort_values("score")
    report.to_csv(os.path.join(CACHE, "null_importance.csv"))
    with open(os.path.join(CACHE, "selected_features.json"), "w") as f:
        json.dump(keep, f, indent=1)
    print(f"kept {len(keep)}/{len(cols)} features -> cache/selected_features.json")
    print("dropped:", [c for c in cols if c not in keep])
