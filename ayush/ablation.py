"""Feature-group ablation with a fixed GPU XGBoost config and fixed folds.

Usage: python ablation.py
"""

import warnings
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from features import TARGET, get_features

warnings.filterwarnings("ignore")
PARAMS = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.03,
              grow_policy="lossguide", max_depth=0, max_leaves=31, min_child_weight=10,
              subsample=0.8, colsample_bytree=0.5, reg_lambda=5, n_estimators=5000,
              tree_method="hist", device="cuda", early_stopping_rounds=200)


def cv_auc(X, y, seed=42):
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = xgb.XGBClassifier(**PARAMS, random_state=seed)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return roc_auc_score(y, oof)


if __name__ == "__main__":
    df, groups = get_features()
    tr = df[df.is_train == 1].reset_index(drop=True)
    y = tr[TARGET].values.astype(int)
    order = ["raw", "base", "delinq", "minpay", "util", "date", "ratio", "app"]
    cols = []
    for g in order:
        cols = cols + groups[g]
        print(f"+ {g:7s} ({len(cols):3d} cols)  AUC {cv_auc(tr[cols], y):.5f}", flush=True)
    for g in order[1:]:
        sub = [c for c in cols if c not in set(groups[g])]
        print(f"- {g:7s} ({len(sub):3d} cols)  AUC {cv_auc(tr[sub], y):.5f}", flush=True)
