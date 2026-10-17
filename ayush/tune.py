"""
Optuna search for the GPU XGBoost models. Early stopping on the fold is acceptable
here because the search only ranks configs; final scores come from train_gbm.py's
honest protocol. The best params are written to cache/best_params_<model>.json after
every improving trial, so the search can be stopped (Ctrl+C) at any time.

Usage: python tune.py xgb|xgbl [n_trials] [all|selected]
"""

import json
import os
import sys
import warnings
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from features import TARGET, CACHE, feature_columns, get_features
from train_gbm import XGB_PARAMS, XGBL_PARAMS, select_columns

warnings.filterwarnings("ignore")
df, _ = get_features()
tr = df[df.is_train == 1].reset_index(drop=True)
FEATS = sys.argv[3] if len(sys.argv) > 3 else "all"
cols = select_columns(df, FEATS)
X, y = tr[cols], tr[TARGET].values.astype(int)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=314).split(X, y))


def cv(params):
    oof = np.zeros(len(y))
    for a, b in FOLDS:
        m = xgb.XGBClassifier(**params, early_stopping_rounds=200, random_state=0)
        m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], verbose=False)
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    return roc_auc_score(y, oof)


def suggest(t, name):
    p = dict(subsample=t.suggest_float("subsample", 0.5, 1.0),
             colsample_bytree=t.suggest_float("colsample_bytree", 0.1, 0.7),
             reg_lambda=t.suggest_float("reg_lambda", 1e-2, 100, log=True),
             reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
             gamma=t.suggest_float("gamma", 0, 5),
             min_child_weight=t.suggest_float("min_child_weight", 1, 200, log=True))
    if name == "xgb":
        p["max_depth"] = t.suggest_int("max_depth", 3, 8)
    else:
        p["max_leaves"] = t.suggest_int("max_leaves", 7, 127, log=True)
    return p


if __name__ == "__main__":
    name = sys.argv[1]
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    base = dict({"xgb": XGB_PARAMS, "xgbl": XGBL_PARAMS}[name], learning_rate=0.03, n_estimators=5000)
    out = os.path.join(CACHE, f"best_params_{name}{'_sel' if FEATS == 'selected' else ''}.json")

    def objective(t):
        return cv(dict(base, **suggest(t, name)))

    def save_best(study, trial):
        print(f"trial {trial.number}: {trial.value:.5f}  best {study.best_value:.5f}", flush=True)
        if study.best_trial.number == trial.number:
            with open(out, "w") as f:
                json.dump(study.best_params, f, indent=1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial({k: v for k, v in base.items() if k in
                         ("subsample", "colsample_bytree", "reg_lambda", "min_child_weight",
                          "max_depth", "max_leaves")} | {"reg_alpha": 1e-3, "gamma": 0.0})
    study.optimize(objective, n_trials=n_trials, callbacks=[save_best])
    print("best", study.best_value, study.best_params, "->", out)
