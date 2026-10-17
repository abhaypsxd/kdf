"""
Optuna search for the GPU models (xgb, xgbl, cat) on any feature set.

The objective follows the honest protocol of train_gbm.py, so tuned scores are not
inflated: early stopping uses an inner 10% holdout of each training fold (never the
validation fold), and every trial is scored on 2 different 5-fold splits. The search uses
learning rate 0.05 for speed; only the ranking of configs matters here.

Runs are resumable: trials are stored in cache/optuna.db, so running the same command
twice gives one longer search. A new search first evaluates previously tuned params for
that model. The best params go to cache/best_params_<model><suffix>.json, which
train_gbm.py picks up automatically for the same --feats/--extra.

Always confirm a new best with train_gbm.py (per-repeat AUC vs the current model)
before trusting it.

Usage: python tune.py xgb|xgbl|cat [n_trials] [feats] [extra feature sets ...]
  e.g. python tune.py xgbl 40 selected forecast
       python tune.py cat 25 sel_v2 forecast dev app2 freq peer shape
"""

import glob
import json
import os
import sys
import warnings
import numpy as np
import optuna
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from features import TARGET, CACHE
from train_gbm import XGB_PARAMS, XGBL_PARAMS, CAT_PARAMS, build_matrix, param_suffix, prepare

warnings.filterwarnings("ignore")
NAME = sys.argv[1]
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
FEATS = sys.argv[3] if len(sys.argv) > 3 else "all"
EXTRA = sys.argv[4:]
SPLIT_SEEDS = (314, 2718)

df, cols = build_matrix(FEATS, EXTRA)
tr, _, X, _, CAT_IDX = prepare(NAME, df, cols)
y = tr[TARGET].values.astype(int)
SPLITS = [list(StratifiedKFold(5, shuffle=True, random_state=s).split(X, y)) for s in SPLIT_SEEDS]


def fit_score(params, a, b, seed):
    ia, ih = train_test_split(a, test_size=0.1, stratify=y[a], random_state=seed)
    if NAME == "cat":
        m = CatBoostClassifier(**params, random_seed=seed, early_stopping_rounds=200, verbose=0)
        m.fit(Pool(X.iloc[ia], y[ia], cat_features=CAT_IDX),
              eval_set=Pool(X.iloc[ih], y[ih], cat_features=CAT_IDX))
        return m.predict_proba(Pool(X.iloc[b], cat_features=CAT_IDX))[:, 1]
    m = xgb.XGBClassifier(**params, random_state=seed, early_stopping_rounds=200)
    m.fit(X.iloc[ia], y[ia], eval_set=[(X.iloc[ih], y[ih])], verbose=False)
    return m.predict_proba(X.iloc[b])[:, 1]


def cv(params):
    aucs = []
    for s, folds in enumerate(SPLITS):
        oof = np.zeros(len(y))
        for f, (a, b) in enumerate(folds):
            oof[b] = fit_score(params, a, b, 1000 * s + f)
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs))


def suggest(t):
    if NAME == "cat":
        return dict(depth=t.suggest_int("depth", 4, 10),
                    l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 50, log=True),
                    random_strength=t.suggest_float("random_strength", 0, 5),
                    bagging_temperature=t.suggest_float("bagging_temperature", 0, 3),
                    border_count=t.suggest_categorical("border_count", [64, 128, 254]))
    p = dict(subsample=t.suggest_float("subsample", 0.5, 1.0),
             colsample_bytree=t.suggest_float("colsample_bytree", 0.05, 0.7, log=True),
             reg_lambda=t.suggest_float("reg_lambda", 1e-2, 100, log=True),
             reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
             gamma=t.suggest_float("gamma", 0, 5),
             min_child_weight=t.suggest_float("min_child_weight", 1, 300, log=True))
    if NAME == "xgb":
        p["max_depth"] = t.suggest_int("max_depth", 3, 9)
    else:
        p["max_leaves"] = t.suggest_int("max_leaves", 7, 255, log=True)
    return p


def known_good():
    """Previously tuned params for this model (any feature set), to seed a new search."""
    keys = {"cat": ("depth", "l2_leaf_reg", "random_strength", "bagging_temperature",
                    "border_count"),
            "xgb": ("subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "gamma",
                    "min_child_weight", "max_depth"),
            "xgbl": ("subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "gamma",
                     "min_child_weight", "max_leaves")}[NAME]
    seeds = []
    for path in sorted(glob.glob(os.path.join(CACHE, f"best_params_{NAME}*.json"))):
        if path.endswith(".rejected.json"):
            continue
        with open(path) as f:
            p = {k: v for k, v in json.load(f).items() if k in keys}
        if len(p) == len(keys) and p not in seeds:
            seeds.append(p)
    return seeds


if __name__ == "__main__":
    suffix = param_suffix(FEATS, EXTRA)
    base = {"xgb": XGB_PARAMS, "xgbl": XGBL_PARAMS, "cat": CAT_PARAMS}[NAME]
    base = dict(base, **({"learning_rate": 0.05, "iterations": 5000} if NAME == "cat"
                         else {"learning_rate": 0.05, "n_estimators": 5000}))
    out = os.path.join(CACHE, f"best_params_{NAME}{suffix}.json")

    def objective(t):
        return cv(dict(base, **suggest(t)))

    def save_best(study, trial):
        print(f"trial {trial.number}: {trial.value:.5f}  best {study.best_value:.5f}", flush=True)
        if study.best_trial.number == trial.number:
            with open(out, "w") as f:
                json.dump(study.best_params, f, indent=1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name=f"v2_{NAME}{suffix}",
                                storage=f"sqlite:///{os.path.join(CACHE, 'optuna.db')}",
                                load_if_exists=True, sampler=optuna.samplers.TPESampler())
    if len(study.trials) == 0:
        seeds = known_good()
        for p in seeds:
            study.enqueue_trial(p)
        print(f"new search '{NAME}{suffix}' on {len(cols)} features, "
              f"starting from {len(seeds)} known-good configs")
    else:
        print(f"resuming '{NAME}{suffix}': {len(study.trials)} trials, best {study.best_value:.5f}")
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[save_best])
    print("best", study.best_value, study.best_params, "->", out)
