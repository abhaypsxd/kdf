"""
GPU gradient-boosting models with an honest CV protocol.

Models (all train on CUDA):
  xgb   - XGBoost, depth-wise trees
  xgbl  - XGBoost, loss-guided (leaf-wise, LightGBM-style) trees
  cat   - CatBoost GPU with native categorical features

For every outer fold the number of boosting rounds is chosen on an inner 10% holdout
of the *training* part, then the model is refit on the full training part with that
round count. The outer validation fold is never used for early stopping, so the OOF
AUC is an unbiased estimate.

Usage:  python train_gbm.py xgb|xgbl|cat [--repeats 3] [--feats all|selected] [--tag name]
                            [--extra forecast]
Writes: preds/<tag or model>.npz with oof (n_train,), test (n_test,), and per-repeat AUCs.
"""

import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from features import TARGET, CAT_COLS, CACHE, HERE, feature_columns, get_features

warnings.filterwarnings("ignore")
PRED_DIR = os.path.join(HERE, "preds")
os.makedirs(PRED_DIR, exist_ok=True)
SEEDS = [42, 7, 2024, 11, 99]

XGB_PARAMS = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.02,
                  max_depth=5, min_child_weight=20, subsample=0.8, colsample_bytree=0.5,
                  reg_lambda=5, n_estimators=10000, tree_method="hist", device="cuda",
                  max_bin=256)
XGBL_PARAMS = dict(XGB_PARAMS, grow_policy="lossguide", max_depth=0, max_leaves=31,
                   min_child_weight=10)
CAT_PARAMS = dict(iterations=10000, learning_rate=0.03, depth=6, l2_leaf_reg=5,
                  loss_function="Logloss", eval_metric="AUC", task_type="GPU", devices="0",
                  border_count=254, metric_period=50)


def load_params(name, feats="all"):
    """Optuna-tuned params (tune.py) override the defaults when present. Params tuned on
    the selected feature set are preferred for --feats selected."""
    base = {"xgb": XGB_PARAMS, "xgbl": XGBL_PARAMS, "cat": CAT_PARAMS}[name]
    path = os.path.join(CACHE, f"best_params_{name}.json")
    sel_path = os.path.join(CACHE, f"best_params_{name}_sel.json")
    if feats == "selected" and os.path.exists(sel_path):
        path = sel_path
    if os.path.exists(path):
        with open(path) as f:
            tuned = json.load(f)
        print(f"using tuned params from {path}: {tuned}")
        return dict(base, **tuned)
    return base


def xgb_fit_predict(Xtr, ytr, Xva, Xte, seed, params):
    Xa, Xb, ya, yb = train_test_split(Xtr, ytr, test_size=0.1, stratify=ytr, random_state=seed)
    m = xgb.XGBClassifier(**params, random_state=seed, early_stopping_rounds=300)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    n = max(100, int(m.best_iteration * 1.1))
    m = xgb.XGBClassifier(**dict(params, n_estimators=n), random_state=seed).fit(Xtr, ytr, verbose=False)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], n


def cat_fit_predict(Xtr, ytr, Xva, Xte, seed, params, cat_idx):
    Xa, Xb, ya, yb = train_test_split(Xtr, ytr, test_size=0.1, stratify=ytr, random_state=seed)
    m = CatBoostClassifier(**params, random_seed=seed, early_stopping_rounds=300)
    m.fit(Pool(Xa, ya, cat_features=cat_idx), eval_set=Pool(Xb, yb, cat_features=cat_idx), verbose=0)
    n = max(100, int(m.get_best_iteration() * 1.1))
    m = CatBoostClassifier(**dict(params, iterations=n), random_seed=seed)
    m.fit(Pool(Xtr, ytr, cat_features=cat_idx), verbose=0)
    return (m.predict_proba(Pool(Xva, cat_features=cat_idx))[:, 1],
            m.predict_proba(Pool(Xte, cat_features=cat_idx))[:, 1], n)


def prepare(name, df, cols):
    tr = df[df.is_train == 1].reset_index(drop=True)
    te = df[df.is_train == 0].reset_index(drop=True)
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    cat_idx = None
    if name == "cat":
        cat_cols = [c for c in cols if c in CAT_COLS or (c.endswith("_app") and c[:-4] in CAT_COLS)]
        for c in cat_cols:
            Xtr[c] = Xtr[c].fillna(-1).astype(int).astype(str)
            Xte[c] = Xte[c].fillna(-1).astype(int).astype(str)
        cat_idx = [cols.index(c) for c in cat_cols]
    return tr, te, Xtr, Xte, cat_idx


def select_columns(df, feats):
    cols = feature_columns(df)
    if feats == "selected":
        with open(os.path.join(CACHE, "selected_features.json")) as f:
            keep = set(json.load(f))
        cols = [c for c in cols if c in keep]
    print(f"using {len(cols)} features ({feats})")
    return cols


def add_extra(df, cols, extra):
    """Merge extra feature sets saved by other scripts (e.g. forecast.py) by ID."""
    import pickle
    for name in extra or []:
        with open(os.path.join(CACHE, f"{name}_features.pkl"), "rb") as f:
            ex = pickle.load(f)
        new = [c for c in ex.columns if c != "ID"]
        df = df.merge(ex, on="ID", how="left")
        cols = cols + new
        print(f"added {len(new)} '{name}' features")
    return df, cols


def run(name, repeats, feats="all", tag=None, extra=None):
    df, _ = get_features()
    cols = select_columns(df, feats)
    df, cols = add_extra(df, cols, extra)
    tr, te, Xtr_all, Xte, cat_idx = prepare(name, df, cols)
    y = tr[TARGET].values.astype(int)
    params = load_params(name, feats)

    oof_all = np.zeros((repeats, len(tr)))
    test_pred = np.zeros(len(te))
    aucs = []
    for r in range(repeats):
        seed = SEEDS[r]
        for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr_all, y)):
            Xa, Xb = Xtr_all.iloc[a], Xtr_all.iloc[b]
            if name == "cat":
                pv, pt, n = cat_fit_predict(Xa, y[a], Xb, Xte, seed + f, params, cat_idx)
            else:
                pv, pt, n = xgb_fit_predict(Xa, y[a], Xb, Xte, seed + f, params)
            oof_all[r, b] = pv
            test_pred += pd.Series(pt).rank(pct=True).values / (5 * repeats)
            print(f"  [{name}] repeat {r} fold {f}: AUC {roc_auc_score(y[b], pv):.5f}  rounds {n}",
                  flush=True)
        aucs.append(roc_auc_score(y, oof_all[r]))
        print(f"[{name}] repeat {r} OOF AUC {aucs[-1]:.5f}", flush=True)

    # average ranks across repeats -> one OOF vector aligned with train rows
    oof = np.mean([pd.Series(o).rank(pct=True).values for o in oof_all], axis=0)
    print(f"[{name}] mean per-repeat AUC {np.mean(aucs):.5f} +- {np.std(aucs):.5f} | "
          f"repeat-averaged OOF AUC {roc_auc_score(y, oof):.5f}")
    out = os.path.join(PRED_DIR, f"{tag or name}.npz")
    np.savez(out, oof=oof, oof_all=oof_all, test=test_pred, aucs=np.array(aucs),
             test_id=te["ID"].values, train_id=tr["ID"].values)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["xgb", "xgbl", "cat"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--feats", choices=["all", "selected"], default="all")
    ap.add_argument("--tag", default=None, help="output name in preds/ (default: model name)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra feature sets from cache/<name>_features.pkl, e.g. forecast")
    args = ap.parse_args()
    run(args.model, args.repeats, args.feats, args.tag, args.extra)
