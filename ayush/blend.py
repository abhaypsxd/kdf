"""
Blend saved OOF/test predictions with rank-averaging and writes submission.csv.

Weights are non-negative and fitted by maximising OOF AUC. To keep the reported
score honest, a nested estimate is also printed: weights fitted on 4/5 of the OOF
rows and scored on the held-out 1/5.

Usage: python blend.py [model names ...]   (default: every preds/*.npz)
"""

import glob
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import DATA_DIR, TARGET
from train_gbm import PRED_DIR, HERE


def rank01(x):
    return (rankdata(x) - 1) / (len(x) - 1)


def fit_weights(P, y):
    k = P.shape[1]

    def loss(w):
        w = np.abs(w)
        return -roc_auc_score(y, P @ (w / w.sum()))

    best = None
    for start in [np.ones(k)] + [np.random.RandomState(s).rand(k) for s in range(4)]:
        res = minimize(loss, start, method="Nelder-Mead", options={"maxiter": 600, "xatol": 1e-4})
        if best is None or res.fun < best.fun:
            best = res
    w = np.abs(best.x)
    return w / w.sum()


def main(names):
    files = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(PRED_DIR, "*.npz"))}
    names = names or sorted(files)
    preds = {n: np.load(files[n]) for n in names}
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), usecols=["ID", TARGET])
    ref = preds[names[0]]
    y = train.set_index("ID").loc[ref["train_id"], TARGET].values
    for n in names:
        assert (preds[n]["train_id"] == ref["train_id"]).all()
        assert (preds[n]["test_id"] == ref["test_id"]).all()

    P = np.column_stack([rank01(preds[n]["oof"]) for n in names])
    Pt = np.column_stack([rank01(preds[n]["test"]) for n in names])
    print("Single models (repeat-averaged OOF AUC):")
    for i, n in enumerate(names):
        print(f"  {n:12s} {roc_auc_score(y, P[:, i]):.5f}   per-repeat {np.round(preds[n]['aucs'], 5)}")
    print("Correlation of OOF ranks:")
    print(pd.DataFrame(P, columns=names).corr().round(3).to_string())

    eq = P.mean(1)
    print(f"Equal-weight rank average OOF AUC: {roc_auc_score(y, eq):.5f}")
    w = fit_weights(P, y) if len(names) > 1 else np.ones(1)
    print("Optimised weights:", dict(zip(names, np.round(w, 3))))
    print(f"Optimised-weight OOF AUC (in-sample): {roc_auc_score(y, P @ w):.5f}")

    nested = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=123).split(P, y):
        wf = fit_weights(P[a], y[a]) if len(names) > 1 else np.ones(1)
        nested[b] = P[b] @ wf
    nested_auc = roc_auc_score(y, nested)
    print(f"Optimised-weight OOF AUC (nested, honest): {nested_auc:.5f}")

    final = Pt @ w
    sub = pd.DataFrame({"ID": ref["test_id"], "24-Month Default Flag": final})
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    sub = sample[["ID"]].merge(sub, on="ID", how="left")
    assert sub["24-Month Default Flag"].notna().all() and len(sub) == len(sample)
    sub.to_csv(os.path.join(HERE, "submission.csv"), index=False)
    pd.DataFrame({"ID": ref["test_id"], "24-Month Default Flag": Pt.mean(1)}).pipe(
        lambda d: sample[["ID"]].merge(d, on="ID")).to_csv(
        os.path.join(HERE, "submission_equal_weight.csv"), index=False)
    rows = [(n, roc_auc_score(y, P[:, i]), w[i]) for i, n in enumerate(names)]
    rows += [("equal_rank_avg", roc_auc_score(y, eq), np.nan),
             ("weighted_blend_nested", nested_auc, np.nan)]
    pd.DataFrame(rows, columns=["Model", "OOF_AUC", "Weight"]).assign(
        Gini=lambda d: 2 * d.OOF_AUC - 1).to_csv(os.path.join(HERE, "validation_metrics.csv"),
                                                 index=False)
    print(f"Wrote submission.csv ({len(sub)} rows), submission_equal_weight.csv, validation_metrics.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
