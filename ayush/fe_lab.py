"""
Feature-engineering lab: test new feature families against the current best feature set
(129 selected + 12 forecast) with the same model, folds and seeds, and report the change
in overall OOF AUC and in AUC *within clean payers* (no missed window in months 6-12;
41% of all defaulters, the weakest-ranked segment).

Families
  te      out-of-fold target encoding of categoricals and their 2-way combinations
          (fitted inside every outer fold, so the validation fold never leaks)
  peer    value relative to peer groups (occupation x age decile, employment x company
          size, industry x age decile): income, balance, limit, draws, repayments, ...
  freq    frequency of each categorical value / combination among all 205k applicants
  dev     recent behaviour vs the customer's own history: last window vs earlier
          windows for pay/due ratio, draws, balance, utilisation; per-window trends
  shape   k-means clusters of the normalised balance path and the pay/due path;
          distance to every centroid
  app2    application-time ratios (limit, debt, rent, dependants vs income, ...)

Unsupervised families are saved to cache/<family>_features.pkl so a winner can be used
directly:  python train_gbm.py xgbl --feats selected --extra forecast peer ...

Usage:  python fe_lab.py                     # base + every family
        python fe_lab.py dev shape           # base + only these families
        python fe_lab.py --combo dev peer    # base + all listed families together
        python fe_lab.py --fair ...          # scale colsample_bytree so every run samples
                                             # the same number of features per tree as base
                                             # (otherwise extra features dilute strong ones)
"""

import json
import os
import pickle
import sys
import time
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from features import BAL_COLS, CACHE, CAT_COLS, TARGET, get_features, load_raw
from train_gbm import SEEDS, add_extra, load_params, select_columns

warnings.filterwarnings("ignore")
FAMILIES = ["te", "peer", "freq", "dev", "shape", "app2"]
REPEATS = 2


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def age_decile(df):
    """5-year age band (fixed edges, so customers.csv and train/test bands match)."""
    return (df["Age (in months)"] // 60).clip(upper=11)


def combos(df):
    """Categorical columns and 2-way combinations used by te / freq."""
    a = age_decile(df).astype(int).astype(str)
    s = lambda c: df[c].astype(int).astype(str)
    out = {c: s(c) for c in ["Occupation Category", "Industry Category", "Employment Type Category",
                             "Company Size Category", "Business Entity Type", "Residence Type",
                             "Application Channel", "Insurance Type", "Residence Ownership Status"]}
    out.update({
        "occ_x_emp": s("Occupation Category") + "_" + s("Employment Type Category"),
        "occ_x_size": s("Occupation Category") + "_" + s("Company Size Category"),
        "ind_x_size": s("Industry Category") + "_" + s("Company Size Category"),
        "occ_x_age": s("Occupation Category") + "_" + a,
        "chan_x_age": s("Application Channel") + "_" + a,
        "res_x_own": s("Residence Type") + "_" + s("Residence Ownership Status"),
        "occ_x_gender": s("Occupation Category") + "_" + s("Gender"),
        "emp_x_ins": s("Employment Type Category") + "_" + s("Insurance Type"),
        "ind_x_occ": s("Industry Category") + "_" + s("Occupation Category"),
    })
    return pd.DataFrame(out, index=df.index)


# ----------------------------------------------------------------------------
# Unsupervised families (computed once on train+test)
# ----------------------------------------------------------------------------
def fam_peer(df):
    a = age_decile(df).astype(int).astype(str)
    s = lambda c: df[c].astype(int).astype(str)
    groups = {
        "occage": s("Occupation Category") + "_" + a,
        "empsize": s("Employment Type Category") + "_" + s("Company Size Category"),
        "indage": s("Industry Category") + "_" + a,
    }
    values = {
        "inc": df["Total Annual Income"].clip(lower=0),
        "bal": df["Outstanding Loan Balance"],
        "lim": df["Desired Credit Limit"],
        "borrow": df["borrow_6"],
        "repay": df["repay_6"],
        "util": df["util_guess"],
        "debt": df["debt_total"],
        "work": df["Work Experience (months)"].clip(lower=0),
        "dbor": df["Declared Borrowing Amount"].clip(lower=0),
    }
    F = {}
    for gname, g in groups.items():
        for vname, v in values.items():
            med = v.groupby(g).transform("median")
            F[f"peer_{vname}_{gname}"] = v / (med.abs() + 1)
            if vname in ("inc", "bal", "util", "debt"):
                F[f"peerrank_{vname}_{gname}"] = v.groupby(g).rank(pct=True)
        F[f"peer_n_{gname}"] = g.map(g.value_counts())
    return pd.DataFrame(F)


def fam_freq(df, customers):
    cc = combos(customers)
    cd = combos(df)
    F = {}
    for c in cd.columns:
        counts = cc[c].value_counts()
        F[f"freq_{c}"] = cd[c].map(counts).fillna(0).values
    return pd.DataFrame(F)


def fam_dev(df):
    n = len(df)
    pr = np.column_stack([np.minimum(df[f"ra_m{k}"] / np.maximum(df[f"due_m{k}"], 1), 5) *
                          (df[f"due_m{k}"] > 0) for k in range(1, 7)])
    ba = np.column_stack([df[f"ba_m{k}"] for k in range(1, 7)]).astype(float)
    ra = np.column_stack([df[f"ra_m{k}"] for k in range(1, 7)]).astype(float)
    bc = np.column_stack([df[f"bc_m{k}"] for k in range(1, 7)]).astype(float)
    rc = np.column_stack([df[f"rc_m{k}"] for k in range(1, 7)]).astype(float)
    B = df[BAL_COLS].values.astype(float)
    lim = df["lim_guess"].values
    util = B / lim[:, None]                              # (n, 7), col 0 = month 12
    t = np.arange(6)[::-1].astype(float)                 # window 1 is the latest -> t = 5
    tc = (t - t.mean()) / ((t - t.mean()) ** 2).sum()
    F = {}
    for name, M in [("pr", pr), ("ba", ba / lim[:, None]), ("bc", bc), ("rc", rc)]:
        hist = M[:, 1:]
        F[f"dev_{name}_last_minus_hist"] = M[:, 0] - hist.mean(1)
        F[f"dev_{name}_last_z"] = (M[:, 0] - hist.mean(1)) / (hist.std(1) + 0.05)
        F[f"dev_{name}_recent2_vs_old3"] = M[:, :2].mean(1) - M[:, 3:].mean(1)
        F[f"dev_{name}_slope"] = M @ tc
        F[f"dev_{name}_max_hist"] = hist.max(1)
    F["dev_pr_min_recent2"] = pr[:, :2].min(1)
    F["dev_pr_drop_from_best"] = pr.max(1) - pr[:, 0]
    F["dev_util_slope"] = util @ (np.arange(7)[::-1] - 3.0) / 28.0
    F["dev_util_last_minus_mean"] = util[:, 0] - util[:, 1:].mean(1)
    F["dev_util_max_jump"] = np.max(util[:, :-1] - util[:, 1:], axis=1)
    F["dev_new_high_now"] = (B[:, 0] >= B[:, 1:].max(1)).astype(int)
    F["dev_n_new_highs"] = sum((B[:, k] > B[:, k + 1:].max(1)).astype(int) for k in range(6))
    F["dev_draw_after_pay"] = ((ba[:, :-1] > 0) & (rc[:, 1:] > 0)).sum(1)  # pays, then draws next window
    F["dev_cycle"] = np.mean(np.minimum(ba, ra) / (np.maximum(ba, ra) + 1), axis=1)  # borrow ~ repay
    F["dev_borrow_months_recent3"] = (ba[:, :3] > 0).sum(1)
    F["dev_borrow_months_old3"] = (ba[:, 3:] > 0).sum(1)
    return pd.DataFrame(F)


def fam_shape(df, k_bal=16, k_pr=12):
    B = df[BAL_COLS].values.astype(float)[:, ::-1]       # oldest -> newest
    path = B / (B.max(1, keepdims=True) + 1)
    pr = np.column_stack([np.minimum(df[f"ra_m{k}"] / np.maximum(df[f"due_m{k}"], 1), 5) *
                          (df[f"due_m{k}"] > 0) for k in range(6, 0, -1)]) / 5
    F = {}
    for name, X, k in [("bal", path, k_bal), ("pr", pr, k_pr)]:
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(X)
        d = km.transform(X)
        F[f"shape_{name}_cluster"] = km.labels_
        for j in range(k):
            F[f"shape_{name}_d{j}"] = d[:, j]
        F[f"shape_{name}_dmin"] = d.min(1)
    return pd.DataFrame(F)


def fam_app2(df):
    inc = df["Declared Annual Income"].clip(lower=0) + 1
    tinc = df["Total Annual Income"].clip(lower=0) + 1
    dbor = df["Declared Borrowing Amount"].clip(lower=0)
    dep = df["Number of Dependents"]
    F = {
        "app2_limit_to_inc": df["Desired Credit Limit"] / inc,
        "app2_debt_plus_limit_to_inc": (dbor + df["Desired Credit Limit"]) / inc,
        "app2_avg_debt_per_loan": dbor / (df["Declared Number of Borrowings"] + 1),
        "app2_inc_per_member": tinc / (dep + 1),
        "app2_rent_per_member": df["Rent Burden"].clip(lower=0) / (dep + 1),
        "app2_disposable": tinc / 12 - df["Rent Burden"].clip(lower=0) - df["min_due_now"],
        "app2_disposable_per_member": (tinc / 12 - df["Rent Burden"].clip(lower=0)) / (dep + 1),
        "app2_age_at_start_job": df["Age (in months)"] + 12 - df["Work Experience (months)"].clip(lower=0),
        "app2_job_since_loan": (df["Work Experience (months)"] < 12).astype(int),
        "app2_moved_recently": (df["Residence Period (in months)"] < 12).astype(int),
        "app2_n_contacts": df[["Home Phone Number Available", "Mobile Phone Number Available",
                               "Employer Phone Number Available", "Driver Licence Number Available"]].sum(1),
        "app2_loans_x_limit": df["Declared Number of Borrowings"] * df["Desired Credit Limit"] / inc,
        "app2_bal_to_declared_debt": df["Outstanding Loan Balance"] / (dbor + 1e4),
    }
    return pd.DataFrame(F)


# ----------------------------------------------------------------------------
# Target encoding, fitted inside each outer fold
# ----------------------------------------------------------------------------
def te_encode(cats_tr, y_tr, cats_apply, m=100, inner=5, seed=0):
    """Smoothed mean encoding. Training rows get inner-fold OOF values; applied rows get the
    mapping fitted on all training rows."""
    prior = y_tr.mean()
    out_tr = np.zeros((len(cats_tr), cats_tr.shape[1]))
    out_ap = np.zeros((len(cats_apply), cats_apply.shape[1]))

    def mapping(c, yy):
        g = pd.DataFrame({"c": c, "y": yy}).groupby("c")["y"].agg(["sum", "count"])
        return (g["sum"] + prior * m) / (g["count"] + m)

    for j, col in enumerate(cats_tr.columns):
        ctr = cats_tr[col].values
        for a, b in StratifiedKFold(inner, shuffle=True, random_state=seed).split(ctr, y_tr):
            out_tr[b, j] = pd.Series(ctr[b]).map(mapping(ctr[a], y_tr[a])).fillna(prior).values
        out_ap[:, j] = pd.Series(cats_apply[col].values).map(mapping(ctr, y_tr)).fillna(prior).values
    names = [f"te_{c}" for c in cats_tr.columns]
    return pd.DataFrame(out_tr, columns=names), pd.DataFrame(out_ap, columns=names)


# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------
def cv_score(X, y, clean, params, cats=None):
    """Mean per-repeat OOF AUC overall and within clean payers."""
    overall, within = [], []
    for r in range(REPEATS):
        seed = SEEDS[r]
        oof = np.zeros(len(y))
        for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
            Xa, Xb = X.iloc[a].reset_index(drop=True), X.iloc[b].reset_index(drop=True)
            if cats is not None:
                ta, tb = te_encode(cats.iloc[a].reset_index(drop=True), y[a],
                                   cats.iloc[b].reset_index(drop=True), seed=seed + f)
                Xa, Xb = pd.concat([Xa, ta], axis=1), pd.concat([Xb, tb], axis=1)
            ia, ih = train_test_split(np.arange(len(a)), test_size=0.1, stratify=y[a],
                                      random_state=seed + f)
            m = xgb.XGBClassifier(**params, random_state=seed + f, early_stopping_rounds=200)
            m.fit(Xa.iloc[ia], y[a][ia], eval_set=[(Xa.iloc[ih], y[a][ih])], verbose=False)
            oof[b] = pd.Series(m.predict_proba(Xb)[:, 1]).rank(pct=True).values
        overall.append(roc_auc_score(y, oof))
        within.append(roc_auc_score(y[clean], oof[clean]))
    return np.array(overall), np.array(within)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    combo = "--combo" in sys.argv
    fair = "--fair" in sys.argv
    fams = args or FAMILIES

    df, _ = get_features()
    cols = select_columns(df, "selected")
    df, cols = add_extra(df, cols, ["forecast"])
    customers, _, _ = load_raw()

    t0 = time.time()
    built = {}
    for fam in fams:
        if fam == "te":
            continue
        path = os.path.join(CACHE, f"{fam}_features.pkl")
        F = {"peer": lambda: fam_peer(df), "freq": lambda: fam_freq(df, customers),
             "dev": lambda: fam_dev(df), "shape": lambda: fam_shape(df),
             "app2": lambda: fam_app2(df)}[fam]()
        F = F.replace([np.inf, -np.inf], np.nan)
        F.insert(0, "ID", df["ID"].values)
        with open(path, "wb") as f:
            pickle.dump(F, f)
        built[fam] = F
        print(f"built {fam}: {F.shape[1] - 1} features -> {path} ({time.time() - t0:.0f}s)", flush=True)

    is_tr = (df["is_train"] == 1).values
    tr = df[is_tr].reset_index(drop=True)
    y = tr[TARGET].values.astype(int)
    clean = (tr["n_miss"] == 0).values
    params = dict(load_params("xgbl", "selected"), learning_rate=0.05, n_estimators=5000)
    cats_all = combos(tr)

    def run(name, extra_fams):
        X = tr[cols].copy()
        for fam in extra_fams:
            if fam != "te":
                X = pd.concat([X, built[fam][is_tr].drop(columns="ID").reset_index(drop=True)], axis=1)
        cats = cats_all if "te" in extra_fams else None
        n_feats = X.shape[1] + (cats.shape[1] if cats is not None else 0)
        p = params
        if fair:  # keep the expected number of features per tree equal to the base run
            p = dict(params, colsample_bytree=params["colsample_bytree"] * len(cols) / n_feats)
        t = time.time()
        o, w = cv_score(X, y, clean, p, cats)
        print(f"{name:22s} overall {o.mean():.5f} {np.round(o, 5)}  clean {w.mean():.5f}  "
              f"({X.shape[1] + (cats.shape[1] if cats is not None else 0)} feats, {time.time() - t:.0f}s)",
              flush=True)
        return o, w

    print(f"\n{REPEATS} repeats x 5 folds, xgbl (tuned params, lr 0.05), base = 129 selected + 12 forecast"
          + (" | fair: colsample scaled to base feature count" if fair else ""))
    base_o, base_w = run("base", [])
    rows = [("base", base_o.mean(), 0.0, 0.0, base_w.mean(), 0.0)]
    runs = [("+".join(fams), fams)] if combo else [(f, [f]) for f in fams]
    for name, ef in runs:
        o, w = run(f"+ {name}", ef)
        d = o - base_o
        rows.append((name, o.mean(), d.mean(), (d > 0).mean(), w.mean(), (w - base_w).mean()))

    rows = [(r[0] + (" (fair)" if fair and r[0] != "base" else ""),) + tuple(r[1:]) for r in rows]
    res = pd.DataFrame(rows, columns=["family", "overall_auc", "delta_overall", "share_repeats_up",
                                      "clean_auc", "delta_clean"])
    print("\n" + res.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    out = os.path.join(CACHE, "fe_lab_results.csv")
    res.to_csv(out, index=False, mode="a", header=not os.path.exists(out))
    print(f"appended to {out}")


if __name__ == "__main__":
    main()
