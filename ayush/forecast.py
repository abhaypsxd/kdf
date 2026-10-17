"""
Next-month behaviour forecaster (self-supervised, no target labels used).

Default = 3 consecutive missed payments in months 13-24, so the most relevant unseen
quantity is how each account behaves in month 13. We cannot observe it, but we can learn
month-to-month dynamics from the windows we do have, on ALL 125k accounts (train+test):

  training rows : history windows t+1..t+4  ->  outcome in window t,   t in {2, 1}
  application   : history windows 1..4      ->  forecast for window 0 (= month 13)

Window 1 is the most recent (ends at month 12). Arrears are recomputed inside the
4-window history so every row uses the same definition. Accounts are split into 5 folds:
an account's forecasts always come from models that never saw that account.

Outputs (cache/forecast_features.pkl, keyed by ID):
  fc_*  forecasts for month 13 (miss prob, borrow prob, arrears, pay/due ratio, balance change)
  sp_*  "surprise" in month 12: actual outcome minus the out-of-fold forecast for window 1

Usage:  python forecast.py
Then:   python train_gbm.py xgbl --feats selected --extra forecast --tag xgbl_sel_fc
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, r2_score

from features import BAL_COLS, CACHE, fit_min_schedule, min_due, get_features

warnings.filterwarnings("ignore")
H = 4  # history length in windows
STATIC = ["Auto-Debit Flag", "Income Proof Submitted", "Total Annual Income", "Desired Credit Limit",
          "Age (in months)", "Payday", "app_dom", "dom_boundary", "payday_gap", "lim_guess",
          "app_p_def12", "Number of Loan Settlements"]
BIN_TARGETS = ["miss", "borrow"]
REG_TARGETS = ["arrears", "payratio", "balchg"]


def window_arrays(df):
    """Per-window arrays indexed by window number 1..6 (index 0 unused for flows)."""
    n = len(df)
    W = {key: np.zeros((n, 7)) for key in ("rc", "ra", "bc", "ba", "due")}
    for k in range(1, 7):
        for key in ("rc", "ra", "bc", "ba", "due"):
            W[key][:, k] = df[f"{key}_m{k}"].values
    B = df[BAL_COLS].values.astype(float)  # B[:, k] = month-end balance before window k
    edges, sched = fit_min_schedule(df)
    W["due"][:, 0] = min_due(B[:, 0], edges, sched)  # minimum due for month 13
    return W, B


def history_features(W, B, df, t):
    """Features from windows t+1..t+H (j=1 is the window right before target t)."""
    X = {}
    carry = np.zeros(len(df))
    for j in range(H, 0, -1):  # oldest -> newest, arrears restarted inside the history
        w = t + j
        owed = (W["due"][:, w] > 0).astype(float)
        carry = np.maximum(0, carry + owed - W["rc"][:, w])
        X[f"h{j}_rc"] = W["rc"][:, w]
        X[f"h{j}_ra"] = W["ra"][:, w]
        X[f"h{j}_bc"] = W["bc"][:, w]
        X[f"h{j}_ba"] = W["ba"][:, w]
        X[f"h{j}_due"] = W["due"][:, w]
        X[f"h{j}_payratio"] = W["ra"][:, w] / np.maximum(W["due"][:, w], 1)
        X[f"h{j}_bal_open"] = B[:, w]
        X[f"h{j}_arrears"] = carry.copy()
    X["arrears_before"] = carry
    X["bal_open_target"] = B[:, t]
    X["due_target"] = W["due"][:, t]
    X["bal_trend"] = (B[:, t] - B[:, t + H]) / (B[:, t + H] + 1e4)
    X["target_cal_month"] = (df["app_month"].values + 12 - t - 1) % 12  # calendar month of window t
    for c in STATIC:
        X[c] = df[c].values
    return pd.DataFrame(X), carry


def outcomes(W, B, t, carry_before):
    owed = (W["due"][:, t] > 0).astype(float)
    return {
        "miss": ((W["rc"][:, t] == 0) & (owed > 0)).astype(int),
        "borrow": (W["ba"][:, t] > 0).astype(int),
        "arrears": np.minimum(np.maximum(0, carry_before + owed - W["rc"][:, t]), 5),
        "payratio": np.clip(W["ra"][:, t] / np.maximum(W["due"][:, t], 1), 0, 5),
        "balchg": np.clip((B[:, t - 1] - B[:, t]) / (B[:, t] + 1e4), -1, 2),
    }


def model(target):
    common = dict(n_estimators=600, learning_rate=0.05, max_depth=7, min_child_weight=20,
                  subsample=0.8, colsample_bytree=0.7, reg_lambda=5, tree_method="hist",
                  device="cuda", random_state=0)
    if target in BIN_TARGETS:
        return xgb.XGBClassifier(objective="binary:logistic", **common)
    return xgb.XGBRegressor(objective="reg:squarederror", **common)


def main():
    df, _ = get_features()
    df = df.reset_index(drop=True)
    W, B = window_arrays(df)
    n = len(df)

    rows = {}
    for t in (2, 1):
        X, carry = history_features(W, B, df, t)
        rows[t] = (X, outcomes(W, B, t, carry))
    X0, _ = history_features(W, B, df, 0)

    fc = {f"fc_{c}": np.zeros(n) for c in BIN_TARGETS + REG_TARGETS}
    oof1 = {c: np.zeros(n) for c in BIN_TARGETS + REG_TARGETS}
    for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(n))):
        Xtr = pd.concat([rows[2][0].iloc[a], rows[1][0].iloc[a]], ignore_index=True)
        for c in BIN_TARGETS + REG_TARGETS:
            ytr = np.concatenate([rows[2][1][c][a], rows[1][1][c][a]])
            m = model(c).fit(Xtr, ytr)
            pred = (lambda X: m.predict_proba(X)[:, 1]) if c in BIN_TARGETS else m.predict
            fc[f"fc_{c}"][b] = pred(X0.iloc[b])
            oof1[c][b] = pred(rows[1][0].iloc[b])
        print(f"fold {f} done", flush=True)

    y1 = rows[1][1]
    print("Forecaster quality on window 1 (out-of-fold, all accounts):")
    for c in BIN_TARGETS:
        print(f"  {c:9s} AUC {roc_auc_score(y1[c], oof1[c]):.4f}")
    for c in REG_TARGETS:
        print(f"  {c:9s} R2  {r2_score(y1[c], oof1[c]):.4f}")

    out = pd.DataFrame({"ID": df["ID"].values, **fc})
    for c in BIN_TARGETS + REG_TARGETS:
        out[f"sp_{c}"] = y1[c] - oof1[c]  # month-12 behaviour vs what was expected
    # chance of three misses in a row starting at month 13, if misses were independent
    out["fc_miss_cube"] = out["fc_miss"] ** 3
    out["fc_miss_x_arrears"] = out["fc_miss"] * (1 + out["fc_arrears"])

    target = "24-Month Default Flag"
    tr = df["is_train"].values == 1
    print("Univariate AUC of new features vs 24-month default (train rows):")
    for c in out.columns[1:]:
        v = out.loc[tr, c].values
        a = roc_auc_score(df.loc[tr, target], v)
        print(f"  {c:20s} {max(a, 1 - a):.4f}")

    path = os.path.join(CACHE, "forecast_features.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"saved {path} ({out.shape[1] - 1} features)")


if __name__ == "__main__":
    main()
