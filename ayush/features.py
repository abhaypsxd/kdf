"""
Feature engineering for the AIFUL 24-month behavioural default model.

Feature groups (each can be toggled for ablation):
  base      - raw columns + per-month flows, balance trajectory, cash-flow ratios
  delinq    - month-by-month payment state machine (miss streaks, cures, patterns)
  minpay    - compliance with AIFUL's minimum repayment schedule (inferred from data)
  util      - credit-limit proxy, headroom, implied interest / principal paydown
  date      - application date decoded from ID (day, weekday, day-of-month, ...)
  app       - transfer scores from customers.csv (12m default / early-closure models)
  ratio     - affordability ratios against income, rent, declared debt
"""

import os
import numpy as np
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "dataset")
CACHE = os.path.join(HERE, "cache")
TARGET = "24-Month Default Flag"
TIME_VARYING = ["Marital Status", "Insurance Type", "Company Size Category"]
CAT_COLS = [
    "Application Channel", "Occupation Category", "Industry Category",
    "Employment Type Category", "Company Size Category", "Business Entity Type",
    "Residence Type", "Insurance Type", "Residence Ownership Status",
]
FLOWS = {
    "ba": "Additional Borrowing Amount",
    "bc": "Additional Borrowing Count",
    "ra": "Repayment Amount",
    "rc": "Repayment Count",
}
BAL_COLS = ["Outstanding Loan Balance"] + [
    f"Balance {k} Month{'s' if k > 1 else ''} Ago" for k in range(1, 7)
]


def _cum_col(prefix, k):
    return f"{prefix} (Past {k} Month{'s' if k > 1 else ''})"


def load_raw():
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return customers, train, test


# ----------------------------------------------------------------------------
# Transfer features from customers.csv
# ----------------------------------------------------------------------------
def _app_matrix(customers):
    """Application-time attributes plus a few ratios trees cannot form on their own."""
    app_cols = [c for c in customers.columns if c not in ("ID", "12-Month Default Flag")]
    X = customers[app_cols].copy()
    day = customers["ID"] // 10000 - 21185
    dt = pd.Timestamp("2018-01-01") + pd.to_timedelta(day, "D")
    X["day"] = day.values
    X["dow"] = dt.dt.dayofweek.values
    X["dom"] = dt.dt.day.values
    inc = X["Declared Annual Income"].clip(lower=0) + 1
    X["limit_to_inc"] = X["Desired Credit Limit"] / inc
    X["debt_to_inc"] = X["Declared Borrowing Amount"].clip(lower=0) / inc
    X["rent_to_inc"] = X["Rent Burden"].clip(lower=0) * 12 / inc
    X["debt_plus_limit_to_inc"] = (X["Declared Borrowing Amount"].clip(lower=0)
                                   + X["Desired Credit Limit"]) / inc
    X["resid_to_age"] = X["Residence Period (in months)"] / X["Age (in months)"]
    X["avg_debt_per_loan"] = X["Declared Borrowing Amount"] / (X["Declared Number of Borrowings"] + 1)
    return X


def application_scores(customers, pop_ids, seed=0):
    """OOF multiclass models on all 205k applications: active@12 / default@12 / closed@12.

    Uses only application-time attributes, so it learns an application scorecard from
    80k customers that never appear in train/test. Two learners (GPU XGBoost on numeric
    codes, GPU CatBoost on native categoricals), each early-stopped on an inner holdout.
    Returned probabilities are OOF for every customer, so train rows never see their
    own label.
    """
    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import train_test_split

    X = _app_matrix(customers)
    y = np.where(customers["12-Month Default Flag"] == 1, 1,
                 np.where(customers["ID"].isin(pop_ids), 0, 2))
    cat_cols = [c for c in X.columns if c in CAT_COLS]
    Xc = X.copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype(int).astype(str)
    cat_idx = [list(X.columns).index(c) for c in cat_cols]

    xgb_params = dict(objective="multi:softprob", learning_rate=0.03, max_depth=6,
                      min_child_weight=50, subsample=0.8, colsample_bytree=0.6, reg_lambda=5,
                      n_estimators=5000, tree_method="hist", device="cuda",
                      early_stopping_rounds=200, random_state=seed)
    cat_params = dict(loss_function="MultiClass", iterations=5000, learning_rate=0.05, depth=7,
                      l2_leaf_reg=5, task_type="GPU", devices="0", early_stopping_rounds=200,
                      random_seed=seed, verbose=0)
    oof_x, oof_c = np.zeros((len(X), 3)), np.zeros((len(X), 3))
    for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
        ia, ih = train_test_split(a, test_size=0.1, stratify=y[a], random_state=seed + f)
        m = xgb.XGBClassifier(**xgb_params)
        m.fit(X.iloc[ia], y[ia], eval_set=[(X.iloc[ih], y[ih])], verbose=False)
        oof_x[b] = m.predict_proba(X.iloc[b])
        c = CatBoostClassifier(**cat_params)
        c.fit(Pool(Xc.iloc[ia], y[ia], cat_features=cat_idx),
              eval_set=Pool(Xc.iloc[ih], y[ih], cat_features=cat_idx))
        oof_c[b] = c.predict_proba(Pool(Xc.iloc[b], cat_features=cat_idx))
        print(f"  app model fold {f}: xgb rounds {m.best_iteration}, cat rounds {c.get_best_iteration()}",
              flush=True)

    from sklearn.metrics import roc_auc_score
    for name, o in [("xgb", oof_x), ("cat", oof_c)]:
        print(f"  app {name}: AUC def12 {roc_auc_score(y == 1, o[:, 1]):.4f} | "
              f"closed {roc_auc_score(y == 2, o[:, 2]):.4f}")
    oof = (oof_x + oof_c) / 2
    out = pd.DataFrame({"ID": customers["ID"].values})
    for name, o in [("", oof), ("_xgb", oof_x), ("_cat", oof_c)]:
        out[f"app_p_def12{name}"] = o[:, 1]
        out[f"app_p_closed12{name}"] = o[:, 2]
        out[f"app_def_vs_active{name}"] = o[:, 1] / (o[:, 1] + o[:, 0])
    out["app_p_active12"] = oof[:, 0]
    return out


# ----------------------------------------------------------------------------
# Minimum repayment schedule (inferred, unsupervised)
# ----------------------------------------------------------------------------
def fit_min_schedule(df):
    """Learn balance -> minimum monthly repayment from clean single-payment months.

    For months with exactly one repayment and no borrowing, the modal payment within a
    balance bin is the contractual minimum. Enforced monotone non-decreasing.
    """
    edges = np.concatenate([np.arange(0, 1_000_001, 20_000),
                            np.arange(1_050_000, 3_000_001, 50_000), [1e12]])
    bal, pay = [], []
    for k in range(1, 7):
        ra = df[_cum_col(FLOWS["ra"], k)] - (df[_cum_col(FLOWS["ra"], k - 1)] if k > 1 else 0)
        rc = df[_cum_col(FLOWS["rc"], k)] - (df[_cum_col(FLOWS["rc"], k - 1)] if k > 1 else 0)
        bc = df[_cum_col(FLOWS["bc"], k)] - (df[_cum_col(FLOWS["bc"], k - 1)] if k > 1 else 0)
        m = (rc == 1) & (bc == 0)
        bal.append(df[BAL_COLS[k]][m].values)
        pay.append(ra[m].values)
    bal, pay = np.concatenate(bal), np.concatenate(pay)
    b = np.digitize(bal, edges) - 1
    s = pd.Series(pay).groupby(b).agg(lambda v: v.value_counts().idxmax())
    sched = pd.Series(np.nan, index=range(len(edges)))
    sched.loc[s.index] = s.values
    sched = sched.ffill().bfill().cummax().values
    return edges, sched


def min_due(balance, edges, sched):
    b = np.clip(np.digitize(balance, edges) - 1, 0, len(sched) - 1)
    due = sched[b]
    return np.where(balance <= 0, 0, due)


# ----------------------------------------------------------------------------
# Main builder
# ----------------------------------------------------------------------------
def build_features(groups=("base", "delinq", "minpay", "arrears", "util", "limit", "date",
                           "app", "ratio")):
    customers, train, test = load_raw()
    train["is_train"] = 1
    test["is_train"] = 0
    df = pd.concat([train, test], ignore_index=True)

    cust = customers.drop(columns=["12-Month Default Flag"]).rename(
        columns={c: f"{c}_app" for c in TIME_VARYING})
    df = df.merge(cust, on="ID", how="left")
    F = {}  # new features
    groups_of = {}

    def tag(g):
        seen = {c for v in groups_of.values() for c in v}
        groups_of[g] = [c for c in F if c not in seen]

    # per-month flows, k=1 most recent window
    m = {}
    for key, prefix in FLOWS.items():
        cum = np.column_stack([df[_cum_col(prefix, k)].values for k in range(1, 7)]).astype(float)
        m[key] = np.diff(np.column_stack([np.zeros(len(df)), cum]), axis=1)  # [:,0]=month1
    B = df[BAL_COLS].values.astype(float)  # B[:,0]=month-end 12, B[:,6]=month-end 6
    ra, rc, ba, bc = m["ra"], m["rc"], m["ba"], m["bc"]
    # minimum due for window k is set by the month-end balance just before it (B[:, k+1])
    edges, sched = fit_min_schedule(df)
    due = np.column_stack([min_due(B[:, k + 1], edges, sched) for k in range(6)])
    day = (df["ID"] // 10000 - 21185).values
    dom = (pd.Timestamp("2018-01-01") + pd.to_timedelta(day, "D")).day.values

    if "base" in groups:
        for key in FLOWS:
            for k in range(6):
                F[f"{key}_m{k+1}"] = m[key][:, k]
        w = np.array([3, 2, 1, 0, -1, -2, -3]) / 28.0  # B index 0 is latest
        F["bal_slope"] = B @ w
        F["bal_mean"] = B.mean(1)
        F["bal_std"] = B.std(1)
        F["bal_min"] = B.min(1)
        F["bal_max"] = B.max(1)
        F["bal_cv"] = F["bal_std"] / (F["bal_mean"] + 1)
        F["bal_chg_1"] = B[:, 0] - B[:, 1]
        F["bal_chg_3"] = B[:, 0] - B[:, 3]
        F["bal_chg_6"] = B[:, 0] - B[:, 6]
        F["bal_pct_1"] = F["bal_chg_1"] / (B[:, 1] + 1)
        F["bal_pct_3"] = F["bal_chg_3"] / (B[:, 3] + 1)
        F["bal_pct_6"] = F["bal_chg_6"] / (B[:, 6] + 1)
        for k in range(6):
            F[f"bal_diff_{k+1}"] = B[:, k] - B[:, k + 1]
        dB = B[:, :-1] - B[:, 1:]
        F["bal_n_up"] = (dB > 0).sum(1)
        F["bal_n_down"] = (dB < 0).sum(1)
        F["bal_accel"] = (B[:, 0] - B[:, 3]) - (B[:, 3] - B[:, 6])
        F["repay_6"] = ra.sum(1)
        F["borrow_6"] = ba.sum(1)
        F["net_flow_6"] = F["repay_6"] - F["borrow_6"]
        F["net_flow_3"] = ra[:, :3].sum(1) - ba[:, :3].sum(1)
        F["repay_trend"] = ra[:, :3].sum(1) / (ra[:, 3:].sum(1) + 1)
        F["borrow_trend"] = ba[:, :3].sum(1) / (ba[:, 3:].sum(1) + 1)
        F["repay_cv"] = ra.std(1) / (ra.mean(1) + 1)
        F["borrow_cv"] = ba.std(1) / (ba.mean(1) + 1)
        F["repay_to_bal"] = F["repay_6"] / (F["bal_mean"] + 1)
        F["borrow_to_bal"] = F["borrow_6"] / (F["bal_mean"] + 1)
        F["repay_borrow_ratio"] = F["repay_6"] / (F["borrow_6"] + 1)
        F["n_borrow_months"] = (bc > 0).sum(1)
        F["avg_draw"] = F["borrow_6"] / (bc.sum(1) + 1)
        F["avg_pay"] = F["repay_6"] / (rc.sum(1) + 1)
        F["last_borrow_month"] = np.where((bc > 0).any(1), (bc > 0).argmax(1) + 1, 7)
        F["borrow_no_pay_months"] = ((ba > 0) & (rc == 0)).sum(1)
        F["borrow_no_pay_m1"] = ((ba[:, 0] > 0) & (rc[:, 0] == 0)).astype(int)

    tag("base")

    if "delinq" in groups:
        miss = rc == 0
        pays = ~miss
        streak = np.zeros(len(df), int)
        alive = np.ones(len(df), bool)
        for k in range(6):
            alive &= miss[:, k]
            streak += alive
        F["miss_streak_now"] = streak  # consecutive missed windows ending at month 12
        run = np.zeros(len(df), int)
        mx = np.zeros(len(df), int)
        for k in range(6):
            run = np.where(miss[:, k], run + 1, 0)
            mx = np.maximum(mx, run)
        F["miss_streak_max"] = mx
        F["n_miss"] = miss.sum(1)
        F["n_miss_3"] = miss[:, :3].sum(1)
        # cure: missed in older window k+1, paid in newer window k
        F["n_cures"] = (miss[:, 1:] & pays[:, :-1]).sum(1)
        F["n_lapses"] = (pays[:, 1:] & miss[:, :-1]).sum(1)
        F["months_since_pay"] = np.where(pays.any(1), pays.argmax(1), 6)
        bits = (miss * (1 << np.arange(6))).sum(1)
        F["miss_pattern6"] = bits
        F["miss_pattern3"] = bits & 7
        F["multi_pay_months"] = (rc >= 2).sum(1)
        F["rc_total"] = rc.sum(1)
        F["first_pay_window_missed"] = miss[:, 5].astype(int)
        if "Auto-Debit Flag" in df:
            F["autodebit_miss_now"] = ((df["Auto-Debit Flag"].values == 1) & miss[:, 0]).astype(int)
            F["autodebit_n_miss"] = df["Auto-Debit Flag"].values * F["n_miss"]

    tag("delinq")

    if "minpay" in groups:
        ratio = np.where(due > 0, ra / np.maximum(due, 1), np.nan)
        for k in range(3):
            F[f"pay_due_ratio_m{k+1}"] = ratio[:, k]
        F["pay_due_ratio_mean"] = np.nanmean(np.where(due > 0, np.minimum(ratio, 10), np.nan), 1)
        F["pay_due_ratio_min"] = np.nanmin(np.where(due > 0, ratio, np.inf), 1)
        F["pay_due_ratio_min"] = np.where(np.isinf(F["pay_due_ratio_min"]), np.nan, F["pay_due_ratio_min"])
        short = (ra > 0) & (ra < due * 0.95)
        exact = (ra > 0) & (np.abs(ra - due) <= 1000)
        F["n_short_pay"] = short.sum(1)
        F["n_exact_min"] = exact.sum(1)
        F["n_over_2x"] = (ra >= 2 * due).sum(1) * (due > 0).any(1)
        F["shortfall_6"] = np.clip(due - ra, 0, None).sum(1)
        F["shortfall_3"] = np.clip(due - ra, 0, None)[:, :3].sum(1)
        F["min_due_now"] = min_due(B[:, 0], edges, sched)
        F["excess_pay_6"] = (ra - due).sum(1)

    tag("minpay")

    if "arrears" in groups:
        # Windows run from the origination day-of-month to the same day next month, but the
        # due date need not sit inside them: a payment near the boundary can slide into the
        # neighbouring window (0 payments then 2). Empty windows are therefore noisy. Track
        # arrears instead: one payment (and one minimum amount) owed per window with a
        # positive balance; extra payments clear arrears but do not bank credit.
        n = len(df)
        owed = (due > 0).astype(float)
        carry, carry_mx = np.zeros(n), np.zeros(n)
        amt, amt_mx = np.zeros(n), np.zeros(n)
        hist, hist_amt = [], []
        for k in range(5, -1, -1):  # oldest -> newest
            carry = np.maximum(0, carry + owed[:, k] - rc[:, k])
            amt = np.maximum(0, amt + due[:, k] - ra[:, k])
            carry_mx = np.maximum(carry_mx, carry)
            amt_months = amt / np.maximum(due[:, k], 1)
            amt_mx = np.maximum(amt_mx, amt_months)
            hist.append(carry)
            hist_amt.append(amt_months)
        hist = np.column_stack(hist[::-1])          # [:, 0] = newest window
        hist_amt = np.column_stack(hist_amt[::-1])
        for k in range(6):
            F[f"arrears_w{k+1}"] = hist[:, k]
            F[f"arrears_amt_w{k+1}"] = hist_amt[:, k]
            F[f"due_m{k+1}"] = due[:, k]
        F["arrears_now"] = carry
        F["arrears_max"] = carry_mx
        F["arrears_amt_now"] = amt
        F["arrears_amt_months_now"] = hist_amt[:, 0]
        F["arrears_amt_months_max"] = amt_mx
        F["arrears_n_windows"] = (hist > 0).sum(1)
        run = np.zeros(n, int)
        alive = np.ones(n, bool)
        for k in range(6):
            alive &= hist[:, k] > 0
            run += alive
        F["arrears_streak_now"] = run
        # cumulative payment gap over the last k windows (0-then-2 nets to zero)
        rc_cum = np.cumsum(rc, 1)
        owed_cum = np.cumsum(owed, 1)
        ra_cum = np.cumsum(ra, 1)
        due_cum = np.cumsum(due, 1)
        for k in (1, 2, 3, 6):
            F[f"cum_gap_{k}"] = owed_cum[:, k - 1] - rc_cum[:, k - 1]
            F[f"cum_pay_ratio_{k}"] = ra_cum[:, k - 1] / np.maximum(due_cum[:, k - 1], 1)
        # an empty window is "caught up" if a neighbouring window has 2+ payments
        miss = (rc == 0) & (owed > 0)
        nb2 = np.zeros_like(miss)
        nb2[:, 1:] |= rc[:, :-1] >= 2   # newer neighbour
        nb2[:, :-1] |= rc[:, 1:] >= 2   # older neighbour
        F["n_true_miss"] = (miss & ~nb2).sum(1)
        F["n_caught_up_miss"] = (miss & nb2).sum(1)
        F["true_miss_now"] = (miss[:, 0] & (rc[:, 1] < 2)).astype(int)
        F["true_miss_last2"] = (miss[:, 0] & miss[:, 1]).astype(int)
        boundary = ((dom <= 5) | (dom >= 26)).astype(int)
        F["dom_boundary"] = boundary
        F["arrears_now_inner"] = carry * (1 - boundary)  # misses away from the boundary are real
        F["n_true_miss_inner"] = F["n_true_miss"] * (1 - boundary)
        F["pay_after_due_ratio_now"] = ra[:, 0] / np.maximum(due[:, 0], 1)

    tag("arrears")

    if "util" in groups:
        lim_proxy = np.maximum(B.max(1), 1)
        desired = df["Desired Credit Limit"].values.astype(float)
        # credit limits are usually round 100k values; the observed peak is a floor on the limit
        lim_round = np.ceil(lim_proxy / 100_000) * 100_000
        F["limit_round"] = lim_round
        F["util_peak"] = B[:, 0] / lim_proxy
        F["util_round"] = B[:, 0] / lim_round
        F["headroom_round"] = lim_round - B[:, 0]
        F["util_desired"] = B[:, 0] / np.maximum(desired, 1)
        F["peak_over_desired"] = lim_proxy / np.maximum(desired, 1)
        F["at_peak_now"] = (B[:, 0] >= 0.98 * lim_proxy).astype(int)
        F["near_round_limit"] = (B[:, 0] >= 0.95 * lim_round).astype(int)
        # B0 = B6 + borrow - repay + interest  (window alignment is approximate)
        implied_int = B[:, 0] - B[:, 6] - ba.sum(1) + ra.sum(1)
        F["implied_int_6"] = implied_int
        F["implied_rate"] = implied_int / np.maximum(B[:, :6].mean(1), 1) / 6
        F["int_share_of_pay"] = implied_int / (ra.sum(1) + 1)
        F["principal_paid_6"] = ra.sum(1) - implied_int
        F["bal_zero_any"] = (B == 0).any(1).astype(int)
        F["bal_small_now"] = (B[:, 0] < 10_000).astype(int)

    tag("util")

    if "limit" in groups:
        # The credit limit is AIFUL's own risk decision but is not given. It is at least the
        # highest balance seen; limits are set in round amounts, usually near the request.
        desired = df["Desired Credit Limit"].values.astype(float)
        bmax = B.max(1)
        peak_intra = np.max(B[:, 1:] + ba, axis=1)  # opening balance + window draws (upper bound)
        lim_guess = np.maximum(desired, np.ceil(bmax / 10_000) * 10_000)
        lim_guess = np.maximum(lim_guess, 10_000)
        F["lim_guess"] = lim_guess
        F["granted_above_desired"] = (bmax > desired * 1.01).astype(int)
        F["util_guess"] = B[:, 0] / lim_guess
        F["util_guess_max"] = bmax / lim_guess
        F["headroom_guess"] = lim_guess - B[:, 0]
        F["peak_intra_to_lim"] = peak_intra / lim_guess
        F["n_win_near_lim"] = (B >= 0.9 * lim_guess[:, None]).sum(1)
        F["borrow_near_lim"] = ((ba > 0) & (B[:, 1:] >= 0.8 * lim_guess[:, None])).sum(1)
        F["draw_to_headroom_m1"] = ba[:, 0] / (lim_guess - B[:, 1] + 1)
        F["draw_to_headroom_3"] = ba[:, :3].sum(1) / (lim_guess - B[:, 3] + 1)
        F["util_change_6"] = (B[:, 0] - B[:, 6]) / lim_guess
        F["headroom_to_income"] = (lim_guess - B[:, 0]) / (df["Total Annual Income"].clip(lower=0) + 1)

    tag("limit")

    if "date" in groups:
        day = (df["ID"] // 10000 - 21185).values
        dt = pd.Timestamp("2018-01-01") + pd.to_timedelta(day, "D")
        F["app_day"] = day
        F["app_seq"] = (df["ID"] % 10000).values
        F["app_dow"] = dt.dayofweek.values
        F["app_dom"] = dt.day.values
        F["app_month"] = dt.month.values
        F["payday_gap"] = (df["Payday"].values - dt.day.values) % 31
        cnt = pd.Series(day).map(pd.Series(customers["ID"] // 10000 - 21185).value_counts())
        F["app_day_volume"] = cnt.values
        F["app_seq_rel"] = F["app_seq"] / F["app_day_volume"]

    tag("date")

    if "ratio" in groups:
        inc = df["Total Annual Income"].astype(float).clip(lower=0)
        dinc = df["Declared Annual Income"].astype(float).clip(lower=0)
        dbor = df["Declared Borrowing Amount"].astype(float).clip(lower=0)
        rent = df["Rent Burden"].astype(float).clip(lower=0)
        F["income_ratio"] = inc / (dinc + 1)
        F["log_income"] = np.log1p(inc)
        F["debt_total"] = B[:, 0] + dbor
        F["dti"] = F["debt_total"] / (inc + 1)
        F["dti_aiful"] = B[:, 0] / (inc + 1)
        F["legal_capacity"] = inc / 3 - F["debt_total"]
        F["min_due_to_income"] = (F.get("min_due_now", B[:, 0] * 0.03)) / (inc / 12 + 1)
        F["rent_to_income"] = rent * 12 / (inc + 1)
        F["declared_dti"] = dbor / (dinc + 1)
        F["limit_to_income"] = df["Desired Credit Limit"] / (dinc + 1)
        F["borrow6_to_income"] = ba.sum(1) / (inc + 1)
        age = df["Age (in months)"].astype(float)
        F["work_exp_neg"] = (df["Work Experience (months)"] < 0).astype(int)
        F["work_to_age"] = df["Work Experience (months)"].clip(lower=0) / (age - 216).clip(lower=12)
        F["resid_to_age"] = df["Residence Period (in months)"] / age
        for c in TIME_VARYING:
            F[f"{c}_changed"] = (df[c] != df[f"{c}_app"]).astype(int)
        F["company_size_delta"] = df["Company Size Category"] - df["Company Size Category_app"]

    tag("ratio")
    out = pd.concat([df, pd.DataFrame(F, index=df.index)], axis=1)
    groups_of["raw"] = [c for c in df.columns if c not in ("ID", TARGET, "is_train")]

    if "app" in groups:
        pop_ids = set(train["ID"]) | set(test["ID"])
        scores = application_scores(customers, pop_ids)
        out = out.merge(scores, on="ID", how="left")
        groups_of["app"] = [c for c in scores.columns if c != "ID"]

    out = out.replace([np.inf, -np.inf], np.nan)
    return out, groups_of


def feature_columns(df):
    drop = {"ID", TARGET, "is_train"}
    return [c for c in df.columns if c not in drop]


def get_features(refresh=False):
    """Build features once and cache them; every training script reads the cache."""
    path = os.path.join(CACHE, "features.pkl")
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    os.makedirs(CACHE, exist_ok=True)
    df, groups = build_features()
    with open(path, "wb") as f:
        pickle.dump((df, groups), f)
    return df, groups


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    df, groups = get_features(refresh=True)
    print(f"cached {df.shape} -> {os.path.join(CACHE, 'features.pkl')}")
    for g, c in groups.items():
        print(f"  {g:7s} {len(c)} cols")
