"""
Credit Scoring Model — Full Pipeline (Optimized v2)
===================================================
Predicts probability of default by month 24 for customers
who survived 12 months on a revolving credit product.

Evaluation: ROC AUC on 5-fold Stratified Cross-Validation
Models: LightGBM + XGBoost + CatBoost + Stacking Ensemble
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
OUTPUT_DIR = os.path.dirname(__file__)


def load_and_merge_data():
    """Load train, test, customers and perform initial clean merge."""
    print("=" * 60)
    print("PHASE 1: Loading & Merging Data")
    print("=" * 60)

    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    print(f"  Customers: {customers.shape}")
    print(f"  Train:     {train.shape}")
    print(f"  Test:      {test.shape}")
    print(f"  Default rate: {train['24-Month Default Flag'].mean():.4f}")

    # Exclude 12-month defaulters (not present in train/test)
    cust_clean = customers[customers["12-Month Default Flag"] == 0].copy()
    cust_clean = cust_clean.drop(columns=["12-Month Default Flag"])

    # Time-varying columns appearing in both datasets
    time_varying_cols = ["Marital Status", "Insurance Type", "Company Size Category"]
    cust_rename = {col: f"{col}_app" for col in time_varying_cols}
    cust_clean = cust_clean.rename(columns=cust_rename)

    # Merge
    train_merged = train.merge(cust_clean, on="ID", how="left")
    test_merged = test.merge(cust_clean, on="ID", how="left")

    print(f"  After merge — Train: {train_merged.shape}, Test: {test_merged.shape}")
    return train_merged, test_merged


def engineer_features(df):
    """Vectorized feature engineering incorporating behavioral and regulatory credit signals."""
    feat = df.copy()

    # ----------------------------------------------------------
    # 1. Derive PER-MONTH values from cumulative columns
    # ----------------------------------------------------------
    for prefix in [
        "Additional Borrowing Amount",
        "Additional Borrowing Count",
        "Repayment Amount",
        "Repayment Count",
    ]:
        cols = [f"{prefix} (Past {i} Month{'s' if i > 1 else ''})" for i in range(1, 7)]
        for i in range(5, 0, -1):
            feat[f"{prefix}_Month{i+1}"] = feat[cols[i]] - feat[cols[i - 1]]
        feat[f"{prefix}_Month1"] = feat[cols[0]]

    # ----------------------------------------------------------
    # 2. Vectorized Balance Trajectory (7 snapshots: M6 -> M12)
    # ----------------------------------------------------------
    balance_cols = [
        "Balance 6 Months Ago",
        "Balance 5 Months Ago",
        "Balance 4 Months Ago",
        "Balance 3 Months Ago",
        "Balance 2 Months Ago",
        "Balance 1 Month Ago",
        "Outstanding Loan Balance",
    ]
    b_mat = feat[balance_cols].values.astype(float)

    # Fast closed-form linear regression slope & intercept over x = [0..6]
    # sum((x - 3)^2) = 28
    weights = np.array([-3, -2, -1, 0, 1, 2, 3]) / 28.0
    feat["balance_slope"] = b_mat.dot(weights)
    feat["balance_intercept"] = b_mat.mean(axis=1) - 3.0 * feat["balance_slope"]

    # Basic balance summary metrics
    feat["balance_mean"] = b_mat.mean(axis=1)
    feat["balance_std"] = b_mat.std(axis=1)
    feat["balance_min"] = b_mat.min(axis=1)
    feat["balance_max"] = b_mat.max(axis=1)
    feat["balance_range"] = feat["balance_max"] - feat["balance_min"]
    feat["balance_cv"] = feat["balance_std"] / (feat["balance_mean"] + 1.0)

    # Balance ratios and recent changes
    feat["balance_ratio_m12_m6"] = feat["Outstanding Loan Balance"] / (feat["Balance 6 Months Ago"] + 1.0)
    feat["balance_ratio_m12_m9"] = feat["Outstanding Loan Balance"] / (feat["Balance 3 Months Ago"] + 1.0)
    feat["balance_pct_change_recent"] = (feat["Outstanding Loan Balance"] - feat["Balance 1 Month Ago"]) / (feat["Balance 1 Month Ago"] + 1.0)

    # Monthly balance differences
    for i in range(6):
        feat[f"balance_change_m{i+1}"] = b_mat[:, 6 - i] - b_mat[:, 5 - i]

    # Balance momentum & acceleration
    feat["balance_momentum_recent_3m"] = feat["Outstanding Loan Balance"] - feat["Balance 3 Months Ago"]
    feat["balance_momentum_earlier_3m"] = feat["Balance 3 Months Ago"] - feat["Balance 6 Months Ago"]
    feat["balance_acceleration"] = feat["balance_momentum_recent_3m"] - feat["balance_momentum_earlier_3m"]

    b_diffs = np.diff(b_mat, axis=1)
    feat["months_balance_increasing"] = (b_diffs > 0).sum(axis=1)
    feat["months_balance_decreasing"] = (b_diffs < 0).sum(axis=1)
    feat["months_balance_stable"] = (b_diffs == 0).sum(axis=1)

    # ----------------------------------------------------------
    # 3. Delinquency & Repayment Signals (Crucial Default Precursors)
    # ----------------------------------------------------------
    repay_month_cols = [f"Repayment Amount_Month{i}" for i in range(1, 7)]
    repay_count_month_cols = [f"Repayment Count_Month{i}" for i in range(1, 7)]

    feat["total_repayment_6m"] = feat["Repayment Amount (Past 6 Months)"]
    feat["avg_repayment_monthly"] = feat["total_repayment_6m"] / 6.0
    feat["repayment_std"] = feat[repay_month_cols].std(axis=1)
    feat["repayment_regularity"] = feat["repayment_std"] / (feat["avg_repayment_monthly"] + 1.0)

    # Delinquency counting
    feat["months_zero_repayment"] = (feat[repay_month_cols] == 0).sum(axis=1)
    feat["months_zero_repayment_count"] = (feat[repay_count_month_cols] == 0).sum(axis=1)
    feat["missed_m1"] = (feat["Repayment Amount_Month1"] == 0).astype(int)
    feat["missed_m2"] = (feat["Repayment Amount_Month2"] == 0).astype(int)
    feat["missed_last_2_months"] = (feat["missed_m1"] & feat["missed_m2"]).astype(int)
    feat["missed_last_3_months"] = (feat["missed_m1"] & feat["missed_m2"] & (feat["Repayment Amount_Month3"] == 0)).astype(int)

    # Consecutive missed months from Month 1 backwards
    m1_0 = (feat["Repayment Amount_Month1"] == 0)
    m2_0 = (feat["Repayment Amount_Month2"] == 0)
    m3_0 = (feat["Repayment Amount_Month3"] == 0)
    m4_0 = (feat["Repayment Amount_Month4"] == 0)
    feat["consecutive_missed_months"] = (
        m1_0.astype(int) +
        (m1_0 & m2_0).astype(int) +
        (m1_0 & m2_0 & m3_0).astype(int) +
        (m1_0 & m2_0 & m3_0 & m4_0).astype(int)
    )

    # Repayment trend
    feat["repay_recent_3m"] = feat["Repayment Amount (Past 3 Months)"]
    feat["repay_earlier_3m"] = feat["total_repayment_6m"] - feat["repay_recent_3m"]
    feat["repay_trend_ratio"] = feat["repay_recent_3m"] / (feat["repay_earlier_3m"] + 1.0)
    feat["repay_to_balance_ratio"] = feat["total_repayment_6m"] / (feat["balance_mean"] + 1.0)

    # Auto-Debit Dishonor Flag (promised automatic bank withdrawal, but payment was 0)
    if "Auto-Debit Flag" in feat.columns:
        feat["auto_debit_dishonor_m1"] = ((feat["Auto-Debit Flag"] == 1) & (feat["Repayment Amount_Month1"] == 0)).astype(int)
        feat["auto_debit_dishonor_recent"] = ((feat["Auto-Debit Flag"] == 1) & (feat["missed_last_2_months"] == 1)).astype(int)

    # ----------------------------------------------------------
    # 4. Borrowing Behavior & Cash Drain
    # ----------------------------------------------------------
    borrow_month_cols = [f"Additional Borrowing Amount_Month{i}" for i in range(1, 7)]
    borrow_count_month_cols = [f"Additional Borrowing Count_Month{i}" for i in range(1, 7)]

    feat["total_borrowing_6m"] = feat["Additional Borrowing Amount (Past 6 Months)"]
    feat["avg_borrowing_monthly"] = feat["total_borrowing_6m"] / 6.0
    feat["borrowing_std"] = feat[borrow_month_cols].std(axis=1)

    feat["months_with_borrowing"] = (feat[borrow_month_cols] > 0).sum(axis=1)
    feat["months_no_borrowing"] = 6 - feat["months_with_borrowing"]
    feat["avg_draw_size"] = feat["total_borrowing_6m"] / (feat["Additional Borrowing Count (Past 6 Months)"] + 1.0)

    feat["borrow_recent_3m"] = feat["Additional Borrowing Amount (Past 3 Months)"]
    feat["borrow_earlier_3m"] = feat["total_borrowing_6m"] - feat["borrow_recent_3m"]
    feat["borrow_trend_ratio"] = feat["borrow_recent_3m"] / (feat["borrow_earlier_3m"] + 1.0)
    feat["recent_borrow_spike"] = (feat["Additional Borrowing Amount_Month1"] > 2.0 * feat["avg_borrowing_monthly"]).astype(int)

    # Distress: Borrowing while not repaying (debt pyramiding)
    feat["borrow_without_repay_m1"] = ((feat["Additional Borrowing Amount_Month1"] > 0) & (feat["Repayment Amount_Month1"] == 0)).astype(int)
    borrow_no_repay_months = 0
    for i in range(1, 7):
        borrow_no_repay_months = borrow_no_repay_months + (
            (feat[f"Additional Borrowing Amount_Month{i}"] > 0) & (feat[f"Repayment Amount_Month{i}"] == 0)
        ).astype(int)
    feat["months_borrow_without_repay"] = borrow_no_repay_months

    # ----------------------------------------------------------
    # 5. Net Cash Flow
    # ----------------------------------------------------------
    feat["net_flow_6m"] = feat["total_repayment_6m"] - feat["total_borrowing_6m"]
    feat["net_flow_recent_3m"] = feat["repay_recent_3m"] - feat["borrow_recent_3m"]
    feat["net_flow_earlier_3m"] = feat["repay_earlier_3m"] - feat["borrow_earlier_3m"]
    feat["net_flow_trend"] = feat["net_flow_recent_3m"] - feat["net_flow_earlier_3m"]
    feat["net_flow_month1"] = feat["Repayment Amount_Month1"] - feat["Additional Borrowing Amount_Month1"]
    feat["repay_borrow_ratio"] = feat["total_repayment_6m"] / (feat["total_borrowing_6m"] + 1.0)
    feat["is_deleveraging"] = (feat["net_flow_6m"] > 0).astype(int)

    # ----------------------------------------------------------
    # 6. Regulatory Debt Burden & Moneylending Act Cap (総量規制)
    # ----------------------------------------------------------
    # In Japan, unsecured borrowing is legally capped at 1/3 (33.3%) of annual income
    declared_borrow = feat["Declared Borrowing Amount"].fillna(0) if "Declared Borrowing Amount" in feat.columns else 0
    total_annual_inc = feat["Total Annual Income"].fillna(0)
    feat["total_known_debt"] = feat["Outstanding Loan Balance"] + declared_borrow
    feat["total_dti"] = feat["total_known_debt"] / (total_annual_inc + 1.0)
    feat["exceeds_legal_debt_limit"] = (feat["total_dti"] >= 0.333).astype(int)
    feat["remaining_legal_capacity"] = (total_annual_inc / 3.0) - feat["total_known_debt"]

    # Minimum payment burden (~3% of outstanding balance in revolving credit)
    min_payment = feat["Outstanding Loan Balance"] * 0.03
    feat["min_payment_amount"] = min_payment
    feat["paid_less_than_min_m1"] = (feat["Repayment Amount_Month1"] < min_payment).astype(int)
    feat["min_payment_to_monthly_income"] = min_payment / (total_annual_inc / 12.0 + 1.0)

    # ----------------------------------------------------------
    # 7. Credit Utilization & Application Attributes
    # ----------------------------------------------------------
    if "Desired Credit Limit" in feat.columns:
        credit_limit = feat["Desired Credit Limit"].replace(0, np.nan)
        feat["current_utilization"] = feat["Outstanding Loan Balance"] / credit_limit
        feat["max_utilization"] = feat["balance_max"] / credit_limit
        feat["is_maxed_out"] = (feat["Outstanding Loan Balance"] >= credit_limit).astype(int)
        feat["utilization_jump_recent"] = (feat["Outstanding Loan Balance"] - feat["Balance 1 Month Ago"]) / credit_limit

    if "Declared Annual Income" in feat.columns:
        # Winsorize extreme income outliers at 99.5th percentile for ratio stability
        inc_cap = feat["Declared Annual Income"].quantile(0.995)
        clean_declared_inc = feat["Declared Annual Income"].clip(upper=inc_cap)
        feat["income_to_credit_limit"] = clean_declared_inc / (feat["Desired Credit Limit"] + 1.0)
        feat["declared_debt_burden"] = declared_borrow / (clean_declared_inc + 1.0)
        feat["income_change_ratio"] = total_annual_inc / (clean_declared_inc + 1.0)

    if "Rent Burden" in feat.columns:
        rent_cap = feat["Rent Burden"].quantile(0.995)
        clean_rent = feat["Rent Burden"].clip(upper=rent_cap)
        feat["rent_to_income"] = (clean_rent * 12.0) / (total_annual_inc + 1.0)

    if "Age (in months)" in feat.columns:
        feat["age_years"] = (feat["Age (in months)"] + 12) / 12.0

    if "Residence Period (in months)" in feat.columns:
        feat["residence_years"] = (feat["Residence Period (in months)"] + 12) / 12.0

    # Work experience clean & flags
    if "Work Experience (months)" in feat.columns:
        feat["is_negative_work_exp"] = (feat["Work Experience (months)"] < 0).astype(int)
        feat["work_exp_clean"] = feat["Work Experience (months)"].clip(lower=0)
        if "Age (in months)" in feat.columns:
            adult_months = (feat["Age (in months)"] - 18 * 12).clip(lower=12)
            feat["work_stability_ratio"] = feat["work_exp_clean"] / adult_months

    # Settlements history
    if "Number of Loan Settlements" in feat.columns:
        feat["has_settled_loans"] = (feat["Number of Loan Settlements"] > 0).astype(int)

    # Contact accessibility score
    contact_cols = [c for c in feat.columns if "Phone Number Available" in c or "Licence" in c]
    if contact_cols:
        feat["contact_score"] = feat[contact_cols].sum(axis=1)

    # Payday flags
    if "Payday" in feat.columns:
        feat["payday_is_25"] = (feat["Payday"] == 25).astype(int)
        feat["payday_is_eom"] = (feat["Payday"].isin([28, 29, 30, 31])).astype(int)

    # Time-varying changes
    for col in ["Marital Status", "Insurance Type", "Company Size Category"]:
        app_col = f"{col}_app"
        if app_col in feat.columns and col in feat.columns:
            feat[f"{col}_changed"] = (feat[col] != feat[app_col]).astype(int)
            if col == "Company Size Category":
                feat["company_size_direction"] = feat[app_col] - feat[col]

    # Log-transforms for long-tailed distributions
    for col in [
        "Outstanding Loan Balance",
        "Total Annual Income",
        "total_repayment_6m",
        "total_borrowing_6m",
        "total_known_debt",
    ]:
        if col in feat.columns:
            feat[f"log_{col}"] = np.log1p(feat[col].clip(lower=0))

    return feat


def rank_average(*predictions):
    """Normalize and rank-average multiple prediction sets."""
    ranked = [rankdata(p) for p in predictions]
    avg_rank = np.mean(ranked, axis=0)
    return (avg_rank - avg_rank.min()) / (avg_rank.max() - avg_rank.min())


def run_pipeline():
    """Main training, cross-validation, and submission pipeline."""
    train_df, test_df = load_and_merge_data()

    print("\n" + "=" * 60)
    print("PHASE 2: Feature Engineering")
    print("=" * 60)
    train_feat = engineer_features(train_df)
    test_feat = engineer_features(test_df)
    print(f"  Engineered Train shape: {train_feat.shape}")
    print(f"  Engineered Test shape:  {test_feat.shape}")

    TARGET = "24-Month Default Flag"
    DROP_COLS = ["ID", TARGET]
    feature_cols = [c for c in train_feat.columns if c not in DROP_COLS]
    feature_cols = train_feat[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Clean non-finite values
    train_feat[feature_cols] = train_feat[feature_cols].replace([np.inf, -np.inf], np.nan)
    test_feat[feature_cols] = test_feat[feature_cols].replace([np.inf, -np.inf], np.nan)

    X = train_feat[feature_cols].values
    y = train_feat[TARGET].values
    X_test = test_feat[feature_cols].values
    test_ids = test_feat["ID"].values

    print("\n" + "=" * 60)
    print("PHASE 3: Modeling Setup")
    print("=" * 60)
    print(f"  Features used: {len(feature_cols)}")
    print(f"  Train samples: {len(X)} (Pos: {y.sum()}, Neg: {(1-y).sum()})")
    print(f"  Test samples:  {len(X_test)}")

    N_FOLDS = 5
    SEED = 42
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Model out-of-fold containers
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))

    test_lgb = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))

    fold_scores = {"lgb": [], "xgb": [], "cat": []}

    # Model Hyperparameters (Optimized for ROC AUC ranking)
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "num_leaves": 47,
        "max_depth": 7,
        "min_child_samples": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": SEED,
        "verbose": -1,
        "n_jobs": -1,
    }

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": SEED,
        "verbosity": 0,
        "tree_method": "hist",
        "early_stopping_rounds": 100,
        "n_jobs": -1,
    }

    cat_params = {
        "iterations": 3000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "subsample": 0.8,
        "random_seed": SEED,
        "verbose": 0,
        "eval_metric": "AUC",
        "early_stopping_rounds": 100,
        "task_type": "CPU",
    }

    print("\n" + "=" * 60)
    print("PHASE 4: 5-Fold Stratified Cross-Validation")
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  --- Fold {fold + 1}/{N_FOLDS} ---")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # 1. LightGBM
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        test_lgb += lgb_model.predict_proba(X_test)[:, 1] / N_FOLDS
        lgb_auc = roc_auc_score(y_val, oof_lgb[val_idx])
        fold_scores["lgb"].append(lgb_auc)

        # 2. XGBoost
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        test_xgb += xgb_model.predict_proba(X_test)[:, 1] / N_FOLDS
        xgb_auc = roc_auc_score(y_val, oof_xgb[val_idx])
        fold_scores["xgb"].append(xgb_auc)

        # 3. CatBoost
        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            verbose=0,
        )
        oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        test_cat += cat_model.predict_proba(X_test)[:, 1] / N_FOLDS
        cat_auc = roc_auc_score(y_val, oof_cat[val_idx])
        fold_scores["cat"].append(cat_auc)

        print(f"    LGB: {lgb_auc:.5f} | XGB: {xgb_auc:.5f} | CAT: {cat_auc:.5f}")

    # ============================================================
    # 5. ENSEMBLE & VALIDATION EVALUATION
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 5: Out-of-Fold Validation & Ensemble Comparison")
    print("=" * 60)

    lgb_oof_auc = roc_auc_score(y, oof_lgb)
    xgb_oof_auc = roc_auc_score(y, oof_xgb)
    cat_oof_auc = roc_auc_score(y, oof_cat)

    print(f"  LightGBM OOF AUC : {lgb_oof_auc:.5f} (CV mean: {np.mean(fold_scores['lgb']):.5f} ± {np.std(fold_scores['lgb']):.5f})")
    print(f"  XGBoost  OOF AUC : {xgb_oof_auc:.5f} (CV mean: {np.mean(fold_scores['xgb']):.5f} ± {np.std(fold_scores['xgb']):.5f})")
    print(f"  CatBoost OOF AUC : {cat_oof_auc:.5f} (CV mean: {np.mean(fold_scores['cat']):.5f} ± {np.std(fold_scores['cat']):.5f})")

    # Simple average
    oof_simple = (oof_lgb + oof_xgb + oof_cat) / 3.0
    simple_auc = roc_auc_score(y, oof_simple)

    # Rank average
    oof_rank = rank_average(oof_lgb, oof_xgb, oof_cat)
    rank_auc = roc_auc_score(y, oof_rank)

    # Stacking with Logistic Regression (Meta-model)
    oof_stack_features = np.column_stack([oof_lgb, oof_xgb, oof_cat])
    test_stack_features = np.column_stack([test_lgb, test_xgb, test_cat])

    meta_model = LogisticRegression(C=1.0, max_iter=1000)
    meta_model.fit(oof_stack_features, y)
    oof_stacked = meta_model.predict_proba(oof_stack_features)[:, 1]
    stacked_auc = roc_auc_score(y, oof_stacked)
    test_stacked = meta_model.predict_proba(test_stack_features)[:, 1]

    print(f"\n  Ensemble Comparison:")
    print(f"  - Simple Average OOF AUC: {simple_auc:.5f}")
    print(f"  - Rank Average   OOF AUC: {rank_auc:.5f}")
    print(f"  - Stacked (LR)   OOF AUC: {stacked_auc:.5f}")

    # Determine winning model
    candidates = {
        "Simple Average": (simple_auc, (test_lgb + test_xgb + test_cat) / 3.0),
        "Rank Average": (rank_auc, rank_average(test_lgb, test_xgb, test_cat)),
        "Stacked LogisticRegression": (stacked_auc, test_stacked),
    }
    best_name = max(candidates.keys(), key=lambda k: candidates[k][0])
    best_auc, best_test_preds = candidates[best_name]

    print(f"\n  >>> WINNING ENSEMBLE: {best_name} (AUC: {best_auc:.5f}, Gini: {2 * best_auc - 1:.5f}) <<<")

    # ============================================================
    # 6. EXPORT SUBMISSIONS & ARTIFACTS
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 6: Generating Output Files")
    print("=" * 60)

    sub_df = pd.DataFrame({"ID": test_ids, "24-Month Default Flag": best_test_preds})
    sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"  Saved main submission: {sub_path} ({len(sub_df)} rows)")

    # Save feature importances
    feat_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": lgb_model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    feat_imp_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    feat_imp.to_csv(feat_imp_path, index=False)
    print(f"  Saved feature importances: {feat_imp_path}")

    # Save validation metrics summary file
    metrics_summary = pd.DataFrame({
        "Model": ["LightGBM", "XGBoost", "CatBoost", "Simple Average", "Rank Average", "Stacked LogisticRegression"],
        "OOF_AUC": [lgb_oof_auc, xgb_oof_auc, cat_oof_auc, simple_auc, rank_auc, stacked_auc],
        "Gini": [2 * x - 1 for x in [lgb_oof_auc, xgb_oof_auc, cat_oof_auc, simple_auc, rank_auc, stacked_auc]],
    })
    metrics_path = os.path.join(OUTPUT_DIR, "validation_metrics.csv")
    metrics_summary.to_csv(metrics_path, index=False)
    print(f"  Saved validation metrics: {metrics_path}")

    print("\nTop 25 Important Features:")
    for _, row in feat_imp.head(25).iterrows():
        print(f"    {row['Feature']:40s}: {int(row['Importance'])}")

    print("\nPipeline finished successfully!")


if __name__ == "__main__":
    run_pipeline()
