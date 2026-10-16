"""
Credit Scoring Model — Full Pipeline (v5 Advanced Architecture)
===============================================================
Predicts probability of default by month 24 for customers
who survived 12 months on a revolving credit product.

Architectural Upgrades:
1. Stage-1 Application Risk Scoring on full 205,637 rows of customers.csv
   (captures origination frailty and survival bias without leakage).
2. Closed-form 2nd-order quadratic curvature and detrended residual volatility.
3. Cash-Cycling (Ponzi repayment / churn) and net principal amortization.
4. Application-to-Month-12 demographic & financial transition drift.
5. Non-linear distance to Japanese Moneylending Act 1/3 cap (総量規制).
6. Diversified tree topologies (shallow additive XGBoost + deep LightGBM).
7. Within-fold rank normalization and calibrated ensemble blending.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
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


# ============================================================
# 1. STAGE-1 APPLICATION RISK MODELING (All 205,637 Customers)
# ============================================================
def train_stage1_application_model(customers, train_ids, test_ids):
    """
    Trains a Stage-1 model on all 205,637 customers.csv records using application attributes.
    Generates out-of-fold predicted default probability and active-status probability.
    """
    print("\n" + "=" * 60)
    print("STAGE 1: Application Risk Scoring (205,637 Records)")
    print("=" * 60)

    app_cols = [c for c in customers.columns if c not in ["ID", "12-Month Default Flag"]]
    app_cols = customers[app_cols].select_dtypes(include=[np.number]).columns.tolist()

    X_app = customers[app_cols].fillna(0).values
    y_app_default = customers["12-Month Default Flag"].values

    active_id_set = set(train_ids).union(set(test_ids))
    y_app_active = customers["ID"].isin(active_id_set).astype(int).values

    skf_app = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_p_default = np.zeros(len(customers))
    oof_p_active = np.zeros(len(customers))

    for fold, (tr_idx, val_idx) in enumerate(skf_app.split(X_app, y_app_default)):
        # Model A: Year-1 Default Risk
        m_def = lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1,
        )
        m_def.fit(X_app[tr_idx], y_app_default[tr_idx])
        oof_p_default[val_idx] = m_def.predict_proba(X_app[val_idx])[:, 1]

        # Model B: Active at M12 Survival Likelihood
        m_act = lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1,
        )
        m_act.fit(X_app[tr_idx], y_app_active[tr_idx])
        oof_p_active[val_idx] = m_act.predict_proba(X_app[val_idx])[:, 1]

    auc_def = roc_auc_score(y_app_default, oof_p_default)
    auc_act = roc_auc_score(y_app_active, oof_p_active)
    print(f"  Stage-1 Year-1 Default Model OOF AUC : {auc_def:.5f}")
    print(f"  Stage-1 Active at M12 Model OOF AUC  : {auc_act:.5f}")

    stage1_df = pd.DataFrame({
        "ID": customers["ID"],
        "p_app_default": oof_p_default,
        "p_app_active": oof_p_active,
        "ratio_app_risk": oof_p_default / (oof_p_active + 0.001),
    })
    return stage1_df


# ============================================================
# 2. DATA LOADING & MERGING
# ============================================================
def load_and_merge_data():
    """Load train, test, customers and perform initial clean merge."""
    print("\n" + "=" * 60)
    print("PHASE 1: Loading & Merging Data")
    print("=" * 60)

    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    print(f"  Customers: {customers.shape}")
    print(f"  Train:     {train.shape}")
    print(f"  Test:      {test.shape}")
    print(f"  Default rate: {train['24-Month Default Flag'].mean():.4f}")

    # Train Stage-1 application baseline risk model on full 205k customers
    stage1_features = train_stage1_application_model(customers, train["ID"].values, test["ID"].values)

    # Exclude 12-month defaulters from customer table for merge
    cust_clean = customers[customers["12-Month Default Flag"] == 0].copy()
    cust_clean = cust_clean.drop(columns=["12-Month Default Flag"])

    # Merge Stage-1 risk scores
    cust_clean = cust_clean.merge(stage1_features, on="ID", how="left")

    # Time-varying columns appearing in both datasets
    time_varying_cols = ["Marital Status", "Insurance Type", "Company Size Category"]
    cust_rename = {col: f"{col}_app" for col in time_varying_cols}
    cust_clean = cust_clean.rename(columns=cust_rename)

    train_merged = train.merge(cust_clean, on="ID", how="left")
    test_merged = test.merge(cust_clean, on="ID", how="left")

    print(f"\n  After merge — Train: {train_merged.shape}, Test: {test_merged.shape}")
    return train_merged, test_merged


# ============================================================
# 3. FEATURE ENGINEERING ENGINE
# ============================================================
def engineer_features(df):
    """Vectorized feature engineering incorporating behavioral, regulatory, and peer signals."""
    feat = df.copy()

    # ----------------------------------------------------------
    # 3.1 Derive PER-MONTH values from cumulative columns
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
    # 3.2 Vectorized Balance Trajectory (Orthogonal Linear & Curvature)
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

    # 1st-order linear slope (weights sum to 0, sum of squares = 28)
    weights_linear = np.array([-3, -2, -1, 0, 1, 2, 3]) / 28.0
    feat["balance_slope"] = b_mat.dot(weights_linear)
    feat["balance_intercept"] = b_mat.mean(axis=1) - 3.0 * feat["balance_slope"]

    # 2nd-order orthogonal quadratic curvature (weights sum to 0, sum of squares = 84)
    # Orthogonal polynomial for 7 points: [5, 0, -3, -4, -3, 0, 5] / 84
    weights_quad = np.array([5, 0, -3, -4, -3, 0, 5]) / 84.0
    feat["balance_curvature"] = b_mat.dot(weights_quad)

    # Detrended residual volatility
    x_grid = np.arange(7)
    trend_matrix = np.outer(feat["balance_intercept"], np.ones(7)) + np.outer(feat["balance_slope"], x_grid)
    residuals = b_mat - trend_matrix
    feat["balance_residual_std"] = residuals.std(axis=1)

    # Summary metrics
    feat["balance_mean"] = b_mat.mean(axis=1)
    feat["balance_std"] = b_mat.std(axis=1)
    feat["balance_min"] = b_mat.min(axis=1)
    feat["balance_max"] = b_mat.max(axis=1)
    feat["balance_range"] = feat["balance_max"] - feat["balance_min"]
    feat["balance_cv"] = feat["balance_std"] / (feat["balance_mean"] + 1.0)

    # Zero-balance milestones
    feat["has_reached_zero_balance"] = (b_mat == 0).any(axis=1).astype(int)
    feat["months_at_zero_balance"] = (b_mat == 0).sum(axis=1)

    # Balance ratios and changes
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
    # 3.3 Delinquency & Repayment Velocity Signals
    # ----------------------------------------------------------
    repay_month_cols = [f"Repayment Amount_Month{i}" for i in range(1, 7)]
    repay_count_month_cols = [f"Repayment Count_Month{i}" for i in range(1, 7)]

    feat["total_repayment_6m"] = feat["Repayment Amount (Past 6 Months)"]
    feat["avg_repayment_monthly"] = feat["total_repayment_6m"] / 6.0
    feat["repayment_std"] = feat[repay_month_cols].std(axis=1)
    feat["repayment_regularity"] = feat["repayment_std"] / (feat["avg_repayment_monthly"] + 1.0)

    feat["months_zero_repayment"] = (feat[repay_month_cols] == 0).sum(axis=1)
    feat["months_zero_repayment_count"] = (feat[repay_count_month_cols] == 0).sum(axis=1)
    feat["missed_m1"] = (feat["Repayment Amount_Month1"] == 0).astype(int)
    feat["missed_m2"] = (feat["Repayment Amount_Month2"] == 0).astype(int)
    feat["missed_last_2_months"] = (feat["missed_m1"] & feat["missed_m2"]).astype(int)
    feat["missed_last_3_months"] = (feat["missed_m1"] & feat["missed_m2"] & (feat["Repayment Amount_Month3"] == 0)).astype(int)

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

    # Repayment velocity drop
    avg_repay_2_6 = (feat["total_repayment_6m"] - feat["Repayment Amount_Month1"]) / 5.0
    feat["repay_velocity_drop"] = feat["Repayment Amount_Month1"] / (avg_repay_2_6 + 1.0)
    feat["repay_collapsed_flag"] = (feat["repay_velocity_drop"] < 0.3).astype(int)

    feat["repay_recent_3m"] = feat["Repayment Amount (Past 3 Months)"]
    feat["repay_earlier_3m"] = feat["total_repayment_6m"] - feat["repay_recent_3m"]
    feat["repay_trend_ratio"] = feat["repay_recent_3m"] / (feat["repay_earlier_3m"] + 1.0)
    feat["repay_to_balance_ratio"] = feat["total_repayment_6m"] / (feat["balance_mean"] + 1.0)

    if "Auto-Debit Flag" in feat.columns:
        feat["auto_debit_dishonor_m1"] = ((feat["Auto-Debit Flag"] == 1) & (feat["Repayment Amount_Month1"] == 0)).astype(int)
        feat["auto_debit_dishonor_recent"] = ((feat["Auto-Debit Flag"] == 1) & (feat["missed_last_2_months"] == 1)).astype(int)

    # ----------------------------------------------------------
    # 3.4 Borrowing Behavior & Cash-Cycling (Ponzi Churn)
    # ----------------------------------------------------------
    borrow_month_cols = [f"Additional Borrowing Amount_Month{i}" for i in range(1, 7)]

    feat["total_borrowing_6m"] = feat["Additional Borrowing Amount (Past 6 Months)"]
    feat["avg_borrowing_monthly"] = feat["total_borrowing_6m"] / 6.0
    feat["borrowing_std"] = feat[borrow_month_cols].std(axis=1)

    feat["months_with_borrowing"] = (feat[borrow_month_cols] > 0).sum(axis=1)
    feat["months_no_borrowing"] = 6 - feat["months_with_borrowing"]
    feat["avg_draw_size"] = feat["total_borrowing_6m"] / (feat["Additional Borrowing Count (Past 6 Months)"] + 1.0)

    # Borrowing acceleration
    borrow_m1_2 = feat["Additional Borrowing Amount (Past 2 Months)"]
    borrow_m5_6 = feat["Additional Borrowing Amount_Month5"] + feat["Additional Borrowing Amount_Month6"]
    feat["borrow_acceleration_ratio"] = borrow_m1_2 / (borrow_m5_6 + 1.0)

    feat["borrow_recent_3m"] = feat["Additional Borrowing Amount (Past 3 Months)"]
    feat["borrow_earlier_3m"] = feat["total_borrowing_6m"] - feat["borrow_recent_3m"]
    feat["borrow_trend_ratio"] = feat["borrow_recent_3m"] / (feat["borrow_earlier_3m"] + 1.0)
    feat["recent_borrow_spike"] = (feat["Additional Borrowing Amount_Month1"] > 2.0 * feat["avg_borrowing_monthly"]).astype(int)

    # Cash-Cycling (Ponzi repayment: borrow and repay roughly equal amounts to stay current)
    cycling_ratios = []
    for i in range(1, 7):
        b_k = feat[f"Additional Borrowing Amount_Month{i}"]
        r_k = feat[f"Repayment Amount_Month{i}"]
        cycle_k = np.minimum(b_k, r_k) / (np.maximum(b_k, r_k) + 1.0)
        cycling_ratios.append(cycle_k)
        feat[f"cash_cycling_m{i}"] = cycle_k

    feat["avg_cash_cycling_6m"] = np.mean(cycling_ratios, axis=0)
    feat["recent_cash_cycling_3m"] = np.mean(cycling_ratios[:3], axis=0)

    # Net principal amortization rate over past 6 months
    feat["net_principal_amortization"] = (feat["total_repayment_6m"] - feat["total_borrowing_6m"]) / (feat["Balance 6 Months Ago"] + 1.0)

    # Distress: Borrowing while not repaying
    feat["borrow_without_repay_m1"] = ((feat["Additional Borrowing Amount_Month1"] > 0) & (feat["Repayment Amount_Month1"] == 0)).astype(int)

    # Net cash deficit streak
    deficit_months = 0
    for i in range(1, 7):
        is_deficit = (feat[f"Additional Borrowing Amount_Month{i}"] > feat[f"Repayment Amount_Month{i}"]).astype(int)
        deficit_months = deficit_months + is_deficit
    feat["deficit_months_count"] = deficit_months

    d1 = (feat["Additional Borrowing Amount_Month1"] > feat["Repayment Amount_Month1"]).astype(int)
    d2 = (feat["Additional Borrowing Amount_Month2"] > feat["Repayment Amount_Month2"]).astype(int)
    d3 = (feat["Additional Borrowing Amount_Month3"] > feat["Repayment Amount_Month3"]).astype(int)
    feat["consecutive_deficit_months"] = d1 + (d1 & d2) + (d1 & d2 & d3)

    # ----------------------------------------------------------
    # 3.5 Net Cash Flow
    # ----------------------------------------------------------
    feat["net_flow_6m"] = feat["total_repayment_6m"] - feat["total_borrowing_6m"]
    feat["net_flow_recent_3m"] = feat["repay_recent_3m"] - feat["borrow_recent_3m"]
    feat["net_flow_earlier_3m"] = feat["repay_earlier_3m"] - feat["borrow_earlier_3m"]
    feat["net_flow_trend"] = feat["net_flow_recent_3m"] - feat["net_flow_earlier_3m"]
    feat["net_flow_month1"] = feat["Repayment Amount_Month1"] - feat["Additional Borrowing Amount_Month1"]
    feat["repay_borrow_ratio"] = feat["total_repayment_6m"] / (feat["total_borrowing_6m"] + 1.0)
    feat["is_deleveraging"] = (feat["net_flow_6m"] > 0).astype(int)

    # ----------------------------------------------------------
    # 3.6 Regulatory Debt Burden & Moneylending Act Cap (総量規制)
    # ----------------------------------------------------------
    declared_borrow = feat["Declared Borrowing Amount"].fillna(0) if "Declared Borrowing Amount" in feat.columns else 0
    total_annual_inc = feat["Total Annual Income"].fillna(0)
    feat["total_known_debt"] = feat["Outstanding Loan Balance"] + declared_borrow
    feat["total_dti"] = feat["total_known_debt"] / (total_annual_inc + 1.0)
    feat["exceeds_legal_debt_limit"] = (feat["total_dti"] >= 0.333).astype(int)
    feat["remaining_legal_capacity"] = (total_annual_inc / 3.0) - feat["total_known_debt"]

    # Non-linear regulatory saturation & distance to ceiling
    feat["regulatory_stress_index"] = 1.0 / (1.0 + np.maximum(0, feat["remaining_legal_capacity"]) / 100000.0)
    feat["months_to_regulatory_ceiling"] = np.maximum(0, feat["remaining_legal_capacity"]) / np.maximum(1.0, feat["balance_slope"])

    min_payment = feat["Outstanding Loan Balance"] * 0.03
    feat["min_payment_amount"] = min_payment
    feat["paid_less_than_min_m1"] = (feat["Repayment Amount_Month1"] < min_payment).astype(int)
    feat["min_payment_to_monthly_income"] = min_payment / (total_annual_inc / 12.0 + 1.0)

    # ----------------------------------------------------------
    # 3.7 Credit Utilization & Application Attributes
    # ----------------------------------------------------------
    if "Desired Credit Limit" in feat.columns:
        credit_limit = feat["Desired Credit Limit"].replace(0, np.nan)
        feat["current_utilization"] = feat["Outstanding Loan Balance"] / credit_limit
        feat["max_utilization"] = feat["balance_max"] / credit_limit
        feat["is_maxed_out"] = (feat["Outstanding Loan Balance"] >= credit_limit).astype(int)
        feat["utilization_jump_recent"] = (feat["Outstanding Loan Balance"] - feat["Balance 1 Month Ago"]) / credit_limit
        feat["utilization_6m_change"] = (feat["Outstanding Loan Balance"] - feat["Balance 6 Months Ago"]) / credit_limit
        feat["credit_headroom"] = credit_limit - feat["Outstanding Loan Balance"]

    if "Declared Annual Income" in feat.columns:
        inc_cap = feat["Declared Annual Income"].quantile(0.995)
        clean_declared_inc = feat["Declared Annual Income"].clip(upper=inc_cap)
        feat["income_to_credit_limit"] = clean_declared_inc / (feat["Desired Credit Limit"] + 1.0)
        feat["declared_debt_burden"] = declared_borrow / (clean_declared_inc + 1.0)
        feat["income_change_ratio"] = total_annual_inc / (clean_declared_inc + 1.0)

        # Transition Drift: Earning Shocks
        feat["income_change_pct"] = (total_annual_inc - clean_declared_inc) / (clean_declared_inc + 1.0)
        feat["income_dropped_10pct"] = (total_annual_inc < 0.90 * clean_declared_inc).astype(int)
        feat["income_dropped_25pct"] = (total_annual_inc < 0.75 * clean_declared_inc).astype(int)

    if "Rent Burden" in feat.columns:
        rent_cap = feat["Rent Burden"].quantile(0.995)
        clean_rent = feat["Rent Burden"].clip(upper=rent_cap)
        feat["rent_to_income"] = (clean_rent * 12.0) / (total_annual_inc + 1.0)

    if "Age (in months)" in feat.columns:
        feat["age_years"] = (feat["Age (in months)"] + 12) / 12.0

    if "Residence Period (in months)" in feat.columns:
        feat["residence_years"] = (feat["Residence Period (in months)"] + 12) / 12.0

    if "Work Experience (months)" in feat.columns:
        feat["is_negative_work_exp"] = (feat["Work Experience (months)"] < 0).astype(int)
        feat["work_exp_clean"] = feat["Work Experience (months)"].clip(lower=0)
        if "Age (in months)" in feat.columns:
            adult_months = (feat["Age (in months)"] - 18 * 12).clip(lower=12)
            feat["work_stability_ratio"] = feat["work_exp_clean"] / adult_months

    if "Number of Loan Settlements" in feat.columns:
        feat["has_settled_loans"] = (feat["Number of Loan Settlements"] > 0).astype(int)

    contact_cols = [c for c in feat.columns if "Phone Number Available" in c or "Licence" in c]
    if contact_cols:
        feat["contact_score"] = feat[contact_cols].sum(axis=1)

    if "Payday" in feat.columns:
        feat["payday_is_25"] = (feat["Payday"] == 25).astype(int)
        feat["payday_is_eom"] = (feat["Payday"].isin([28, 29, 30, 31])).astype(int)

    # ----------------------------------------------------------
    # 3.8 Peer Groupby Aggregations (Relative Cohort Standing)
    # ----------------------------------------------------------
    if "Industry Category" in feat.columns:
        ind_mean_inc = feat.groupby("Industry Category")["Total Annual Income"].transform("mean")
        feat["income_ratio_to_industry_mean"] = feat["Total Annual Income"] / (ind_mean_inc + 1.0)
        feat["income_diff_to_industry_mean"] = feat["Total Annual Income"] - ind_mean_inc
        ind_med_debt = feat.groupby("Industry Category")["total_known_debt"].transform("median")
        feat["debt_ratio_to_industry_median"] = feat["total_known_debt"] / (ind_med_debt + 1.0)

    if "Occupation Category" in feat.columns:
        occ_mean_inc = feat.groupby("Occupation Category")["Total Annual Income"].transform("mean")
        feat["income_ratio_to_occ_mean"] = feat["Total Annual Income"] / (occ_mean_inc + 1.0)
        feat["income_diff_to_occ_mean"] = feat["Total Annual Income"] - occ_mean_inc
        occ_med_debt = feat.groupby("Occupation Category")["total_known_debt"].transform("median")
        feat["debt_ratio_to_occ_median"] = feat["total_known_debt"] / (occ_med_debt + 1.0)

    if "age_years" in feat.columns:
        age_bracket = (feat["age_years"] // 10).clip(2, 6)
        age_med_inc = feat.groupby(age_bracket)["Total Annual Income"].transform("median")
        feat["income_ratio_to_age_median"] = feat["Total Annual Income"] / (age_med_inc + 1.0)
        age_med_debt = feat.groupby(age_bracket)["total_known_debt"].transform("median")
        feat["debt_ratio_to_age_median"] = feat["total_known_debt"] / (age_med_debt + 1.0)
        if "work_exp_clean" in feat.columns:
            age_med_work = feat.groupby(age_bracket)["work_exp_clean"].transform("median")
            feat["work_exp_ratio_to_age_median"] = feat["work_exp_clean"] / (age_med_work + 1.0)

    # ----------------------------------------------------------
    # 3.9 Transition Drift: Employment & Job Shocks
    # ----------------------------------------------------------
    # In Japan: Insurance Type 1 = Company Social Insurance (employed)
    # Insurance Type 3/4 = National Health Insurance (self-employed / unemployed / lost job)
    if "Insurance Type_app" in feat.columns and "Insurance Type" in feat.columns:
        feat["insurance_type_changed"] = (feat["Insurance Type"] != feat["Insurance Type_app"]).astype(int)
        feat["insurance_job_loss_shock"] = (
            (feat["Insurance Type_app"] == 1) & (feat["Insurance Type"].isin([0, 3, 4]))
        ).astype(int)

    if "Company Size Category_app" in feat.columns and "Company Size Category" in feat.columns:
        feat["company_size_changed"] = (feat["Company Size Category"] != feat["Company Size Category_app"]).astype(int)
        # Larger integer in Company Size Category = smaller company
        feat["company_size_downgraded"] = (
            feat["Company Size Category"] > feat["Company Size Category_app"]
        ).astype(int)

    if "Marital Status_app" in feat.columns and "Marital Status" in feat.columns:
        feat["marital_status_changed"] = (feat["Marital Status"] != feat["Marital Status_app"]).astype(int)

    # Log-transforms for long-tailed volume features
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


# ============================================================
# 4. MAIN CROSS-VALIDATION PIPELINE
# ============================================================
def run_pipeline():
    """Main training, cross-validation, and submission pipeline."""
    train_df, test_df = load_and_merge_data()

    print("\n" + "=" * 60)
    print("PHASE 2: Feature Engineering & Domain Expansion")
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
    print("PHASE 3: Modeling Setup & Cross-Validation")
    print("=" * 60)
    print(f"  Features used: {len(feature_cols)}")
    print(f"  Train samples: {len(X)} (Pos: {y.sum()}, Neg: {(1-y).sum()})")
    print(f"  Test samples:  {len(X_test)}")

    N_FOLDS = 5
    SEED = 42
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Containers for raw probabilities and within-fold normalized ranks
    oof_lgb_raw = np.zeros(len(X))
    oof_xgb_raw = np.zeros(len(X))
    oof_cat_raw = np.zeros(len(X))

    oof_lgb_rank = np.zeros(len(X))
    oof_xgb_rank = np.zeros(len(X))
    oof_cat_rank = np.zeros(len(X))

    test_lgb_rank = np.zeros(len(X_test))
    test_xgb_rank = np.zeros(len(X_test))
    test_cat_rank = np.zeros(len(X_test))

    fold_scores = {"lgb": [], "xgb": [], "cat": []}

    # Diversified Topologies:
    # 1. LightGBM: Interaction specialist (deeper trees, leaves=55)
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "num_leaves": 55,
        "max_depth": 7,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "random_state": SEED,
        "verbose": -1,
        "n_jobs": -1,
    }

    # 2. XGBoost: Shallow additive specialist (max_depth=5 prevents high-order noise)
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "random_state": SEED,
        "verbosity": 0,
        "tree_method": "hist",
        "early_stopping_rounds": 100,
        "n_jobs": -1,
    }

    # 3. CatBoost: Symmetric/oblivious tree specialist
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

    print("\n  Executing 5-Fold Stratified Cross-Validation with Within-Fold Ranking...")

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
        p_val_lgb = lgb_model.predict_proba(X_val)[:, 1]
        p_test_lgb = lgb_model.predict_proba(X_test)[:, 1]

        oof_lgb_raw[val_idx] = p_val_lgb
        oof_lgb_rank[val_idx] = rankdata(p_val_lgb) / len(p_val_lgb)
        test_lgb_rank += (rankdata(p_test_lgb) / len(p_test_lgb)) / N_FOLDS
        lgb_auc = roc_auc_score(y_val, p_val_lgb)
        fold_scores["lgb"].append(lgb_auc)

        # 2. XGBoost
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        p_val_xgb = xgb_model.predict_proba(X_val)[:, 1]
        p_test_xgb = xgb_model.predict_proba(X_test)[:, 1]

        oof_xgb_raw[val_idx] = p_val_xgb
        oof_xgb_rank[val_idx] = rankdata(p_val_xgb) / len(p_val_xgb)
        test_xgb_rank += (rankdata(p_test_xgb) / len(p_test_xgb)) / N_FOLDS
        xgb_auc = roc_auc_score(y_val, p_val_xgb)
        fold_scores["xgb"].append(xgb_auc)

        # 3. CatBoost
        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            verbose=0,
        )
        p_val_cat = cat_model.predict_proba(X_val)[:, 1]
        p_test_cat = cat_model.predict_proba(X_test)[:, 1]

        oof_cat_raw[val_idx] = p_val_cat
        oof_cat_rank[val_idx] = rankdata(p_val_cat) / len(p_val_cat)
        test_cat_rank += (rankdata(p_test_cat) / len(p_test_cat)) / N_FOLDS
        cat_auc = roc_auc_score(y_val, p_val_cat)
        fold_scores["cat"].append(cat_auc)

        print(f"    LGB: {lgb_auc:.5f} | XGB: {xgb_auc:.5f} | CAT: {cat_auc:.5f}")

    # ============================================================
    # 5. ENSEMBLE & OPTIMAL WEIGHTING EVALUATION
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Out-of-Fold Validation & Ensemble Comparison")
    print("=" * 60)

    lgb_oof_auc = roc_auc_score(y, oof_lgb_raw)
    xgb_oof_auc = roc_auc_score(y, oof_xgb_raw)
    cat_oof_auc = roc_auc_score(y, oof_cat_raw)

    print(f"  LightGBM OOF AUC : {lgb_oof_auc:.5f} (CV mean: {np.mean(fold_scores['lgb']):.5f} ± {np.std(fold_scores['lgb']):.5f})")
    print(f"  XGBoost  OOF AUC : {xgb_oof_auc:.5f} (CV mean: {np.mean(fold_scores['xgb']):.5f} ± {np.std(fold_scores['xgb']):.5f})")
    print(f"  CatBoost OOF AUC : {cat_oof_auc:.5f} (CV mean: {np.mean(fold_scores['cat']):.5f} ± {np.std(fold_scores['cat']):.5f})")

    # Simple average of raw probabilities
    oof_simple = (oof_lgb_raw + oof_xgb_raw + oof_cat_raw) / 3.0
    simple_auc = roc_auc_score(y, oof_simple)

    # Within-fold rank average (eliminates inter-fold calibration shifts)
    oof_within_fold_rank = (oof_lgb_rank + oof_xgb_rank + oof_cat_rank) / 3.0
    within_fold_rank_auc = roc_auc_score(y, oof_within_fold_rank)

    # Nelder-Mead Optimal Weighted Blend on Within-Fold Ranks
    def loss_func(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        blend = w[0] * oof_xgb_rank + w[1] * oof_cat_rank + w[2] * oof_lgb_rank
        return -roc_auc_score(y, blend)

    opt_res = minimize(loss_func, [0.45, 0.35, 0.20], method="Nelder-Mead")
    opt_weights = opt_res.x / np.sum(opt_res.x)
    oof_opt_weighted = opt_weights[0] * oof_xgb_rank + opt_weights[1] * oof_cat_rank + opt_weights[2] * oof_lgb_rank
    opt_weighted_auc = roc_auc_score(y, oof_opt_weighted)

    # Stacked Logistic Regression on standardized log-odds
    def to_logit(p):
        p_c = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.log(p_c / (1.0 - p_c))

    oof_stack_logits = np.column_stack([
        to_logit(oof_lgb_raw),
        to_logit(oof_xgb_raw),
        to_logit(oof_cat_raw),
    ])
    test_stack_logits = np.column_stack([
        to_logit(test_lgb_rank),
        to_logit(test_xgb_rank),
        to_logit(test_cat_rank),
    ])

    meta_model = LogisticRegression(C=1.0, max_iter=1000)
    meta_model.fit(oof_stack_logits, y)
    oof_stacked = meta_model.predict_proba(oof_stack_logits)[:, 1]
    stacked_auc = roc_auc_score(y, oof_stacked)
    test_stacked = meta_model.predict_proba(test_stack_logits)[:, 1]

    print(f"\n  Ensemble Comparison:")
    print(f"  - Simple Average              OOF AUC: {simple_auc:.5f}")
    print(f"  - Within-Fold Rank Average    OOF AUC: {within_fold_rank_auc:.5f}")
    print(f"  - Optimal Weighted Rank Blend OOF AUC: {opt_weighted_auc:.5f} (Weights: XGB={opt_weights[0]:.3f}, CAT={opt_weights[1]:.3f}, LGB={opt_weights[2]:.3f})")
    print(f"  - Stacked Logit (LR)          OOF AUC: {stacked_auc:.5f}")

    candidates = {
        "Simple Average": (simple_auc, (test_lgb_rank + test_xgb_rank + test_cat_rank) / 3.0),
        "Within-Fold Rank Average": (within_fold_rank_auc, (test_lgb_rank + test_xgb_rank + test_cat_rank) / 3.0),
        "Optimal Weighted Rank Blend": (opt_weighted_auc, opt_weights[0] * test_xgb_rank + opt_weights[1] * test_cat_rank + opt_weights[2] * test_lgb_rank),
        "Stacked Logit LogisticRegression": (stacked_auc, test_stacked),
    }
    best_name = max(candidates.keys(), key=lambda k: candidates[k][0])
    best_auc, best_test_preds = candidates[best_name]

    print(f"\n  >>> WINNING ENSEMBLE: {best_name} (AUC: {best_auc:.5f}, Gini: {2 * best_auc - 1:.5f}) <<<")

    # ============================================================
    # 6. EXPORT SUBMISSIONS & ARTIFACTS
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 5: Generating Output Files")
    print("=" * 60)

    sub_df = pd.DataFrame({"ID": test_ids, "24-Month Default Flag": best_test_preds})
    sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"  Saved main submission: {sub_path} ({len(sub_df)} rows)")

    feat_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": lgb_model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    feat_imp_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    feat_imp.to_csv(feat_imp_path, index=False)
    print(f"  Saved feature importances: {feat_imp_path}")

    metrics_summary = pd.DataFrame({
        "Model": ["LightGBM", "XGBoost", "CatBoost", "Simple Average", "Within-Fold Rank Average", "Optimal Weighted Rank Blend", "Stacked Logit LogisticRegression"],
        "OOF_AUC": [lgb_oof_auc, xgb_oof_auc, cat_oof_auc, simple_auc, within_fold_rank_auc, opt_weighted_auc, stacked_auc],
        "Gini": [2 * x - 1 for x in [lgb_oof_auc, xgb_oof_auc, cat_oof_auc, simple_auc, within_fold_rank_auc, opt_weighted_auc, stacked_auc]],
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
