"""
Credit Scoring Model — Full Training Data Pipeline
===================================================
1. Evaluates 5-fold Stratified CV to discover optimal early-stopping iterations
   for LightGBM, XGBoost, and CatBoost.
2. Computes the median optimal iterations (scaled for 100% data).
3. Trains final models on 100% of the training data (all 93,987 samples).
4. Generates test predictions and saves to submission_full_train.csv.
5. Compares alignment (correlation) with the 5-fold bagged submission.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Allow importing from honey folder
sys.path.insert(0, os.path.dirname(__file__))
from credit_model import load_and_merge_data, engineer_features, rank_average

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = os.path.dirname(__file__)


def main():
    print("=" * 65)
    print("FULL TRAINING DATA PIPELINE (100% Training Dataset)")
    print("=" * 65)

    # 1. Load and engineer features
    train_df, test_df = load_and_merge_data()
    train_feat = engineer_features(train_df)
    test_feat = engineer_features(test_df)

    TARGET = "24-Month Default Flag"
    DROP_COLS = ["ID", TARGET]
    feature_cols = [c for c in train_feat.columns if c not in DROP_COLS]
    feature_cols = train_feat[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    train_feat[feature_cols] = train_feat[feature_cols].replace([np.inf, -np.inf], np.nan)
    test_feat[feature_cols] = test_feat[feature_cols].replace([np.inf, -np.inf], np.nan)

    X_full = train_feat[feature_cols].values
    y_full = train_feat[TARGET].values
    X_test = test_feat[feature_cols].values
    test_ids = test_feat["ID"].values

    print(f"\nTotal training records: {len(X_full)} (Defaults: {y_full.sum()})")
    print(f"Total test records:     {len(X_test)}")
    print(f"Total features:         {len(feature_cols)}")

    # 2. Stage 1: Run 5-fold CV to capture best_iteration per model
    print("\n" + "=" * 65)
    print("STEP 1: Determining Optimal Iterations via 5-Fold Stratified CV")
    print("=" * 65)

    N_FOLDS = 5
    SEED = 42
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    lgb_best_iters = []
    xgb_best_iters = []
    cat_best_iters = []

    lgb_base_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "num_leaves": 55,
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

    xgb_base_params = {
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

    cat_base_params = {
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

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        X_tr, X_val = X_full[train_idx], X_full[val_idx]
        y_tr, y_val = y_full[train_idx], y_full[val_idx]

        # LightGBM
        m_lgb = lgb.LGBMClassifier(**lgb_base_params)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
        best_lgb_it = m_lgb.best_iteration_
        lgb_best_iters.append(best_lgb_it)

        # XGBoost
        m_xgb = xgb.XGBClassifier(**xgb_base_params)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        best_xgb_it = m_xgb.best_iteration
        xgb_best_iters.append(best_xgb_it)

        # CatBoost
        m_cat = CatBoostClassifier(**cat_base_params)
        m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        best_cat_it = m_cat.get_best_iteration()
        cat_best_iters.append(best_cat_it)

        print(f"  Fold {fold+1}: LightGBM={best_lgb_it:4d} iters | XGBoost={best_xgb_it:4d} iters | CatBoost={best_cat_it:4d} iters")

    # Compute Median Optimal Iterations
    lgb_med = int(np.median(lgb_best_iters))
    xgb_med = int(np.median(xgb_best_iters))
    cat_med = int(np.median(cat_best_iters))

    # Scale by 1.10 to account for 25% larger dataset (100% vs 80%)
    lgb_full_iters = int(lgb_med * 1.10)
    xgb_full_iters = int(xgb_med * 1.10)
    cat_full_iters = int(cat_med * 1.10)

    print("\nOptimal Iteration Summary:")
    print(f"  LightGBM : Folds={lgb_best_iters} -> Median={lgb_med} -> Full Data Target={lgb_full_iters}")
    print(f"  XGBoost  : Folds={xgb_best_iters} -> Median={xgb_med} -> Full Data Target={xgb_full_iters}")
    print(f"  CatBoost : Folds={cat_best_iters} -> Median={cat_med} -> Full Data Target={cat_full_iters}")

    # 3. Stage 2: Train Final Models on 100% of Training Data
    print("\n" + "=" * 65)
    print("STEP 2: Training on 100% of Train Data (93,987 Samples)")
    print("=" * 65)

    # 3.1 LightGBM Full
    print(f"  [1/3] Training LightGBM on full dataset ({lgb_full_iters} trees)...")
    lgb_full_params = lgb_base_params.copy()
    lgb_full_params["n_estimators"] = lgb_full_iters
    full_lgb_model = lgb.LGBMClassifier(**lgb_full_params)
    full_lgb_model.fit(X_full, y_full)
    test_preds_lgb = full_lgb_model.predict_proba(X_test)[:, 1]

    # 3.2 XGBoost Full
    print(f"  [2/3] Training XGBoost on full dataset ({xgb_full_iters} trees)...")
    xgb_full_params = xgb_base_params.copy()
    xgb_full_params["n_estimators"] = xgb_full_iters
    xgb_full_params.pop("early_stopping_rounds", None)
    full_xgb_model = xgb.XGBClassifier(**xgb_full_params)
    full_xgb_model.fit(X_full, y_full, verbose=False)
    test_preds_xgb = full_xgb_model.predict_proba(X_test)[:, 1]

    # 3.3 CatBoost Full
    print(f"  [3/3] Training CatBoost on full dataset ({cat_full_iters} trees)...")
    cat_full_params = cat_base_params.copy()
    cat_full_params["iterations"] = cat_full_iters
    cat_full_params.pop("early_stopping_rounds", None)
    full_cat_model = CatBoostClassifier(**cat_full_params)
    full_cat_model.fit(X_full, y_full, verbose=0)
    test_preds_cat = full_cat_model.predict_proba(X_test)[:, 1]

    # 4. Ensemble Full Models (Optimal Weighted Rank Blend)
    print("\n" + "=" * 65)
    print("STEP 3: Ensembling Full Data Predictions (Optimal Weighted Rank Blend)")
    print("=" * 65)
    r_xgb = rankdata(test_preds_xgb) / len(test_preds_xgb)
    r_cat = rankdata(test_preds_cat) / len(test_preds_cat)
    r_lgb = rankdata(test_preds_lgb) / len(test_preds_lgb)
    test_full_rank = 0.429 * r_xgb + 0.376 * r_cat + 0.195 * r_lgb

    sub_full = pd.DataFrame({
        "ID": test_ids,
        "24-Month Default Flag": test_full_rank
    })
    sub_path = os.path.join(OUTPUT_DIR, "submission_full_train.csv")
    sub_full.to_csv(sub_path, index=False)
    print(f"  Saved Full Train submission: {sub_path} ({len(sub_full)} rows)")

    # 5. Compare with 5-fold Bagged Submission
    bagged_sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
    if os.path.exists(bagged_sub_path):
        sub_bagged = pd.read_csv(bagged_sub_path)
        corr, _ = spearmanr(sub_full["24-Month Default Flag"], sub_bagged["24-Month Default Flag"])
        print("\n" + "=" * 65)
        print("COMPARISON: Full Training Data vs 5-Fold Bagging")
        print("=" * 65)
        print(f"  Spearman Rank Correlation: {corr:.5f}")
        diff = np.abs(sub_full["24-Month Default Flag"].values - sub_bagged["24-Month Default Flag"].values)
        print(f"  Mean Absolute Difference : {diff.mean():.5f}")
        print(f"  Max Absolute Difference  : {diff.max():.5f}")

    print("\nFull training data pipeline complete!")


if __name__ == "__main__":
    main()
