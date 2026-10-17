# AIFUL 24-Month Default Model (ayush/)

Behavioural credit-scoring pipeline for the AIFUL hackathon: rank customers who survived
their first 12 months by their probability of default by month 24 (metric: ROC AUC).

## Results

| Version | OOF AUC (honest) | Kaggle public |
|---|---|---|
| First GPU pipeline (196 features, 4-model blend) | 0.7549 | 0.76686 |
| + arrears, limit and better application-model features, feature pruning, TabM | 0.7564 | 0.76731 |
| + self-supervised month-13 forecaster, 5 seeds on best model | **0.7572** | **0.76813** |

For reference: honey's pipeline scores 0.7522 OOF; the public leaderboard leader is 0.77152.
The public leaderboard uses ~15.6k rows (~1.3k defaults), so it moves about ±0.007 by chance.
Select submissions by OOF score, not public score.

**OOF AUC (honest)** means the validation fold is never used for early stopping or blend
weights (details under [Validation](#validation)).

## Key findings about the data

1. **The ID encodes the application date.** `ID // 10000 - 21185` is the day of 2018 (0–364),
   and `ID % 10000` is the order within that day. It shows a weekly cycle and a New Year dip,
   and Sep–Dec loans default less (~7.5% vs ~8.5%).
2. **Many "missed payments" are window artefacts.** Windows run from the loan's start day to the
   same day next month, so a payment near the boundary can slide into the next window
   (0 payments, then 2). The average number of empty windows ranges from 0.43 to 0.78 by
   start day of month. Misses that are caught up the next window default at 6%, the same as
   clean payers; real misses default at 10%.
3. **41% of defaulters were clean payers** (no missed windows), and the model ranks them weakly
   (AUC 0.667 within that group). For them, the application score and credit utilisation are the
   strongest signals. Currently delinquent customers are already ranked well (AUC 0.80–0.94).
4. **The minimum-payment schedule can be recovered from the data:** about ¥4k under ¥100k
   balance, ¥8k under ¥200k, ¥11k under ¥400k, ¥13k under ¥500k, and so on.
5. **The data dictionary's units are wrong for `customers.csv`:** Declared Annual Income and
   Rent Burden are already in yen, not ¥10k / ¥1k.
6. **No repeat-customer leak:** only 93 of 205k applications share a full profile.

## Pipeline

```
customers.csv ─┐
train.csv ─────┼─► features.py ──► cache/features.pkl (255 features)
test.csv ──────┘        │
                        ├─► select_features.py ──► cache/selected_features.json (129 kept)
                        ├─► forecast.py ─────────► cache/forecast_features.pkl (12 features)
                        ▼
        train_gbm.py (xgb / xgbl / cat) · train_tabm.py · train_nn.py
                        │  preds/<model>.npz  (OOF + test predictions)
                        ▼
                    blend.py ──► submission.csv, submission_equal_weight.csv, validation_metrics.csv
```

Everything trains on the GPU: XGBoost (CUDA), CatBoost (GPU) and PyTorch.

## Preprocessing

| Step | Treatment |
|---|---|
| Join | train + test stacked and merged with `customers.csv` on `ID`. The application-time values of Marital Status, Insurance Type and Company Size are kept as `*_app`. |
| Cumulative → monthly | The "Past k Months" columns are differenced into amounts and counts per month (no negative differences exist). |
| Bad values | inf → NaN. Income, debt and rent clipped at ≥ 0 inside ratios. Negative work experience (7,588 rows) is flagged and clipped to 0 in ratios; the raw value is kept. |
| Missing values | **GBMs:** native NaN handling. **Neural networks:** median fill. |
| Outliers / scaling | **GBMs:** none needed (trees split on rank order). **Neural networks:** quantile-normal transform (`train_nn.py`), piecewise-linear quantile encoding (TabM), `log1p` + standardisation for the monthly sequences. |
| Categories | **XGBoost:** numeric codes. **CatBoost:** native string categories. |
| Imbalance (8.3%) | No resampling or class weights, because AUC only depends on ranking. Folds are stratified; early stopping uses AUC. |

## Features (`features.py`, plus `forecast.py`)

| Group | # | Content |
|---|---|---|
| `raw` | 64 | Original columns from train/test and `customers.csv` |
| `base` | 62 | Monthly flows; balance slope/mean/std/min/max and changes; net cash flow; repayment and borrowing trends; months borrowing without paying |
| `delinq` | 14 | Missed-window streaks, lapses, cures, missed-payment patterns, auto-debit misses |
| `minpay` | 12 | Inferred minimum due; payment/minimum ratios; short, exact-minimum and over-paying months; shortfall |
| `arrears` | 41 | Arrears that ignore window artefacts: one payment and one minimum owed per window, extra payments clear arrears. Count and amount arrears, streaks, cumulative gaps, real vs caught-up misses, boundary-day flag, per-window arrears and minimum due |
| `util` | 14 | Peak and rounded utilisation, headroom, implied interest, principal paid |
| `limit` | 12 | Estimated credit limit (≥ peak balance, near the desired limit); utilisation against it, borrowing near the limit, draws against remaining headroom |
| `date` | 8 | Application day, weekday, day of month, month, order within the day, payday gap |
| `ratio` | 18 | Debt-to-income, the 1/3-of-income legal cap (総量規制), rent/income, work/age, changes since application |
| `app` | 10 | Application model on **all 205k customers** (GPU XGBoost + CatBoost, out-of-fold, 3 classes: active / default / closed at month 12) |
| `forecast` | 12 | Month-13 forecasts and month-12 surprise features (below) |

Key single-feature AUCs: `arrears_amt_months_now` 0.667 and `arrears_now` 0.661, against 0.620
for the old missed-streak count. `fc_arrears` scores 0.683.

### Self-supervised next-month forecaster (`forecast.py`)
Default means 3 consecutive missed payments in months 13–24, so month-13 behaviour matters most.
GPU XGBoost models learn month-to-month dynamics on **all 125k accounts (train + test,
no target labels used)**:

- **Training rows:** windows t+1..t+4 → outcome in window t, for t ∈ {2, 1}.
- **Application:** windows 1..4 → forecast month 13 (miss, borrowing, arrears,
  payment/minimum ratio, balance change).
- **Surprise features:** month-12 actual minus the out-of-fold forecast.
- **Folds by account:** an account's forecast never comes from a model trained on that account.

| Check | Result |
|---|---|
| Forecast quality on month 12 | miss AUC 0.82, borrowing AUC 0.88, arrears R² 0.73 |
| Calibration | forecast month-13 miss rate 9.3%, against 8.9% / 9.3% actual in months 11–12 |
| Train/test shift | train-vs-test classifier AUC 0.499 (no shift) |
| Gain | +0.0004 to +0.0009 per model; about 75% of the latest blend gain |

### Feature selection (`select_features.py`)
Null importance: a feature is kept if its real gain beats the 75th percentile of its gain
across 30 shuffled-label runs. 129 of 255 are kept, and the selected set beats the full set
on every repeat.

## Models

| Model | Description | Repeat-averaged OOF |
|---|---|---|
| `xgbl_sel_fc` | XGBoost, leaf-wise (LightGBM-style), selected + forecast features, 5 seeds | **0.7569** |
| `xgb_sel_fc` | XGBoost, depth-wise, selected + forecast features | 0.7562 |
| `xgbl_sel` / `xgb_sel` | The same without forecast features | 0.7562 / 0.7552 |
| `cat` | CatBoost GPU, native categories | 0.7534 |
| `tabm` | TabM: 32 ensemble members sharing one MLP, piecewise-linear embeddings, plus a GRU over the 6 monthly windows | 0.7491 |

LightGBM was replaced by XGBoost leaf-wise trees because the pip LightGBM build has no GPU
support. Settings were tuned with Optuna (`tune.py`, saved to `cache/best_params_*.json`).

### Validation
- **Folds:** 5-fold stratified CV, repeated 3–5 times with different seeds.
- **Training length:** each fold picks it on an **inner 10% holdout** of its training part,
  then refits on the whole training part. The validation fold is never used for stopping.
- **Ranking:** neural-network predictions are ranked within each fold before pooling, because
  their scale shifts between folds.
- **Blending:** rank-average weights fitted with Nelder–Mead. The reported blend score is
  **nested** (weights fitted on 4/5 of the rows, scored on the other 1/5).

### Current blend (nested OOF 0.75718)
Weights: `xgbl_sel_fc` 0.62 · `xgbl_sel` 0.22 · `tabm` 0.13 · `cat` 0.03.
The equal-weight version (0.75670) is written as `submission_equal_weight.csv`: the hedge
to use as the second final pick.

## What did not help

| Idea | Result |
|---|---|
| Daily vintage averages (default rate per application day) | ±0 |
| Implied interest rate as a risk grade | Clusters at the legal 18%; no signal |
| More near-identical models in the blend | +0.00004 (they correlate at 0.98–0.99) |
| ARIMA per customer | Not feasible: 6–7 points per series |
| Transformer over the sequence | Not worth it for 6 steps; the GRU already covers it |

## How to run (PowerShell, venv `C:\venvs\prism`)

```powershell
cd ayush
python features.py                    # build cache/features.pkl (~5-10 min)
python select_features.py 30          # feature pruning
python forecast.py                    # month-13 forecaster features
python tune.py xgbl 40 selected       # optional re-tuning

python train_gbm.py xgbl --repeats 5 --feats selected --extra forecast --tag xgbl_sel_fc
python train_gbm.py xgb  --repeats 3 --feats selected --extra forecast --tag xgb_sel_fc
python train_gbm.py xgbl --repeats 3 --feats selected --tag xgbl_sel
python train_gbm.py cat  --repeats 3
python train_tabm.py --repeats 1

python blend.py xgbl_sel_fc xgbl_sel xgb_sel_fc xgb_sel cat tabm
```

Requirements: `xgboost==3.0.5` (built for CUDA 12; 3.4 needs a CUDA 13 driver and silently
falls back to the CPU), `catboost`, `torch` (CUDA), `optuna`, `scikit-learn`, `pandas`.
`cache/` and `preds/` are git-ignored (`features.pkl` is about 200 MB).

## Files

| File | Purpose |
|---|---|
| `features.py` | Preprocessing, all feature groups, application model, feature cache |
| `forecast.py` | Self-supervised month-13 forecaster |
| `select_features.py` | Null-importance feature selection |
| `ablation.py` | Feature-group ablation |
| `tune.py` | Optuna tuning for the XGBoost models |
| `train_gbm.py` | XGBoost / CatBoost training with honest CV |
| `train_tabm.py` | TabM + GRU deep model |
| `train_nn.py` | Earlier MLP + GRU network (superseded by TabM) |
| `blend.py` | Rank blending, nested score, writes the submissions |
