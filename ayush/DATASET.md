# AIFUL Dataset Guide

What the data represents, what every column means, the quirks found in it, and the features
built from it. Pipeline and results are in [README.md](README.md).

## 1. What the data represents

Anonymised accounts from **AIFUL's unsecured revolving credit product** (a Japanese consumer
"card loan"), opened in **2018**.

- **Credit limit:** set by AIFUL at contract time; the customer can borrow up to it at any time
  (app transfer, convenience-store ATMs, AIFUL machines).
- **Minimum repayment:** due every month, based on the balance. Customers may pay more, at any
  time. Interest is deducted first and the rest reduces the balance.
- **Delinquent:** no payment by the monthly due date.
- **Default:** **no payment for 3 consecutive months**. The contract is terminated.

### Timeline

```
Month 0        Months 1-5      Months 6-12              Month 12            Months 13-24
application -> (no data)    -> behaviour window      -> observation point -> outcome window
customers.csv                  train.csv / test.csv     (reference date)    target: default by 24?
```

**Task:** for customers who reached month 12 **without defaulting** (and still have an open,
non-zero account), rank their probability of being in default by month 24. Metric: ROC AUC.

### Files and population

| File | Rows | Content |
|---|---|---|
| `customers.csv` | 205,637 | Application data for **every** 2018 customer |
| `train.csv` | 93,987 | Month-12 snapshot + months 6–12 behaviour + **target** |
| `test.csv` | 31,330 | Same without the target |
| `sample_submission.csv` | 31,330 | `ID, 24-Month Default Flag` |

`customers.csv` breaks down as:

| Group | Rows | In train/test? |
|---|---|---|
| Active at month 12 | 125,317 | Yes (93,987 train + 31,330 test) |
| Defaulted within 12 months (`12-Month Default Flag = 1`) | 19,900 | No |
| Closed or fully repaid before month 12 | 60,420 | No |

- **Target rate:** 8.27% of train (7,771 defaults).
- **Join key:** `ID`, present in all files.
- **Train/test split:** random, with test at about 25% of every application month.

## 2. `customers.csv`: application data (captured at month 0)

| Column | Meaning | Values |
|---|---|---|
| `ID` | Customer identifier (**encodes the application date**, see §4) | |
| `12-Month Default Flag` | In default at month 12 | 1 = default, 0 = not |
| `Application Channel` | How the customer applied | 1 PC, 17 iPhone, 18 Android, 20 contract office |
| `Gender` | | 1 male, 2 female |
| `Marital Status` | At application | 1 single, 2 married |
| `Age (in months)` | Age at application | 228–840 (19–70 years) |
| `Residence Period (in months)` | Time at current address | |
| `Occupation Category` | Job type | Codes 140–1540 (table below) |
| `Industry Category` | Employer's industry | 1–19, 99 (table below) |
| `Employment Type Category` | Contract type | 1–8 (table below) |
| `Insurance Type` | Health insurance | 0 unknown, 1 company social, 3 company national, 4 self-employed national |
| `Company Size Category` | Employer size | 1–9 (table below) |
| `Business Entity Type` | Employer's legal form | 0 unknown, 1–4 corporation types, 5–7 sole-proprietor types, 9 other |
| `Declared Annual Income` | Self-declared income | **yen** (see §4) |
| `Payday` | Salary day of month | 1–31 |
| `Declared Number of Borrowings` | Loans with other lenders | 0–9 |
| `Declared Borrowing Amount` | Total owed to other lenders | yen |
| `Desired Credit Limit` | Limit the customer asked for | yen, 10k–8M |
| `Residence Type` | | 1 own house, 2 own condo, 3 public rental, 4 private rental, 5 municipal rental, 6 dormitory/sublet |
| `Rent Burden` | Monthly rent | **yen** (see §4) |
| `Residence Ownership Status` | | 0 unknown, 1 sole, 2 joint, 3 not owner |
| `Number of Dependents` / `Number of Dependent Children` | People supported | |
| `Home` / `Mobile` / `Employer Phone Number Available`, `Driver Licence Number Available` | Contact details given | 0/1 |

**Lookup tables**

| Occupation code | | Employment type | | Company size | |
|---|---|---|---|---|---|
| 140 | Admin / business owner | 1 | Permanent | 1 | Government |
| 240 | Sales (fixed salary) | 2 | Agency temp | 2 | Listed company |
| 340 | Sales (commission) | 3 | Seconded | 3 | 500+ employees |
| 440 | Factory / skilled trade | 4 | Contract | 4 | 100+ |
| 540 | Retail / customer service | 5 | Part-time / casual | 5 | 50+ |
| 640 | Labourer / security | 6 | President / representative | 6 | 20+ |
| 740 | Driver / transport | 7 | Commissioned | 7 | 5+ |
| 840 | Healthcare | 8 | Fixed-term contract | 8 | Fewer than 5 |
| 940 | Licensed professional | | | 9 | Self-employed |
| 1040 | Civil servant | | | | |
| 1140 | IT | | | | |
| 1240 | Arts & design | | | | |
| 1340 | Service industry | | | | |
| 1440 | Media / entertainment | | | | |
| 1540 | Hospitality / nightlife | | | | |

Industry codes: 1 manufacturing, 2 finance, 3 securities, 4 insurance, 5 construction,
6 real estate, 7 telecom, 8 wholesale/retail, 9 transport, 10 printing, 11 broadcasting,
12 publishing, 13 advertising, 14 research, 15 healthcare, 16 education, 17 government/associations,
18 agriculture/mining, 19 student, 99 other.

## 3. `train.csv` / `test.csv`: behaviour at the month-12 reference date

**Reference date:** exactly 12 months after the loan was issued.
- **Balances** are at the **last month-end before** it.
- **Flow columns** (borrowing, repayment) are **cumulative windows ending on it**: "Past 3 Months"
  includes Past 1 and Past 2.

Example from the data dictionary: loan issued 2018-01-05, so the reference date is 2019-01-05.
"Outstanding Loan Balance" is the 2018-12-31 balance; "Repayment Amount (Past 1 Month)" covers
2018-12-05 → 2019-01-05.

| Column | Meaning |
|---|---|
| `24-Month Default Flag` | **Target** (train only): in default at month 24 |
| `Marital Status`, `Insurance Type`, `Company Size Category` | Value **at month 12**; can differ from the application value |
| `Work Experience (months)` | Job tenure at month 12 |
| `Income Proof Submitted` | Payslips etc. provided (0/1) |
| `Total Annual Income` | Income from proof documents, or self-declared if none (yen) |
| `Auto-Debit Flag` | Repays by automatic bank withdrawal (0/1) |
| `Number of Loan Settlements` | Loan settlements in the first 12 months |
| `Outstanding Loan Balance` | Balance at the last month-end (month 12) |
| `Balance 1 … 6 Months Ago` | Balances at the 6 earlier month-ends (months 11 … 6) |
| `Additional Borrowing Amount (Past 1…6 Months)` | Cumulative new borrowing (yen) |
| `Additional Borrowing Count (Past 1…6 Months)` | Cumulative number of draws |
| `Repayment Amount (Past 1…6 Months)` | Cumulative repayments (yen) |
| `Repayment Count (Past 1…6 Months)` | Cumulative number of repayments |

## 4. Data quality issues and findings

| Finding | Evidence / handling |
|---|---|
| **The ID encodes the application date** | `ID // 10000 − 21185` = day of 2018 (0–364); `ID % 10000` = order within the day. There's a weekly cycle (Mondays busiest), a New Year dip, and Sep–Dec loans default less (~7.5% vs ~8.5%). |
| **The dictionary's units are wrong** | It says Declared Annual Income is in ¥10k and Rent Burden in ¥1k, but the medians are 3,000,000 and 11,000: both are already **yen**. |
| **Missed windows are partly artefacts** | Windows run from the loan's start day to the same day next month, so payments near the boundary slide between windows (0, then 2). The average number of empty windows ranges from **0.43 to 0.78** by start day. Caught-up misses default at 6%; real misses at 10%. |
| Negative work experience | 7,588 rows; flagged, and clipped to 0 in ratios |
| Zero income / extreme outliers | 69 rows with income = 0; some values up to ¥9.5e10. Clipped at ≥ 0 in ratios; trees are unaffected by outliers. |
| Missing values | None in the raw files; ratio features can produce some |
| Cumulative columns | Internally consistent: no negative month-on-month differences |
| Interest rate | Implied interest is about 1.5% a month (18% a year, the legal cap for this loan size): no hidden risk grade |
| Minimum payment schedule | Can be recovered from repayments: ~¥4k under ¥100k, ¥8k under ¥200k, ¥11k under ¥400k, ¥13k under ¥500k, … |
| Repeat customers | Only 93 of 205k share a full profile: no exploitable link |

**Where the defaults are** (train):

| Segment | Share of defaulters | Default rate |
|---|---|---|
| Clean payers (no missed windows) | 41% | 5.3% |
| Missed before, paying now | 27% | 8.8% |
| Missed the last window | 19% | 20% |
| Missed the last 2 windows | 8% | 60% |

## 5. Preprocessing

| Step | Treatment |
|---|---|
| Join | train + test stacked and merged with `customers.csv` on `ID`. Application-time values of the 3 time-varying columns are kept as `*_app`. |
| Cumulative → monthly | `month_k = Past_k − Past_(k−1)` for all four flow types |
| Bad values | inf → NaN. Income, debt and rent clipped ≥ 0 in ratios. Negative work experience is flagged. |
| Missing values | **Trees:** native. **Neural networks:** median fill. |
| Scaling | **Trees:** none. **Neural networks:** quantile-normal or piecewise-linear quantile encoding; `log1p` + standardisation for sequences. |
| Categories | **XGBoost:** numeric codes. **CatBoost:** native categories. |
| Imbalance | None needed: AUC is rank-based. Folds are stratified. |

## 6. Feature engineering (`features.py`, `forecast.py`)

Notation: window k = 1 is the most recent month (ending at month 12), and k = 6 the oldest.
`B_k` = month-end balance before window k (`B_0` = month 12).

| Group | # | Main features |
|---|---|---|
| **raw** | 64 | All original columns |
| **base** | 62 | **Monthly flows:** `ra/rc/ba/bc_m1..6` (repayment and borrowing amounts and counts). **Balance shape:** slope (least-squares over 7 points), mean, std, min, max, coefficient of variation. **Balance changes:** 1 / 3 / 6-month changes, absolute and %, monthly diffs, months up and down, acceleration. **Cash flow:** net flow, repayment and borrowing trends (last 3 ÷ prior 3), repayment/balance, repayment/borrowing, average draw and payment size, months borrowing without paying. |
| **delinq** | 14 | Missed-window streak now and longest, number missed, recoveries, lapses, months since last payment, missed-payment pattern (6-bit code), auto-debit misses |
| **minpay** | 12 | Minimum due inferred per balance level; payment/minimum ratio per month, mean and min; months short, at exactly the minimum, or over 2×; shortfall; excess paid |
| **arrears** | 41 | **Artefact-free delinquency.** One payment and one minimum are owed per window with a positive balance; extra payments clear arrears but do not bank credit. Features: arrears by count and amount (now, max, streak, per window), cumulative gap over the last 1/2/3/6 windows (0-then-2 nets to 0), real misses vs. caught-up misses, boundary-day flag, misses away from the boundary |
| **util** | 14 | Utilisation against the peak balance and against a rounded limit, headroom, implied interest and rate, interest share of payments, principal paid |
| **limit** | 12 | Estimated credit limit = max(desired limit, peak balance rounded up to ¥10k); utilisation against it, peak and current; borrowing while near the limit; draws ÷ remaining headroom; headroom ÷ income |
| **date** | 8 | From the ID: application day, order within the day, weekday, day of month, month, daily application volume, payday gap |
| **ratio** | 18 | Verified ÷ declared income, debt-to-income (AIFUL only and total), remaining room under the **legal 1/3-of-income cap** (総量規制), minimum due ÷ monthly income, rent ÷ income, work tenure ÷ adult age, residence ÷ age, changes since application in marital status, insurance and company size |
| **app** | 10 | **Application model on all 205k customers:** GPU XGBoost + CatBoost, out-of-fold, predicting active / default / closed at month 12 from application data only. Probability of early default, of closure, and early-default vs. survival odds. |
| **forecast** | 12 | **Self-supervised month-13 forecaster** trained on all 125k accounts (no target labels): forecast miss, borrowing, arrears, payment/minimum ratio and balance change, plus month-12 "surprise" (actual minus forecast) |

**Feature selection:** the null-importance test keeps 129 of 255 base features; the 12
forecast features are added on top, for 141 in the best model.

### Most important features (best model, mean |SHAP|)

1. `fc_arrears`: forecast month-13 arrears
2. `bal_chg_1` / `bal_pct_1`: balance change in the last month
3. `Work Experience (months)`
4. `income_ratio`: verified ÷ declared income
5. `bal_diff_1`, `bal_diff_2`: recent month-on-month balance changes
6. `app_p_def12`: application profile resembles early defaulters
7. `app_month`: application month (vintage)
8. `Age (in months)`
9. `borrow_trend`: recent vs. earlier borrowing
10. `int_share_of_pay`, `cum_gap_2/3`, `arrears_now`, `util_round`

Share of importance by group: base 20%, raw 18%, arrears 14%, forecast 11%, ratio 10%,
app 8%, util 7%, minpay 4%, date 3%, limit 3%, old delinq 2%.
The full ranking is in [feature_importance.csv](feature_importance.csv).
