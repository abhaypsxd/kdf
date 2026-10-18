"""
Figures for README.md / the presentation, written to plots/*.png.

Static PNGs on a light surface (slides and GitHub). Palette: reference data-viz palette
(series blue #2a78d6, orange #eb6834; text #0b0b0b / #52514e; surface #fcfcfb).
Needs the caches from features.py and forecast.py.

Usage:  python make_plots.py
"""

import json
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import CACHE, HERE, TARGET, get_features, load_raw

warnings.filterwarnings("ignore")
OUT = os.path.join(HERE, "plots")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def bars(ax, labels, values, title, fmt="{:.1f}%", color=BLUE, highlight=None, xlabel=None,
         ylabel="Default by month 24"):
    """Vertical bars with a 2px surface gap and a value label on each bar."""
    colors = [ORANGE if highlight is not None and i in highlight else color for i in range(len(values))]
    b = ax.bar(range(len(values)), values, color=colors, edgecolor=SURFACE, linewidth=2, width=0.75)
    ax.set_xticks(range(len(values)), labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(values) * 1.18)
    for rect, v in zip(b, values):
        ax.text(rect.get_x() + rect.get_width() / 2, v + max(values) * 0.02, fmt.format(v),
                ha="center", va="bottom", fontsize=10, color=INK2)


def rate(y, groups):
    """Default rate (%) per group; ordered bands (pd.cut / qcut) keep their order."""
    g = pd.Series(groups).reset_index(drop=True)
    return pd.Series(y).groupby(g, observed=True, sort=True).mean() * 100


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=160)
    plt.close(fig)
    print("wrote plots/" + name)


def main():
    os.makedirs(OUT, exist_ok=True)
    df, groups = get_features()
    customers, _, _ = load_raw()
    with open(os.path.join(CACHE, "forecast_features.pkl"), "rb") as f:
        df = df.merge(pickle.load(f), on="ID", how="left")
    tr = df[df.is_train == 1].reset_index(drop=True)
    y = tr[TARGET].values

    # 1. Population: where the 205k applications went, and the target balance
    pop_ids = set(df["ID"])
    status = np.where(customers["12-Month Default Flag"] == 1, "Defaulted\nby month 12",
                      np.where(customers["ID"].isin(pop_ids), "Active at\nmonth 12", "Closed / repaid\nby month 12"))
    counts = pd.Series(status).value_counts().reindex(
        ["Active at\nmonth 12", "Closed / repaid\nby month 12", "Defaulted\nby month 12"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    bars(a, counts.index, counts.values / 1000, "205,637 applications by month-12 status",
         fmt="{:.0f}k", ylabel="Customers (thousands)", highlight=None)
    a.text(0, counts.values[0] / 1000 * 0.5, "scored:\ntrain 94.0k\ntest 31.3k", ha="center",
           va="center", color=SURFACE, fontsize=10, fontweight="bold")
    n_def = int(y.sum())
    bars(b, ["No default", "Default"], [len(y) - n_def, n_def], "Train set: 24-month default (8.27%)",
         fmt="{:,.0f}", ylabel="Customers", highlight=[1])
    save(fig, "01_population.png")

    # 2. Who defaults: application profile (small multiples)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    age = pd.cut(tr["Age (in months)"] / 12, [18, 25, 30, 40, 50, 60, 80],
                 labels=["<25", "25-30", "30-40", "40-50", "50-60", "60+"])
    r = rate(y, age)
    bars(axes[0, 0], r.index.astype(str), r.values, "By age", highlight=[0])
    res = {1: "Own\nhouse", 2: "Own\ncondo", 3: "Public\nrental", 4: "Private\nrental",
           5: "Municipal\nrental", 6: "Dormitory"}
    r = rate(y, tr["Residence Type"]).rename(index=res)
    bars(axes[0, 1], r.index, r.values, "By housing", highlight=[2])
    axes[0, 1].tick_params(axis="x", labelsize=9)
    ch = {1: "PC", 17: "iPhone", 18: "Android", 20: "Contract\noffice"}
    r = rate(y, tr["Application Channel"]).rename(index=ch)
    bars(axes[1, 0], r.index, r.values, "By application channel", highlight=[1])
    r = rate(y, tr["Declared Number of Borrowings"].clip(upper=4)).rename(index={4: "4+"})
    bars(axes[1, 1], r.index.astype(str), r.values, "By loans held with other lenders",
         highlight=[len(r) - 1])
    save(fig, "02_profile.png")

    # 3. Vintage effect from the ID date
    r = rate(y, tr["app_month"])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    bars(ax, months, r.values, "Default rate by application month (decoded from the ID)",
         highlight=[8, 9, 10])
    save(fig, "03_vintage.png")

    # 4. The missed-payment artefact
    rc = np.column_stack([tr[f"rc_m{k}"] for k in range(1, 7)])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    d = pd.cut(tr["app_dom"], [0, 5, 10, 15, 20, 25, 31], labels=["1-5", "6-10", "11-15", "16-20", "21-25", "26-31"])
    m = pd.Series((rc == 0).sum(1)).groupby(d.values).mean()
    bars(a, m.index.astype(str), m.values, "Empty payment windows depend on loan start day",
         fmt="{:.2f}", ylabel="Avg empty windows (of 6)", xlabel="Day of month the loan started",
         highlight=[0, 5])
    seg = np.where(tr["n_miss"] == 0, "Never\nmissed",
                   np.where(tr["n_true_miss"] == 0, "Only caught-up\nmisses", "Genuine\nmisses"))
    r = rate(y, seg).reindex(["Never\nmissed", "Only caught-up\nmisses", "Genuine\nmisses"])
    bars(b, r.index, r.values, "Caught-up misses behave like perfect payers", highlight=[2])
    save(fig, "04_missed_payment_artefact.png")

    # 5. Arrears: the artefact-free delinquency feature
    r = rate(y, tr["arrears_now"].clip(upper=4)).rename(index={4.0: "4+"})
    fig, ax = plt.subplots(figsize=(8, 4))
    bars(ax, [str(int(i)) if i != "4+" else i for i in r.index], r.values,
         "Default rate by payments in arrears at month 12", xlabel="Payments behind (arrears_now)")
    save(fig, "05_arrears.png")

    # 6. Consumer behaviour: utilisation, stuck revolvers, minimum payment
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    u = pd.cut(tr["util_guess"], [-0.01, 0.25, 0.5, 0.75, 0.9, 10], labels=["<25%", "25-50%", "50-75%", "75-90%", "90-100%"])
    r = rate(y, u)
    bars(axes[0], r.index.astype(str), r.values, "Utilisation of estimated limit", highlight=[4])
    bc = pd.cut(tr["bal_pct_3"], [-10, -0.2, -0.05, 0.05, 0.2, 100],
                labels=["fell >20%", "fell 5-20%", "flat ±5%", "rose 5-20%", "rose >20%"])
    r = rate(y, bc)
    bars(axes[1], r.index.astype(str), r.values, "Balance change, last 3 months", highlight=[2])
    axes[1].tick_params(axis="x", labelsize=9)
    r = rate(y, tr["n_exact_min"].clip(upper=6))
    bars(axes[2], r.index.astype(int).astype(str), r.values, "Months paying exactly the minimum",
         xlabel="Months (of 6)", highlight=[1, 2, 3])
    save(fig, "06_behaviour.png")

    # 7. Application-risk score (model trained on all 205k applications)
    r = rate(y, pd.qcut(tr["app_p_def12"], 5, labels=["1\nsafest", "2", "3", "4", "5\nriskiest"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    bars(ax, r.index.astype(str), r.values, "Default rate by application-risk score (app_p_def12)",
         xlabel="Fifth of customers", highlight=[4])
    save(fig, "07_application_risk.png")

    # 8. Null-importance selection: kept vs dropped per family
    keep = set(json.load(open(os.path.join(CACHE, "selected_features.json"))))
    names = {"raw": "Raw columns", "base": "Monthly flows & balance", "arrears": "Arrears",
             "ratio": "Affordability ratios", "util": "Utilisation", "delinq": "Missed-window counts",
             "minpay": "Minimum payment", "limit": "Credit limit", "app": "Application risk", "date": "Date from ID"}
    rows = [(names[g], sum(c in keep for c in groups[g]), len(groups[g])) for g in names]
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(9, 5))
    lab = [r[0] for r in rows]
    kept = np.array([r[1] for r in rows])
    tot = np.array([r[2] for r in rows])
    ax.barh(lab, kept, color=BLUE, edgecolor=SURFACE, linewidth=2, label="Kept")
    ax.barh(lab, tot - kept, left=kept, color=GREY, edgecolor=SURFACE, linewidth=2, label="Dropped")
    for i, (k, t) in enumerate(zip(kept, tot)):
        ax.text(t + 0.8, i, f"{k}/{t}", va="center", fontsize=10, color=INK2)
    ax.set_title("Null-importance selection: 129 of 255 features kept")
    ax.set_xlabel("Features")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, tot.max() * 1.15)
    save(fig, "08_null_importance_selection.png")

    # 9. Feature importance: top features and family shares (mean |SHAP|, best model)
    fi = pd.read_csv(os.path.join(HERE, "feature_importance.csv"))
    top = fi.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["mean_abs_shap"], color=[ORANGE if g == "forecast" else BLUE for g in top["group"]],
            edgecolor=SURFACE, linewidth=2)
    ax.set_title("Top 15 features (mean |SHAP|); orange = forecaster output")
    ax.set_xlabel("Mean |SHAP| contribution")
    ax.grid(axis="y", visible=False)
    save(fig, "09_top_features.png")

    fam = {"base": "Monthly flows & balance", "raw": "Raw columns", "arrears": "Arrears",
           "forecast": "Month-13 forecast", "ratio": "Affordability ratios", "app": "Application risk",
           "util": "Utilisation", "minpay": "Minimum payment", "date": "Date from ID",
           "limit": "Credit limit", "delinq": "Missed-window counts"}
    s = fi.groupby("group")["mean_abs_shap"].sum()
    s = (s / s.sum() * 100).rename(index=fam).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(s.index, s.values, color=[ORANGE if i in ("Arrears", "Month-13 forecast") else BLUE for i in s.index],
            edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(s.values):
        ax.text(v + 0.3, i, f"{v:.0f}%", va="center", fontsize=10, color=INK2)
    ax.set_title("Share of model importance by feature family")
    ax.set_xlabel("Share of total mean |SHAP| (%)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, s.max() * 1.15)
    save(fig, "10_importance_by_family.png")

    # 11. Forecaster: default rate by forecast month-13 arrears
    q = pd.qcut(tr["fc_arrears"].rank(method="first"), 10, labels=[str(i) for i in range(1, 11)])
    r = rate(y, q)
    fig, ax = plt.subplots(figsize=(9, 4))
    bars(ax, r.index.astype(str), r.values, "Default rate by forecast month-13 arrears (fc_arrears)",
         xlabel="Tenth of customers, lowest to highest forecast arrears", highlight=[9])
    save(fig, "11_forecast_arrears.png")


if __name__ == "__main__":
    main()
