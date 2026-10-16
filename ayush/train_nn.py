"""
Neural model for ensemble diversity: GRU over the 6 monthly windows + MLP over all
tabular features (quantile-normalised). Same honest CV protocol as train_gbm.py:
epochs are chosen on an inner 10% holdout of the training fold.

Usage:  python train_nn.py [--repeats 3]
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score

from features import get_features
from features import TARGET, FLOWS, BAL_COLS, feature_columns
from train_gbm import SEEDS, PRED_DIR

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sequence_tensor(df):
    """(n, 6, 7): per window k (oldest first) -> opening balance, closing balance,
    borrow amt/count, repay amt/count, missed flag. Log-scaled amounts."""
    steps = []
    for k in range(6, 0, -1):
        def per(prefix):
            cur = df[f"{prefix} (Past {k} Month{'s' if k > 1 else ''})"].values
            prev = df[f"{prefix} (Past {k-1} Month{'s' if k > 2 else ''})"].values if k > 1 else 0
            return cur - prev
        ra, rc = per(FLOWS["ra"]), per(FLOWS["rc"])
        ba, bc = per(FLOWS["ba"]), per(FLOWS["bc"])
        b_open, b_close = df[BAL_COLS[k]].values, df[BAL_COLS[k - 1]].values
        steps.append(np.column_stack([
            np.log1p(b_open), np.log1p(b_close), np.log1p(ba), np.log1p(bc),
            np.log1p(ra), np.log1p(rc), (rc == 0).astype(float)]))
    x = np.stack(steps, axis=1).astype(np.float32)
    mu = x.reshape(-1, x.shape[2]).mean(0)
    sd = x.reshape(-1, x.shape[2]).std(0) + 1e-6
    return (x - mu) / sd


class Net(nn.Module):
    def __init__(self, n_tab, n_seq, hid=64):
        super().__init__()
        self.gru = nn.GRU(n_seq, hid, batch_first=True, bidirectional=True)
        self.tab = nn.Sequential(
            nn.Linear(n_tab, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.2))
        self.head = nn.Sequential(nn.Linear(128 + 2 * hid, 64), nn.SiLU(), nn.Dropout(0.1),
                                  nn.Linear(64, 1))

    def forward(self, xt, xs):
        _, h = self.gru(xs)
        h = torch.cat([h[0], h[1]], 1)
        return self.head(torch.cat([self.tab(xt), h], 1)).squeeze(1)


def predict(model, xt, xs, bs=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xt), bs):
            out.append(torch.sigmoid(model(xt[i:i + bs], xs[i:i + bs])).cpu().numpy())
    return np.concatenate(out)


def train_one(xt, xs, y, seed, epochs=None, xt_val=None, xs_val=None, y_val=None):
    torch.manual_seed(seed)
    model = Net(xt.shape[1], xs.shape[2]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    max_ep = epochs or 40
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max_ep * (len(y) // 1024 + 1))
    lossf = nn.BCEWithLogitsLoss()
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    best, best_ep, bad = -1, 0, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(max_ep):
        model.train()
        perm = torch.randperm(len(y), generator=g).to(DEV)
        for i in range(0, len(y), 1024):
            idx = perm[i:i + 1024]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(xt[idx], xs[idx]), yt[idx])
            loss.backward()
            opt.step()
            sched.step()
        if y_val is not None:
            auc = roc_auc_score(y_val, predict(model, xt_val, xs_val))
            if auc > best:
                best, best_ep, bad = auc, ep + 1, 0
            else:
                bad += 1
                if bad >= 6:
                    break
    return model, best_ep


def run(repeats):
    df, _ = get_features()
    cols = feature_columns(df)
    is_tr = (df.is_train == 1).values
    y_all = df[TARGET].values
    tab = df[cols].astype(float)
    tab = tab.fillna(tab.median())
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=10**9,
                             random_state=0)
    tab = qt.fit_transform(tab.values).astype(np.float32)  # unsupervised, train+test
    seq = sequence_tensor(df)
    T = lambda a: torch.tensor(a, device=DEV)
    xt_tr, xs_tr, y = tab[is_tr], seq[is_tr], y_all[is_tr].astype(int)
    xt_te, xs_te = T(tab[~is_tr]), T(seq[~is_tr])

    oof_all = np.zeros((repeats, len(y)))
    test_pred = np.zeros(int((~is_tr).sum()))
    aucs = []
    for r in range(repeats):
        seed = SEEDS[r]
        for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(xt_tr, y)):
            ia, ih = train_test_split(a, test_size=0.1, stratify=y[a], random_state=seed + f)
            _, ep = train_one(T(xt_tr[ia]), T(xs_tr[ia]), y[ia], seed + f,
                              xt_val=T(xt_tr[ih]), xs_val=T(xs_tr[ih]), y_val=y[ih])
            ep = max(ep, 3)
            preds_v, preds_t = [], []
            for s in range(3):  # small seed bag on the refit
                model, _ = train_one(T(xt_tr[a]), T(xs_tr[a]), y[a], seed + f + 100 * s, epochs=ep)
                preds_v.append(predict(model, T(xt_tr[b]), T(xs_tr[b])))
                preds_t.append(predict(model, xt_te, xs_te))
            pv, pt = np.mean(preds_v, 0), np.mean(preds_t, 0)
            oof_all[r, b] = pv
            test_pred += pd.Series(pt).rank(pct=True).values / (5 * repeats)
            print(f"  [nn] repeat {r} fold {f}: AUC {roc_auc_score(y[b], pv):.5f} epochs {ep}", flush=True)
        aucs.append(roc_auc_score(y, oof_all[r]))
        print(f"[nn] repeat {r} OOF AUC {aucs[-1]:.5f}", flush=True)
    oof = np.mean([pd.Series(o).rank(pct=True).values for o in oof_all], axis=0)
    print(f"[nn] mean per-repeat AUC {np.mean(aucs):.5f} +- {np.std(aucs):.5f} | "
          f"repeat-averaged OOF AUC {roc_auc_score(y, oof):.5f}")
    np.savez(os.path.join(PRED_DIR, "nn.npz"), oof=oof, oof_all=oof_all, test=test_pred,
             aucs=np.array(aucs), test_id=df.loc[~is_tr, "ID"].values,
             train_id=df.loc[is_tr, "ID"].values)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    run(ap.parse_args().repeats)
