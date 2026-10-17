"""
TabM-style deep model (Gorishniy et al., 2024, "TabM: Advancing Tabular Deep Learning
with Parameter-Efficient Ensembling") with a GRU branch over the monthly history.

  numeric features --PLE (quantile piecewise-linear encoding)--> per-feature linear embed
  6 monthly windows (balances, flows, arrears, pay/due) --> bidirectional GRU
  concat --> TabM-mini: k members share one MLP; each member scales the input by its own
             adapter vector and has its own output head. Prediction = mean over members.

Same honest CV protocol as the GBMs: the best epoch is chosen on an inner 10% holdout of
the training fold; the outer fold is only predicted.

Usage:  python train_tabm.py [--repeats 3] [--feats all|selected] [--k 32] [--tag tabm]
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from features import TARGET, BAL_COLS, get_features
from train_gbm import SEEDS, PRED_DIR, select_columns

DEV = "cuda"
assert torch.cuda.is_available(), "CUDA GPU not visible to PyTorch"


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def ple_edges(x, n_bins):
    """Quantile bin edges per feature (unsupervised, computed on train+test)."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.nanquantile(x.astype(np.float64), qs, axis=0).T  # (F, n_bins+1)
    return np.maximum.accumulate(edges, axis=1)


def sequence_tensor(df):
    """(n, 6, 9) per window, oldest first: opening/closing balance, draws (amt, count),
    repayments (amt, count), arrears count, arrears in months of minimum, pay/due ratio."""
    steps = []
    for k in range(6, 0, -1):
        due = df[f"due_m{k}"].values
        steps.append(np.column_stack([
            np.log1p(df[BAL_COLS[k]].values), np.log1p(df[BAL_COLS[k - 1]].values),
            np.log1p(df[f"ba_m{k}"].values), np.log1p(df[f"bc_m{k}"].values),
            np.log1p(df[f"ra_m{k}"].values), np.log1p(df[f"rc_m{k}"].values),
            np.minimum(df[f"arrears_w{k}"].values, 6), np.minimum(df[f"arrears_amt_w{k}"].values, 6),
            np.minimum(df[f"ra_m{k}"].values / np.maximum(due, 1), 5) * (due > 0)]))
    x = np.stack(steps, axis=1).astype(np.float32)
    flat = x.reshape(-1, x.shape[2])
    return (x - flat.mean(0)) / (flat.std(0) + 1e-6)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class PLEEmbedding(nn.Module):
    def __init__(self, edges, d):
        super().__init__()
        width = edges[:, 1:] - edges[:, :-1]
        # zero-width bins (repeated quantiles) become a step at `lo` instead of dividing by 0
        width = np.where(width > 0, width, 1e-6 * np.maximum(np.abs(edges[:, :-1]), 1.0))
        self.register_buffer("lo", torch.tensor(edges[:, :-1], dtype=torch.float32))  # (F, T)
        self.register_buffer("w", torch.tensor(width, dtype=torch.float32))
        n_f, n_t = self.lo.shape
        self.weight = nn.Parameter(torch.randn(n_f, n_t, d) * (1 / n_t ** 0.5))
        self.bias = nn.Parameter(torch.zeros(n_f, d))

    def forward(self, x):                                                  # x: (B, F)
        enc = ((x[:, :, None] - self.lo) / self.w).clamp(0, 1)             # (B, F, T)
        return torch.relu(torch.einsum("bft,ftd->bfd", enc, self.weight) + self.bias).flatten(1)


class TabM(nn.Module):
    def __init__(self, edges, n_seq, k=32, d_emb=8, d_hidden=512, dropout=0.15, gru_hid=48):
        super().__init__()
        self.k = k
        self.emb = PLEEmbedding(edges, d_emb)
        self.gru = nn.GRU(n_seq, gru_hid, batch_first=True, bidirectional=True)
        d_in = edges.shape[0] * d_emb + 2 * gru_hid
        # TabM-mini: one multiplicative adapter per member on the input, shared backbone
        self.adapter = nn.Parameter(torch.sign(torch.randn(k, d_in)))  # random +-1 init
        self.backbone = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head_w = nn.Parameter(torch.randn(k, d_hidden) * d_hidden ** -0.5)
        self.head_b = nn.Parameter(torch.zeros(k))

    def forward(self, xt, xs):                                             # -> (B, k) logits
        _, h = self.gru(xs)
        z = torch.cat([self.emb(xt), h[0], h[1]], 1)                       # (B, D)
        z = z[:, None, :] * self.adapter[None]                             # (B, k, D)
        z = self.backbone(z)                                               # (B, k, H)
        return (z * self.head_w[None]).sum(-1) + self.head_b


def predict(model, xt, xs, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xt), bs):
            out.append(torch.sigmoid(model(xt[i:i + bs], xs[i:i + bs])).mean(1).cpu().numpy())
    return np.concatenate(out)


def train_fold(edges, xt, xs, y, xt_h, xs_h, y_h, seed, k, max_epochs=40, patience=6, bs=512):
    torch.manual_seed(seed)
    model = TabM(edges, xs.shape[2], k=k).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    lossf = nn.BCEWithLogitsLoss()
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(y), generator=g).to(DEV)
        for i in range(0, len(y) - 1, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            logits = model(xt[idx], xs[idx])                               # (B, k)
            loss = lossf(logits, yt[idx, None].expand_as(logits))          # every member learns
            loss.backward()
            opt.step()
        auc = roc_auc_score(y_h, predict(model, xt_h, xs_h))
        if auc > best:
            best, bad = auc, 0
            best_state = {n: p.detach().clone() for n, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1


def run(repeats, feats, k, tag):
    df, _ = get_features()
    cols = select_columns(df, feats)
    is_tr = (df.is_train == 1).values
    tab = df[cols].astype(np.float32).values
    tab = np.where(np.isnan(tab), np.nanmedian(tab, axis=0), tab).astype(np.float32)
    edges = ple_edges(tab, n_bins=16)
    seq = sequence_tensor(df)
    T = lambda a: torch.tensor(a, device=DEV)
    y = df.loc[is_tr, TARGET].values.astype(int)
    xt_tr, xs_tr = tab[is_tr], seq[is_tr]
    xt_te, xs_te = T(tab[~is_tr]), T(seq[~is_tr])

    oof_all = np.zeros((repeats, len(y)))
    test_pred = np.zeros(int((~is_tr).sum()))
    aucs = []
    for r in range(repeats):
        seed = SEEDS[r]
        for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(xt_tr, y)):
            t0 = time.time()
            ia, ih = train_test_split(a, test_size=0.1, stratify=y[a], random_state=seed + f)
            model, hold_auc, n_ep = train_fold(edges, T(xt_tr[ia]), T(xs_tr[ia]), y[ia],
                                               T(xt_tr[ih]), T(xs_tr[ih]), y[ih], seed + f, k)
            pv = predict(model, T(xt_tr[b]), T(xs_tr[b]))
            oof_all[r, b] = pd.Series(pv).rank(pct=True).values  # NN output scale shifts per fold
            test_pred += pd.Series(predict(model, xt_te, xs_te)).rank(pct=True).values / (5 * repeats)
            print(f"  [{tag}] repeat {r} fold {f}: AUC {roc_auc_score(y[b], pv):.5f}  "
                  f"(holdout {hold_auc:.5f}, {n_ep} epochs, {time.time() - t0:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
        aucs.append(roc_auc_score(y, oof_all[r]))
        print(f"[{tag}] repeat {r} OOF AUC {aucs[-1]:.5f}", flush=True)
    oof = np.mean([pd.Series(o).rank(pct=True).values for o in oof_all], axis=0)
    print(f"[{tag}] mean per-repeat AUC {np.mean(aucs):.5f} +- {np.std(aucs):.5f} | "
          f"repeat-averaged OOF AUC {roc_auc_score(y, oof):.5f}")
    out = os.path.join(PRED_DIR, f"{tag}.npz")
    np.savez(out, oof=oof, oof_all=oof_all, test=test_pred, aucs=np.array(aucs),
             test_id=df.loc[~is_tr, "ID"].values, train_id=df.loc[is_tr, "ID"].values)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--feats", choices=["all", "selected"], default="all")
    ap.add_argument("--k", type=int, default=32, help="number of TabM ensemble members")
    ap.add_argument("--tag", default="tabm")
    args = ap.parse_args()
    run(args.repeats, args.feats, args.k, args.tag)
