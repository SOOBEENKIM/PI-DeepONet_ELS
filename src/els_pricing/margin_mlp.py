"""[5] Stage-2 margin MLP (Section 6).

Target decomposition (feedback):  fair = mc + recent_margin + resid
  - recent_margin_i : causal per-issuer anchor = mean(fair-mc) over the SAME
    issuer's products issued in the past <=90 days (global 90d mean as backup).
    It is an anchor, NOT a model input, and never looks into the future.
  - resid = fair - mc - recent_margin  is what the MLP predicts, from features
    only (recent_margin / mc are NOT inputs).
  - margin_hat = recent_margin + resid_hat ;  Final = mc + margin_hat.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import config as C
from . import features as F
from . import splits as S
from .deeponet import r2, mape

WINDOW_DAYS = 90


def compute_recent_margin(df: pd.DataFrame) -> pd.Series:
    """Causal per-issuer 90-day mean of (fair - mc); global 90d backup."""
    d = df.sort_values("isu_ord")
    ordv = d["isu_ord"].to_numpy()
    margin = (d["fair"] - d["mc"]).to_numpy()
    issuer = d["issuer"].astype(str).to_numpy()

    out = np.zeros(len(d))
    iss_q = defaultdict(deque)      # issuer -> deque of (ord, margin)
    iss_sum = defaultdict(float)
    g_q = deque()                   # global (ord, margin)
    g_sum = 0.0
    for i in range(len(d)):
        o = ordv[i]
        q = iss_q[issuer[i]]
        while q and o - q[0][0] > WINDOW_DAYS:
            iss_sum[issuer[i]] -= q.popleft()[1]
        while g_q and o - g_q[0][0] > WINDOW_DAYS:
            g_sum -= g_q.popleft()[1]
        if q:
            out[i] = iss_sum[issuer[i]] / len(q)
        elif g_q:
            out[i] = g_sum / len(g_q)
        else:
            out[i] = 0.0
        q.append((o, margin[i])); iss_sum[issuer[i]] += margin[i]
        g_q.append((o, margin[i])); g_sum += margin[i]
    return pd.Series(out, index=d.index).reindex(df.index)


class MarginMLP(nn.Module):
    def __init__(self, n_in, hidden=(256, 128)):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_fold(X, y, tr, va, epochs=120, bs=1024, lr=1e-3, device="cpu",
               huber=True, seed=0):
    torch.manual_seed(seed)
    std = F.Standardizer().fit(X[tr])
    Xt = torch.tensor(std.transform(X), dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    model = MarginMLP(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
    lf = nn.HuberLoss(delta=0.02) if huber else nn.MSELoss()
    tr_t = torch.tensor(tr, device=device)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=device)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lf(model(Xt[b]), yt[b])
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            va_loss = lf(model(Xt[va]), yt[va]).item()
        if va_loss < best - 1e-7:
            best, best_state, bad = va_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 15:
                break
    if best_state:
        model.load_state_dict(best_state)
    with torch.no_grad():
        pred = model(Xt).cpu().numpy()
    return pred


def run(df, device="cpu"):
    df = df[np.isfinite(df["mc"]) & np.isfinite(df["fair"])].copy()
    df["recent_margin"] = compute_recent_margin(df)
    df["resid"] = df["fair"] - df["mc"] - df["recent_margin"]

    # build tabular features with a FIXED vocab from the whole set (one-hot cols stable)
    X, cols, vocab = F.tabular_matrix(df)
    y = df["resid"].to_numpy(float)

    folds = S.walk_forward_folds(df)
    pos = {ix: k for k, ix in enumerate(df.index)}
    resid_hat = np.full(len(df), np.nan)
    for fi, (tr_all, te) in enumerate(folds):
        tr, va = S.train_valid_split(tr_all, valid_frac=0.1, seed=fi)
        tr_p = np.array([pos[i] for i in tr])
        va_p = np.array([pos[i] for i in va])
        te_p = np.array([pos[i] for i in te])
        pred_all = train_fold(X, y, tr_p, va_p, device=device, seed=fi)
        resid_hat[te_p] = pred_all[te_p]
        yt = y[te_p]
        print(f"  [fold{fi}] test={len(te)}  resid R2={r2(yt, pred_all[te_p]):.4f}")

    df["resid_hat"] = resid_hat
    df["margin_hat"] = df["recent_margin"] + df["resid_hat"]
    df["final"] = df["mc"] + df["margin_hat"]
    if "deeponet_mc_hat" in df:
        df["final_fast"] = df["deeponet_mc_hat"] + df["margin_hat"]

    m = np.isfinite(resid_hat)
    f_true = df["fair"].to_numpy()[m]
    print("\n[margin_mlp] OOS vs FAIR:")
    print(f"  MC only        R2={r2(f_true, df['mc'].to_numpy()[m]):.4f}  "
          f"MAPE={mape(f_true, df['mc'].to_numpy()[m])*100:.2f}%")
    print(f"  MC+recent      R2={r2(f_true, (df['mc']+df['recent_margin']).to_numpy()[m]):.4f}  "
          f"MAPE={mape(f_true, (df['mc']+df['recent_margin']).to_numpy()[m])*100:.2f}%")
    print(f"  Final(margin)  R2={r2(f_true, df['final'].to_numpy()[m]):.4f}  "
          f"MAPE={mape(f_true, df['final'].to_numpy()[m])*100:.2f}%")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    src = C.CACHE_DIR / "product_deeponet.parquet"
    df = pd.read_parquet(src if src.exists() else C.MC_MASTER)
    out = run(df, device=device)
    out.to_parquet(C.CACHE_DIR / "product_final.parquet")
    print(f"[margin_mlp] -> {C.CACHE_DIR / 'product_final.parquet'}")


if __name__ == "__main__":
    main()
