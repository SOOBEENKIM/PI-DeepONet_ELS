"""[4] Stage-1 DeepONet-Curve surrogate for the MC theoretical value (Section 5).

Branch (market state): 1D-CNN over the yield curve u0..u9, concatenated with
vol/corr (7) -> MLP -> B_theta(m) in R^p.
Trunk  (contract):     MLP over strk_0..11,B,coupon,tenor (15) -> T_theta(y) in R^p.
Output: <B_theta(m), T_theta(y)> + b0  ~ mc.

Target is the MC theoretical value (NOT fair). Loss is selectable
(mse / mae / mape) to benchmark the cost function (Section 5.2 feedback).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import config as C
from . import features as F
from . import splits as S

P = 192  # branch/trunk embedding dim


class BranchNet(nn.Module):
    def __init__(self, n_curve=10, n_vol=7, p=P):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(32 + n_vol, 128), nn.Tanh(),
            nn.Linear(128, p), nn.Tanh(),
        )

    def forward(self, curve, vol):
        z = self.cnn(curve.unsqueeze(1)).squeeze(-1)   # (n,32)
        return self.mlp(torch.cat([z, vol], dim=1))    # (n,p)


class TrunkNet(nn.Module):
    def __init__(self, n_in=15, p=P):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_in, p), nn.Tanh(),
            nn.Linear(p, p), nn.Tanh(),
            nn.Linear(p, p), nn.Tanh(),
        )

    def forward(self, y):
        return self.mlp(y)


class DeepONet(nn.Module):
    def __init__(self, y0=1.0):
        super().__init__()
        self.branch = BranchNet()
        self.trunk = TrunkNet()
        self.b0 = nn.Parameter(torch.tensor(float(y0)))

    def forward(self, curve, vol, y):
        b = self.branch(curve, vol)
        t = self.trunk(y)
        return (b * t).sum(dim=1) + self.b0


def _loss_fn(kind):
    if kind == "mse":
        return lambda p, y: torch.mean((p - y) ** 2)
    if kind == "mae":
        return lambda p, y: torch.mean(torch.abs(p - y))
    if kind == "mape":
        return lambda p, y: torch.mean(torch.abs(p - y) / torch.clamp(torch.abs(y), min=1e-3))
    raise ValueError(kind)


def _prep(df, tr_idx):
    """Fit standardizers on train, return tensor-builder."""
    curve_s = F.Standardizer().fit(df.loc[tr_idx, F.U_COLS].to_numpy(float))
    vol_s = F.Standardizer().fit(df.loc[tr_idx, F.VOL_COLS].to_numpy(float))
    trunk_s = F.Standardizer().fit(df.loc[tr_idx, F.TRUNK_COLS].to_numpy(float))

    def build(idx):
        curve = curve_s.transform(df.loc[idx, F.U_COLS].to_numpy(float))
        vol = vol_s.transform(df.loc[idx, F.VOL_COLS].to_numpy(float))
        trunk = trunk_s.transform(df.loc[idx, F.TRUNK_COLS].to_numpy(float))
        y = df.loc[idx, "mc"].to_numpy(float)
        t = lambda a: torch.tensor(a, dtype=torch.float32)
        return t(curve), t(vol), t(trunk), t(y)

    return build


def train_fold(df, tr_idx, va_idx, loss="mse", epochs=200, bs=2048, lr=1e-3,
               device="cpu", seed=0, patience=20, clip=5.0, verbose=False):
    """Train one fold. Targets (mc) are standardised with TRAIN statistics so
    MSE/MAE/MAPE all optimise on the same scale -> a fair loss comparison.
    Predictions are returned on the original scale by predict()."""
    torch.manual_seed(seed)
    build = _prep(df, tr_idx)
    cu_tr, vo_tr, tr_tr, y_tr = (x.to(device) for x in build(tr_idx))
    cu_va, vo_va, tr_va, y_va = (x.to(device) for x in build(va_idx))

    y_mu, y_sd = float(y_tr.mean()), float(y_tr.std() + 1e-8)
    norm = lambda t: (t - y_mu) / y_sd

    model = DeepONet(y0=0.0).to(device)   # standardised space -> bias starts at 0
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
    lf = _loss_fn(loss)

    n = len(y_tr)
    best_va, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss_v = lf(model(cu_tr[b], vo_tr[b], tr_tr[b]), norm(y_tr[b]))
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            va = lf(model(cu_va, vo_va, tr_va), norm(y_va)).item()
        if va < best_va - 1e-6:
            best_va, best_state, bad = va, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 20 == 0:
            print(f"    ep{ep} valid_{loss}={va:.5f}")
    if best_state:
        model.load_state_dict(best_state)
    return model, build, y_mu, y_sd


def predict(model, build, idx, y_mu=0.0, y_sd=1.0, device="cpu"):
    cu, vo, tr, _ = (x.to(device) for x in build(idx))
    model.eval()
    with torch.no_grad():
        return model(cu, vo, tr).cpu().numpy() * y_sd + y_mu


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - p) ** 2).sum() / ss if ss > 0 else np.nan


def mape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return np.mean(np.abs(p - y) / np.clip(np.abs(y), 1e-3, None))


def run(df, loss="mse", device="cpu", epochs=200):
    df = df[np.isfinite(df["mc"]) & np.isfinite(df["fair"])].copy()
    folds = S.walk_forward_folds(df)
    oos = np.full(len(df), np.nan)
    pos = {ix: k for k, ix in enumerate(df.index)}
    for fi, (tr_all, te) in enumerate(folds):
        tr, va = S.train_valid_split(tr_all, valid_frac=0.1, seed=fi)
        model, build, y_mu, y_sd = train_fold(df, tr, va, loss=loss, epochs=epochs, device=device, seed=fi)
        pred = predict(model, build, te, y_mu, y_sd, device=device)
        for ix, pv in zip(te, pred):
            oos[pos[ix]] = pv
        y = df.loc[te, "mc"].to_numpy()
        print(f"  [fold{fi}] test={len(te)}  R2={r2(y,pred):.4f}  MAPE={mape(y,pred)*100:.2f}%")
    m = np.isfinite(oos)
    y = df["mc"].to_numpy()[m]
    metrics = dict(loss=loss, R2=r2(y, oos[m]), MAPE=mape(y, oos[m]) * 100,
                   bp=float(np.mean(np.abs(oos[m] - y)) * 1e4),
                   Spearman=float(pd.Series(y).corr(pd.Series(oos[m]), method="spearman")))
    print(f"[deeponet:{loss}] OOS  R2(mc)={metrics['R2']:.4f}  MAPE={metrics['MAPE']:.2f}%  "
          f"bp={metrics['bp']:.1f}  Spearman={metrics['Spearman']:.4f}")
    df["deeponet_mc_hat"] = oos
    return df, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="mae", choices=["mse", "mae", "mape"])
    ap.add_argument("--all-losses", action="store_true")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    print(f"[deeponet] device={device}")
    df = pd.read_parquet(C.MC_MASTER)

    if args.all_losses:
        results = []
        for ls in ["mse", "mae", "mape"]:
            _, m = run(df, loss=ls, device=device, epochs=args.epochs)
            results.append(m)
        print("\n[cost-function benchmark] (same folds/seed/preprocessing, standardised target)")
        print(pd.DataFrame(results).to_string(index=False))
    else:
        out, _ = run(df, loss=args.loss, device=device, epochs=args.epochs)
        out.to_parquet(C.CACHE_DIR / "product_deeponet.parquet")
        print(f"[deeponet] -> {C.CACHE_DIR / 'product_deeponet.parquet'}")


if __name__ == "__main__":
    main()
