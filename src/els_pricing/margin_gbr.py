"""[5b] Stage-2 margin GradientBoostingRegressor (sklearn tree) — comparison
variant for margin_mlp.py.

Same target decomposition, features, and walk-forward folds/valid split as
margin_mlp; only the Stage-2 model differs (sklearn GradientBoostingRegressor
instead of MarginMLP). No xgboost dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from . import config as C
from . import features as F
from . import splits as S
from .deeponet import r2, mape
from .margin_mlp import compute_recent_margin


def _huber(y_true, y_pred, delta=0.02):
    err = y_true - y_pred
    a = np.abs(err)
    q = np.minimum(a, delta)
    l = a - q
    return float(np.mean(0.5 * q ** 2 + delta * l))


def train_fold(X, y, tr, va, seed=0, n_estimators=2000, patience=50,
               max_depth=3, learning_rate=0.05, subsample=0.8):
    """Fit the full n_estimators once, then replay validation huber loss
    incrementally via staged_predict (O(N) total, not the O(N^2) you get
    from calling predict() after each warm_start increment) to find the
    tree count where `patience` consecutive trees fail to improve it —
    the same patience logic as margin_mlp.train_fold's bad>=15, at tree
    granularity. Returns predictions from that best checkpoint, not the
    final (possibly overfit) 2000-tree ensemble."""
    std = F.Standardizer().fit(X[tr])
    Xt = std.transform(X)
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample,
        loss="huber", alpha=0.9, random_state=seed)
    model.fit(Xt[tr], y[tr])

    best, best_n, bad, stopped_at = np.inf, 0, 0, n_estimators
    for i, p in enumerate(model.staged_predict(Xt[va]), start=1):
        va_loss = _huber(y[va], p)
        if va_loss < best - 1e-7:
            best, best_n, bad = va_loss, i, 0
        else:
            bad += 1
            if bad >= patience:
                stopped_at = i
                break

    pred_all = None
    for i, p in enumerate(model.staged_predict(Xt), start=1):
        pred_all = p
        if i == best_n:
            break
    return pred_all, best_n, stopped_at


def run(df):
    df = df[np.isfinite(df["mc"]) & np.isfinite(df["fair"])].copy()
    df["recent_margin"] = compute_recent_margin(df)
    df["resid"] = df["fair"] - df["mc"] - df["recent_margin"]

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
        pred_all, best_n, stopped_at = train_fold(X, y, tr_p, va_p, seed=fi)
        resid_hat[te_p] = pred_all[te_p]
        yt = y[te_p]
        early_stop = stopped_at < 2000
        print(f"  [fold{fi}] test={len(te)}  best_n_trees={best_n}  "
              f"patience_broke_at={stopped_at}  early_stop={early_stop}  "
              f"resid R2={r2(yt, pred_all[te_p]):.4f}")

    df["resid_hat"] = resid_hat
    df["margin_hat"] = df["recent_margin"] + df["resid_hat"]
    df["final"] = df["mc"] + df["margin_hat"]

    m = np.isfinite(resid_hat)
    f_true = df["fair"].to_numpy()[m]
    print(f"\n[margin_gbr] OOS resid R2={r2(y[m], resid_hat[m]):.4f}")
    print("[margin_gbr] OOS vs FAIR:")
    print(f"  MC only        R2={r2(f_true, df['mc'].to_numpy()[m]):.4f}  "
          f"MAPE={mape(f_true, df['mc'].to_numpy()[m]) * 100:.2f}%")
    print(f"  MC+recent      R2={r2(f_true, (df['mc'] + df['recent_margin']).to_numpy()[m]):.4f}  "
          f"MAPE={mape(f_true, (df['mc'] + df['recent_margin']).to_numpy()[m]) * 100:.2f}%")
    print(f"  Final(tree)    R2={r2(f_true, df['final'].to_numpy()[m]):.4f}  "
          f"MAPE={mape(f_true, df['final'].to_numpy()[m]) * 100:.2f}%")
    return df


def main():
    src = C.CACHE_DIR / "product_deeponet.parquet"
    df = pd.read_parquet(src if src.exists() else C.MC_MASTER)
    out = run(df)
    out.to_parquet(C.CACHE_DIR / "product_final_gbr.parquet")
    print(f"[margin_gbr] -> {C.CACHE_DIR / 'product_final_gbr.parquet'}")


if __name__ == "__main__":
    main()
