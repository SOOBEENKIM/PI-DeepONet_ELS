"""[8] Metrics — %error centric (Section 8).

  - MAPE / bp error / MAE / RMSE
  - moneyness-bucket errors (bucket by MC price ~ how deep the worst sits)
  - calibration slope/intercept (pred vs actual) + low/high skew
  - per-stage R^2: Stage-1 (mc reproduction), Stage-2 (resid), Final (fair)
  - Spearman rank correlation (ordering quality)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .deeponet import r2, mape


def bp(y, p):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))) * 1e4)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def calibration(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    A = np.vstack([y, np.ones_like(y)]).T
    slope, intercept = np.linalg.lstsq(A, p, rcond=None)[0]
    lo = p[y < np.median(y)] - y[y < np.median(y)]
    hi = p[y >= np.median(y)] - y[y >= np.median(y)]
    return dict(slope=float(slope), intercept=float(intercept),
                low_bias=float(lo.mean()), high_bias=float(hi.mean()))


def spearman(y, p):
    y, p = pd.Series(np.asarray(y)), pd.Series(np.asarray(p))
    return float(y.corr(p, method="spearman"))


def moneyness_buckets(df, pred_col, actual_col="fair", by="mc", n=6):
    d = df[np.isfinite(df[pred_col]) & np.isfinite(df[actual_col])].copy()
    d["bucket"] = pd.qcut(d[by], n, duplicates="drop")
    rows = []
    for b, g in d.groupby("bucket", observed=True):
        y, p = g[actual_col].to_numpy(), g[pred_col].to_numpy()
        rows.append(dict(bucket=str(b), n=len(g), mid=float(g[by].mean()),
                         MAPE=mape(y, p) * 100, bp=bp(y, p),
                         bias=float(np.mean(p - y))))
    return pd.DataFrame(rows)


def report(df):
    d = df.copy()
    oos = np.isfinite(d["final"]) if "final" in d else np.isfinite(d["deeponet_mc_hat"])
    d = d[oos].copy()
    fair = d["fair"].to_numpy()
    print("=" * 64)
    print(f"OOS products: {len(d):,}")
    # per-stage
    if "deeponet_mc_hat" in d:
        mc, mh = d["mc"].to_numpy(), d["deeponet_mc_hat"].to_numpy()
        print("\n[Stage-1 DeepONet vs MC theoretical value]  (our essence)")
        print(f"  R2={r2(mc, mh):.4f}  MAPE={mape(mc, mh)*100:.2f}%  bp={bp(mc, mh):.1f}  "
              f"Spearman={spearman(mc, mh):.4f}")
    if "resid_hat" in d:
        rr, rh = d["resid"].to_numpy(), d["resid_hat"].to_numpy()
        print("\n[Stage-2 margin MLP vs residual]")
        print(f"  R2={r2(rr, rh):.4f}  RMSE={rmse(rr, rh):.5f}")
    if "final" in d:
        fh = d["final"].to_numpy()
        print("\n[Final = MC + margin  vs FAIR]")
        print(f"  R2={r2(fair, fh):.4f}  MAPE={mape(fair, fh)*100:.2f}%  bp={bp(fair, fh):.1f}  "
              f"RMSE={rmse(fair, fh):.5f}  Spearman={spearman(fair, fh):.4f}")
        print("  calibration:", {k: round(v, 4) for k, v in calibration(fair, fh).items()})
        print("\n[Moneyness buckets — Final vs FAIR]")
        print(moneyness_buckets(d, "final").to_string(index=False))
    if "final_fast" in d:
        ff = d["final_fast"].to_numpy()
        print("\n[Final_fast = DeepONet + margin  vs FAIR]")
        print(f"  R2={r2(fair, ff):.4f}  MAPE={mape(fair, ff)*100:.2f}%  bp={bp(fair, ff):.1f}")
    print("=" * 64)


def main():
    df = pd.read_parquet(C.CACHE_DIR / "product_final.parquet")
    report(df)


if __name__ == "__main__":
    main()
