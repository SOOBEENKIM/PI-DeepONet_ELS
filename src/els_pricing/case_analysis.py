"""[10] Low-price-region error analysis (Section 10).

Observation to investigate: DeepONet reproduction error grows for low-priced
(deep, near-barrier) products. Produces:
  A. error curve by MC-price bucket (does %error rise as price falls?)
  B. correlation of |error| with structural drivers (b_over_k, worst return,
     tenor, vol, sample density)
  C. extreme-case table (largest |error| products with their structure)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .deeponet import mape


def bucket_error(df, pred="deeponet_mc_hat", actual="mc", n=8):
    d = df[np.isfinite(df[pred]) & np.isfinite(df[actual])].copy()
    d["err"] = np.abs(d[pred] - d[actual])
    d["ape"] = d["err"] / np.clip(np.abs(d[actual]), 1e-3, None)
    d["bucket"] = pd.qcut(d[actual], n, duplicates="drop")
    d["_bias"] = d[pred] - d[actual]
    g = d.groupby("bucket", observed=True).agg(
        n=("err", "size"), mid_price=(actual, "mean"),
        MAPE_pct=("ape", lambda s: s.mean() * 100),
        MAE=("err", "mean"), bias=("_bias", "mean"),
    )
    return g.reset_index()


def driver_corr(df, pred="deeponet_mc_hat", actual="mc"):
    d = df[np.isfinite(df[pred]) & np.isfinite(df[actual])].copy()
    d["abserr"] = np.abs(d[pred] - d[actual])
    drivers = ["b_over_k", "tenor", "sig_eff", "coupon", "stepdown", "mc",
               "sig3", "nobs", "cpn_spread"]
    rows = []
    for c in drivers:
        if c in d:
            rows.append((c, float(d["abserr"].corr(d[c], method="spearman"))))
    return pd.DataFrame(rows, columns=["driver", "spearman_vs_abserr"]).sort_values(
        "spearman_vs_abserr", key=lambda s: s.abs(), ascending=False)


def extreme_cases(df, pred="deeponet_mc_hat", actual="mc", k=10):
    d = df[np.isfinite(df[pred]) & np.isfinite(df[actual])].copy()
    d["abserr"] = np.abs(d[pred] - d[actual])
    cols = ["item", "issuer", "isu_dt", actual, pred, "abserr", "fair",
            "b_over_k", "tenor", "sig_eff", "coupon", "nobs", "knock_in"]
    cols = [c for c in cols if c in d]
    return d.nlargest(k, "abserr")[cols]


def main():
    src = C.CACHE_DIR / "product_final.parquet"
    if not src.exists():
        src = C.CACHE_DIR / "product_deeponet.parquet"
    df = pd.read_parquet(src)
    print("A. Error by MC-price bucket (low price -> ?):")
    print(bucket_error(df).to_string(index=False))
    print("\nB. |error| vs structural drivers (Spearman):")
    print(driver_corr(df).to_string(index=False))
    print("\nC. Top-10 extreme error cases:")
    print(extreme_cases(df).to_string(index=False))


if __name__ == "__main__":
    main()
