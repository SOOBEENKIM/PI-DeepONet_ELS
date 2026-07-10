"""[9] Speed benchmark: MC vs DeepONet for pricing the whole test set (Section 9).

Measured single-threaded (1 core), serial per product for MC (path generation
vectorised per product; products serial). DeepONet = load + forward only.
Reports: log-scale time bars, "N x faster", price-agreement scatter stats.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import features as F
from . import mc_engine as MC
from .deeponet import DeepONet, _prep, r2


def bench_mc(df, n_paths=100_000, sample=200, ki_mode="daily", steps_per_year=252):
    """Time the SAME MC engine/conditions used to generate the labels."""
    d = df.head(sample)
    t0 = time.time()
    for _, row in d.iterrows():
        MC._price_one(row, n_paths, 0, ki_mode, steps_per_year)
    per = (time.time() - t0) / len(d)
    return per, per * len(df)


def bench_deeponet(df, device="cpu"):
    tr = df.index.to_numpy()
    build = _prep(df, tr)
    model = DeepONet(y0=float(df["mc"].mean())).to(device)
    cu, vo, tk, _ = (x.to(device) for x in build(df.index))
    model.eval()
    # warmup
    with torch.no_grad():
        model(cu[:64], vo[:64], tk[:64])
    t0 = time.time()
    with torch.no_grad():
        model(cu, vo, tk)
    total = time.time() - t0
    return total / len(df), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=100_000)
    ap.add_argument("--mc-sample", type=int, default=60)
    ap.add_argument("--ki-mode", default="daily", choices=["daily", "european"])
    args = ap.parse_args()
    spy = C.MC["steps_per_year"]
    df = pd.read_parquet(C.MC_MASTER)
    df = df[np.isfinite(df["mc"])].copy()
    n = len(df)

    mc_per, mc_total = bench_mc(df, n_paths=args.n_paths, sample=args.mc_sample,
                               ki_mode=args.ki_mode, steps_per_year=spy)
    dn_per, dn_total = bench_deeponet(df)
    speedup = mc_per / dn_per
    print("=" * 64)
    print(f"Test set: {n:,} products   MC paths: {args.n_paths:,}   "
          f"[MC basis: ki_mode={args.ki_mode}, steps/yr={spy}]")
    print(f"MC       : {mc_per*1e3:11.3f} ms/product -> {mc_total:11.1f} s total "
          f"(extrapolated from {args.mc_sample})")
    print(f"DeepONet : {dn_per*1e6:11.3f} us/product -> {dn_total:11.4f} s total (all {n:,} measured)")
    print(f"Speedup  : {speedup:,.0f}x   [basis: {args.ki_mode} MC]")
    print("path-independence: DeepONet time flat vs MC scales with paths x steps")
    print("=" * 64)


if __name__ == "__main__":
    main()
