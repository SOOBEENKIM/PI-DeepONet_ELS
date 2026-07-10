"""Orchestrate the full pipeline (Appendix).

    python -m els_pricing.run_all --n-paths 100000 --jobs 32 --loss mse
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import config as C
from . import data_prep, market_data, mc_engine, deeponet, margin_mlp, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=100_000)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--loss", default="mae", choices=["mse", "mae", "mape"])
    ap.add_argument("--vol-source", default="auto")
    ap.add_argument("--ki-mode", default="daily", choices=["daily", "european"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--skip-mc", action="store_true", help="reuse cached MC")
    args = ap.parse_args()
    import torch
    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"

    print("### [1] data_prep");  data_prep.build().to_parquet(C.PRODUCT_MASTER)
    print("### [2] market_data"); market_data.build(args.vol_source).to_parquet(C.MARKET_MASTER)
    if not args.skip_mc:
        print("### [3] mc_engine")
        mc_engine._bs_selftest()
        df = pd.read_parquet(C.MARKET_MASTER)
        mc_engine.run_batch_parallel(df, n_paths=args.n_paths, jobs=args.jobs,
                                     ki_mode=args.ki_mode).to_parquet(C.MC_MASTER)
    print("### [4] deeponet")
    dn, _ = deeponet.run(pd.read_parquet(C.MC_MASTER), loss=args.loss, device=device, epochs=args.epochs)
    dn.to_parquet(C.CACHE_DIR / "product_deeponet.parquet")
    print("### [5] margin_mlp")
    fin = margin_mlp.run(dn, device=device)
    fin.to_parquet(C.CACHE_DIR / "product_final.parquet")
    print("### [8] metrics")
    metrics.report(fin)


if __name__ == "__main__":
    main()
