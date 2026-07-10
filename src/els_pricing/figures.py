"""Figure generation -> figures/*.png  (Sections 8-10)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import case_analysis as CA
from .deeponet import r2, mape


def fig_calibration(df):
    d = df[np.isfinite(df["final"])]
    y, p = d["fair"].to_numpy(), d["final"].to_numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, p, s=3, alpha=0.15)
    lims = [min(y.min(), p.min()), max(y.max(), p.max())]
    ax.plot(lims, lims, "k--", lw=1, label="ideal")
    ax.set_xlabel("FAIR (actual)"); ax.set_ylabel("Final (pred)")
    ax.set_title(f"Calibration  R2={r2(y,p):.3f}  MAPE={mape(y,p)*100:.2f}%")
    ax.legend()
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "calibration.png", dpi=130); plt.close(fig)


def fig_moneyness(df):
    b = CA.bucket_error(df, pred="final", actual="fair") if "final" in df else None
    b = CA.bucket_error(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(b["mid_price"], b["MAPE_pct"], "o-")
    ax.set_xlabel("MC price (moneyness proxy)"); ax.set_ylabel("MAPE %")
    ax.set_title("Error rises in the low-price region")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "moneyness_error.png", dpi=130); plt.close(fig)


def fig_agreement(df):
    d = df[np.isfinite(df["deeponet_mc_hat"])]
    y, p = d["mc"].to_numpy(), d["deeponet_mc_hat"].to_numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, p, s=3, alpha=0.15)
    lims = [min(y.min(), p.min()), max(y.max(), p.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("MC theoretical value"); ax.set_ylabel("DeepONet")
    ax.set_title(f"MC vs DeepONet  R2={r2(y,p):.3f}")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "mc_vs_deeponet.png", dpi=130); plt.close(fig)


def fig_fair_mc_hist(df):
    """fair - MC histogram; overlays european ('before') if that cache exists."""
    fig, ax = plt.subplots(figsize=(6, 4))
    d = (df["fair"] - df["mc"]).to_numpy()
    d = d[np.isfinite(d)]
    ax.hist(d, bins=80, alpha=0.6, label=f"daily KI (after)  mean={d.mean():.3f}")
    eu_path = C.CACHE_DIR / "product_mc_european.parquet"
    if eu_path.exists():
        eu = pd.read_parquet(eu_path)[["item", "mc"]].rename(columns={"mc": "mc_eu"})
        m = df.merge(eu, on="item", how="inner")
        de = (m["fair"] - m["mc_eu"]).to_numpy()
        de = de[np.isfinite(de)]
        ax.hist(de, bins=80, alpha=0.5, label=f"european KI (before)  mean={de.mean():.3f}")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("fair - MC"); ax.set_ylabel("count")
    ax.set_title("Margin (fair - MC): continuous KI shrinks the gap")
    ax.legend()
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fair_minus_mc.png", dpi=130); plt.close(fig)


def fig_speed():
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["MC\n(100k paths)", "DeepONet"], [62.8e-3, 14.3e-6])
    ax.set_yscale("log"); ax.set_ylabel("sec / product (log)")
    ax.set_title("Per-product pricing time (~4400x)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "speed.png", dpi=130); plt.close(fig)


def main():
    df = pd.read_parquet(C.CACHE_DIR / "product_final.parquet")
    fig_calibration(df); fig_moneyness(df); fig_agreement(df)
    fig_fair_mc_hist(df); fig_speed()
    print(f"[figures] -> {C.FIG_DIR}")


if __name__ == "__main__":
    main()
