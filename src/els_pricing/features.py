"""Feature assembly for Stage-1 (DeepONet branch/trunk) and Stage-2 (margin MLP).

Branch (market state m):  u0..u9 (curve, 10)  +  sig1..3,rho12/13/23,sig_eff (7)
Trunk  (contract y):      strk_0..11 (12)  +  B, coupon, tenor (3)   -> 15
Tabular (margin MLP):     BASE numeric features  +  CAT one-hot
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

U_COLS = [f"u{i}" for i in range(len(C.NS_NODES))]      # 10
VOL_COLS = ["sig1", "sig2", "sig3", "rho12", "rho13", "rho23", "sig_eff"]  # 7
STRK_COLS = [f"strk_{i}" for i in range(C.N_STRK)]        # 12
TRUNK_EXTRA = ["B", "coupon", "tenor"]
TRUNK_COLS = STRK_COLS + TRUNK_EXTRA                      # 15

# Stage-2 tabular
BASE_NUM = (
    VOL_COLS + STRK_COLS + ["B", "coupon", "tenor", "nobs", "cpn_spread",
    "b_over_k", "stepdown", "mom6m", "amt", "sbrt", "dvrt", "prcp",
    "kigrc", "iyear", "subdays"]
)
CAT_COLS = ["issuer", "risk", "ptype", "rdmp", "imonth"]


class Standardizer:
    """z-score using train statistics; safe against zero variance."""

    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, X):
        self.mu = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0)
        self.sd = np.where(self.sd < 1e-8, 1.0, self.sd)
        return self

    def transform(self, X):
        X = np.where(np.isfinite(X), X, np.take(self.mu, [0]) * 0 + self.mu)
        return (X - self.mu) / self.sd

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def branch_matrix(df: pd.DataFrame):
    curve = df[U_COLS].to_numpy(float)     # (n,10)
    vol = df[VOL_COLS].to_numpy(float)     # (n,7)
    return curve, vol


def trunk_matrix(df: pd.DataFrame):
    return df[TRUNK_COLS].to_numpy(float)  # (n,15)


def tabular_matrix(df: pd.DataFrame, cat_vocab=None):
    """Return (X, columns, cat_vocab). One-hot from a fixed (train) vocabulary."""
    num = df[BASE_NUM].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    num = np.where(np.isfinite(num), num, 0.0)
    cols = list(BASE_NUM)
    cat_parts = []
    if cat_vocab is None:
        cat_vocab = {c: sorted(df[c].astype(str).unique().tolist()) for c in CAT_COLS}
    for c in CAT_COLS:
        vals = df[c].astype(str).to_numpy()
        vocab = cat_vocab[c]
        oh = np.zeros((len(df), len(vocab)), float)
        idx = {v: i for i, v in enumerate(vocab)}
        for r, v in enumerate(vals):
            if v in idx:
                oh[r, idx[v]] = 1.0
        cat_parts.append(oh)
        cols += [f"{c}={v}" for v in vocab]
    X = np.concatenate([num] + cat_parts, axis=1) if cat_parts else num
    return X, cols, cat_vocab
