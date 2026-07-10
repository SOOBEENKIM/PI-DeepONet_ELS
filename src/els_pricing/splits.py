"""[7] Walk-forward folds (test is always future) + random train/valid split.

Section 7:
  - Sort by isu_ord. Cumulative train [0,cut); test [cut, cut+0.1) for
    cut in {0.6,0.7,0.8,0.9}  -> 4 folds, OOS covers 60..100%.
  - Inside a fold's train block, valid is drawn at RANDOM (not by time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward_folds(df: pd.DataFrame, cuts=(0.6, 0.7, 0.8, 0.9), test_w=0.1):
    order = df.sort_values("isu_ord").index.to_numpy()
    n = len(order)
    folds = []
    for cut in cuts:
        a = int(round(cut * n))
        b = int(round(min(cut + test_w, 1.0) * n))
        if b <= a:
            continue
        folds.append((order[:a], order[a:b]))
    return folds


def train_valid_split(train_idx, valid_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.array(train_idx)
    rng.shuffle(idx)
    k = int(round(valid_frac * len(idx)))
    return idx[k:], idx[:k]   # (train, valid)
