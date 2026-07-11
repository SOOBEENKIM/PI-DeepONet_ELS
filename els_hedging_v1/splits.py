"""7. walk-forward 4-fold 분할 (누적 train 60/70/80/90% -> 다음 10% test)"""
import numpy as np
import pandas as pd

from . import config as C


def sort_chronological(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("isu_ord", kind="stable").reset_index(drop=True)


def walk_forward_folds(n: int, fractions=None):
    """포지션 인덱스 기준 (train_slice, test_slice) 리스트. df는 반드시 isu_ord 오름차순 정렬된 상태여야 함."""
    fractions = fractions or C.WF_CUM_FRACTIONS
    cuts = [int(round(f * n)) for f in fractions]
    cuts[-1] = n
    folds = []
    for i in range(len(cuts) - 1):
        train_end, test_end = cuts[i], cuts[i + 1]
        if train_end <= 0 or test_end <= train_end:
            continue
        folds.append((np.arange(0, train_end), np.arange(train_end, test_end)))
    return folds
