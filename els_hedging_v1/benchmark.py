"""6. Stage-2 XGB(잔차, 비교용) + Direct 벤치(ridge/xgb) + 하이브리드 조립.

전처리 통계(스케일러·원핫)는 반드시 train fold로만 fit (walk-forward look-ahead 방지).

주의: RobustScaler/median-fillna/RidgeCV/time-decay/early-stopping 등 "안정화" 시도는
stage1_xgb(0.896->0.835), xgb_hybrid(0.338->0.160)를 오히려 악화시켜 원복함
(V1_피드백반영_지시서 §1.2). 이 파일은 최초 재현 버전 그대로 유지.
Stage-2 기본 모델은 margin_mlp.py(MLP)로 이전 - 여기 train_stage2_xgb는 비교용으로만 남김.
"""
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from . import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("benchmark")

XGB_PARAMS = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, random_state=C.SEED,
                   n_jobs=4, reg_lambda=1.0)


class TabularEncoder:
    """BASE(수치, 표준화) + CAT(원핫) 인코더. train으로만 fit."""

    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, df):
        self.scaler.fit(df[self.num_cols].fillna(0.0))
        self.ohe.fit(df[self.cat_cols].astype(str))
        return self

    def transform(self, df):
        num = self.scaler.transform(df[self.num_cols].fillna(0.0))
        cat = self.ohe.transform(df[self.cat_cols].astype(str))
        return np.concatenate([num, cat], axis=1)


def train_stage2_xgb(train_df, num_cols=None, cat_cols=None, target_col="resid"):
    """비교용(참고): margin_mlp.py가 기본 Stage-2."""
    num_cols = num_cols or C.BASE_NUM_COLS
    cat_cols = cat_cols or C.CAT_COLS
    enc = TabularEncoder(num_cols, cat_cols).fit(train_df)
    X = enc.transform(train_df)
    y = train_df[target_col].to_numpy()
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)
    return model, enc


def predict_tabular(model, enc, df):
    return model.predict(enc.transform(df))


def train_stage1_xgb(train_df):
    """xgb_hybrid의 Stage-1: branch+trunk+aux 플랫 피처로 MC 회귀."""
    cols = C.BRANCH_COLS + C.TRUNK_COLS + ["r"]
    enc = TabularEncoder(cols, []).fit(train_df)
    X = enc.transform(train_df)
    y = train_df["MC"].to_numpy()
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)
    return model, enc


def train_bench_ridge(train_df):
    cols = C.BASE_NUM_COLS + C.REG_COLS
    enc = TabularEncoder(cols, C.CAT_COLS).fit(train_df)
    X = enc.transform(train_df)
    y = train_df["fair"].to_numpy()
    model = Ridge(alpha=1.0, random_state=C.SEED)
    model.fit(X, y)
    return model, enc


def train_bench_xgb(train_df):
    cols = C.BASE_NUM_COLS + C.REG_COLS
    enc = TabularEncoder(cols, C.CAT_COLS).fit(train_df)
    X = enc.transform(train_df)
    y = train_df["fair"].to_numpy()
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)
    return model, enc
