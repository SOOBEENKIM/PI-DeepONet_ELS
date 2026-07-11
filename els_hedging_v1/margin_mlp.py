"""Stage-2 (margin/잔차) — 간단한 MLP. 교수님 피드백: tree는 외삽 불가 -> MLP를 기본으로.

타깃 = margin = fair - MC - recent_margin (기존 'resid'와 동일 정의).
입력은 BASE+CAT만 (MC·recent_margin은 앵커이지 피처가 아님 - §V1_빌드_지시서 5).
train/valid는 랜덤 셔플(시간순 X) - §V1_피드백반영_지시서 1.5.
"""
import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from . import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("margin_mlp")

torch.manual_seed(C.SEED)


class MarginMLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64, 32)):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class MarginEncoder:
    """BASE(수치, 표준화) + CAT(원핫) + 타깃 표준화. train으로만 fit."""

    def __init__(self, num_cols=None, cat_cols=None):
        self.num_cols = num_cols or C.BASE_NUM_COLS
        self.cat_cols = cat_cols or C.CAT_COLS
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.target_scaler = StandardScaler()

    def fit(self, df, target_col=None):
        self.scaler.fit(df[self.num_cols].fillna(0.0))
        self.ohe.fit(df[self.cat_cols].astype(str))
        if target_col is not None:
            self.target_scaler.fit(df[[target_col]].to_numpy())
        return self

    def transform(self, df):
        num = self.scaler.transform(df[self.num_cols].fillna(0.0)).astype(np.float32)
        cat = self.ohe.transform(df[self.cat_cols].astype(str)).astype(np.float32)
        return torch.from_numpy(np.concatenate([num, cat], axis=1))

    def transform_target(self, y):
        return self.target_scaler.transform(y.reshape(-1, 1)).astype(np.float32).ravel()

    def inverse_target(self, y_std):
        return self.target_scaler.inverse_transform(y_std.reshape(-1, 1)).ravel()

    @property
    def in_dim(self):
        return len(self.num_cols) + sum(len(c) for c in self.ohe.categories_)


def train_margin_mlp(train_df, target_col="resid", val_frac=0.1, epochs=200, lr=1e-3,
                      batch_size=512, patience=20, grad_clip=5.0, lr_gamma=0.97,
                      shuffle_split=True, seed=C.SEED, verbose=False):
    n = len(train_df)
    n_val = max(int(n * val_frac), 1)
    if shuffle_split:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[val_idx]
    else:
        tr_df, va_df = train_df.iloc[:-n_val], train_df.iloc[-n_val:]

    enc = MarginEncoder().fit(tr_df, target_col=target_col)
    X_tr, X_va = enc.transform(tr_df), enc.transform(va_df)
    y_tr = torch.from_numpy(enc.transform_target(tr_df[target_col].to_numpy(dtype=np.float32)))
    y_va = torch.from_numpy(enc.transform_target(va_df[target_col].to_numpy(dtype=np.float32)))

    model = MarginMLP(enc.in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_gamma)
    loss_fn = nn.MSELoss()

    best_val, best_state, bad_epochs = np.inf, {k: v.clone() for k, v in model.state_dict().items()}, 0
    n_tr = len(tr_df)
    for epoch in range(epochs):
        model.train()
        perm_b = torch.randperm(n_tr)
        for i in range(0, n_tr, batch_size):
            idx = perm_b[i:i + batch_size]
            opt.zero_grad()
            pred = model(X_tr[idx])
            loss = loss_fn(pred, y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va), y_va).item()
        if np.isfinite(val_loss) and val_loss < best_val - 1e-9:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
        if verbose and epoch % 10 == 0:
            log.info(f"epoch {epoch} val_mse={val_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e}")
        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, enc


def predict_margin_mlp(model, enc, df):
    X = enc.transform(df)
    model.eval()
    with torch.no_grad():
        pred_std = model(X).numpy()
    return enc.inverse_target(pred_std)


if __name__ == "__main__":
    """python -m els_hedging_v1.margin_mlp : Stage-2 MLP만 독립적으로 walk-forward 진단."""
    import pandas as pd
    from .splits import sort_chronological, walk_forward_folds
    from .metrics import compute_metrics, summarize_folds

    ml = pd.read_parquet(C.CACHE_DIR / "dataset_ml.parquet")
    ml = sort_chronological(ml)
    folds = walk_forward_folds(len(ml))

    fold_metrics = []
    for fid, (tr_idx, te_idx) in enumerate(folds, start=1):
        train_df = ml.iloc[tr_idx].reset_index(drop=True)
        test_df = ml.iloc[te_idx].reset_index(drop=True)
        model, enc = train_margin_mlp(train_df, target_col="resid")
        pred = predict_margin_mlp(model, enc, test_df)
        m = compute_metrics(test_df["resid"], pred)
        log.info(f"[fold {fid}] stage2_margin_mlp: R2={m['r2']:.4f} MAPE={m['mape']:.4%} "
                  f"MAE={m['mae']:.5f} n={m['n']}")
        fold_metrics.append(m)

    pooled = summarize_folds(fold_metrics)
    log.info(f"풀링 결과: R2={pooled['r2']:.4f} MAPE={pooled['mape']:.4%} MAE={pooled['mae']:.5f} n={pooled['n']}")
