"""6. Stage-1 DeepONet-Curve: branch(1D-CNN 곡선 + vol/corr) x trunk(계약) -> MC 재현.

우리 본질은 이론가(Stage-1) — MC 재현 R2/%error가 핵심 지표.
손실함수 벤치: MSE/MAE/MAPE 세 가지로 학습·비교 가능 (--all-losses).
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from . import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deeponet")

N_CURVE = len(C.NS_TENOR_NODES)          # u0..9 = 10
N_VOLCORR = len(C.BRANCH_COLS) - N_CURVE  # sig1..3,rho12/13/23,sig_eff = 7
N_TRUNK = len(C.TRUNK_COLS)               # strk_0..11 + B + coupon + tenor = 15
EMB_DIM = 32
LOSS_TYPES = ("mse", "mae", "mape")

torch.manual_seed(C.SEED)


class DeepONetCurve(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.branch_cnn = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
        )
        cnn_out_dim = 16 * N_CURVE
        self.branch_mlp = nn.Sequential(
            nn.Linear(cnn_out_dim + N_VOLCORR, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, emb_dim),
        )
        self.trunk_mlp = nn.Sequential(
            nn.Linear(N_TRUNK, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, emb_dim),
        )
        self.aux_bias = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
        self.out_bias = nn.Parameter(torch.tensor(1.0))

    def forward(self, curve, volcorr, trunk, aux_r):
        x = curve.unsqueeze(1)                  # (B,1,10)
        x = self.branch_cnn(x)                  # (B,16,10)
        x = x.flatten(1)                         # (B,160)
        x = torch.cat([x, volcorr], dim=1)
        branch_emb = self.branch_mlp(x)          # (B,emb_dim)
        trunk_emb = self.trunk_mlp(trunk)        # (B,emb_dim)
        dot = (branch_emb * trunk_emb).sum(dim=1, keepdim=True)
        bias = self.aux_bias(aux_r)
        return (dot + bias + self.out_bias).squeeze(1)


class DeepONetDataset:
    """스케일러 fit(train만) + 텐서 변환 헬퍼. 타깃(MC)도 train 통계로 표준화."""

    def __init__(self):
        self.curve_scaler = StandardScaler()
        self.volcorr_scaler = StandardScaler()
        self.trunk_scaler = StandardScaler()
        self.aux_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def fit(self, df, target_col=None):
        self.curve_scaler.fit(df[C.U_COLS].to_numpy())
        self.volcorr_scaler.fit(df[C.BRANCH_COLS[N_CURVE:]].to_numpy())
        self.trunk_scaler.fit(df[C.TRUNK_COLS].to_numpy())
        self.aux_scaler.fit(df[["r"]].to_numpy())
        if target_col is not None:
            self.target_scaler.fit(df[[target_col]].to_numpy())
        return self

    def transform(self, df):
        curve = self.curve_scaler.transform(df[C.U_COLS].to_numpy()).astype(np.float32)
        volcorr = self.volcorr_scaler.transform(df[C.BRANCH_COLS[N_CURVE:]].to_numpy()).astype(np.float32)
        trunk = self.trunk_scaler.transform(df[C.TRUNK_COLS].to_numpy()).astype(np.float32)
        aux = self.aux_scaler.transform(df[["r"]].to_numpy()).astype(np.float32)
        return (torch.from_numpy(curve), torch.from_numpy(volcorr),
                torch.from_numpy(trunk), torch.from_numpy(aux))

    def transform_target(self, y):
        return self.target_scaler.transform(y.reshape(-1, 1)).astype(np.float32).ravel()

    def inverse_target(self, y_std):
        return self.target_scaler.inverse_transform(y_std.reshape(-1, 1)).ravel()

    def target_affine_torch(self):
        """표준화 타깃 -> 실스케일 역변환의 (mean, scale) torch 상수 (미분가능 아핀)."""
        mean = torch.tensor(float(self.target_scaler.mean_[0]))
        scale = torch.tensor(float(self.target_scaler.scale_[0]))
        return mean, scale


def _real_scale_loss(pred_std, y_std, mean, scale, loss_type):
    """표준화 공간 예측/타깃을 실스케일로 역변환한 뒤 손실 계산 (MSE/MAE/MAPE 비교가 경제적으로 의미있게).
    역변환이 순수 아핀(affine)이라 미분가능."""
    pred_real = pred_std * scale + mean
    y_real = y_std * scale + mean
    diff = pred_real - y_real
    if loss_type == "mse":
        return (diff ** 2).mean()
    if loss_type == "mae":
        return diff.abs().mean()
    if loss_type == "mape":
        return (diff.abs() / y_real.abs().clamp_min(1e-6)).mean()
    raise ValueError(f"unknown loss_type: {loss_type}")


def train_deeponet(train_df, target_col="MC", val_frac=0.1, epochs=200, lr=1e-3,
                    batch_size=512, patience=20, grad_clip=5.0, lr_gamma=0.97,
                    loss_type="mse", shuffle_split=True, seed=C.SEED, verbose=False):
    """train_df 내부에서 train/valid를 분리해 학습 (test/OOS는 walk-forward가 별도로 관리, 미래 유지).

    안정화: 타깃 train 통계로 표준화(학습은 표준화 공간, 손실은 실스케일로 역변환해 계산) +
    grad clip(max_norm=5) + LR 지수감쇠(ExponentialLR) + epochs 200 / patience 20.
    검증분할: shuffle_split=True(기본)면 랜덤 셔플(시간순 아님) - 국면 과적합 방지, 검증 자체의
    미래참조는 무방(§V1_피드백반영_지시서 1.5). test fold 자체는 항상 미래로 유지되므로 룩어헤드 아님.
    loss_type: 'mse'|'mae'|'mape' - 셋 다 실스케일에서 계산해 직접 비교 가능.
    """
    # 재현성 픽스: 모듈 top-level 1회 시딩만으로는 fold/호출 순서에 따라 결과가 갈림
    # (다른 torch 모델이 그 사이에 얼마나 난수를 소비했는지에 좌우됨) - 매 호출 시작부에서 재시딩.
    torch.manual_seed(seed)
    n = len(train_df)
    n_val = max(int(n * val_frac), 1)
    if shuffle_split:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[val_idx]
    else:
        tr_df, va_df = train_df.iloc[:-n_val], train_df.iloc[-n_val:]

    scaler = DeepONetDataset().fit(tr_df, target_col=target_col)
    curve_tr, vc_tr, trunk_tr, aux_tr = scaler.transform(tr_df)
    curve_va, vc_va, trunk_va, aux_va = scaler.transform(va_df)
    y_tr = torch.from_numpy(scaler.transform_target(tr_df[target_col].to_numpy(dtype=np.float32)))
    y_va = torch.from_numpy(scaler.transform_target(va_df[target_col].to_numpy(dtype=np.float32)))
    t_mean, t_scale = scaler.target_affine_torch()

    model = DeepONetCurve()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_gamma)

    best_val, best_state, bad_epochs = np.inf, {k: v.clone() for k, v in model.state_dict().items()}, 0
    n_tr = len(tr_df)
    for epoch in range(epochs):
        model.train()
        perm_b = torch.randperm(n_tr)
        for i in range(0, n_tr, batch_size):
            idx = perm_b[i:i + batch_size]
            opt.zero_grad()
            pred = model(curve_tr[idx], vc_tr[idx], trunk_tr[idx], aux_tr[idx])
            loss = _real_scale_loss(pred, y_tr[idx], t_mean, t_scale, loss_type)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(curve_va, vc_va, trunk_va, aux_va)
            val_loss = _real_scale_loss(val_pred, y_va, t_mean, t_scale, loss_type).item()
        # best_state 가드: 검증 개선이 한 번도 없어도(NaN 포함) 항상 마지막으로 저장된 상태가 있음
        # (초기값 자체가 첫 epoch 이전 가중치 clone이므로 None이 될 수 없음)
        if np.isfinite(val_loss) and val_loss < best_val - 1e-9:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
        if verbose and epoch % 10 == 0:
            log.info(f"epoch {epoch} val_{loss_type}={val_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e}")
        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, scaler


def predict_deeponet(model, scaler, df):
    curve, vc, trunk, aux = scaler.transform(df)
    model.eval()
    with torch.no_grad():
        pred_std = model(curve, vc, trunk, aux).numpy()
    return scaler.inverse_target(pred_std)


def _loss_comparison_main():
    """python -m els_hedging_v1.deeponet --all-losses
    walk-forward 4-fold x {mse,mae,mape} Stage-1 DeepONet 학습·비교, R2/MAE/MAPE 리포트."""
    from .splits import sort_chronological, walk_forward_folds
    from .metrics import compute_metrics

    don = pd.read_parquet(C.CACHE_DIR / "dataset_deeponet.parquet")
    don = sort_chronological(don)
    folds = walk_forward_folds(len(don))

    rows = []
    for loss_type in LOSS_TYPES:
        for fid, (tr_idx, te_idx) in enumerate(folds, start=1):
            train_df = don.iloc[tr_idx].reset_index(drop=True)
            test_df = don.iloc[te_idx].reset_index(drop=True)
            t0 = time.perf_counter()
            model, scaler = train_deeponet(train_df, target_col="MC", loss_type=loss_type)
            train_sec = time.perf_counter() - t0
            pred = predict_deeponet(model, scaler, test_df)
            m = compute_metrics(test_df["MC"], pred)
            m.update(loss_type=loss_type, fold=fid, train_seconds=round(train_sec, 1))
            log.info(f"[loss={loss_type} fold={fid}] R2={m['r2']:.4f} MAPE={m.get('mape', float('nan')):.4%} "
                      f"MAE={m['mae']:.5f} ({train_sec:.1f}s)")
            rows.append(m)

    res = pd.DataFrame(rows)
    pooled = res.groupby("loss_type").apply(
        lambda g: pd.Series({
            "r2": np.average(g["r2"], weights=g["n"]),
            "mae": np.average(g["mae"], weights=g["n"]),
            "mape": np.average(g["mape"], weights=g["n"]) if "mape" in g else np.nan,
            "rmse": np.average(g["rmse"], weights=g["n"]),
            "spearman": np.average(g["spearman"], weights=g["n"]),
        }), include_groups=False
    ).reset_index()
    log.info("\n손실함수별 Stage-1 DeepONet 풀링 결과:\n" + pooled.to_string(index=False))

    res.to_csv(C.OUT_DIR / "stage1_loss_comparison_folds.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(C.OUT_DIR / "stage1_loss_comparison.csv", index=False, encoding="utf-8-sig")
    log.info(f"저장 완료 -> {C.OUT_DIR / 'stage1_loss_comparison.csv'}")
    return pooled


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-losses", action="store_true", help="MSE/MAE/MAPE 손실 비교 실행")
    args = ap.parse_args()
    if args.all_losses:
        _loss_comparison_main()
    else:
        log.info("단독 실행: --all-losses 플래그로 손실함수 비교를 실행하세요. "
                  "일반 학습은 run_all.py를 통해 walk-forward로 수행됩니다.")
