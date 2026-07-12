"""PI-DeepONet: 다자산 Black-Scholes PDE 잔차를 물리정규화(soft penalty)로 추가한 DeepONet.

기존 DeepONetCurve(deeponet.py)와 달리 trunk를 상태좌표 (x1,x2,x3,tau)로 재구성해 PDE 잔차를
autograd로 계산할 수 있게 한다. 물리정규화 아이디어는 Wang & Perdikaris, "Learning the solution
operator of parametric PDEs with physics-informed DeepONets" (Sci. Adv. 2021)을 참조해 ELS
이론가(worst-of 다자산 BS)에 독립 구현 (팀원 PI-DeepONet-ELS repo 미참조).

- Branch(연산자 입력 u): 계약+시장 요약 (기존 trunk+branch 전체: strk_0..11,B,coupon,tenor +
  u0..9,sig1..3,rho12/13/23,sig_eff) -> 계수 b_k(u).
- Trunk(좌표 입력 y=(x1,x2,x3,tau), 로그가격+잔여시간, 물리단위 그대로/비표준화) -> 기저 t_k(y).
  2차미분까지 필요하므로 활성함수는 tanh(매끄러움 보존, 마지막층은 linear).
- V(y;u) = sum_k b_k(u)*t_k(y) + bias(r) + out_bias.
- L = L_data(발행시점 x=0,tau=tenor 에서 MC 앵커) + lambda_pde*L_pde(다자산 BS PDE 잔차, collocation)
      + lambda_term*L_term(tau=0 단일시점 근사 payoff, 작은 가중치).
- autocall/KI(경로의존·불연속)는 물리로 넣지 않음 - MC 데이터손실이 담당. PDE는 관측일 사이
  연속영역의 매끄러운 BS 동역학만 규제(§0 V1_빌드 대비 PI_DeepONet_빌드_지시서).
- tau=만기까지 잔여시간(τ=T-t) 좌표이므로 dV/dt = -dV/dtau로 치환해 PDE에 대입.
"""
import argparse
import json
import logging
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from . import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pi_deeponet")

N_CURVE = len(C.NS_TENOR_NODES)                     # u0..9 = 10
N_VOLCORR = len(C.BRANCH_COLS) - N_CURVE             # sig1..3,rho12/13/23,sig_eff = 7
CONTRACT_COLS = C.TRUNK_COLS                         # strk_0..11,B,coupon,tenor = 15 (구 trunk -> 신 branch)
N_CONTRACT = len(CONTRACT_COLS)
N_ASSET = 3
N_COORD = N_ASSET + 1                                # x1,x2,x3,tau
EMB_DIM = 32
LOSS_TYPES = ("mse", "mae")
SIG_COLS = ["sig1", "sig2", "sig3", "rho12", "rho13", "rho23"]

X_LOW, X_HIGH = float(np.log(0.3)), float(np.log(1.7))   # collocation 로그가격 도메인(배리어~ATM 커버)
N_COL_DEFAULT = 24         # 상품당 collocation점. 논문/지시서 예시(64)에서 compute 비용으로 축소 - §runtime 문서화
N_COL_TERM_DEFAULT = 8     # tau=0 terminal 근사 점

LAMBDA_GRID = (0.0, 0.01, 0.1, 1.0)
DATA_FRAC_GRID = (0.25, 0.5, 1.0)

CACHE_PATH = C.OUT_DIR / "_pi_deeponet_runs_cache.jsonl"

torch.manual_seed(C.SEED)


class PIDeepONet(nn.Module):
    """branch: curve(1D-CNN)+volcorr+contract -> b_k(u). trunk: (x1,x2,x3,tau) -> t_k(y), tanh(미분가능)."""

    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.branch_cnn = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
        )
        cnn_out_dim = 16 * N_CURVE
        branch_in_dim = cnn_out_dim + N_VOLCORR + N_CONTRACT
        self.branch_mlp = nn.Sequential(
            nn.Linear(branch_in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, emb_dim),
        )
        # trunk는 PDE 잔차 위해 2차미분까지 매끄러워야 함(§1) -> 은닉층 tanh, 출력층은 linear(그 자체로 매끄러움)
        self.trunk_mlp = nn.Sequential(
            nn.Linear(N_COORD, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, emb_dim),
        )
        self.aux_bias = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
        self.out_bias = nn.Parameter(torch.tensor(1.0))

    def branch_forward(self, curve, volcorr, contract):
        x = curve.unsqueeze(1)
        x = self.branch_cnn(x).flatten(1)
        x = torch.cat([x, volcorr, contract], dim=1)
        return self.branch_mlp(x)                      # (B, emb_dim)

    def trunk_forward(self, y):
        return self.trunk_mlp(y)                        # (N, emb_dim)

    def value_flat(self, branch_emb_flat, y_flat, aux_scaled_flat):
        """branch_emb_flat/y_flat/aux_scaled_flat: 행 단위 1:1 대응(broadcast)된 flat 텐서."""
        trunk_emb = self.trunk_forward(y_flat)
        dot = (branch_emb_flat * trunk_emb).sum(-1)
        bias = self.aux_bias(aux_scaled_flat.unsqueeze(-1)).squeeze(-1)
        return dot + bias + self.out_bias


class PIDeepONetDataset:
    """branch(curve+volcorr+contract) 표준화(train만 fit). trunk 좌표(x,tau)와 PDE 계수(sigma/rho/r)는
    물리단위 그대로 유지 - PDE가 물리스케일에서 성립해야 잔차가 의미있음(표준화 금지)."""

    def __init__(self):
        self.curve_scaler = StandardScaler()
        self.volcorr_scaler = StandardScaler()
        self.contract_scaler = StandardScaler()
        self.aux_scaler = StandardScaler()       # r의 NN 안정화용 표준화(신경망 bias 입력 전용)
        self.target_scaler = StandardScaler()

    def fit(self, df, target_col="MC"):
        self.curve_scaler.fit(df[C.U_COLS].to_numpy())
        self.volcorr_scaler.fit(df[C.BRANCH_COLS[N_CURVE:]].to_numpy())
        self.contract_scaler.fit(df[CONTRACT_COLS].to_numpy())
        self.aux_scaler.fit(df[["r"]].to_numpy())
        self.target_scaler.fit(df[[target_col]].to_numpy())
        return self

    def transform_branch(self, df):
        curve = self.curve_scaler.transform(df[C.U_COLS].to_numpy()).astype(np.float32)
        volcorr = self.volcorr_scaler.transform(df[C.BRANCH_COLS[N_CURVE:]].to_numpy()).astype(np.float32)
        contract = self.contract_scaler.transform(df[CONTRACT_COLS].to_numpy()).astype(np.float32)
        aux = self.aux_scaler.transform(df[["r"]].to_numpy()).astype(np.float32)
        return (torch.from_numpy(curve), torch.from_numpy(volcorr),
                torch.from_numpy(contract), torch.from_numpy(aux))

    def transform_target(self, y):
        return self.target_scaler.transform(y.reshape(-1, 1)).astype(np.float32).ravel()

    def inverse_target(self, y_std):
        return self.target_scaler.inverse_transform(y_std.reshape(-1, 1)).ravel()

    def target_affine_torch(self):
        mean = torch.tensor(float(self.target_scaler.mean_[0]))
        scale = torch.tensor(float(self.target_scaler.scale_[0]))
        return mean, scale


def payoff_terminal(x1, x2, x3, K, coupon):
    """tau=0 만기 근사 payoff(원금=1 기준). autocall/KI 경로의존 세부는 무시(§3 - 데이터가 담당)."""
    worst = torch.minimum(torch.minimum(torch.exp(x1), torch.exp(x2)), torch.exp(x3))
    redemption = torch.ones_like(worst) + coupon
    return torch.where(worst >= K, redemption, worst)


def pde_residual(model, branch_emb_flat, x1, x2, x3, tau,
                  sig1, sig2, sig3, rho12, rho13, rho23, r_raw, aux_scaled_flat):
    """다자산 BS PDE 잔차 (§2):
    dV/dt + sum_i(r-1/2 sigma_i^2) dV/dx_i + 1/2 sum_ij rho_ij sigma_i sigma_j d2V/dx_i dx_j - rV = 0
    trunk 좌표는 tau=잔여시간(T-t)이므로 dV/dt = -dV/dtau 로 치환.
    x1,x2,x3,tau는 requires_grad=True leaf 텐서여야 함.
    """
    y = torch.stack([x1, x2, x3, tau], dim=1)
    V = model.value_flat(branch_emb_flat, y, aux_scaled_flat)

    ones = torch.ones_like(V)
    dVdx1, dVdx2, dVdx3, dVdtau = torch.autograd.grad(
        V, [x1, x2, x3, tau], grad_outputs=ones, create_graph=True)

    d2x1 = torch.autograd.grad(dVdx1, [x1, x2, x3], grad_outputs=torch.ones_like(dVdx1), create_graph=True)
    d2x2 = torch.autograd.grad(dVdx2, [x1, x2, x3], grad_outputs=torch.ones_like(dVdx2), create_graph=True)
    d2x3 = torch.autograd.grad(dVdx3, [x1, x2, x3], grad_outputs=torch.ones_like(dVdx3), create_graph=True)

    d2V_dx1dx1 = d2x1[0]
    d2V_dx2dx2 = d2x2[1]
    d2V_dx3dx3 = d2x3[2]
    d2V_dx1dx2 = 0.5 * (d2x1[1] + d2x2[0])
    d2V_dx1dx3 = 0.5 * (d2x1[2] + d2x3[0])
    d2V_dx2dx3 = 0.5 * (d2x2[2] + d2x3[1])

    drift = (r_raw - 0.5 * sig1 ** 2) * dVdx1 + (r_raw - 0.5 * sig2 ** 2) * dVdx2 + (r_raw - 0.5 * sig3 ** 2) * dVdx3
    diffusion = 0.5 * (sig1 ** 2 * d2V_dx1dx1 + sig2 ** 2 * d2V_dx2dx2 + sig3 ** 2 * d2V_dx3dx3
                        + 2 * rho12 * sig1 * sig2 * d2V_dx1dx2
                        + 2 * rho13 * sig1 * sig3 * d2V_dx1dx3
                        + 2 * rho23 * sig2 * sig3 * d2V_dx2dx3)
    return -dVdtau + drift + diffusion - r_raw * V


def train_pi_deeponet(train_df, target_col="MC", lambda_pde=0.0, lambda_term=0.0,
                       n_col=N_COL_DEFAULT, n_col_term=N_COL_TERM_DEFAULT, data_frac=1.0,
                       val_frac=0.1, epochs=100, lr=1e-3, batch_size=256, patience=15,
                       grad_clip=5.0, lr_gamma=0.97, loss_type="mae", shuffle_split=True,
                       seed=C.SEED, verbose=False):
    """재현성 픽스(v2 관례): 함수 시작부에서 torch/numpy 모두 재시딩. 이 함수 호출은 프로세스 내
    다른 어떤 호출과도 RNG 상태를 공유하지 않음(v2 §torch.manual_seed 1회시딩 버그 교훈)."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    n = len(train_df)
    n_val = max(int(n * val_frac), 1)
    perm = rng.permutation(n)
    if shuffle_split:
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
    else:
        tr_idx, val_idx = np.arange(0, n - n_val), np.arange(n - n_val, n)
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[val_idx].reset_index(drop=True)
    n_tr = len(tr_df)

    # data_frac: L_data에 쓸 라벨 서브셋(나머지 행은 collocation/PDE 전용 - §5 데이터효율)
    n_labeled = max(int(round(n_tr * data_frac)), 1)
    labeled_idx = rng.permutation(n_tr)[:n_labeled]
    labeled_mask_tr = np.zeros(n_tr, dtype=bool)
    labeled_mask_tr[labeled_idx] = True
    labeled_tr_t = torch.from_numpy(labeled_mask_tr)

    scaler = PIDeepONetDataset().fit(tr_df, target_col=target_col)
    curve_tr, vc_tr, contract_tr, aux_tr = scaler.transform_branch(tr_df)
    curve_va, vc_va, contract_va, aux_va = scaler.transform_branch(va_df)
    y_tr_std = torch.from_numpy(scaler.transform_target(tr_df[target_col].to_numpy(dtype=np.float32)))
    y_va_std = torch.from_numpy(scaler.transform_target(va_df[target_col].to_numpy(dtype=np.float32)))
    t_mean, t_scale = scaler.target_affine_torch()

    tenor_tr_t = torch.from_numpy(tr_df["tenor"].to_numpy(dtype=np.float32))
    tenor_va_t = torch.from_numpy(va_df["tenor"].to_numpy(dtype=np.float32))
    r_raw_tr_t = torch.from_numpy(tr_df["r"].to_numpy(dtype=np.float32))
    sig_tr_t = torch.from_numpy(tr_df[SIG_COLS].to_numpy(dtype=np.float32))
    K_tr_t = torch.from_numpy(tr_df["strk_11"].to_numpy(dtype=np.float32))
    coupon_tr_t = torch.from_numpy(tr_df["coupon"].to_numpy(dtype=np.float32))

    model = PIDeepONet()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_gamma)

    def _anchor_V(curve, vc, contract, aux, tenor):
        branch_emb = model.branch_forward(curve, vc, contract)
        B = curve.shape[0]
        y = torch.zeros(B, N_COORD)
        y[:, N_ASSET] = tenor
        V = model.value_flat(branch_emb, y, aux.squeeze(-1))
        return V, branch_emb

    def _data_loss(V_std_pred, y_std_true, mask):
        if mask is not None:
            if mask.sum() == 0:
                # data_frac<1.0인 소배치(특히 마지막 partial batch)에서 라벨이 0개로 뽑힐 수 있음.
                # 그래프에서 완전히 끊긴 torch.zeros(())를 반환하면 loss=l_data뿐인 경우(plain, lambda_pde=0)
                # backward()가 "does not require grad" 에러로 죽는다 - 예측치에 0을 곱해 그래프 연결은
                # 유지한 채(dL/dparam=0, 값은 정확히 0) 안전하게 무손실 처리.
                return (V_std_pred * 0.0).sum()
            V_std_pred = V_std_pred[mask]
            y_std_true = y_std_true[mask]
        diff = V_std_pred - y_std_true
        return diff.abs().mean() if loss_type == "mae" else (diff ** 2).mean()

    def _pde_loss(branch_emb, tenor, sig, r_raw, aux, n_pts):
        B = branch_emb.shape[0]
        x1 = rng.uniform(X_LOW, X_HIGH, size=(B, n_pts)).astype(np.float32)
        x2 = rng.uniform(X_LOW, X_HIGH, size=(B, n_pts)).astype(np.float32)
        x3 = rng.uniform(X_LOW, X_HIGH, size=(B, n_pts)).astype(np.float32)
        tau_frac = rng.uniform(0.0, 1.0, size=(B, n_pts)).astype(np.float32)
        tenor_np = tenor.detach().numpy()
        tau = tau_frac * tenor_np[:, None]
        x1_t = torch.from_numpy(x1.reshape(-1)).requires_grad_(True)
        x2_t = torch.from_numpy(x2.reshape(-1)).requires_grad_(True)
        x3_t = torch.from_numpy(x3.reshape(-1)).requires_grad_(True)
        tau_t = torch.from_numpy(tau.reshape(-1)).requires_grad_(True)

        branch_flat = branch_emb.repeat_interleave(n_pts, dim=0)
        aux_flat = aux.squeeze(-1).repeat_interleave(n_pts, dim=0)
        sig_flat = sig.repeat_interleave(n_pts, dim=0)
        r_flat = r_raw.repeat_interleave(n_pts, dim=0)

        res = pde_residual(model, branch_flat, x1_t, x2_t, x3_t, tau_t,
                            sig_flat[:, 0], sig_flat[:, 1], sig_flat[:, 2],
                            sig_flat[:, 3], sig_flat[:, 4], sig_flat[:, 5],
                            r_flat, aux_flat)
        return (res ** 2).mean()

    def _term_loss(branch_emb, sig3, K, coupon, aux, n_pts):
        B = branch_emb.shape[0]
        x1 = rng.uniform(X_LOW, X_HIGH, size=(B, n_pts)).astype(np.float32)
        x2 = rng.uniform(X_LOW, X_HIGH, size=(B, n_pts)).astype(np.float32)
        x3 = rng.uniform(X_LOW, X_HIGH, size=(B, n_pts)).astype(np.float32)
        x1_t = torch.from_numpy(x1.reshape(-1))
        x2_t = torch.from_numpy(x2.reshape(-1))
        x3_t = torch.from_numpy(x3.reshape(-1))
        tau_t = torch.zeros_like(x1_t)
        y = torch.stack([x1_t, x2_t, x3_t, tau_t], dim=1)
        branch_flat = branch_emb.repeat_interleave(n_pts, dim=0)
        aux_flat = aux.squeeze(-1).repeat_interleave(n_pts, dim=0)
        V = model.value_flat(branch_flat, y, aux_flat)
        K_flat = K.repeat_interleave(n_pts, dim=0)
        coupon_flat = coupon.repeat_interleave(n_pts, dim=0)
        target = payoff_terminal(x1_t, x2_t, x3_t, K_flat, coupon_flat)
        target_std = (target - t_mean) / t_scale
        diff = V - target_std
        return diff.abs().mean() if loss_type == "mae" else (diff ** 2).mean()

    best_val = np.inf
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    bad_epochs = 0
    pde_log = []
    n_batches_total = 0
    for epoch in range(epochs):
        model.train()
        perm_b = torch.randperm(n_tr)
        l_pde_last = torch.zeros(())
        l_term_last = torch.zeros(())
        for i in range(0, n_tr, batch_size):
            idx = perm_b[i:i + batch_size]
            opt.zero_grad()

            V_std, branch_emb = _anchor_V(curve_tr[idx], vc_tr[idx], contract_tr[idx], aux_tr[idx], tenor_tr_t[idx])
            mask = labeled_tr_t[idx]
            loss = _data_loss(V_std, y_tr_std[idx], mask)

            if lambda_pde > 0:
                l_pde_last = _pde_loss(branch_emb, tenor_tr_t[idx], sig_tr_t[idx], r_raw_tr_t[idx], aux_tr[idx], n_col)
                loss = loss + lambda_pde * l_pde_last
            if lambda_term > 0:
                l_term_last = _term_loss(branch_emb, sig_tr_t[idx][:, :3], K_tr_t[idx], coupon_tr_t[idx], aux_tr[idx], n_col_term)
                loss = loss + lambda_term * l_term_last

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
            n_batches_total += 1
        scheduler.step()

        model.eval()
        V_va_std, _ = _anchor_V(curve_va, vc_va, contract_va, aux_va, tenor_va_t)
        val_loss = _data_loss(V_va_std, y_va_std, None).item()
        pde_log.append({"epoch": epoch, "val_data_loss": val_loss,
                         "l_pde": float(l_pde_last.detach()), "l_term": float(l_term_last.detach())})
        if np.isfinite(val_loss) and val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if verbose and epoch % 10 == 0:
            log.info(f"epoch {epoch} val_{loss_type}={val_loss:.6f} l_pde={float(l_pde_last):.6f} "
                      f"l_term={float(l_term_last):.6f} lr={scheduler.get_last_lr()[0]:.2e}")
        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, pde_log


def predict_pi_deeponet(model, scaler, df):
    curve, vc, contract, aux = scaler.transform_branch(df)
    tenor = torch.from_numpy(df["tenor"].to_numpy(dtype=np.float32))
    model.eval()
    with torch.no_grad():
        branch_emb = model.branch_forward(curve, vc, contract)
        B = curve.shape[0]
        y = torch.zeros(B, N_COORD)
        y[:, N_ASSET] = tenor
        V_std = model.value_flat(branch_emb, y, aux.squeeze(-1)).numpy()
    return scaler.inverse_target(V_std)


# ---------------------------------------------------------------------------
# 실험 오케스트레이션 (§5,6,7,8) - MC 재실행 없음, dataset_deeponet.parquet만 재사용
# ---------------------------------------------------------------------------

def _load_don():
    from .splits import sort_chronological
    don = pd.read_parquet(C.CACHE_DIR / "dataset_deeponet.parquet")
    return sort_chronological(don)


def _append_cache(record):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _load_cache():
    """(tag,fold,data_frac) 별로 마지막(최신) 기록만 남긴다 - 서로 다른 CLI 호출(정합체크 +
    lambda-scan 등)이 같은 config를 중복 실행해도 pooled 집계에서 이중계산되지 않도록."""
    if not CACHE_PATH.exists():
        return []
    with open(CACHE_PATH, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]
    dedup = {}
    for r in raw:
        key = (r.get("tag"), r.get("fold"), r.get("data_frac"))
        dedup[key] = r
    return list(dedup.values())


def _run_one(don_df, folds, fold_ids, lambda_pde, lambda_term, data_frac, loss_type,
             epochs, patience, batch_size, n_col, tag, resume=True):
    """folds 중 fold_ids에 해당하는 fold만 학습·평가해 레코드 리스트 반환(+JSONL cache 저장).
    resume=True(기본)면 (tag,fold,data_frac)이 이미 캐시에 있으면 재학습 없이 캐시값을 재사용
    - 장시간 실험이 중간에 죽어도(background 킬 등) 이어서 돌릴 수 있게(§재현성과 무관, 순수 체크포인트)."""
    from .metrics import compute_metrics
    cached_by_key = {}
    if resume:
        for r in _load_cache():
            cached_by_key[(r.get("tag"), r.get("fold"), r.get("data_frac"))] = r

    records = []
    for fid in fold_ids:
        key = (tag, fid, data_frac)
        if resume and key in cached_by_key:
            log.info(f"[{tag}] fold={fid} data_frac={data_frac}: 캐시 재사용(스킵) "
                      f"R2={cached_by_key[key]['r2']:.4f}")
            records.append(cached_by_key[key])
            continue
        tr_idx, te_idx = folds[fid - 1]
        train_df = don_df.iloc[tr_idx].reset_index(drop=True)
        test_df = don_df.iloc[te_idx].reset_index(drop=True)
        t0 = time.perf_counter()
        model, scaler, pde_log = train_pi_deeponet(
            train_df, target_col="MC", lambda_pde=lambda_pde, lambda_term=lambda_term,
            data_frac=data_frac, loss_type=loss_type, epochs=epochs, patience=patience,
            batch_size=batch_size, n_col=n_col)
        train_sec = time.perf_counter() - t0
        pred = predict_pi_deeponet(model, scaler, test_df)
        m = compute_metrics(test_df["MC"], pred)
        rec = {"tag": tag, "fold": fid, "lambda_pde": lambda_pde, "lambda_term": lambda_term,
               "data_frac": data_frac, "loss_type": loss_type, "train_seconds": round(train_sec, 1),
               "n_epochs_ran": len(pde_log), "final_l_pde": pde_log[-1]["l_pde"] if pde_log else 0.0,
               **m}
        log.info(f"[{tag}] fold={fid} lambda_pde={lambda_pde} data_frac={data_frac}: "
                  f"R2={m['r2']:.4f} MAPE={m.get('mape', float('nan')):.4%} "
                  f"bp={m.get('bp_error', float('nan')):.1f} Spearman={m['spearman']:.4f} "
                  f"({train_sec:.1f}s, {len(pde_log)} epochs)")
        records.append(rec)
        _append_cache(rec)
    return records


def _consistency_check(don_df, folds, lambda_pde, epochs, patience, batch_size, n_col):
    """§9 최소성공 기준: lambda_pde=0일 때 코드경로상 PDE/terminal 항이 전혀 계산되지 않는지
    (연산 자체 미실행 - 근사적 0이 아니라 완전한 부재) 확인 + walk-forward 4-fold 베이스라인 확보.

    주의: 신 아키텍처(trunk=좌표)는 구 DeepONetCurve(trunk=계약항)와 파라미터화 자체가 다르므로
    lambda_pde=0에서도 두 모델의 예측치가 '수치적으로 동일'할 수는 없다 - 여기서 검증하는 '정합'은
    (a) lambda_pde=0일 때 PDE/terminal 손실이 정말로 손실 그래프에서 완전히 빠지는지(구현 정합),
    (b) 그 상태로 학습한 성능이 구 DeepONet과 합리적으로 comparable한지(§보조 참고), 두 가지다.
    """
    log.info("=== 정합체크 (a): lambda_pde=0 코드경로 - PDE/terminal 항 연산 자체 미실행 검증 ===")
    small = don_df.iloc[:300].reset_index(drop=True)
    _, _, pde_log = train_pi_deeponet(small, target_col="MC", lambda_pde=0.0, lambda_term=0.0,
                                        epochs=2, batch_size=64, patience=5)
    all_zero = all(r["l_pde"] == 0.0 and r["l_term"] == 0.0 for r in pde_log)
    log.info(f"lambda_pde=0 학습 로그의 l_pde/l_term 전부 0.0(연산 미실행) = {all_zero}")
    if not all_zero:
        raise RuntimeError("정합체크 실패: lambda_pde=0인데 l_pde/l_term이 0이 아님 - 구현 버그")
    log.info("=> (a) 통과: lambda_pde=0이면 총손실 = L_data 그 자체(부동소수점 근사가 아닌 코드경로 보장)")

    tag = "plain(lambda=0)" if lambda_pde == 0.0 else f"PI(lambda_pde={lambda_pde})"
    log.info(f"=== 정합체크 (b): {tag} walk-forward 4-fold 베이스라인 ===")
    records = _run_one(don_df, folds, [1, 2, 3, 4], lambda_pde=lambda_pde, lambda_term=0.0, data_frac=1.0,
                        loss_type="mae", epochs=epochs, patience=patience, batch_size=batch_size,
                        n_col=n_col, tag=tag)
    from .metrics import summarize_folds
    pooled = summarize_folds(records)
    log.info(f"=> {tag} pooled: R2={pooled['r2']:.4f} MAPE={pooled['mape']:.4%} "
              f"bp={pooled['bp_error']:.1f} Spearman={pooled['spearman']:.4f} n={pooled['n']}")
    return records


def _lambda_scan(don_df, folds, epochs, patience, batch_size, n_col):
    from .metrics import summarize_folds

    log.info("=== §5.1 lambda_pde 그리드 스캔 (plain vs PI, walk-forward 4-fold) ===")
    all_records = []
    for lam in LAMBDA_GRID:
        tag = "plain(lambda=0)" if lam == 0.0 else f"PI(lambda_pde={lam})"
        recs = _run_one(don_df, folds, [1, 2, 3, 4], lambda_pde=lam, lambda_term=0.0, data_frac=1.0,
                         loss_type="mae", epochs=epochs, patience=patience, batch_size=batch_size,
                         n_col=n_col, tag=tag)
        all_records.extend(recs)

    pooled_by_lam = {lam: summarize_folds([r for r in all_records if r["lambda_pde"] == lam])
                      for lam in LAMBDA_GRID}
    best_lam = max([l for l in LAMBDA_GRID if l > 0], key=lambda l: pooled_by_lam[l]["r2"])
    log.info("lambda 그리드 pooled R2: " + str({l: round(pooled_by_lam[l]["r2"], 4) for l in LAMBDA_GRID}))
    log.info(f"최적 lambda_pde(양수 중) = {best_lam} (R2={pooled_by_lam[best_lam]['r2']:.4f})")

    log.info("=== §5.2 데이터효율 (라벨 25/50/100%, plain vs PI, walk-forward 4-fold) ===")
    eff_records = []
    for frac in DATA_FRAC_GRID:
        for lam, kind in [(0.0, "plain"), (best_lam, "PI")]:
            tag = f"dataeff_{kind}_frac{frac}"
            if lam == 0.0 and frac == 1.0:
                # plain@100%는 위 lambda-grid에서 이미 계산된 것과 동일 config -> 재사용
                recs = [dict(r, tag=tag) for r in all_records if r["lambda_pde"] == 0.0]
                for r in recs:
                    _append_cache(r)
            elif lam == best_lam and frac == 1.0:
                recs = [dict(r, tag=tag) for r in all_records if r["lambda_pde"] == best_lam]
                for r in recs:
                    _append_cache(r)
            else:
                recs = _run_one(don_df, folds, [1, 2, 3, 4], lambda_pde=lam, lambda_term=0.0, data_frac=frac,
                                 loss_type="mae", epochs=epochs, patience=patience, batch_size=batch_size,
                                 n_col=n_col, tag=tag)
            eff_records.extend(recs)

    with open(C.OUT_DIR / "_pi_deeponet_best_lambda.json", "w", encoding="utf-8") as f:
        json.dump({"best_lambda_pde": best_lam}, f)
    log.info(f"lambda-scan 완료. best_lambda_pde={best_lam} 저장 -> _pi_deeponet_best_lambda.json")
    return all_records, eff_records


def _build_comparison_tables():
    from .metrics import summarize_folds
    cache = _load_cache()
    if not cache:
        raise RuntimeError("캐시가 비어 있음 - 먼저 --lambda-pde/--lambda-scan을 실행하세요.")

    # plain-vs-PI lambda 그리드 표 (data_frac==1.0인 레코드만)
    grid_records = [r for r in cache if r.get("data_frac") == 1.0 and str(r.get("tag", "")).startswith(("plain", "PI("))]
    tags = sorted(set(r["tag"] for r in grid_records),
                  key=lambda t: (0, 0) if t.startswith("plain") else (1, float(t.split("=")[-1].rstrip(")"))))
    rows = []
    for tag in tags:
        recs = [r for r in grid_records if r["tag"] == tag]
        if not recs:
            continue
        pooled = summarize_folds(recs)
        rows.append({"tag": tag, "lambda_pde": recs[0]["lambda_pde"],
                      "R2": round(pooled["r2"], 4), "MAPE": round(pooled["mape"], 4),
                      "bp_error": round(pooled["bp_error"], 1), "MAE": round(pooled["mae"], 5),
                      "RMSE": round(pooled["rmse"], 5), "Spearman": round(pooled["spearman"], 4),
                      "n": pooled["n"], "mean_train_seconds": round(np.mean([r["train_seconds"] for r in recs]), 1)})
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(C.OUT_DIR / "pi_deeponet_comparison.csv", index=False, encoding="utf-8-sig")
    log.info("\nplain vs PI(lambda_pde 그리드) 비교표:\n" + comparison_df.to_string(index=False))

    # 데이터효율 표
    eff_records = [r for r in cache if str(r.get("tag", "")).startswith("dataeff_")]
    eff_tags = sorted(set(r["tag"] for r in eff_records))
    rows2 = []
    for tag in eff_tags:
        recs = [r for r in eff_records if r["tag"] == tag]
        if not recs:
            continue
        pooled = summarize_folds(recs)
        kind = "PI" if "_PI_" in tag else "plain"
        frac = float(tag.split("frac")[-1])
        rows2.append({"tag": tag, "kind": kind, "data_frac": frac, "lambda_pde": recs[0]["lambda_pde"],
                       "R2": round(pooled["r2"], 4), "MAPE": round(pooled["mape"], 4),
                       "bp_error": round(pooled["bp_error"], 1), "MAE": round(pooled["mae"], 5),
                       "RMSE": round(pooled["rmse"], 5), "Spearman": round(pooled["spearman"], 4), "n": pooled["n"]})
    dataeff_df = pd.DataFrame(rows2).sort_values(["kind", "data_frac"])
    dataeff_df.to_csv(C.OUT_DIR / "pi_deeponet_data_efficiency.csv", index=False, encoding="utf-8-sig")
    log.info("\n데이터효율(25/50/100% 라벨) 비교표:\n" + dataeff_df.to_string(index=False))
    return comparison_df, dataeff_df


def _reproducibility_check(best_lam, epochs, patience, batch_size, n_col, fold=4):
    """canonical config(최적 lambda_pde, 단일 fold) 별도 프로세스 2회 실행 -> R2 등 전부 일치 확인."""
    log.info(f"=== 재현성 검증: lambda_pde={best_lam}, fold={fold} config를 별도 프로세스 2회 실행 ===")
    import os
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "KMP_DUPLICATE_LIB_OK": "TRUE"}
    outs = []
    for i in range(2):
        out_path = C.OUT_DIR / f"_pi_repro_run{i + 1}.json"
        cmd = [sys.executable, "-m", "els_hedging_v1.pi_deeponet", "--repro-run",
               "--lambda-pde", str(best_lam), "--fold", str(fold),
               "--epochs", str(epochs), "--patience", str(patience),
               "--batch-size", str(batch_size), "--n-col", str(n_col), "--out", str(out_path)]
        subprocess.run(cmd, check=True, env=env)
        with open(out_path, encoding="utf-8") as f:
            outs.append(json.load(f))

    keys = ["r2", "mae", "rmse", "mape", "bp_error", "spearman"]
    match_flags = {k: (outs[0][k] == outs[1][k]) for k in keys}
    all_match = all(match_flags.values())
    df = pd.DataFrame([
        {"run": "run1", **{k: outs[0][k] for k in keys}},
        {"run": "run2", **{k: outs[1][k] for k in keys}},
        {"run": "exact_match", **match_flags},
    ])
    df.to_csv(C.OUT_DIR / "pi_deeponet_reproducibility_check.csv", index=False, encoding="utf-8-sig")
    log.info(f"재현성 검증 완료(전부 일치={all_match}):\n" + df.to_string(index=False))
    return all_match, df


def _make_figures(best_lam, epochs, patience, batch_size, n_col, fold=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    don_df = _load_don()
    from .splits import walk_forward_folds
    folds = walk_forward_folds(len(don_df))
    tr_idx, te_idx = folds[fold - 1]
    train_df = don_df.iloc[tr_idx].reset_index(drop=True)
    test_df = don_df.iloc[te_idx].reset_index(drop=True)

    fig_dir = C.OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"그림용 재학습: plain(lambda=0) vs PI(lambda_pde={best_lam}), fold={fold}")
    m_plain, s_plain, _ = train_pi_deeponet(train_df, target_col="MC", lambda_pde=0.0, lambda_term=0.0,
                                              epochs=epochs, patience=patience, batch_size=batch_size, n_col=n_col)
    pred_plain = predict_pi_deeponet(m_plain, s_plain, test_df)

    m_pi, s_pi, pde_log_pi = train_pi_deeponet(train_df, target_col="MC", lambda_pde=best_lam, lambda_term=0.0,
                                                 epochs=epochs, patience=patience, batch_size=batch_size, n_col=n_col,
                                                 verbose=True)
    pred_pi = predict_pi_deeponet(m_pi, s_pi, test_df)
    true = test_df["MC"].to_numpy()

    # 1) plain-vs-PI 산점도
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, pred, title in zip(axes, [pred_plain, pred_pi], ["plain DeepONet(lambda=0)", f"PI-DeepONet(lambda_pde={best_lam})"]):
        ax.scatter(true, pred, s=6, alpha=0.4)
        lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("MC(true)"); ax.set_ylabel("predicted"); ax.set_title(title)
    fig.suptitle(f"plain vs PI-DeepONet: 예측 vs MC (fold {fold})")
    fig.tight_layout()
    fig.savefig(fig_dir / "pi_deeponet_01_plain_vs_pi_scatter.png", dpi=130)
    plt.close(fig)

    # 2) L_pde 수렴
    epochs_x = [r["epoch"] for r in pde_log_pi]
    l_pde_y = [r["l_pde"] for r in pde_log_pi]
    val_y = [r["val_data_loss"] for r in pde_log_pi]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(epochs_x, l_pde_y, color="tab:red", label="L_pde(마지막 배치)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("L_pde", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(epochs_x, val_y, color="tab:blue", label="val L_data")
    ax2.set_ylabel("val L_data(표준화)", color="tab:blue")
    fig.suptitle(f"PI-DeepONet 학습 수렴 (lambda_pde={best_lam}, fold {fold})")
    fig.tight_layout()
    fig.savefig(fig_dir / "pi_deeponet_02_pde_convergence.png", dpi=130)
    plt.close(fig)

    # 3) 데이터효율 곡선
    cache = _load_cache()
    eff_records = [r for r in cache if str(r.get("tag", "")).startswith("dataeff_")]
    from .metrics import summarize_folds
    fig, ax = plt.subplots(figsize=(7, 5))
    for kind, marker in [("plain", "o"), ("PI", "s")]:
        fracs, r2s = [], []
        for frac in DATA_FRAC_GRID:
            recs = [r for r in eff_records if f"_{kind}_frac{frac}" in r["tag"]]
            if recs:
                fracs.append(frac); r2s.append(summarize_folds(recs)["r2"])
        if fracs:
            ax.plot(fracs, r2s, marker=marker, label=kind)
    ax.set_xlabel("L_data에 사용한 MC 라벨 비율"); ax.set_ylabel("이론가(Stage-1) pooled R2")
    ax.set_title("데이터효율: plain vs PI-DeepONet"); ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "pi_deeponet_03_data_efficiency.png", dpi=130)
    plt.close(fig)

    log.info(f"그림 저장 완료 -> {fig_dir}/pi_deeponet_0{{1,2,3}}_*.png")


def _repro_run_main(args):
    """--repro-run: 단일 config/fold 학습 후 raw(비반올림) 지표를 JSON으로 저장 (재현성 검증용 서브프로세스)."""
    from .metrics import compute_metrics
    from .splits import walk_forward_folds
    don_df = _load_don()
    folds = walk_forward_folds(len(don_df))
    tr_idx, te_idx = folds[args.fold - 1]
    train_df = don_df.iloc[tr_idx].reset_index(drop=True)
    test_df = don_df.iloc[te_idx].reset_index(drop=True)
    model, scaler, _ = train_pi_deeponet(train_df, target_col="MC", lambda_pde=args.lambda_pde, lambda_term=0.0,
                                           epochs=args.epochs, patience=args.patience,
                                           batch_size=args.batch_size, n_col=args.n_col)
    pred = predict_pi_deeponet(model, scaler, test_df)
    m = compute_metrics(test_df["MC"], pred)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(m, f)
    log.info(f"[repro-run] fold={args.fold} lambda_pde={args.lambda_pde} R2={m['r2']!r} -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-pde", type=float, default=None,
                     help="단일 lambda_pde로 walk-forward 4-fold 실행 + (0이면) 정합체크")
    ap.add_argument("--lambda-scan", action="store_true", help="lambda_pde 그리드 + 데이터효율 실험")
    ap.add_argument("--report", action="store_true", help="비교표·그림·재현성검증 산출")
    ap.add_argument("--repro-run", action="store_true", help=argparse.SUPPRESS)  # 내부용(서브프로세스)
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-col", type=int, default=N_COL_DEFAULT)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.repro_run:
        _repro_run_main(args)
        return

    from .splits import walk_forward_folds
    don_df = _load_don()
    folds = walk_forward_folds(len(don_df))
    log.info(f"dataset_deeponet 로드: {len(don_df)} rows (MC 재실행 없음, 기존 캐시 재사용)")

    if args.lambda_pde is not None:
        _consistency_check(don_df, folds, args.lambda_pde, args.epochs, args.patience, args.batch_size, args.n_col)
    elif args.lambda_scan:
        _lambda_scan(don_df, folds, args.epochs, args.patience, args.batch_size, args.n_col)
    elif args.report:
        _build_comparison_tables()
        best_lam_path = C.OUT_DIR / "_pi_deeponet_best_lambda.json"
        if best_lam_path.exists():
            with open(best_lam_path, encoding="utf-8") as f:
                best_lam = json.load(f)["best_lambda_pde"]
        else:
            best_lam = LAMBDA_GRID[1]
            log.warning(f"best_lambda 캐시 없음 - 기본값 {best_lam} 사용")
        _reproducibility_check(best_lam, args.epochs, args.patience, args.batch_size, args.n_col, fold=args.fold)
        _make_figures(best_lam, args.epochs, args.patience, args.batch_size, args.n_col, fold=args.fold)
    else:
        log.info("--lambda-pde X / --lambda-scan / --report 중 하나를 지정하세요.")


if __name__ == "__main__":
    main()
