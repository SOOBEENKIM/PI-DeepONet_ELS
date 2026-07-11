"""3. MC 이론가: worst-of 스텝다운 오토콜 3자산 GBM, 경로 10만, 일별 KI 감시, KR NS 할인.

python -m els_hedging_v1.mc_engine --n-paths 100000 --jobs 14
"""
import os

# BLAS(MKL/OpenBLAS)가 워커 프로세스 내부에서 자체적으로 멀티스레딩하면
# multiprocessing.Pool의 프로세스 병렬성과 겹쳐 코어 과다구독(oversubscription)이 발생한다.
# numpy import 전에 반드시 단일 스레드로 고정 (각 워커=프로세스 병렬성만 사용).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import logging
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as C
from .market_data import _ns_basis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mc_engine")

STEPS_PER_YEAR = 252


def ns_zero_rate_scalar(b0, b1, b2, lam, tau):
    tau = max(tau, 1e-6)
    basis = _ns_basis(np.array([tau]), lam)[0]
    return float(basis @ np.array([b0, b1, b2]))


CHUNK_DAYS = 21  # 월~월내 청크 단위로 벡터화(파이썬 루프 오버헤드 축소), 메모리 절약을 위해 작게 유지
DTYPE = np.float32


def simulate_worst_of_autocall(sig, corr, r_flat, ns_params, tenor, obs_t, obs_strikes,
                                B, coupon, n_paths, seed, steps_per_year=STEPS_PER_YEAR,
                                chunk_days=CHUNK_DAYS):
    """단일 상품 worst-of 오토콜 MC (일별 KI 감시, 관측일 정확 시뮬).

    sig: (3,) 연율화 vol.  corr: (3,3) 상관행렬.  r_flat: 드리프트용 단일금리.
    ns_params: (b0,b1,b2,lam) - 관측일별 정확 할인계수 계산용.
    obs_t: (nobs,) 관측시점(년, 오름차순, 마지막=만기).  obs_strikes: (nobs,) 콜/행사 스트라이크.
    B: 낙인배리어(0~1) 또는 nan(=KI 없음).  coupon: 연 쿠폰율.
    반환: (MC가격, 소요초)

    구현 노트: 시간축을 chunk_days 단위로 묶어 (chunk,n_paths,3) 텐서에 대해
    cumsum/누적최소를 한번에 벡터화 계산 -> 파이썬 레벨 스텝 루프(최대 ~1260회)를
    청크 루프(최대 ~60회)로 축소해 numpy 호출 오버헤드를 크게 줄인다.
    수학적으로는 일별(스텝별) 순차 시뮬레이션과 동일.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    n_steps = max(int(round(tenor * steps_per_year)), len(obs_t))
    dt = tenor / n_steps
    obs_step_idx = np.clip(np.round(obs_t / tenor * n_steps).astype(int), 1, n_steps)
    for i in range(1, len(obs_step_idx)):
        if obs_step_idx[i] <= obs_step_idx[i - 1]:
            obs_step_idx[i] = obs_step_idx[i - 1] + 1
    n_steps = max(n_steps, obs_step_idx[-1])

    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # float32 변환/부동소수 절삭 등으로 인해 여전히 PSD가 아닌 극단적 예외 케이스 대비:
        # 대각에 작은 jitter를 추가해 재시도(값 자체에 미치는 영향은 무시할 수준).
        jitter = 1e-8
        corr_fixed = corr.copy()
        for _ in range(6):
            try:
                L = np.linalg.cholesky(corr_fixed)
                break
            except np.linalg.LinAlgError:
                corr_fixed = corr_fixed + np.eye(3) * jitter
                jitter *= 10
        else:
            raise
    L = L.astype(DTYPE)
    drift = ((r_flat - 0.5 * sig ** 2) * dt).astype(DTYPE)   # (3,)
    vol_dt = (sig * np.sqrt(dt)).astype(DTYPE)               # (3,)

    cum_log = np.zeros((n_paths, 3), dtype=DTYPE)
    running_min_log = np.zeros(n_paths, dtype=DTYPE)
    has_ki = np.isfinite(B)
    log_B = np.log(B) if has_ki else -np.inf

    nobs = len(obs_t)
    closed_at_k = np.full(n_paths, -1, dtype=np.int32)
    active = np.ones(n_paths, dtype=bool)
    payoff = np.zeros(n_paths, dtype=DTYPE)

    obs_ptr = 0
    step = 0
    while step < n_steps:
        c = min(chunk_days, n_steps - step)
        Z = rng.standard_normal((c, n_paths, 3)).astype(DTYPE)
        corr_shock = Z @ L.T                                  # (c,n_paths,3)
        increments = drift + vol_dt * corr_shock
        step_cum = np.cumsum(increments, axis=0) + cum_log[None, :, :]   # (c,n_paths,3) 절대 누적 로그가격
        local_worst = step_cum.min(axis=2)                    # (c,n_paths)
        local_running_min = np.minimum.accumulate(local_worst, axis=0)
        combined_running_min = np.minimum(local_running_min, running_min_log[None, :])

        # 이 청크 안에 속하는 관측일 처리
        while obs_ptr < nobs and obs_step_idx[obs_ptr] <= step + c:
            k = obs_ptr
            local_idx = obs_step_idx[k] - step - 1
            worst_level = np.exp(step_cum[local_idx].min(axis=1))
            called_now = active & (worst_level >= obs_strikes[k])

            if k == nobs - 1:  # 만기
                run_min_at_k = combined_running_min[local_idx]
                ki_hit = has_ki & (run_min_at_k < log_B)
                mat_call = active & (worst_level >= obs_strikes[k])
                mat_ki_loss = active & ~mat_call & ki_hit
                mat_protect = active & ~mat_call & ~ki_hit

                payoff[mat_call] = 1.0 + coupon * obs_t[k]
                payoff[mat_ki_loss] = worst_level[mat_ki_loss]
                payoff[mat_protect] = 1.0
                closed_at_k[mat_call | mat_ki_loss | mat_protect] = k
                active[:] = False
            else:
                payoff[called_now] = 1.0 + coupon * obs_t[k]
                closed_at_k[called_now] = k
                active[called_now] = False
            obs_ptr += 1

        cum_log = step_cum[-1]
        running_min_log = combined_running_min[-1]
        step += c

    df_k = np.array([np.exp(-ns_zero_rate_scalar(*ns_params, t) * t) for t in obs_t])
    disc = df_k[closed_at_k].astype(DTYPE)
    mc_price = float(np.mean(payoff.astype(np.float64) * disc.astype(np.float64)))
    elapsed = time.perf_counter() - t0
    return mc_price, elapsed


def _bs_digital_validate():
    """1자산·무KI 단순 디지털콜 케이스로 엔진 검증 (BS 닫힌해와 비교)."""
    sig0, r0, T0, K0, coupon0 = 0.25, 0.03, 1.0, 1.0, 0.08
    sig = np.array([sig0, sig0, sig0])
    corr = np.eye(3) + (np.ones((3, 3)) - np.eye(3)) * 0.999999  # 사실상 완전상관 -> 단일자산과 동일
    obs_t = np.array([T0])
    obs_strikes = np.array([K0])
    ns_params = (r0, 0.0, 0.0, 1.5)  # 평평한 커브(r0)
    mc, elapsed = simulate_worst_of_autocall(sig, corr, r0, ns_params, T0, obs_t, obs_strikes,
                                              B=np.nan, coupon=coupon0, n_paths=200_000, seed=1)
    d2 = (np.log(1.0 / K0) + (r0 - 0.5 * sig0 ** 2) * T0) / (sig0 * np.sqrt(T0))
    bs = np.exp(-r0 * T0) * (1.0 + coupon0 * norm.cdf(d2))
    rel_err = abs(mc - bs) / bs
    log.info(f"[BS 검증] MC={mc:.6f} BS={bs:.6f} rel_err={rel_err:.4%} ({elapsed:.2f}s)")
    assert rel_err < 0.01, "MC 엔진이 BS 닫힌해와 1% 이상 차이 - 로직 점검 필요"
    return mc, bs


def _load_full_schedule() -> dict:
    """SCHD_INFO에서 SCHD_TYPE==1 전체(12 패딩 없이) 관측일/스트라이크 스케줄 로드."""
    sc = pd.read_csv(C.SCHD_INFO_CSV, encoding="utf-8", low_memory=False)
    t1 = sc[sc["SCHD_TYPE"] == 1].sort_values(["ITEM_CD", "SEQ"])
    t1 = t1[["ITEM_CD", "SEQ", "EXER_DT", "STRK_1"]].copy()
    t1["EXER_DT"] = pd.to_datetime(t1["EXER_DT"], format="mixed")
    out = {}
    for item, g in t1.groupby("ITEM_CD"):
        strikes = g["STRK_1"].to_numpy(dtype=float) / 100.0
        if np.isnan(strikes).any():
            # 원본 SCHD_INFO 일부 SEQ STRK_1 결측(데이터 품질) -> 인접 관측치로 보간
            strikes = pd.Series(strikes).ffill().bfill().to_numpy()
        out[item] = (g["EXER_DT"].to_numpy(), strikes)
    return out


def _worker(args):
    (item, sig, corr, r_flat, ns_params, tenor, obs_t, obs_strikes, B, coupon,
     n_paths, seed) = args
    try:
        mc, elapsed = simulate_worst_of_autocall(sig, corr, r_flat, ns_params, tenor, obs_t,
                                                  obs_strikes, B, coupon, n_paths, seed)
        return item, mc, elapsed, None
    except Exception as e:  # noqa: BLE001
        return item, np.nan, 0.0, str(e)


def build_product_mc(n_paths=C.MC_N_PATHS, jobs=4, limit=None, checkpoint_every=2000,
                      out_name="product_mc.parquet"):
    log.info("BS 단순케이스 엔진 검증 중...")
    _bs_digital_validate()

    market = pd.read_parquet(C.CACHE_DIR / "product_market.parquet")
    if limit:
        market = market.iloc[:limit].copy()
    log.info(f"product_market 로드: {len(market)} rows, n_paths={n_paths}, jobs={jobs}")

    schedules = _load_full_schedule()
    isu_dt_map = dict(zip(market["item"], market["ISU_DT"]))

    tasks = []
    skipped_no_sched = 0
    for _, row in market.iterrows():
        item = row["item"]
        if item not in schedules:
            skipped_no_sched += 1
            continue
        exer_dt, strikes = schedules[item]
        isu_dt = isu_dt_map[item]
        obs_t = np.array([(pd.Timestamp(d) - isu_dt).days / 365.25 for d in exer_dt])
        valid = obs_t > 0
        obs_t, strikes = obs_t[valid], strikes[valid]
        if len(obs_t) == 0:
            skipped_no_sched += 1
            continue

        sig = np.array([row["sig1"], row["sig2"], row["sig3"]])
        corr = np.array([[1.0, row["rho12"], row["rho13"]],
                          [row["rho12"], 1.0, row["rho23"]],
                          [row["rho13"], row["rho23"], 1.0]])
        # 수치 안정성: 상관행렬이 PSD가 아니면 근사 보정
        eigval, eigvec = np.linalg.eigh(corr)
        if (eigval < -1e-8).any():
            eigval = np.clip(eigval, 1e-6, None)
            corr = eigvec @ np.diag(eigval) @ eigvec.T
            d = np.sqrt(np.diag(corr))
            corr = corr / np.outer(d, d)

        ns_params = (row["ns_b0"], row["ns_b1"], row["ns_b2"], row["ns_lam"])
        seed = (C.MC_SEED + abs(hash(item))) % (2 ** 32 - 1)
        tasks.append((item, sig, corr, float(row["r"]), ns_params, float(row["tenor"]),
                      obs_t, strikes, float(row["B"]), float(row["coupon"]), n_paths, seed))

    log.info(f"스케줄 없음으로 스킵: {skipped_no_sched}, MC 실행 대상: {len(tasks)}")

    results = []
    t_start = time.time()
    with Pool(processes=jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, tasks, chunksize=4)):
            results.append(res)
            if (i + 1) % 500 == 0 or (i + 1) == len(tasks):
                elapsed_total = time.time() - t_start
                avg_ms = np.mean([r[2] for r in results[-500:]]) * 1000
                eta_min = (len(tasks) - (i + 1)) * (elapsed_total / (i + 1)) / 60
                log.info(f"[{i+1}/{len(tasks)}] 상품당 평균 {avg_ms:.1f}ms, "
                          f"경과 {elapsed_total/60:.1f}min, ETA {eta_min:.1f}min")
            if (i + 1) % checkpoint_every == 0:
                _save_checkpoint(results, market, out_name)

    _save_checkpoint(results, market, out_name, final=True)
    return results


def _save_checkpoint(results, market, out_name, final=False):
    res_df = pd.DataFrame(results, columns=["item", "MC", "mc_seconds", "mc_error"])
    n_err = res_df["mc_error"].notna().sum()
    merged = market.merge(res_df, on="item", how="inner")
    merged.to_parquet(C.CACHE_DIR / out_name, index=False)
    tag = "최종" if final else "체크포인트"
    log.info(f"[{tag}] {len(merged)} rows 저장 -> {C.CACHE_DIR / out_name} (에러 {n_err}건)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=C.MC_N_PATHS)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default="product_mc.parquet")
    args = ap.parse_args()
    build_product_mc(n_paths=args.n_paths, jobs=args.jobs, limit=args.limit, out_name=args.out)
