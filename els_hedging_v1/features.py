"""4. recent_margin(발행전 90일 fair-MC 인과평균) + REG 피처 -> product_features.parquet"""
import logging

import numpy as np
import pandas as pd

from . import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("features")

WINDOW_DAYS = 90


def _causal_window_stats(isu_ord: np.ndarray, values: np.ndarray, window_days: int = WINDOW_DAYS):
    """정렬된 isu_ord 기준, [isu_ord_i-window, isu_ord_i) 인과 구간 평균/카운트.
    이력 없으면(window 내 0건) 전역 확장 인과평균(모든 과거)으로 백필, 그마저 없으면 0.
    반환은 원래(미정렬) 순서.
    """
    order = np.argsort(isu_ord, kind="stable")
    ord_sorted = isu_ord[order]
    val_sorted = values[order]

    cumsum = np.concatenate([[0.0], np.cumsum(val_sorted)])
    n = len(ord_sorted)

    lo = np.searchsorted(ord_sorted, ord_sorted - window_days, side="left")
    hi = np.searchsorted(ord_sorted, ord_sorted, side="left")  # 자기 자신 포함 이전 첫 위치(동일일 제외)

    win_sum = cumsum[hi] - cumsum[lo]
    win_cnt = hi - lo

    global_sum = cumsum[hi]
    global_cnt = hi

    mean_sorted = np.where(win_cnt > 0, win_sum / np.maximum(win_cnt, 1),
                            np.where(global_cnt > 0, global_sum / np.maximum(global_cnt, 1), 0.0))
    cnt_sorted = win_cnt.astype(float)

    mean_out = np.empty(n)
    cnt_out = np.empty(n)
    mean_out[order] = mean_sorted
    cnt_out[order] = cnt_sorted
    return mean_out, cnt_out


def build_product_features():
    mc = pd.read_parquet(C.CACHE_DIR / "product_mc.parquet")
    mc = mc.sort_values("isu_ord").reset_index(drop=True)
    log.info(f"product_mc 로드: {len(mc)} rows")

    n_before = len(mc)
    mc = mc.dropna(subset=["MC"]).reset_index(drop=True)
    if len(mc) < n_before:
        log.warning(f"MC 계산 실패(mc_error) 상품 {n_before - len(mc)}건 제외")

    isu_ord = mc["isu_ord"].to_numpy()
    margin = (mc["fair"] - mc["MC"]).to_numpy()
    sig_mean = mc["sig_mean"].to_numpy()

    recent_margin, issue_intensity = _causal_window_stats(isu_ord, margin)
    recent_mktvol, _ = _causal_window_stats(isu_ord, sig_mean)

    mc["recent_margin"] = recent_margin
    mc["recent_mktvol"] = recent_mktvol
    mc["issue_intensity"] = issue_intensity

    u_cols = C.U_COLS
    mc["curve_level"] = mc[u_cols].mean(axis=1)
    mc["curve_slope"] = mc["u9"] - mc["u0"]
    mc["curve_curv"] = 2 * mc["u4"] - mc["u0"] - mc["u9"]

    log.info(f"recent_margin 통계: mean={mc['recent_margin'].mean():.5f}, "
              f"std={mc['recent_margin'].std():.5f}, 0-백필 건수={ (mc['issue_intensity']==0).sum() }")

    mc.to_parquet(C.CACHE_DIR / "product_features.parquet", index=False)
    log.info(f"saved -> {C.CACHE_DIR / 'product_features.parquet'} ({len(mc)} rows)")
    return mc


if __name__ == "__main__":
    build_product_features()
