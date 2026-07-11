"""저가 케이스 분석 (교수님 최신 요청):
- 공정가(fair) 세그먼트별 오차(mean fair-MC, MAPE) - "저가일수록 오차 증가" 정량화
- |오차| vs (coupon, sig_eff, sig3, b_over_k, tenor, cpn_spread) 상관 (드라이버 탐색)
- 극단 케이스 표

IV(내재변동성) 미보유로 세그먼트별 vol 스케일링 보정은 보류 - 패턴 제시까지만.
"""
import logging

import numpy as np
import pandas as pd

from . import config as C
from .metrics import moneyness_segment_report, calibration_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("case_analysis")

DRIVER_COLS = ["coupon", "sig_eff", "sig3", "b_over_k", "tenor", "cpn_spread"]


def _load_isu_prc():
    """product_features/mc에는 없는 ISU_PRC_DETAIL(액면가, KRW)을 원본에서 읽어 KRW 환산에 사용.
    MC 재실행/재계산 아님 - 순수 조회용 조인."""
    ac = pd.read_csv(C.AUTO_CALL_CSV, encoding="utf-8", low_memory=False)[["ITEM_CD", "ISU_PRC_DETAIL"]]
    return ac.rename(columns={"ITEM_CD": "item"})


def run_case_analysis(n_bins=10, n_extreme=20):
    ml = pd.read_parquet(C.CACHE_DIR / "dataset_ml.parquet")
    isu_prc = _load_isu_prc()
    df = ml.merge(isu_prc, on="item", how="left")

    df["err_krw"] = (df["fair"] - df["MC"]) * df["ISU_PRC_DETAIL"]
    df["abs_err"] = (df["fair"] - df["MC"]).abs()
    df["ape"] = df["abs_err"] / df["fair"].clip(lower=1e-6)

    log.info(f"fair-MC(KRW) 평균={df['err_krw'].mean():.1f}, 중앙값={df['err_krw'].median():.1f} "
              f"(참고: 팀 궤적 평균 약 -512 KRW, MC>FAIR 방향 - IV 미사용이 원인)")

    # ---- 1. 공정가 세그먼트별 오차 ----
    seg = moneyness_segment_report(df["fair"], df["MC"], df["fair"], n_bins=n_bins)
    log.info("\n공정가(fair) 세그먼트별 오차 (저가일수록 오차 증가 패턴 확인용):\n" + seg.to_string(index=False))
    seg.to_csv(C.OUT_DIR / "case_fair_segment_report.csv", index=False, encoding="utf-8-sig")

    calib = calibration_stats(df["fair"], df["MC"])
    log.info(f"fair vs MC calibration: slope={calib['slope']:.4f} intercept={calib['intercept']:.4f} "
              f"(slope<1 -> 저가 과대추정/고가 과소추정 = 평균회귀 경향)")

    # ---- 2. 드라이버 상관 ----
    driver_rows = []
    for c in DRIVER_COLS:
        if c not in df.columns:
            continue
        sub = df[[c, "abs_err", "ape"]].dropna()
        driver_rows.append({
            "driver": c,
            "corr_with_abs_err": float(sub[c].corr(sub["abs_err"])),
            "corr_with_ape": float(sub[c].corr(sub["ape"])),
        })
    driver_df = pd.DataFrame(driver_rows).sort_values("corr_with_ape", key=lambda s: s.abs(), ascending=False)
    log.info("\n|오차| 드라이버 상관계수 (내림차순):\n" + driver_df.to_string(index=False))
    driver_df.to_csv(C.OUT_DIR / "case_driver_correlations.csv", index=False, encoding="utf-8-sig")

    # ---- 3. 극단 케이스 표 ----
    extreme_cols = ["item", "fair", "MC", "err_krw", "ape"] + DRIVER_COLS
    extreme_cols = [c for c in extreme_cols if c in df.columns]
    extreme = df.reindex(df["ape"].sort_values(ascending=False).index)[extreme_cols].head(n_extreme)
    log.info(f"\n극단 케이스 상위 {n_extreme}건 (MAPE 기준):\n" + extreme.to_string(index=False))
    extreme.to_csv(C.OUT_DIR / "case_extreme_examples.csv", index=False, encoding="utf-8-sig")

    log.info(f"저장 완료 -> {C.OUT_DIR}/case_fair_segment_report.csv, case_driver_correlations.csv, "
              f"case_extreme_examples.csv")
    return seg, driver_df, extreme


if __name__ == "__main__":
    run_case_analysis()
