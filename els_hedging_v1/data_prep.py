"""1. 전처리: raw 3 CSV -> product_master.parquet"""
import logging
import numpy as np
import pandas as pd

from . import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("data_prep")


def _load_raw():
    ac = pd.read_csv(C.AUTO_CALL_CSV, encoding="utf-8", low_memory=False)
    sc = pd.read_csv(C.SCHD_INFO_CSV, encoding="utf-8", low_memory=False)
    ud = pd.read_csv(C.UDLY_INFO_CSV, encoding="utf-8", low_memory=False)
    return ac, sc, ud


def _schedule_vectors(sc: pd.DataFrame) -> pd.DataFrame:
    """SCHD_TYPE==1 rows -> strk_0..11 (/100, 12패딩 ffill), B(=BARR_1 최소/100), nobs."""
    t1 = sc[sc["SCHD_TYPE"] == 1].sort_values(["ITEM_CD", "SEQ"])

    def build(g: pd.DataFrame):
        strikes = (g["STRK_1"].to_numpy(dtype=float) / 100.0)
        nobs = len(strikes)
        if np.isnan(strikes).any():
            # 원본 SCHD_INFO의 일부 SEQ에 STRK_1 결측(데이터 품질 이슈) -> 같은 상품 내
            # 인접 관측치로 ffill/bfill (극소수: 스텝다운 스케줄이라 인접값이 최선의 근사)
            s = pd.Series(strikes)
            strikes = s.ffill().bfill().to_numpy()
        vec = np.full(C.N_STRK_PAD, np.nan)
        n = min(nobs, C.N_STRK_PAD)
        vec[:n] = strikes[:n]
        if n < C.N_STRK_PAD and n > 0:
            vec[n:] = vec[n - 1]
        barr_vals = g["BARR_1"].dropna()
        b = barr_vals.min() / 100.0 if len(barr_vals) else np.nan
        strk1_first = g["STRK_1"].iloc[0]
        strk1_last = g["STRK_1"].iloc[-1]

        # 실제 상환평가일(EXER_DT) - MC 시뮬레이션에 필요 (관측일 기준 정확 시뮬)
        exer = pd.to_datetime(g["EXER_DT"], format="mixed").to_numpy()
        obsdt = np.full(C.N_STRK_PAD, np.datetime64("NaT"), dtype="datetime64[ns]")
        obsdt[:n] = exer[:n]
        if n < C.N_STRK_PAD and n > 0:
            obsdt[n:] = obsdt[n - 1]

        out = {f"strk_{i}": vec[i] for i in range(C.N_STRK_PAD)}
        out.update({f"obsdt_{i}": obsdt[i] for i in range(C.N_STRK_PAD)})
        out.update(dict(nobs=nobs, B=b, STRK_1_first=strk1_first, STRK_1_last=strk1_last))
        return pd.Series(out)

    return t1.groupby("ITEM_CD").apply(build, include_groups=False).reset_index()


def _ticker_map_check(ud: pd.DataFrame) -> pd.DataFrame:
    """3자산 전부 티커 매핑 가능한 item만 표시 (mapped_all bool) + tickers 리스트."""
    ud = ud.copy()
    ud["ticker"] = ud["UDLY_NM"].map(C.TICKER_MAP)

    def build(g: pd.DataFrame):
        tickers = g["ticker"].tolist()
        names = g["UDLY_NM"].tolist()
        mapped_all = all(pd.notna(t) for t in tickers)
        return pd.Series({
            "tickers": tickers if mapped_all else None,
            "udly_names": names,
            "mapped_all": mapped_all,
        })

    return ud.groupby("ITEM_CD").apply(build, include_groups=False).reset_index()


def build_product_master() -> pd.DataFrame:
    ac, sc, ud = _load_raw()
    n0 = len(ac)
    log.info(f"raw AUTO_CALL rows = {n0}")

    df = ac.copy()

    # STCK_MTHD (worst-of) : SCHD_INFO 기준, item당 전 row가 ALL_MIN 이어야 통과
    stck = sc.groupby("ITEM_CD")["STCK_MTHD"].apply(lambda s: (s == "ALL_MIN").all())
    df = df.merge(stck.rename("all_min"), on="ITEM_CD", how="left")

    mask = (
        (df["PRODUCT_TYPE"] == "ELS")
        & (df["all_min"] == True)  # noqa: E712
        & (df["UDRL_CNT"] == 3)
        & (df["OPT_TYPE"] == "STEP")
        & (df["CUR_CD"] == "KRW")
    )
    log.info(f"after ELS/worst-of/3star/STEP/KRW filter: {mask.sum()} (dropped {n0 - mask.sum()})")
    df = df[mask].copy()

    # KNCK_IN_YN==1: "3-star ELS"의 canonical 형태(낙인배리어 보유 step-down worst-of KI 오토콜)만 채택.
    # 팀 목표 23,151건 대비 진단 결과 KNCK_IN_YN==1 필터가 23,580건(+1.8%)으로 가장 근접했고,
    # KNCK_IN_YN==0(24,447건)은 BARR_1이 전부 결측 -> B(낙인배리어) 피처/§3 MC의 KI 로직 자체가
    # 성립하지 않는 별도 상품군이므로 제외. (KNCK_IN_YN==0 포함시 48,027건, B 결측 51%)
    n_before_ki = len(df)
    df = df[df["KNCK_IN_YN"] == 1].copy()
    log.info(f"after KNCK_IN_YN==1 필터(팀 노션 3-star KI ELS 정의 정렬): {len(df)} "
              f"(dropped {n_before_ki - len(df)}, 팀 목표 23,151과 차이 = {len(df) - 23151})")

    df["ISU_DT"] = pd.to_datetime(df["ISU_DT"], format="mixed")
    df["MAT_DT"] = pd.to_datetime(df["MAT_DT"], format="mixed")
    df["SUB_START_DT"] = pd.to_datetime(df["SUB_START_DT"], format="mixed")
    df["SUB_END_DT"] = pd.to_datetime(df["SUB_END_DT"], format="mixed")

    df["fair"] = df["FAIR_VALUE"] / df["ISU_PRC_DETAIL"]
    df["tenor"] = (df["MAT_DT"] - df["ISU_DT"]).dt.days / 365.25

    n_before_range = len(df)
    df = df[df["fair"].between(*C.FAIR_RANGE) & df["tenor"].between(*C.TENOR_RANGE)]
    log.info(f"after fair{C.FAIR_RANGE}/tenor{C.TENOR_RANGE} range filter: {len(df)} "
              f"(dropped {n_before_range - len(df)})")

    # 티커 매핑 가능 여부
    tick = _ticker_map_check(ud)
    n_before_tick = len(df)
    df = df.merge(tick, on="ITEM_CD", how="left")
    unmapped = df[df["mapped_all"] != True]  # noqa: E712
    if len(unmapped):
        bad_names = pd.Series([n for row in unmapped["udly_names"].dropna() for n in row
                                if n not in C.TICKER_MAP]).value_counts()
        log.info(f"티커 미매핑으로 제외: {len(unmapped)}건. 상위 미매핑 기초자산:\n{bad_names.head(20)}")
    df = df[df["mapped_all"] == True].copy()  # noqa: E712
    log.info(f"after ticker-mappable filter: {len(df)} (dropped {n_before_tick - len(df)})")

    # 스케줄 벡터 병합
    sv = _schedule_vectors(sc)
    n_before_sv = len(df)
    df = df.merge(sv, on="ITEM_CD", how="inner")
    log.info(f"after schedule-vector merge: {len(df)} (dropped {n_before_sv - len(df)})")

    # 파생 피처
    df["coupon"] = df["ANL_RTRN"] / 100.0
    df["Kfirst"] = df["strk_0"]
    df["K"] = df["strk_11"]
    df["b_over_k"] = df["B"] / df["K"]
    df["stepdown"] = df["Kfirst"] - df["K"]
    df["sub_days"] = (df["SUB_END_DT"] - df["SUB_START_DT"]).dt.days
    df["ISU_DT_year"] = df["ISU_DT"].dt.year
    df["ISU_DT_month"] = df["ISU_DT"].dt.month

    # 관측일 시간오프셋(년, ISU_DT 기준) - MC 엔진에서 정확한 관측일 시뮬레이션에 사용
    obst_cols = []
    for i in range(C.N_STRK_PAD):
        col = f"obs_t_{i}"
        df[col] = (pd.to_datetime(df[f"obsdt_{i}"]) - df["ISU_DT"]).dt.days / 365.25
        obst_cols.append(col)

    # 키
    df["item"] = df["ITEM_CD"]
    df["isu_ord"] = df["ISU_DT"].map(lambda d: d.toordinal())
    df["issuer"] = df["ISU_ORG"]

    keep_cols = [
        "item", "isu_ord", "issuer", "ITEM_CD", "ISU_ORG", "ISU_DT", "MAT_DT",
        "SUB_START_DT", "SUB_END_DT", "tickers", "udly_names",
        "fair", "tenor", "coupon", "Kfirst", "K", "B", "b_over_k", "stepdown",
        "nobs", "sub_days", "ISU_DT_year", "ISU_DT_month",
        "STRK_1_first", "STRK_1_last", "CPN_YN",
        "ACT_ISU_AMT", "SB_RT", "DV_RT", "PRCP_GRTE_RT", "KNCK_IN_GRC_PRD",
        "RISK_GRADE", "PRODUCT_TYPE", "RDMP_TYPE",
    ] + C.STRK_COLS + obst_cols
    out = df[keep_cols].reset_index(drop=True)

    log.info(f"최종 product_master 건수 = {len(out)} (팀 목표 23,151과 차이 = {len(out) - 23151})")
    log.info(f"진단용 CPN_YN 분포(쿠폰유형 필터가 원인일 가능성): "
              f"{out['CPN_YN'].value_counts().to_dict()}")
    out.to_parquet(C.CACHE_DIR / "product_master.parquet", index=False)
    log.info(f"saved -> {C.CACHE_DIR / 'product_master.parquet'}")
    return out


if __name__ == "__main__":
    build_product_master()
