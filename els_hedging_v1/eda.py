"""발표보완 §4: 필터 cascade 표 + EDA 그림 (연도별 발행건수/구조유형/공정가·만기·쿠폰 분포/기초자산 top15).

MC 재실행 없음 - product_master/product_features(캐시)와 원본 CSV(조회만)만 사용.
"""
import logging
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eda")

FIG_DIR = C.OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def _save(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    log.info(f"saved -> {path}")


def _save_table_image(df, name, title=None, figsize=None, colwidths=None):
    figsize = figsize or (min(1.1 * len(df.columns) + 1, 16), 0.42 * (len(df) + 1) + 0.6)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, pad=14)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center",
                    colWidths=colwidths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", weight="bold")
    _save(fig, name)


def build_filter_cascade(data_prep_log="data_prep_run.log", market_data_log="market_data_run2.log"):
    """data_prep.py/market_data.py 재실행 로그를 파싱해 단계별 탈락 cascade 표 생성 (MC 재실행 아님)."""
    dp = open(C.ROOT / data_prep_log, encoding="utf-8", errors="replace").read()
    md = open(C.ROOT / market_data_log, encoding="utf-8", errors="replace").read()

    def grab(pattern, text):
        m = re.search(pattern, text)
        return int(m.group(1).replace(",", "")) if m else None

    raw_n = grab(r"raw AUTO_CALL rows = (\d+)", dp)
    after_base = grab(r"after ELS/worst-of/3star/STEP/KRW filter: (\d+)", dp)
    after_ki = grab(r"KNCK_IN_YN==1.*?: (\d+)", dp)
    after_range = grab(r"after fair.*?range filter: (\d+)", dp)
    after_ticker = grab(r"after ticker-mappable filter: (\d+)", dp)
    after_sched = grab(r"after schedule-vector merge: (\d+)", dp)
    after_mkt = grab(r"시장데이터 결측 제거: \d+ -> (\d+)", md)

    rows = [
        ("0. 원본 DART 공시(AUTO_CALL)", raw_n, None),
        ("1. ELS·worst-of(ALL_MIN)·3기초자산·STEP·KRW", after_base, raw_n - after_base),
        ("2. 낙인배리어 보유(KNCK_IN_YN==1, 3-star KI ELS 정의)", after_ki, after_base - after_ki),
        ("3. fair[0.70,1.05] & tenor[0.5,5]년 범위", after_range, after_ki - after_range),
        ("4. 3기초자산 전부 티커 매핑 가능", after_ticker, after_range - after_ticker),
        ("5. 상환평가 스케줄 존재", after_sched, after_ticker - after_sched),
        ("6. 발행 전 180영업일 시장데이터(vol/corr) 확보", after_mkt, after_sched - after_mkt),
    ]
    df = pd.DataFrame(rows, columns=["단계", "건수", "탈락"])
    df["탈락"] = df["탈락"].fillna(0).astype(int)
    log.info("\n필터 cascade:\n" + df.to_string(index=False))
    df.to_csv(C.OUT_DIR / "filter_cascade.csv", index=False, encoding="utf-8-sig")
    _save_table_image(df, "eda_00_filter_cascade.png", title="필터 Cascade (원본 -> 최종 23,479건)")
    return df


def _load_step_lizard_at_ki_stage():
    """KI필터 직후(=STEP 제한 이전) 모집단에서 STEP/LIZARD 구조유형 분포 (참고용, 최종셋은 전량 STEP)."""
    ac = pd.read_csv(C.AUTO_CALL_CSV, encoding="utf-8", low_memory=False)
    sc = pd.read_csv(C.SCHD_INFO_CSV, encoding="utf-8", low_memory=False)
    stck = sc.groupby("ITEM_CD")["STCK_MTHD"].apply(lambda s: (s == "ALL_MIN").all())
    ac = ac.merge(stck.rename("all_min"), on="ITEM_CD", how="left")
    mask = ((ac["PRODUCT_TYPE"] == "ELS") & (ac["all_min"] == True) & (ac["UDRL_CNT"] == 3)  # noqa: E712
            & (ac["CUR_CD"] == "KRW") & (ac["KNCK_IN_YN"] == 1))
    return ac.loc[mask, "OPT_TYPE"].value_counts()


def build_eda_figures():
    pf = pd.read_parquet(C.CACHE_DIR / "product_features.parquet")
    pf["year"] = pf["ISU_DT_year"] if "ISU_DT_year" in pf.columns else pd.to_datetime(pf["ISU_DT"]).dt.year

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    yc = pf["year"].value_counts().sort_index()
    ax.bar(yc.index.astype(str), yc.values, color="#2980b9")
    ax.set_title("연도별 발행건수 (최종 필터셋)")
    ax.tick_params(axis="x", rotation=45)

    ax = axes[0, 1]
    step_liz = _load_step_lizard_at_ki_stage()
    ax.bar(step_liz.index, step_liz.values, color=["#27ae60", "#e67e22"])
    ax.set_title("구조유형 분포 (KI보유 모집단, STEP필터 이전)\n※ 최종셋은 스펙상 전량 STEP")
    for i, v in enumerate(step_liz.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")

    ax = axes[0, 2]
    ax.hist(pf["fair"], bins=60, color="#8e44ad")
    ax.axvline(pf["fair"].mean(), color="red", ls="--", lw=1.2, label=f"평균={pf['fair'].mean():.3f}")
    ax.set_title("공정가(fair) 분포")
    ax.legend()

    ax = axes[1, 0]
    ax.hist(pf["tenor"], bins=40, color="#16a085")
    ax.set_title("만기(tenor, 년) 분포")

    ax = axes[1, 1]
    ax.hist(pf["coupon"] * 100, bins=40, color="#c0392b")
    ax.set_title("쿠폰(연, %) 분포")

    ax = axes[1, 2]
    names = [n for row in pf["udly_names"] for n in row]
    top15 = pd.Series(names).value_counts().head(15).sort_values()
    ax.barh(top15.index, top15.values, color="#34495e")
    ax.set_title("기초자산 Top15 (출현빈도)")

    _save(fig, "eda_01_overview.png")


if __name__ == "__main__":
    build_filter_cascade()
    build_eda_figures()
