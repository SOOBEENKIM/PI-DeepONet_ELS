"""팀원과 동일 구성의 그림 4종:
1. 속도(MC vs DeepONet, 로그축) + price agreement scatter + price diff 히스토그램
2. fair-MC 히스토그램 (KRW)
3. per-stage R2 (손실함수 변형별 Stage-1 + Stage1/2/Final)
4. 저가 케이스 분석 (세그먼트별 MAPE + 드라이버 상관)

선행 스크립트(speed_benchmark/deeponet --all-losses/metrics/case_analysis) 산출물을 읽어서 그린다.
"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("figures")

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


def fig_speed():
    summary_f = C.OUT_DIR / "speed_benchmark_summary.csv"
    detail_f = C.OUT_DIR / "speed_benchmark_detail.parquet"
    if not summary_f.exists():
        log.warning("speed_benchmark 산출물 없음 - speed_benchmark.py 먼저 실행 필요. 스킵.")
        return
    summary = pd.read_csv(summary_f).iloc[0]
    detail = pd.read_parquet(detail_f)
    mc_total = summary["mc_total_sec(소표본환산)"]
    don_total = summary["deeponet_total_sec(배치forward)"]
    speedup = summary["speedup_x(소표본환산기준)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    bars = ax.bar(["MC (100k경로,\n소표본n=200 환산)", f"DeepONet\n(배치forward,{summary['loss_type']}손실)"],
                   [mc_total, don_total], color=["#c0392b", "#2980b9"])
    ax.set_yscale("log")
    ax.set_ylabel("총 소요시간(초, log)")
    ax.set_title(f"OOS 전체 pricing 총시간\n({int(summary['n_products'])}건, {speedup:,.0f}배)")
    for b, v in zip(bars, [mc_total, don_total]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}s", ha="center", va="bottom")

    ax = axes[1]
    ax.scatter(detail["true_mc"], detail["pred_mc"], s=4, alpha=0.3, color="#2980b9")
    lo, hi = detail["true_mc"].min(), detail["true_mc"].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("MC (실측)")
    ax.set_ylabel("DeepONet 예측")
    ax.set_title(f"Price agreement (corr={summary['price_corr']:.4f})")

    ax = axes[2]
    diff = detail["pred_mc"] - detail["true_mc"]
    ax.hist(diff, bins=60, color="#7f8c8d")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("DeepONet - MC")
    ax.set_title(f"Price diff 분포 (MAE={summary['price_mae']:.5f})")

    _save(fig, "01_speed_and_agreement.png")


def fig_fair_minus_mc():
    from .case_analysis import _load_isu_prc
    ml = pd.read_parquet(C.CACHE_DIR / "dataset_ml.parquet")
    isu_prc = _load_isu_prc()
    df = ml.merge(isu_prc, on="item", how="left")
    err_krw = (df["fair"] - df["MC"]) * df["ISU_PRC_DETAIL"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(err_krw, bins=80, color="#8e44ad", alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(err_krw.mean(), color="red", lw=1.5, ls="--",
               label=f"평균={err_krw.mean():.0f} KRW")
    ax.set_xlabel("fair - MC (KRW, 액면가 환산)")
    ax.set_ylabel("빈도")
    ax.set_title("fair - MC 분포 (MC > fair 경향 = IV 미사용)")
    ax.legend()
    _save(fig, "02_fair_minus_mc_histogram.png")


def fig_per_stage_r2():
    loss_f = C.OUT_DIR / "stage1_loss_comparison.csv"
    final_f = C.OUT_DIR / "final_comparison_table.csv"

    n_panels = int(loss_f.exists()) + int(final_f.exists())
    if n_panels == 0:
        log.warning("per-stage R2 산출물 없음 - deeponet --all-losses / metrics.py 먼저 실행. 스킵.")
        return
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    ai = 0

    if loss_f.exists():
        loss_df = pd.read_csv(loss_f)
        ax = axes[ai]; ai += 1
        ax.bar(loss_df["loss_type"], loss_df["r2"], color=["#2980b9", "#27ae60", "#e67e22"])
        ax.axhspan(0.92, 0.99, color="green", alpha=0.1, label="팀목표(0.92~0.99)")
        ax.set_ylabel("Stage-1 R2 (MC 재현)")
        ax.set_title("손실함수별 Stage-1 DeepONet R2")
        ax.legend()

    if final_f.exists():
        table = pd.read_csv(final_f)
        ax = axes[ai]; ai += 1
        order = ["stage1_deeponet", "stage1_xgb", "stage2_margin_mlp", "stage2_resid",
                  "final_deeponet_hybrid", "bench_ridge", "bench_gbm", "deeponet_direct"]
        table = table[table["model"].isin(order)].set_index("model").reindex(order).dropna(how="all")
        colors = ["#2980b9" if "stage1" in m else "#27ae60" if "stage2" in m else
                  "#e67e22" if "final" in m else "#95a5a6" for m in table.index]
        ax.bar(table.index, table["R2"], color=colors)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticklabels(table.index, rotation=40, ha="right")
        ax.set_ylabel("R2")
        ax.set_title("Stage1 / Stage2 / Final / Direct 벤치 R2")

    _save(fig, "03_per_stage_r2.png")


def fig_low_price_case():
    seg_f = C.OUT_DIR / "case_fair_segment_report.csv"
    drv_f = C.OUT_DIR / "case_driver_correlations.csv"
    if not seg_f.exists():
        log.warning("case_analysis 산출물 없음 - case_analysis.py 먼저 실행 필요. 스킵.")
        return
    seg = pd.read_csv(seg_f)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.bar(range(len(seg)), seg["MAPE"] * 100, color="#c0392b")
    ax.set_xticks(range(len(seg)))
    ax.set_xticklabels([f"{m:.3f}" for m in seg["seg_mid"]], rotation=45, ha="right")
    ax.set_xlabel("공정가(fair) 세그먼트 중앙값")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("공정가 세그먼트별 MAPE (저가일수록 오차 증가)")

    if drv_f.exists():
        drv = pd.read_csv(drv_f)
        ax = axes[1]
        ax.barh(drv["driver"], drv["corr_with_ape"], color="#16a085")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("corr(|%error|, driver)")
        ax.set_title("오차 드라이버 상관계수")
    _save(fig, "04_low_price_case_analysis.png")


def fig_signed_bias(n_bins=10):
    """§V1_발표보완_지시서 1: 부호 있는 편향(over/underprice). 교수님 Q2 대응.
    저가구간 양수(과대평가)->고가구간 음수(과소평가) 패턴 확인용."""
    oos_f = C.OUT_DIR / "oos_predictions.parquet"
    if not oos_f.exists():
        log.warning("oos_predictions.parquet 없음 - metrics.py 먼저 실행 필요. 스킵.")
        return
    oos = pd.read_parquet(oos_f)

    def _signed_bias_by_segment(true_vals, pred_vals, seg_vals, n_bins):
        df = pd.DataFrame({"true": true_vals, "pred": pred_vals, "seg": seg_vals}).dropna()
        bins = pd.qcut(df["seg"], n_bins, duplicates="drop")
        g = df.groupby(bins, observed=True).apply(
            lambda d: pd.Series({"mid": d["seg"].mean(), "bias": (d["pred"] - d["true"]).mean()}),
            include_groups=False)
        return g.sort_values("mid")

    final_bias = _signed_bias_by_segment(oos["fair_true"], oos["final_deeponet_hybrid"], oos["fair_true"], n_bins)
    mc_bias = _signed_bias_by_segment(oos["fair_true"], oos["mc_true"], oos["fair_true"], n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = ["#c0392b" if b > 0 else "#2980b9" for b in final_bias["bias"]]
    ax.bar(range(len(final_bias)), final_bias["bias"], color=colors)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(final_bias)))
    ax.set_xticklabels([f"{m:.3f}" for m in final_bias["mid"]], rotation=45, ha="right")
    ax.set_xlabel("공정가(fair) 세그먼트 중앙값")
    ax.set_ylabel("mean(예측 - 실제)")
    ax.set_title("Final 모델: 공정가 구간별 편향\n(저가 과대평가/양수 -> 고가 과소평가/음수)")

    ax = axes[1]
    colors2 = ["#c0392b" if b > 0 else "#2980b9" for b in mc_bias["bias"]]
    ax.bar(range(len(mc_bias)), mc_bias["bias"], color=colors2)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(mc_bias)))
    ax.set_xticklabels([f"{m:.3f}" for m in mc_bias["mid"]], rotation=45, ha="right")
    ax.set_xlabel("공정가(fair) 세그먼트 중앙값")
    ax.set_ylabel("mean(MC - fair)")
    ax.set_title("MC 자체의 편향\n(전 구간 MC>fair 과대평가, 저가일수록 더 큼)")

    _save(fig, "05_signed_bias_by_segment.png")


def fig_pred_vs_actual_scatter():
    """§V1_발표보완_지시서 2: predicted vs actual FAIR 산점도, 모델별 서브플롯."""
    oos_f = C.OUT_DIR / "oos_predictions.parquet"
    if not oos_f.exists():
        log.warning("oos_predictions.parquet 없음 - metrics.py 먼저 실행 필요. 스킵.")
        return
    oos = pd.read_parquet(oos_f)

    models = [("final_deeponet_hybrid", "Final (DeepONet hybrid)"),
              ("final_xgb_hybrid", "Final (XGB hybrid)"),
              ("deeponet_direct", "DeepONet direct")]
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.5))
    lo, hi = oos["fair_true"].min(), oos["fair_true"].max()

    for ax, (col, title) in zip(axes, models):
        seg = oos["fair_true"]
        sc = ax.scatter(oos["fair_true"], oos[col], c=seg, cmap="viridis", s=5, alpha=0.4)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="y=x (이상)")
        r2 = 1 - np.sum((oos[col] - oos["fair_true"]) ** 2) / np.sum((oos["fair_true"] - oos["fair_true"].mean()) ** 2)
        ax.set_xlabel("실제 fair")
        ax.set_ylabel("예측")
        ax.set_title(f"{title}\nR2={r2:.3f} (왼쪽상단에 가까울수록 좋음)")
        ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=axes[-1], label="fair(세그먼트)")

    _save(fig, "06_pred_vs_actual_fair_scatter.png")


def make_all_figures():
    fig_speed()
    fig_fair_minus_mc()
    fig_per_stage_r2()
    fig_low_price_case()
    fig_signed_bias()
    fig_pred_vs_actual_scatter()
    log.info(f"전체 그림 저장 완료 -> {FIG_DIR}")


if __name__ == "__main__":
    make_all_figures()
