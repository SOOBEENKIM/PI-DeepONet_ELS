"""7. 평가지표: R2/MAE/RMSE/Spearman + %error(MAPE·bp) + moneyness 세그먼트 + calibration.

피드백 헤드라인: R2만 보지 말 것 -> %error(현업 관점)로 "어이없는 값 안 나온다"를 보이는 프레임워크.
이 모듈의 main()이 walk-forward 전체 오케스트레이션(Stage1 DeepONet/Stage2 MLP/Final/direct벤치)을
수행하며 §V1_피드백반영_지시서의 재실행 순서 중 `python -m els_hedging_v1.metrics` 단계에 해당.
"""
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from . import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("metrics")


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return dict(r2=np.nan, mae=np.nan, rmse=np.nan, mape=np.nan, bp_error=np.nan,
                     bias_bp=np.nan, spearman=np.nan, n=len(y_true))
    err = y_pred - y_true
    ape = np.abs(err) / np.clip(np.abs(y_true), 1e-6, None)
    return dict(
        r2=r2_score(y_true, y_pred),
        mae=mean_absolute_error(y_true, y_pred),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mape=float(np.mean(ape)),
        bp_error=float(np.mean(np.abs(err)) * 10000),   # 평균 절대오차, bp(=1/10000) 단위
        bias_bp=float(np.mean(err) * 10000),             # 부호있는 편향, bp 단위
        spearman=spearmanr(y_true, y_pred).correlation,
        n=len(y_true),
    )


def summarize_folds(fold_metrics: list[dict]) -> dict:
    """폴드별 dict 리스트 -> n 가중평균."""
    df = pd.DataFrame(fold_metrics)
    w = df["n"].to_numpy()
    out = {}
    for c in ["r2", "mae", "rmse", "mape", "bp_error", "bias_bp", "spearman"]:
        if c in df.columns:
            out[c] = float(np.average(df[c], weights=w))
    out["n"] = int(w.sum())
    return out


def calibration_stats(y_true, y_pred) -> dict:
    """pred = intercept + slope*true 회귀. 이상적으로 slope=1, intercept=0.
    slope<1이면 저가에서 과대예측/고가에서 과소예측(평균회귀) 경향."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return dict(slope=np.nan, intercept=np.nan)
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    return dict(slope=float(slope), intercept=float(intercept))


def moneyness_segment_report(y_true, y_pred, segment_values, n_bins=5) -> pd.DataFrame:
    """segment_values(예: fair 또는 moneyness) 분위수 구간별 MAPE·bias·mean(true-pred).
    '저가일수록 오차 증가' 패턴 정량화용."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    seg = np.asarray(segment_values, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(seg)
    y_true, y_pred, seg = y_true[mask], y_pred[mask], seg[mask]

    try:
        bins = pd.qcut(seg, n_bins, duplicates="drop")
    except ValueError:
        bins = pd.cut(seg, n_bins)

    rows = []
    for b in sorted(bins.unique(), key=lambda x: x.left):
        m = bins == b
        if m.sum() == 0:
            continue
        yt, yp = y_true[m], y_pred[m]
        err = yp - yt
        ape = np.abs(err) / np.clip(np.abs(yt), 1e-6, None)
        rows.append({
            "segment": str(b),
            "seg_mid": (b.left + b.right) / 2,
            "n": int(m.sum()),
            "mean_true": float(yt.mean()),
            "mean_pred": float(yp.mean()),
            "mean_err(true-pred)": float(-err.mean()),
            "MAPE": float(ape.mean()),
            "bias_bp": float(err.mean() * 10000),
        })
    return pd.DataFrame(rows)


def _in_range(val, rng):
    lo, hi = rng
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if lo is None:
        return "OK" if val <= hi else "FAIL"
    if hi is None:
        return "OK" if val >= lo else "FAIL"
    return "OK" if lo <= val <= hi else "FAIL"


def build_comparison_table(results: dict) -> pd.DataFrame:
    """results: {row_name: {"r2":..,"spearman":..,...}} -> 팀 목표치 비교 DataFrame.
    이론가(Stage-1)를 최상단에 두고 %error(MAPE/bp) 열을 R2와 나란히 표시."""
    rows = []
    mapping = {
        "stage1_deeponet": ("stage1_deeponet_r2", "r2"),
        "stage1_xgb": ("stage1_xgb_r2", "r2"),
        "stage2_margin_mlp": ("stage2_resid_r2", "r2"),
        "stage2_resid": ("stage2_resid_r2", "r2"),
        "final_deeponet_hybrid": ("final_deeponet_hybrid_r2", "r2"),
        "bench_ridge": ("direct_ridge_r2", "r2"),
        "bench_gbm": ("direct_tree_r2", "r2"),
        "deeponet_direct": ("direct_deeponet_r2", "r2"),
    }
    order = ["stage1_deeponet", "stage1_xgb", "stage2_margin_mlp", "stage2_resid",
             "final_deeponet_hybrid", "final_xgb_hybrid", "bench_ridge", "bench_gbm", "deeponet_direct"]
    for name in order:
        if name not in results:
            continue
        m = results[name]
        target_key, metric_key = mapping.get(name, (None, "r2"))
        target_rng = C.TEAM_TARGETS.get(target_key) if target_key else None
        row = {
            "model": name,
            "R2": round(m.get("r2", np.nan), 4),
            "MAPE": round(m.get("mape", np.nan), 4) if np.isfinite(m.get("mape", np.nan)) else np.nan,
            "bp_error": round(m.get("bp_error", np.nan), 1),
            "MAE": round(m.get("mae", np.nan), 5),
            "RMSE": round(m.get("rmse", np.nan), 5),
            "Spearman": round(m.get("spearman", np.nan), 4),
            "n": m.get("n"),
            "팀목표": f"{target_rng}" if target_rng else "-",
            "판정": _in_range(m.get("r2", np.nan), target_rng) if target_rng else "-",
        }
        rows.append(row)
    for name, m in results.items():
        if name not in order:
            target_key, _ = mapping.get(name, (None, "r2"))
            target_rng = C.TEAM_TARGETS.get(target_key) if target_key else None
            rows.append({
                "model": name, "R2": round(m.get("r2", np.nan), 4),
                "MAPE": round(m.get("mape", np.nan), 4) if np.isfinite(m.get("mape", np.nan)) else np.nan,
                "bp_error": round(m.get("bp_error", np.nan), 1),
                "MAE": round(m.get("mae", np.nan), 5), "RMSE": round(m.get("rmse", np.nan), 5),
                "Spearman": round(m.get("spearman", np.nan), 4), "n": m.get("n"),
                "팀목표": f"{target_rng}" if target_rng else "-", "판정": "-",
            })
    return pd.DataFrame(rows)


def _load_merged():
    ml = pd.read_parquet(C.CACHE_DIR / "dataset_ml.parquet")
    don_df = pd.read_parquet(C.CACHE_DIR / "dataset_deeponet.parquet")
    overlap = [c for c in don_df.columns if c in ml.columns and c not in ("item", "isu_ord")]
    don_only = don_df.drop(columns=overlap)
    merged = ml.merge(don_only, on=["item", "isu_ord"], how="inner")
    from .splits import sort_chronological
    return sort_chronological(merged)


def _run_fold(train_df, test_df, fold_id, loss_type="mae"):
    """loss_type: DeepONet(Stage-1/direct) 대표 손실. §V1_발표보완_지시서 5 - MAE 최적본을 대표로 통일."""
    from . import benchmark as bm
    from . import deeponet as don
    from . import margin_mlp as mmlp

    out = {}

    # ---- Stage-1: DeepONet(이론가, 핵심) vs XGB(하이브리드 비교용) ----
    model_don, scaler_don = don.train_deeponet(train_df, target_col="MC", loss_type=loss_type)
    mc_hat_don = don.predict_deeponet(model_don, scaler_don, test_df)
    out["stage1_deeponet"] = compute_metrics(test_df["MC"], mc_hat_don)

    model_s1xgb, enc_s1xgb = bm.train_stage1_xgb(train_df)
    mc_hat_xgb = bm.predict_tabular(model_s1xgb, enc_s1xgb, test_df)
    out["stage1_xgb"] = compute_metrics(test_df["MC"], mc_hat_xgb)

    # ---- Stage-2: MLP(기본, 교수님 피드백) vs XGB(비교용) ----
    model_mlp, enc_mlp = mmlp.train_margin_mlp(train_df, target_col="resid")
    margin_hat_mlp = mmlp.predict_margin_mlp(model_mlp, enc_mlp, test_df)
    out["stage2_margin_mlp"] = compute_metrics(test_df["resid"], margin_hat_mlp)

    model_s2xgb, enc_s2xgb = bm.train_stage2_xgb(train_df)
    margin_hat_xgb = bm.predict_tabular(model_s2xgb, enc_s2xgb, test_df)
    out["stage2_resid"] = compute_metrics(test_df["resid"], margin_hat_xgb)

    # ---- Final 하이브리드 조립 (부차) ----
    final_don_hybrid = mc_hat_don + test_df["recent_margin"].to_numpy() + margin_hat_mlp
    out["final_deeponet_hybrid"] = compute_metrics(test_df["fair"], final_don_hybrid)

    final_xgb_hybrid = mc_hat_xgb + test_df["recent_margin"].to_numpy() + margin_hat_xgb
    out["final_xgb_hybrid"] = compute_metrics(test_df["fair"], final_xgb_hybrid)

    # ---- Direct 벤치 (원복된 최초 버전) ----
    model_ridge, enc_ridge = bm.train_bench_ridge(train_df)
    pred_ridge = bm.predict_tabular(model_ridge, enc_ridge, test_df)
    out["bench_ridge"] = compute_metrics(test_df["fair"], pred_ridge)

    model_bgxgb, enc_bgxgb = bm.train_bench_xgb(train_df)
    pred_bgxgb = bm.predict_tabular(model_bgxgb, enc_bgxgb, test_df)
    out["bench_gbm"] = compute_metrics(test_df["fair"], pred_bgxgb)

    model_don_direct, scaler_don_direct = don.train_deeponet(train_df, target_col="fair", loss_type=loss_type)
    pred_don_direct = don.predict_deeponet(model_don_direct, scaler_don_direct, test_df)
    out["deeponet_direct"] = compute_metrics(test_df["fair"], pred_don_direct)

    # 전 모델 원시 예측치 보존 (부호편향/산점도/agreement 그림용 - §V1_발표보완_지시서 1,2)
    raw = pd.DataFrame({
        "item": test_df["item"].to_numpy(), "isu_ord": test_df["isu_ord"].to_numpy(),
        "fold": fold_id,
        "mc_true": test_df["MC"].to_numpy(), "fair_true": test_df["fair"].to_numpy(),
        "recent_margin": test_df["recent_margin"].to_numpy(),
        "mc_hat_deeponet": mc_hat_don, "mc_hat_xgb": mc_hat_xgb,
        "margin_hat_mlp": margin_hat_mlp, "margin_hat_xgb": margin_hat_xgb,
        "final_deeponet_hybrid": final_don_hybrid, "final_xgb_hybrid": final_xgb_hybrid,
        "deeponet_direct": pred_don_direct, "bench_ridge": pred_ridge, "bench_gbm": pred_bgxgb,
    })

    for k, v in out.items():
        log.info(f"[fold {fold_id}] {k}: R2={v['r2']:.4f} MAPE={v.get('mape', float('nan')):.4%} "
                  f"bp_err={v.get('bp_error', float('nan')):.1f} Spearman={v['spearman']:.4f} n={v['n']}")
    return out, raw


def main(loss_type="mae"):
    df = _load_merged()
    log.info(f"모델링 데이터셋 로드: {len(df)} rows, OOS 시작 60% 지점부터 walk-forward 평가 "
              f"(Stage-2=MLP 기본, Stage-1=DeepONet[{loss_type}] 대표/핵심지표)")

    from .splits import walk_forward_folds
    folds = walk_forward_folds(len(df))
    all_fold_results, all_raw = [], []
    for fid, (tr_idx, te_idx) in enumerate(folds, start=1):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        test_df = df.iloc[te_idx].reset_index(drop=True)
        log.info(f"=== Fold {fid}: train={len(train_df)} test={len(test_df)} ===")
        res, raw = _run_fold(train_df, test_df, fid, loss_type=loss_type)
        all_fold_results.append(res)
        all_raw.append(raw)

    model_names = all_fold_results[0].keys()
    pooled = {name: summarize_folds([fr[name] for fr in all_fold_results]) for name in model_names}
    table = build_comparison_table(pooled)
    log.info("\n최종 per-stage / 벤치 비교표 (OOS 40% 풀링, %error 포함):\n" + table.to_string(index=False))
    table.to_csv(C.OUT_DIR / "final_comparison_table.csv", index=False, encoding="utf-8-sig")
    table.to_csv(C.OUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

    oos = pd.concat(all_raw, ignore_index=True)
    oos.to_parquet(C.OUT_DIR / "oos_predictions.parquet", index=False)
    # 하위호환: 기존 stage1_final_oos_predictions.parquet 스키마도 함께 저장
    oos.rename(columns={"mc_hat_deeponet": "mc_hat", "final_deeponet_hybrid": "final_hat"})[
        ["mc_true", "mc_hat", "fair_true", "final_hat"]
    ].to_parquet(C.OUT_DIR / "stage1_final_oos_predictions.parquet", index=False)

    # ---- 이론가(Stage-1) 전용 심화 리포트: calibration + moneyness 세그먼트 ----
    mc_true_all, mc_hat_all = oos["mc_true"].to_numpy(), oos["mc_hat_deeponet"].to_numpy()
    fair_true_all, final_hat_all = oos["fair_true"].to_numpy(), oos["final_deeponet_hybrid"].to_numpy()

    stage1_calib = calibration_stats(mc_true_all, mc_hat_all)
    final_calib = calibration_stats(fair_true_all, final_hat_all)
    log.info(f"Stage-1(DeepONet[{loss_type}]) calibration: slope={stage1_calib['slope']:.4f} "
              f"intercept={stage1_calib['intercept']:.4f} (이상=slope 1, intercept 0)")
    log.info(f"Final(hybrid) calibration: slope={final_calib['slope']:.4f} "
              f"intercept={final_calib['intercept']:.4f}")

    stage1_seg = moneyness_segment_report(mc_true_all, mc_hat_all, mc_true_all, n_bins=5)
    stage1_seg.insert(0, "which", "stage1_MC")
    final_seg = moneyness_segment_report(fair_true_all, final_hat_all, fair_true_all, n_bins=5)
    final_seg.insert(0, "which", "final_fair")
    seg_report = pd.concat([stage1_seg, final_seg], ignore_index=True)
    log.info("\nmoneyness(가격) 세그먼트별 %error:\n" + seg_report.to_string(index=False))

    calib_df = pd.DataFrame([
        {"which": "stage1_MC", **stage1_calib},
        {"which": "final_fair", **final_calib},
    ])
    calib_df.to_csv(C.OUT_DIR / "calibration_report.csv", index=False, encoding="utf-8-sig")
    seg_report.to_csv(C.OUT_DIR / "moneyness_segment_report.csv", index=False, encoding="utf-8-sig")

    log.info(f"저장 완료 -> {C.OUT_DIR}/final_comparison_table.csv, model_comparison.csv, "
              f"oos_predictions.parquet, calibration_report.csv, moneyness_segment_report.csv")
    return table, seg_report, calib_df


if __name__ == "__main__":
    main()
