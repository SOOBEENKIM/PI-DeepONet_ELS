"""7. walk-forward 4-fold 평가 오케스트레이터: Stage1/Stage2/Final/Direct 벤치 전부 학습·평가."""
import logging

import numpy as np
import pandas as pd

from . import config as C
from . import benchmark as bm
from . import deeponet as don
from .splits import sort_chronological, walk_forward_folds
from .metrics import compute_metrics, summarize_folds, build_comparison_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_all")


def load_merged():
    ml = pd.read_parquet(C.CACHE_DIR / "dataset_ml.parquet")
    don_df = pd.read_parquet(C.CACHE_DIR / "dataset_deeponet.parquet")
    overlap = [c for c in don_df.columns if c in ml.columns and c not in ("item", "isu_ord")]
    don_only = don_df.drop(columns=overlap)
    merged = ml.merge(don_only, on=["item", "isu_ord"], how="inner")
    return sort_chronological(merged)


def run_fold(train_df, test_df, fold_id):
    out = {}

    # ---- Stage-1 DeepONet (MC 재현) ----
    model_don, scaler_don = don.train_deeponet(train_df, target_col="MC")
    mc_hat_don_test = don.predict_deeponet(model_don, scaler_don, test_df)
    out["stage1_deeponet"] = compute_metrics(test_df["MC"], mc_hat_don_test)

    # ---- Stage-1 XGB (xgb_hybrid용) ----
    model_s1xgb, enc_s1xgb = bm.train_stage1_xgb(train_df)
    mc_hat_xgb_test = bm.predict_tabular(model_s1xgb, enc_s1xgb, test_df)
    out["stage1_xgb"] = compute_metrics(test_df["MC"], mc_hat_xgb_test)

    # ---- Stage-2 XGB (resid) ----
    model_s2, enc_s2 = bm.train_stage2_xgb(train_df)
    resid_hat_test = bm.predict_tabular(model_s2, enc_s2, test_df)
    out["stage2_resid"] = compute_metrics(test_df["resid"], resid_hat_test)

    # ---- 하이브리드 조립 (Stage1_hat + recent_margin + Stage2_hat) ----
    final_don_hybrid = mc_hat_don_test + test_df["recent_margin"].to_numpy() + resid_hat_test
    out["final_deeponet_hybrid"] = compute_metrics(test_df["fair"], final_don_hybrid)

    final_xgb_hybrid = mc_hat_xgb_test + test_df["recent_margin"].to_numpy() + resid_hat_test
    out["final_xgb_hybrid"] = compute_metrics(test_df["fair"], final_xgb_hybrid)

    # ---- Direct 벤치 ----
    model_ridge, enc_ridge = bm.train_bench_ridge(train_df)
    pred_ridge = bm.predict_tabular(model_ridge, enc_ridge, test_df)
    out["bench_ridge"] = compute_metrics(test_df["fair"], pred_ridge)

    model_bgxgb, enc_bgxgb = bm.train_bench_xgb(train_df)
    pred_bgxgb = bm.predict_tabular(model_bgxgb, enc_bgxgb, test_df)
    out["bench_gbm"] = compute_metrics(test_df["fair"], pred_bgxgb)

    model_don_direct, scaler_don_direct = don.train_deeponet(train_df, target_col="fair")
    pred_don_direct = don.predict_deeponet(model_don_direct, scaler_don_direct, test_df)
    out["deeponet_direct"] = compute_metrics(test_df["fair"], pred_don_direct)

    for k, v in out.items():
        log.info(f"[fold {fold_id}] {k}: R2={v['r2']:.4f} MAE={v['mae']:.5f} "
                  f"RMSE={v['rmse']:.5f} Spearman={v['spearman']:.4f} n={v['n']}")
    return out


def main():
    df = load_merged()
    log.info(f"모델링 데이터셋 로드: {len(df)} rows, OOS 시작 60% 지점부터 walk-forward 평가")

    folds = walk_forward_folds(len(df))
    all_fold_results = []
    for fid, (tr_idx, te_idx) in enumerate(folds, start=1):
        train_df, test_df = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
        log.info(f"=== Fold {fid}: train={len(train_df)} test={len(test_df)} "
                  f"(train {train_df['isu_ord'].min()}~{train_df['isu_ord'].max()}, "
                  f"test {test_df['isu_ord'].min()}~{test_df['isu_ord'].max()}) ===")
        res = run_fold(train_df, test_df, fid)
        all_fold_results.append(res)

    model_names = all_fold_results[0].keys()
    pooled = {name: summarize_folds([fr[name] for fr in all_fold_results]) for name in model_names}

    table = build_comparison_table(pooled)
    log.info("\n" + "최종 per-stage / 벤치 R2 비교표 (OOS 40% 풀링):\n" + table.to_string(index=False))

    table.to_csv(C.OUT_DIR / "final_comparison_table.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_fold_results).to_json(C.OUT_DIR / "fold_results.json", orient="records", indent=2)
    log.info(f"저장 완료 -> {C.OUT_DIR / 'final_comparison_table.csv'}")
    return table, all_fold_results


if __name__ == "__main__":
    main()
