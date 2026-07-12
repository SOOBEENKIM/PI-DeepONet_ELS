"""V2 (els-hedging-v2 브랜치): Stage-2를 "MC 잔차보정"으로 재정의.

교수님 방식 — V1의 마진 분해(fair - MC - recent_margin)를 전부 버리고, fair·recent_margin·마진
개념 자체를 무시한다. Stage-2는 순수하게 Stage-1 DeepONet의 MC 근사오차를 보정하는
잔차모델이다:

    Stage-2 타깃  : resid_mc = MC - DeepONet예측(MC)         (fold의 train 구간에서 in-sample로 계산)
    Stage-2 입력  : BASE+CAT만 (V1과 동일 원칙 - MC/recent_margin/margin류는 피처가 아니라 앵커)
    Final 이론가  : DeepONet예측(MC) + Stage-2예측(resid_mc)  (≈ MC 재현을 정밀화한 이론가)

Stage-2 기본 모델은 MLP(margin_mlp.py 재사용 - 이미 target_col 파라미터화되어 있어 그대로
resid_mc를 태우면 됨), tree(XGB, benchmark.py의 train_stage2_xgb 재사용)는 비교용.

⚠ MC 재실행 없음: mc_engine.py를 다시 돌리지 않는다. 기존 data/cache/product_mc.parquet가
반영된 dataset_ml.parquet / dataset_deeponet.parquet(V1이 이미 생성) 그대로 재사용한다.
Stage-1 DeepONet만 각 walk-forward fold에서 새로 학습한다 (train 구간의 in-sample 예측이
있어야 resid_mc 타깃을 만들 수 있기 때문 - 기존 파이프라인도 fold마다 DeepONet을 새로 학습하는
방식과 동일한 관례).
"""
import argparse
import logging

import numpy as np
import pandas as pd

from . import config as C
from . import benchmark as bm
from . import deeponet as don
from . import margin_mlp as mmlp
from .splits import sort_chronological, walk_forward_folds
from .metrics import compute_metrics, summarize_folds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stage2_mc_residual")

TABLE_ORDER = [
    "stage1_deeponet_alone",
    "final_theory_mlp",
    "stage2_mc_resid_mlp",
    "final_theory_xgb",
    "stage2_mc_resid_xgb",
]


def _load_merged():
    ml = pd.read_parquet(C.CACHE_DIR / "dataset_ml.parquet")
    don_df = pd.read_parquet(C.CACHE_DIR / "dataset_deeponet.parquet")
    overlap = [c for c in don_df.columns if c in ml.columns and c not in ("item", "isu_ord")]
    don_only = don_df.drop(columns=overlap)
    merged = ml.merge(don_only, on=["item", "isu_ord"], how="inner")
    return sort_chronological(merged)


def run_fold(train_df, test_df, fold_id, loss_type="mae"):
    out = {}

    # ---- Stage-1 DeepONet (이론가 본질, MC 재현) - fold마다 새로 학습 ----
    model_don, scaler_don = don.train_deeponet(train_df, target_col="MC", loss_type=loss_type)
    mc_hat_train = don.predict_deeponet(model_don, scaler_don, train_df)
    mc_hat_test = don.predict_deeponet(model_don, scaler_don, test_df)
    out["stage1_deeponet_alone"] = compute_metrics(test_df["MC"], mc_hat_test)

    # ---- Stage-2 타깃: resid_mc = MC - DeepONet예측 (fair/recent_margin/마진 전부 무시) ----
    train_df = train_df.copy()
    train_df["resid_mc"] = train_df["MC"].to_numpy() - mc_hat_train
    resid_mc_true_test = test_df["MC"].to_numpy() - mc_hat_test  # 진단용(Stage-2 자체 성능 참고)

    # ---- Stage-2 MLP (기본, 교수님 방식) ----
    model_mlp, enc_mlp = mmlp.train_margin_mlp(train_df, target_col="resid_mc")
    resid_hat_mlp = mmlp.predict_margin_mlp(model_mlp, enc_mlp, test_df)
    out["stage2_mc_resid_mlp"] = compute_metrics(resid_mc_true_test, resid_hat_mlp)
    final_theory_mlp = mc_hat_test + resid_hat_mlp
    out["final_theory_mlp"] = compute_metrics(test_df["MC"], final_theory_mlp)

    # ---- Stage-2 XGB (비교용) ----
    model_xgb, enc_xgb = bm.train_stage2_xgb(train_df, target_col="resid_mc")
    resid_hat_xgb = bm.predict_tabular(model_xgb, enc_xgb, test_df)
    out["stage2_mc_resid_xgb"] = compute_metrics(resid_mc_true_test, resid_hat_xgb)
    final_theory_xgb = mc_hat_test + resid_hat_xgb
    out["final_theory_xgb"] = compute_metrics(test_df["MC"], final_theory_xgb)

    for name in TABLE_ORDER:
        v = out[name]
        log.info(f"[fold {fold_id}] {name}: R2={v['r2']:.4f} MAPE={v.get('mape', float('nan')):.4%} "
                  f"bp_err={v.get('bp_error', float('nan')):.1f} Spearman={v['spearman']:.4f} n={v['n']}")

    raw = pd.DataFrame({
        "item": test_df["item"].to_numpy(), "isu_ord": test_df["isu_ord"].to_numpy(),
        "fold": fold_id,
        "mc_true": test_df["MC"].to_numpy(),
        "mc_hat_deeponet": mc_hat_test,
        "resid_mc_true": resid_mc_true_test,
        "resid_mc_hat_mlp": resid_hat_mlp, "resid_mc_hat_xgb": resid_hat_xgb,
        "final_theory_mlp": final_theory_mlp, "final_theory_xgb": final_theory_xgb,
    })
    return out, raw


def build_comparison_table(pooled: dict) -> pd.DataFrame:
    rows = []
    for name in TABLE_ORDER:
        if name not in pooled:
            continue
        m = pooled[name]
        rows.append({
            "model": name,
            "R2": round(m.get("r2", np.nan), 4),
            "MAPE": round(m.get("mape", np.nan), 4) if np.isfinite(m.get("mape", np.nan)) else np.nan,
            "bp_error": round(m.get("bp_error", np.nan), 1),
            "MAE": round(m.get("mae", np.nan), 5),
            "RMSE": round(m.get("rmse", np.nan), 5),
            "Spearman": round(m.get("spearman", np.nan), 4),
            "n": m.get("n"),
        })
    return pd.DataFrame(rows)


def main(loss_type="mae"):
    df = _load_merged()
    log.info(f"모델링 데이터셋 로드: {len(df)} rows (V2: Stage-2=MC잔차보정 MLP기본, "
              f"Stage-1=DeepONet[{loss_type}], MC 재실행 없음)")

    folds = walk_forward_folds(len(df))
    all_fold_results, all_raw = [], []
    for fid, (tr_idx, te_idx) in enumerate(folds, start=1):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        test_df = df.iloc[te_idx].reset_index(drop=True)
        log.info(f"=== Fold {fid}: train={len(train_df)} test={len(test_df)} ===")
        res, raw = run_fold(train_df, test_df, fid, loss_type=loss_type)
        all_fold_results.append(res)
        all_raw.append(raw)

    model_names = all_fold_results[0].keys()
    pooled = {name: summarize_folds([fr[name] for fr in all_fold_results]) for name in model_names}
    table = build_comparison_table(pooled)
    log.info("\nV2 Stage-2(MC 잔차보정) walk-forward 비교표 (DeepONet 단독 vs 보정 후 이론가, "
              "OOS 40% 풀링):\n" + table.to_string(index=False))

    table.to_csv(C.OUT_DIR / "stage2_v2_comparison_table.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_fold_results).to_json(C.OUT_DIR / "stage2_v2_fold_results.json", orient="records", indent=2)
    oos = pd.concat(all_raw, ignore_index=True)
    oos.to_parquet(C.OUT_DIR / "stage2_v2_oos_predictions.parquet", index=False)

    log.info(f"저장 완료 -> {C.OUT_DIR}/stage2_v2_comparison_table.csv, "
              f"stage2_v2_fold_results.json, stage2_v2_oos_predictions.parquet")
    return table, all_fold_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss-type", type=str, default="mae", choices=["mse", "mae", "mape"],
                     help="Stage-1 DeepONet 대표 손실 (기본 mae, §V1_발표보완_지시서 5)")
    args = ap.parse_args()
    main(loss_type=args.loss_type)
