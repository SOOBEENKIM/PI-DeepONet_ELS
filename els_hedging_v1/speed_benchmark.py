"""속도 실험 v2 (§V1_발표보완_지시서 3): DeepONet은 배치 forward로 재측정(per-call 오버헤드 제거),
MC는 기존 실측 mc_seconds에서 소표본(200건) 추출 -> 상품당 평균 x 전체 건수로 환산 (MC 재실행 없음).

측정조건: CPU 1코어(torch.set_num_threads(1)), 대표 DeepONet = MAE 손실 최적본.
"""
import logging
import time

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import deeponet as don
from .splits import sort_chronological, walk_forward_folds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("speed_benchmark")

MC_SAMPLE_N = 200


def _batch_infer_seconds(model, scaler, df):
    """전체 배치를 한 번에 forward (모델 로드/스케일러 fit 제외, 순수 추론시간). CPU 1코어."""
    torch.set_num_threads(1)
    curve, vc, trunk, aux = scaler.transform(df)  # 텐서 준비(전처리)는 타이밍 제외
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        pred_std = model(curve, vc, trunk, aux).numpy()
    pred = scaler.inverse_target(pred_std)
    elapsed = time.perf_counter() - t0
    return elapsed, pred


def run_speed_benchmark(loss_type="mae", seed=C.SEED):
    don_df = pd.read_parquet(C.CACHE_DIR / "dataset_deeponet.parquet")
    mc_df = pd.read_parquet(C.CACHE_DIR / "product_mc.parquet")[["item", "mc_seconds"]]
    don_df = sort_chronological(don_df)
    folds = walk_forward_folds(len(don_df))

    total_don_sec = 0.0
    all_pred_mc, all_true_mc = [], []
    oos_items = []

    for fid, (tr_idx, te_idx) in enumerate(folds, start=1):
        train_df = don_df.iloc[tr_idx].reset_index(drop=True)
        test_df = don_df.iloc[te_idx].reset_index(drop=True)
        log.info(f"[fold {fid}] Stage-1 DeepONet(loss={loss_type}) 학습(train={len(train_df)}) 후 "
                  f"test={len(test_df)} 배치추론 실측...")

        model, scaler = don.train_deeponet(train_df, target_col="MC", loss_type=loss_type)
        elapsed, pred = _batch_infer_seconds(model, scaler, test_df)

        total_don_sec += elapsed
        all_pred_mc.append(pred)
        all_true_mc.append(test_df["MC"].to_numpy())
        oos_items.append(test_df["item"])
        log.info(f"[fold {fid}] DeepONet 배치추론 {len(test_df)}건 -> {elapsed*1000:.2f}ms "
                  f"({elapsed/len(test_df)*1e6:.2f}us/건)")

    pred_mc = np.concatenate(all_pred_mc)
    true_mc = np.concatenate(all_true_mc)
    oos_item_s = pd.concat(oos_items, ignore_index=True)
    n_oos = len(oos_item_s)

    # ---- MC: 기존 실측치에서 소표본(200건) 추출 -> 상품당 평균 x 전체 건수로 환산 (재시뮬레이션 없음) ----
    oos_mc_seconds = oos_item_s.to_frame("item").merge(mc_df, on="item", how="left")["mc_seconds"]
    rng = np.random.RandomState(seed)
    sample = oos_mc_seconds.dropna().sample(min(MC_SAMPLE_N, oos_mc_seconds.notna().sum()),
                                              random_state=rng)
    mc_per_product_sample = float(sample.mean())
    mc_total_estimated = mc_per_product_sample * n_oos
    mc_total_exact = float(oos_mc_seconds.sum())  # 참고: 전량 실측 합(더 정확하나 팀 방법론과는 다름)

    total_don_us_per_product = total_don_sec / n_oos * 1e6
    speedup_sample = mc_total_estimated / total_don_sec if total_don_sec > 0 else np.nan
    speedup_exact = mc_total_exact / total_don_sec if total_don_sec > 0 else np.nan

    summary = pd.DataFrame([{
        "n_products": n_oos,
        "loss_type": loss_type,
        "mc_per_product_sec(소표본n=200)": mc_per_product_sample,
        "mc_total_sec(소표본환산)": mc_total_estimated,
        "mc_total_sec(전수실측,참고)": mc_total_exact,
        "deeponet_total_sec(배치forward)": total_don_sec,
        "deeponet_per_product_us": total_don_us_per_product,
        "speedup_x(소표본환산기준)": speedup_sample,
        "speedup_x(전수실측기준,참고)": speedup_exact,
        "price_corr": float(np.corrcoef(true_mc, pred_mc)[0, 1]),
        "price_mae": float(np.mean(np.abs(pred_mc - true_mc))),
        "price_bias(pred-true)": float(np.mean(pred_mc - true_mc)),
    }])
    log.info("\n=== 속도 실험 요약 v2 (배치 forward, OOS 40% 전체, CPU 1코어) ===\n"
              + summary.T.to_string())
    log.info("(참고) 팀 궤적: MC ~5.46s/상품, DeepONet ~4.26us/상품, ~130만배")

    summary.to_csv(C.OUT_DIR / "speed_benchmark_summary.csv", index=False, encoding="utf-8-sig")
    detail = pd.DataFrame({"true_mc": true_mc, "pred_mc": pred_mc})
    detail.to_parquet(C.OUT_DIR / "speed_benchmark_detail.parquet", index=False)
    log.info(f"저장 완료 -> {C.OUT_DIR / 'speed_benchmark_summary.csv'}, speed_benchmark_detail.parquet")
    return summary, detail


if __name__ == "__main__":
    run_speed_benchmark()
