"""5. 데이터셋 3종(ml/deeponet) 구성 -> dataset_ml.parquet, dataset_deeponet.parquet

Stage-2 타깃 resid = fair - MC - recent_margin.
Stage-2 입력은 BASE+CAT만 (REG/MC/recent_margin 제외).
"""
import logging

import pandas as pd

from . import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("datasets")

ID_COLS = ["item", "isu_ord"]
ANCHOR_COLS = ["fair", "MC", "recent_margin"]


def build_datasets():
    feat = pd.read_parquet(C.CACHE_DIR / "product_features.parquet")
    log.info(f"product_features 로드: {len(feat)} rows")

    feat = feat.copy()
    feat["resid"] = feat["fair"] - feat["MC"] - feat["recent_margin"]

    # ---- ml (tabular, Stage-2) : BASE + REG + CAT + target/anchor ----
    ml_cols = ID_COLS + C.BASE_NUM_COLS + C.REG_COLS + C.CAT_COLS + ANCHOR_COLS + ["resid"]
    ml_cols = list(dict.fromkeys(ml_cols))  # 중복 제거(순서 유지) - REG_COLS/ANCHOR_COLS에 recent_margin 중복
    ml_cols = [c for c in ml_cols if c in feat.columns]
    missing = set(ID_COLS + C.BASE_NUM_COLS + C.REG_COLS + C.CAT_COLS) - set(feat.columns)
    if missing:
        log.warning(f"ml 데이터셋에서 누락된 컬럼: {missing}")
    ml_df = feat[ml_cols].copy()
    ml_df.to_parquet(C.CACHE_DIR / "dataset_ml.parquet", index=False)
    log.info(f"dataset_ml 저장: {ml_df.shape}")

    # ---- deeponet: branch(u0..9 + sig/rho/sig_eff) + trunk(strk_0..11+B+coupon+tenor) + aux(r) ----
    don_cols = ID_COLS + C.BRANCH_COLS + C.TRUNK_COLS + ["r"] + ANCHOR_COLS + ["resid"]
    don_cols = list(dict.fromkeys(don_cols))  # 중복 제거(순서 유지)
    don_cols = [c for c in don_cols if c in feat.columns]
    don_df = feat[don_cols].copy()
    don_df.to_parquet(C.CACHE_DIR / "dataset_deeponet.parquet", index=False)
    log.info(f"dataset_deeponet 저장: {don_df.shape}")

    log.info(f"resid 통계: mean={ml_df['resid'].mean():.5f} std={ml_df['resid'].std():.5f}")
    return ml_df, don_df


if __name__ == "__main__":
    build_datasets()
