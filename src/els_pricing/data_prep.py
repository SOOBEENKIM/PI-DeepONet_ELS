"""[1] Raw 3 CSV -> product_master.parquet  (Section 2).

Independent reimplementation. Builds, per surviving product:
  - fair (label)      = FAIR_VALUE / ISU_PRC_DETAIL
  - tenor (years)     = (MAT_DT - ISU_DT)/365.25   (NOT DT_DIFF)
  - strk_0..11        = per-step call strikes (%/100), SEQ-ordered, len-12 padded
  - cpn_0..11         = per-step cumulative coupon (PMT_1), len-12 padded
  - texer_0..11       = per-step exercise time in years from issue, len-12 padded
  - B, knock_in       = KI / loss barrier and KI flag
  - derived features  = Kfirst,K,b_over_k,stepdown,nobs, coupon(ANL_RTRN)
  - issuer, isu_dt, isu_ord, underlyings

Real-data note: BARR_1 is ~92% empty, so the loss barrier B is taken from the
final maturity strike (Digital_Call_Put row) when BARR_1 is absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

N = C.N_STRK


def _load_auto_call() -> pd.DataFrame:
    ac = pd.read_csv(C.CSV_AUTO_CALL, low_memory=False)
    ac["ISU_DT"] = pd.to_datetime(ac["ISU_DT"], errors="coerce")
    ac["MAT_DT"] = pd.to_datetime(ac["MAT_DT"], errors="coerce")
    ac["tenor"] = (ac["MAT_DT"] - ac["ISU_DT"]).dt.days / 365.25
    ac["fair"] = ac["FAIR_VALUE"] / ac["ISU_PRC_DETAIL"]
    return ac


def _filter_auto_call(ac: pd.DataFrame) -> pd.DataFrame:
    f = C.FILTER
    m = (
        (ac["PRODUCT_TYPE"] == f["product_type"])
        & (ac["OPT_TYPE"] == f["opt_type"])
        & (ac["CUR_CD"] == f["cur_cd"])
        & (ac["UDRL_CNT"] == f["udrl_cnt"])
        & (ac["CPN_YN"] == f["cpn_yn"])
        & ac["fair"].between(f["fair_lo"], f["fair_hi"])
        & ac["tenor"].between(f["tenor_lo"], f["tenor_hi"])
        & ac["ISU_DT"].notna()
        & ac["MAT_DT"].notna()
    )
    return ac.loc[m].copy()


def _pad(vec: list[float], n: int = N, fill: str = "ffill") -> list[float]:
    """Pad/truncate to length n. ffill repeats last element; else pad with last."""
    vec = list(vec[:n])
    if not vec:
        return [np.nan] * n
    while len(vec) < n:
        vec.append(vec[-1])
    return vec


def _build_schedule_vectors(sc: pd.DataFrame, items: set, isu_dt: dict) -> pd.DataFrame:
    """From SCHD_TYPE==1 rows build per-product strike/coupon/exercise vectors."""
    s = sc[(sc["SCHD_TYPE"] == 1) & (sc["ITEM_CD"].isin(items)) & (sc["STCK_MTHD"] == "ALL_MIN")].copy()
    s["EXER_DT"] = pd.to_datetime(s["EXER_DT"], errors="coerce")
    s = s.sort_values(["ITEM_CD", "SEQ"])

    rows = []
    for item, g in s.groupby("ITEM_CD", sort=False):
        g = g.dropna(subset=["STRK_1", "EXER_DT"])
        if g.empty:
            continue
        strikes = (g["STRK_1"].to_numpy(float) / 100.0).tolist()
        # cumulative coupon: PMT_1 (already cumulative in this dataset); fallback 0
        cpn = g["PMT_1"].fillna(0.0).to_numpy(float).tolist()
        d0 = isu_dt[item]
        texer = ((g["EXER_DT"] - d0).dt.days / 365.25).clip(lower=0).to_numpy(float).tolist()
        nobs = len(strikes)
        # loss barrier B: min explicit BARR_1 if present, else final strike
        barr = g["BARR_1"].dropna()
        B = float(barr.min()) / 100.0 if len(barr) else float(strikes[-1])
        rec = {"item": item, "nobs": nobs, "B": B,
               "strk_last": strikes[-1], "strk_first": strikes[0],
               # full variable-length schedule for the MC engine
               "strk_all": strikes, "cpn_all": cpn, "texer_all": texer}
        sp, cp, tp = _pad(strikes), _pad(cpn), _pad(texer)
        for i in range(N):
            rec[f"strk_{i}"] = sp[i]
            rec[f"cpn_{i}"] = cp[i]
            rec[f"texer_{i}"] = tp[i]
        rows.append(rec)
    return pd.DataFrame(rows)


def _build_underlyings(ud: pd.DataFrame, items: set) -> pd.DataFrame:
    u = ud[ud["ITEM_CD"].isin(items)].copy()
    g = u.groupby("ITEM_CD").agg(
        udly_names=("UDLY_NM", lambda s: list(s)),
        udly_ids=("UDLY_ID", lambda s: list(s)),
        wtd=("WTD_RTO", lambda s: list(s)),
    )
    return g.reset_index().rename(columns={"ITEM_CD": "item"})


def build() -> pd.DataFrame:
    print("[data_prep] loading AUTO_CALL ...")
    ac = _load_auto_call()
    print(f"  rows={len(ac):,}")

    # STCK_MTHD lives in SCHD; find worst-of items first
    print("[data_prep] loading SCHD (worst-of item set) ...")
    sc = pd.read_csv(C.CSV_SCHD, low_memory=False)
    worst_items = set(sc.loc[sc["STCK_MTHD"] == C.FILTER["stck_mthd"], "ITEM_CD"].unique())

    kept = _filter_auto_call(ac)
    kept = kept[kept["ITEM_CD"].isin(worst_items)].copy()
    print(f"[data_prep] after filter (Section 2.2): {len(kept):,} products")

    kept["item"] = kept["ITEM_CD"]
    kept["issuer"] = kept["ISU_ORG"].astype(str)
    kept["isu_ord"] = kept["ISU_DT"].map(pd.Timestamp.toordinal)
    kept["coupon"] = kept["ANL_RTRN"] / 100.0
    isu_dt = dict(zip(kept["item"], kept["ISU_DT"]))
    items = set(kept["item"])

    print("[data_prep] building schedule vectors ...")
    schd = _build_schedule_vectors(sc, items, isu_dt)
    print("[data_prep] building underlyings ...")
    udl = _build_underlyings(pd.read_csv(C.CSV_UDLY, low_memory=False), items)

    base_cols = [
        "item", "issuer", "ISU_DT", "MAT_DT", "isu_ord", "tenor", "fair",
        "ISU_PRC_DETAIL", "FAIR_VALUE", "coupon", "KNCK_IN_YN", "RISK_GRADE",
        "RDMP_TYPE", "PRODUCT_TYPE", "ACT_ISU_AMT", "SB_RT", "DV_RT",
        "PRCP_GRTE_RT", "KNCK_IN_GRC_PRD",
    ]
    base = kept[base_cols].rename(columns={
        "ISU_DT": "isu_dt", "MAT_DT": "mat_dt", "ISU_PRC_DETAIL": "isu_prc",
        "KNCK_IN_YN": "knock_in", "RISK_GRADE": "risk", "RDMP_TYPE": "rdmp",
        "PRODUCT_TYPE": "ptype", "ACT_ISU_AMT": "amt", "SB_RT": "sbrt",
        "DV_RT": "dvrt", "PRCP_GRTE_RT": "prcp", "KNCK_IN_GRC_PRD": "kigrc",
    })

    df = base.merge(schd, on="item", how="inner").merge(udl, on="item", how="left")
    df = df[df["nobs"] >= 1].copy()

    # derived scalar features
    df["Kfirst"] = df["strk_first"]
    df["K"] = df["strk_last"]
    df["b_over_k"] = df["B"] / df["K"]
    df["stepdown"] = df["strk_first"] - df["strk_last"]
    df["iyear"] = df["isu_dt"].dt.year
    df["imonth"] = df["isu_dt"].dt.month
    df["subdays"] = (df["mat_dt"] - df["isu_dt"]).dt.days

    print(f"[data_prep] product_master: {len(df):,} rows, {df.shape[1]} cols")
    return df.reset_index(drop=True)


def main() -> None:
    df = build()
    df.to_parquet(C.PRODUCT_MASTER)
    print(f"[data_prep] -> {C.PRODUCT_MASTER}")
    print(df[["item", "issuer", "isu_dt", "tenor", "fair", "nobs", "B", "K", "b_over_k"]].head())


if __name__ == "__main__":
    main()
