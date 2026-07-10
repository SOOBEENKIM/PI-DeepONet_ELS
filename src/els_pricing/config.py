"""Central configuration: paths, filter thresholds, tenor nodes, ticker map."""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (project root = two levels up from this file: src/els_pricing/config.py)
# ---------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
SRC_DIR = PKG_DIR.parent
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", SRC_DIR.parent))

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR = DATA_DIR / "out"
FIG_DIR = PROJECT_DIR / "figures"
for _d in (CACHE_DIR, OUT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Raw CSVs
CSV_AUTO_CALL = RAW_DIR / "LAKE_V2_DART_AUTO_CALL.csv"
CSV_SCHD = RAW_DIR / "LAKE_V2_DART_SCHD_INFO.csv"
CSV_UDLY = RAW_DIR / "LAKE_V2_DART_UDLY_INFO.csv"

# Cache / output artifacts
PRODUCT_MASTER = CACHE_DIR / "product_master.parquet"
MARKET_MASTER = CACHE_DIR / "product_market.parquet"   # + sig/rho/u0..9/r/mom6m
MC_MASTER = CACHE_DIR / "product_mc.parquet"            # + mc column

# ---------------------------------------------------------------------------
# Sample filter (Section 2.2) — 3-star worst-of stepdown autocall, KRW
# ---------------------------------------------------------------------------
FILTER = dict(
    product_type="ELS",
    stck_mthd="ALL_MIN",   # worst-of
    udrl_cnt=3,            # 3-star
    opt_type="STEP",       # stepdown
    cur_cd="KRW",
    cpn_yn=0,              # exclude conditional-coupon monthly products
    fair_lo=0.70, fair_hi=1.05,
    tenor_lo=0.5, tenor_hi=5.0,
)

# Fixed contract-schedule vector length (Section 2.4)
N_STRK = 12

# ---------------------------------------------------------------------------
# MC engine defaults (Section 4.1)
# ---------------------------------------------------------------------------
MC = dict(
    n_paths=100_000,
    steps_per_year=252,
    seed=0,
    vol_window=180,       # trading days of history for vol/corr
    mom_window=126,       # 6m momentum window
)

# ---------------------------------------------------------------------------
# DeepONet knobs (Section 4d) — tune here if R^2 is insufficient
# ---------------------------------------------------------------------------
DEEPONET = dict(
    P=192,                       # branch/trunk embedding dim
    branch_cnn=(16, 32),
    trunk_hidden=(192, 192, 192),
    epochs=200,
    patience=20,
    lr=1e-3,
    grad_clip=5.0,
)

# ---------------------------------------------------------------------------
# Yield curve (Section 3.3) — FRED KR series and NS tenor nodes
# ---------------------------------------------------------------------------
FRED_SERIES = {
    "short": "IRSTCI01KRM156N",   # call / interbank (<24h)  ~ short end
    "mid": "IR3TIB01KRM156N",     # 3M interbank
    "long": "IRLTLT01KRM156N",    # 10Y government bond
}
FRED_TENORS = {"short": 1 / 12, "mid": 0.25, "long": 10.0}  # years
NS_NODES = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]  # u0..u9
NS_LAMBDA = 0.7  # fixed NS decay (0.5~1.0 recommended)

# ---------------------------------------------------------------------------
# Underlying -> yahoo ticker map (Section 3.1)
# ---------------------------------------------------------------------------
INDEX_TICKER = {
    "EURO STOXX 50": "^STOXX50E",
    "S&P 500": "^GSPC",
    "HSCEI": "^HSCE",
    "항셍": "^HSI",
    "HANG SENG": "^HSI",
    "KOSPI 200": "^KS200",
    "니케이225": "^N225",
    "NIKKEI 225": "^N225",
    "나스닥100": "^NDX",
    "NASDAQ 100": "^NDX",
    "DAX": "^GDAXI",
    "CSI 300": "000300.SS",
    "EURO STOXX BANKS": "^SX7E",
}
# Single-stock name -> yahoo ticker (major ones seen in the data)
STOCK_TICKER = {
    "테슬라": "TSLA", "엔비디아": "NVDA", "애플": "AAPL", "아마존닷컴": "AMZN",
    "넷플릭스": "NFLX", "메타 플랫폼스": "META", "브로드컴": "AVGO", "인텔": "INTC",
    "마이크론 테크놀로지": "MU", "ADVANCED MICRO DEVICES INC": "AMD",
    "팔란티어 테크놀로지스": "PLTR",
    "삼성전자": "005930.KS", "SK 하이닉스": "000660.KS", "현대자동차": "005380.KS",
    "네이버": "035420.KS", "LG화학": "051910.KS", "포스코홀딩스": "005490.KS",
    "LG전자": "066570.KS", "SK텔레콤": "017670.KS",
}

# Asset-class annualised vol used ONLY as a labelled fallback when price history
# cannot be fetched (yahoo unreachable). Clearly flagged via `vol_source` column.
FALLBACK_VOL = {"index": 0.20, "stock": 0.38}
FALLBACK_CORR = 0.55  # default pairwise correlation in fallback mode


def describe() -> None:
    print("PROJECT_DIR :", PROJECT_DIR)
    print("RAW_DIR     :", RAW_DIR, "(exists)" if RAW_DIR.exists() else "(MISSING)")
    print("CACHE_DIR   :", CACHE_DIR)
    print("OUT_DIR     :", OUT_DIR)


if __name__ == "__main__":
    describe()
