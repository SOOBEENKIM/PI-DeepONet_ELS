"""ELS_hedging_V1 공통 설정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "lake_v2_dart_sba" / "lake_v2_dart_sba"
CACHE_DIR = ROOT / "data" / "cache"
OUT_DIR = ROOT / "data" / "out"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUTO_CALL_CSV = RAW_DIR / "LAKE_V2_DART_AUTO_CALL.csv"
SCHD_INFO_CSV = RAW_DIR / "LAKE_V2_DART_SCHD_INFO.csv"
UDLY_INFO_CSV = RAW_DIR / "LAKE_V2_DART_UDLY_INFO.csv"

SEED = 42
N_STRK_PAD = 12

# ---- 1. 전처리 필터 ----
FAIR_RANGE = (0.70, 1.05)
TENOR_RANGE = (0.5, 5.0)

# ---- 2. 시장데이터 ----
VOL_WINDOW_DAYS = 180          # 180 영업일
MOM_WINDOW_DAYS = 126           # 6개월 모멘텀
NS_TENOR_NODES = [0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10]
FRED_SERIES = {
    "call": "IRSTCI01KRM156N",   # 콜금리 (<24h)
    "m3": "IR3TIB01KRM156N",     # 3개월
    "y10": "IRLTLT01KRM156N",    # 10년
}
FRED_TENORS = {"call": 1 / 365.0, "m3": 0.25, "y10": 10.0}

# 기초자산명 -> yahoo finance 티커. 지수/해외종목 우선, 국내종목은 FDR 폴백 가능(6자리 코드).
TICKER_MAP = {
    # 지수
    "EURO STOXX 50 지수": "^STOXX50E",
    "S&P 500 지수": "^GSPC",
    "HSCEI 지수": "^HSCE",
    "KOSPI 200 지수": "^KS200",
    "니케이225 지수": "^N225",
    "항셍 지수": "^HSI",
    "나스닥100 지수": "^NDX",
    "DAX 지수": "^GDAXI",
    "CAC40 지수": "^FCHI",
    "FTSE100 지수": "^FTSE",
    "TAIEX": "^TWII",
    "ASX 200 지수": "^AXJO",
    "Hang Seng TECH 지수": "^HSTECH",
    "CSI 300 지수": "000300.SS",
    "KOSPI 지수": "^KS11",
    "KOSDAQ 150 지수": "229200.KS",       # KODEX KOSDAQ150 ETF (프록시)
    "KOSPI 200 레버리지 지수": "122630.KS",  # KODEX 레버리지 ETF (프록시)
    "FTSE China A50 지수": "2823.HK",       # iShares FTSE A50 China ETF (프록시)
    "KRX300": "375500.KS",                  # KODEX KRX300 ETF (프록시)
    "EURO STOXX BANKS 지수": "EXX1.DE",     # iShares STOXX Europe 600 Banks 프록시 아님. 실패시 제외.

    # 국내 개별주 (yahoo .KS, FDR 코드는 앞 6자리)
    "삼성전자": "005930.KS",
    "SK 하이닉스": "000660.KS",
    "현대자동차": "005380.KS",
    "네이버": "035420.KS",
    "LG화학": "051910.KS",
    "LG전자": "066570.KS",
    "SK텔레콤": "017670.KS",
    "KB금융지주": "105560.KS",
    "한국전력공사": "015760.KS",
    "카카오": "035720.KS",
    "KT&G": "033780.KS",
    "삼성생명보험": "032830.KS",
    "현대모비스": "012330.KS",
    "삼성SDI": "006400.KS",
    "기아": "000270.KS",
    "SK이노베이션": "096770.KS",
    "하나금융지주": "086790.KS",
    "삼성화재해상보험": "000810.KS",
    "아모레퍼시픽": "090430.KS",
    "신한금융지주회사": "055550.KS",
    "SK": "034730.KS",
    "한화에어로스페이스": "012450.KS",
    "LG유플러스": "032640.KS",
    "셀트리온": "068270.KS",
    "삼성바이오로직스": "207940.KS",
    "LG에너지솔루션": "373220.KS",
    "LG디스플레이": "034220.KS",
    "포스코퓨처엠": "003670.KS",
    "포스코홀딩스": "005490.KS",
    "한국가스공사": "036460.KS",
    "삼성물산": "028260.KS",
    "삼성SDS": "018260.KS",
    "HD현대중공업": "329180.KS",
    "이마트": "139480.KS",
    "LG생활건강": "051900.KS",
    "우리금융지주": "316140.KS",
    "LG": "003550.KS",
    "케이티": "030200.KS",
    "에이치엘만도": "204320.KS",
    "중소기업은행": "024110.KS",
    "삼성전기": "009150.KS",
    "S-OIL": "010950.KS",
    "CJ오쇼핑": "035760.KS",
    "현대홈쇼핑": "057050.KS",
    "엔씨소프트": "036570.KS",
    "넷마블": "251270.KS",
    "한온시스템": "018880.KS",
    "현대글로비스": "086280.KS",
    "한화오션": "042660.KS",
    "호텔신라": "008770.KS",
    "GS 보통주": "078930.KS",
    "현대제철": "004020.KS",
    "CJ 보통주": "001040.KS",
    "신세계": "004170.KS",
    "현대위아": "011210.KS",
    "에이치디한국조선해양": "009540.KS",
    "디엘": "000210.KS",
    "롯데케미칼": "011170.KS",
    "씨제이": "001040.KS",
    "현대백화점": "069960.KS",
    "지에스홈쇼핑": "007070.KS",
    "지에스리테일": "007070.KS",
    "두산에너빌리티": "034020.KS",
    "고려아연": "010130.KS",
    "LG이노텍": "011070.KS",
    "롯데쇼핑": "023530.KS",
    "한화생명보험": "088350.KS",
    "미래에셋증권": "006800.KS",
    "KB손해보험": "002550.KS",

    # 해외 개별주 (달러 표시, 원화 상품에도 그대로 사용 - 환리스크는 별도 미반영)
    "테슬라": "TSLA",
    "엔비디아": "NVDA",
    "ADVANCED MICRO DEVICES INC": "AMD",
    "팔란티어 테크놀로지스": "PLTR",
    "마이크론 테크놀로지": "MU",
    "애플": "AAPL",
    "아마존닷컴": "AMZN",
    "넷플릭스": "NFLX",
    "메타 플랫폼스": "META",
    "브로드컴": "AVGO",
    "인텔": "INTC",
    "알파벳 클래스A": "GOOGL",
    "스타벅스": "SBUX",
    "iShares China Large-Cap ETF": "FXI",
    "마이크로소프트": "MSFT",
    "퀄컴": "QCOM",
    "BOEING CO": "BA",
    "오라클": "ORCL",
    "Gilead Sciences": "GILD",
    "General Motors": "GM",
    "텐센트 홀딩스": "0700.HK",
    "알리바바그룹 홀딩스": "BABA",
    "우버 테크놀로지스": "UBER",
    "나이키": "NKE",
    "BANK OF AMERICA CORP": "BAC",
    "씨티그룹": "C",
    "일라이릴리": "LLY",
    "월트디즈니": "DIS",
    "TSMC ADR": "TSM",
    "APPLIED MATERIALS INC": "AMAT",
    "페이팔": "PYPL",
    "JPMorgan Chase": "JPM",
    "EXXON MOBIL CORP": "XOM",
    "AMGEN INC": "AMGN",
    "AT&T": "T",
    "Goldman Sachs Group": "GS",
    "닌텐도": "7974.T",
    "맥도날드": "MCD",
    "General Electric": "GE",
    "소프트뱅크그룹": "9984.T",
    "쇼피파이": "SHOP",
    "T-Mobile US": "TMUS",
    "월마트": "WMT",
    "버크셔 해서웨이 B": "BRK-B",
    "CVS Health Corp": "CVS",
    "ARM HOLDINGS PLC ADR": "ARM",
    "IBM CORP": "IBM",
    "귀주모태주": "600519.SS",
    "도미노 피자": "DPZ",
    "도쿄일렉트론": "8035.T",
    "미쓰비시중공업": "7011.T",
    "ACTIVISION BLIZZARD INC": "ATVI",
    "항서제약": "600276.SS",
    "브렌트유 최근월 선물": "BZ=F",
    "런던 금 가격지수": "GC=F",
    "런던 은 가격지수": "SI=F",
}

# yahoo 티커 -> FinanceDataReader 코드 (국내 상장물만, .KS/.KQ 접미 제거)
def fdr_fallback_code(yahoo_ticker: str):
    if yahoo_ticker.endswith(".KS") or yahoo_ticker.endswith(".KQ"):
        return yahoo_ticker.split(".")[0]
    return None

# ---- 5. 피처 그룹 ----
STRK_COLS = [f"strk_{i}" for i in range(N_STRK_PAD)]
U_COLS = [f"u{i}" for i in range(len(NS_TENOR_NODES))]

BASE_NUM_COLS = (
    ["sig1", "sig2", "sig3", "rho12", "rho13", "rho23", "sig_mean", "rho", "r",
     "B", "K", "Kfirst", "coupon", "tenor", "nobs", "cpn_spread", "b_over_k",
     "stepdown", "mom6m", "ACT_ISU_AMT", "SB_RT", "DV_RT", "PRCP_GRTE_RT",
     "KNCK_IN_GRC_PRD", "ISU_DT_year", "sub_days"]
    + STRK_COLS
)
REG_COLS = ["recent_margin", "recent_mktvol", "curve_level", "curve_slope",
            "curve_curv", "issue_intensity"]
CAT_COLS = ["ISU_ORG", "RISK_GRADE", "PRODUCT_TYPE", "RDMP_TYPE", "ISU_DT_month"]

BRANCH_COLS = U_COLS + ["sig1", "sig2", "sig3", "rho12", "rho13", "rho23", "sig_eff"]
TRUNK_COLS = STRK_COLS + ["B", "coupon", "tenor"]

# ---- 3. MC 엔진 ----
MC_N_PATHS = 100_000
MC_SEED = 20240601

# ---- 7. walk-forward ----
WF_CUM_FRACTIONS = [0.60, 0.70, 0.80, 0.90, 1.00]

# ---- 8. 팀원 목표치 ----
TEAM_TARGETS = {
    "stage1_deeponet_r2": (0.92, 0.99),
    "stage1_xgb_r2": (0.83, 0.90),
    "stage2_resid_r2": (0.49, 0.49),
    "final_deeponet_hybrid_r2": (0.72, 0.74),
    "final_deeponet_hybrid_spearman": (0.85, 0.87),
    "direct_ridge_r2": (0.56, 0.56),
    "direct_tree_r2": (0.26, 0.33),
    "direct_deeponet_r2": (None, 0.0),
}
