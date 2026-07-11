"""2. 시장데이터: 180일 vol/corr + KR Nelson-Siegel 금리커브 -> product_market.parquet"""
import logging
import time
import warnings

import numpy as np
import pandas as pd

from . import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("market_data")
warnings.filterwarnings("ignore")

FETCH_START = "2013-01-01"


def _sanitize(ticker: str) -> str:
    return ticker.replace("^", "IDX_").replace("=", "_").replace(".", "_")


def _fetch_one_ticker(ticker: str, end: str) -> pd.Series | None:
    """yfinance -> 실패시 FinanceDataReader(국내) 폴백. 종가 Series(index=date) 반환."""
    cache_f = C.CACHE_DIR / f"px_{_sanitize(ticker)}.parquet"
    if cache_f.exists():
        s = pd.read_parquet(cache_f)["close"]
        s.index = pd.to_datetime(s.index)
        return s

    import yfinance as yf

    px = None
    for attempt in range(4):
        try:
            df = yf.download(ticker, start=FETCH_START, end=end, auto_adjust=True,
                              progress=False, threads=False)
            if df is not None and not df.empty:
                col = df["Close"]
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
                px = col.dropna()
                if len(px):
                    break
        except Exception as e:  # noqa: BLE001
            log.warning(f"[{ticker}] yfinance 시도 {attempt+1} 실패: {e}")
        time.sleep(2 ** attempt)

    if px is None or px.empty:
        fdr_code = C.fdr_fallback_code(ticker)
        if fdr_code is not None:
            try:
                import FinanceDataReader as fdr
                df = fdr.DataReader(fdr_code, FETCH_START, end)
                if df is not None and not df.empty:
                    px = df["Close"].dropna()
                    log.info(f"[{ticker}] FDR 폴백 성공 ({fdr_code})")
            except Exception as e:  # noqa: BLE001
                log.warning(f"[{ticker}] FDR 폴백 실패: {e}")

    if px is None or px.empty:
        log.warning(f"[{ticker}] 가격 fetch 완전 실패 -> 제외")
        return None

    px.index = pd.to_datetime(px.index)
    px.name = "close"
    px.to_frame().to_parquet(cache_f)
    return px


def fetch_all_prices(tickers: list[str], end: str) -> dict:
    from tqdm import tqdm
    out = {}
    for t in tqdm(tickers, desc="fetch px"):
        s = _fetch_one_ticker(t, end)
        if s is not None:
            out[t] = s
    return out


def _ns_basis(tau: np.ndarray, lam: float) -> np.ndarray:
    """Nelson-Siegel basis matrix columns: [1, f1(tau), f2(tau)]."""
    tau = np.asarray(tau, dtype=float)
    x = tau / lam
    f1 = np.where(x > 1e-8, (1 - np.exp(-x)) / x, 1.0)
    f2 = f1 - np.exp(-x)
    return np.stack([np.ones_like(tau), f1, f2], axis=-1)


def build_ns_curve(min_date: str, max_date: str, lam: float = 1.5):
    """FRED 3점(콜/3M/10Y) 월간 -> 일간 ffill -> 월별 NS(beta0,1,2) 적합."""
    import pandas_datareader.data as web

    start = pd.Timestamp(min_date) - pd.DateOffset(years=1)
    end = pd.Timestamp(max_date) + pd.DateOffset(months=1)

    series = {}
    for key, code in C.FRED_SERIES.items():
        try:
            s = web.DataReader(code, "fred", start, end)[code]
        except Exception as e:
            log.warning(f"FRED {code} fetch 실패, 재시도: {e}")
            time.sleep(3)
            s = web.DataReader(code, "fred", start, end)[code]
        series[key] = s / 100.0  # % -> decimal
    rates = pd.DataFrame(series).sort_index()
    rates = rates.ffill().dropna()

    tenors = np.array([C.FRED_TENORS["call"], C.FRED_TENORS["m3"], C.FRED_TENORS["y10"]])
    A = _ns_basis(tenors, lam)          # 3x3
    A_inv = np.linalg.inv(A)
    y = rates[["call", "m3", "y10"]].to_numpy()
    beta = y @ A_inv.T                  # n_months x 3

    curve = pd.DataFrame(beta, index=rates.index, columns=["b0", "b1", "b2"])
    curve["lam"] = lam

    # 일간 ffill (월초 기준일 -> 매일)
    daily_idx = pd.date_range(curve.index.min(), pd.Timestamp(max_date) + pd.Timedelta(days=5), freq="D")
    curve_daily = curve.reindex(curve.index.union(daily_idx)).sort_index().ffill().reindex(daily_idx)
    curve_daily.index.name = "date"
    return curve_daily


def ns_rate(beta0, beta1, beta2, lam, tau):
    basis = _ns_basis(np.atleast_1d(tau), lam)
    b = np.stack([beta0, beta1, beta2], axis=-1)
    if b.ndim == 1:
        return float(basis[0] @ b)
    return np.einsum("nk,nk->n", basis if basis.shape[0] == b.shape[0] else np.tile(basis, (b.shape[0], 1)), b)


def discount_factor(beta0, beta1, beta2, lam, tau):
    """연속복리 할인계수 exp(-r(tau)*tau)."""
    r = ns_rate(beta0, beta1, beta2, lam, tau)
    return np.exp(-r * np.atleast_1d(tau))


def _asset_market_features(item_dates, item_tickers, px_logret: dict, px_dates_ord: dict, px_close_cache: dict):
    """상품별 (180일 vol/corr, mom6m) 계산 - numpy 기반 루프."""
    n = len(item_dates)
    sig = np.full((n, 3), np.nan)
    rho_raw = np.full((n, 3), np.nan)  # 12,13,23 in ORIGINAL (unsorted) asset order
    mom = np.full((n, 3), np.nan)
    n_common = np.full(n, 0, dtype=int)

    for i in range(n):
        cutoff_ord = item_dates[i]
        tks = item_tickers[i]
        slices = []
        ok = True
        for t in tks:
            if t not in px_logret:
                ok = False
                break
            dates_ord = px_dates_ord[t]
            pos = np.searchsorted(dates_ord, cutoff_ord, side="left")
            if pos < 30:
                ok = False
                break
            slices.append((dates_ord[:pos], px_logret[t][:pos]))
        if not ok:
            continue

        # 공통 거래일 정렬(교집합) 후 최근 180개
        common = slices[0][0]
        for d, _ in slices[1:]:
            common = np.intersect1d(common, d, assume_unique=True)
        if len(common) < 30:
            continue
        common_win = common[-C.VOL_WINDOW_DAYS:]
        n_common[i] = len(common_win)

        rets = np.empty((len(common_win), 3))
        for k, (d, r) in enumerate(slices):
            idx = np.searchsorted(d, common_win)
            rets[:, k] = r[idx]

        vol = rets.std(axis=0, ddof=1) * np.sqrt(252)
        sig[i] = vol
        if len(common_win) >= 2:
            corr = np.corrcoef(rets.T)
            rho_raw[i] = [corr[0, 1], corr[0, 2], corr[1, 2]]

        mom_win = common[-C.MOM_WINDOW_DAYS:]
        if len(mom_win) >= 2:
            for k, (d, _) in enumerate(slices):
                dates_ord = px_dates_ord[tks[k]]
                start_pos = np.searchsorted(dates_ord, mom_win[0])
                end_pos = np.searchsorted(dates_ord, mom_win[-1])
                p_all = px_close_cache[tks[k]]
                if start_pos < len(p_all) and end_pos < len(p_all) and p_all[start_pos] > 0:
                    mom[i, k] = np.log(p_all[end_pos] / p_all[start_pos])

    return sig, rho_raw, mom, n_common


def build_product_market():
    master = pd.read_parquet(C.CACHE_DIR / "product_master.parquet")
    log.info(f"product_master 로드: {len(master)} rows")

    all_tickers = sorted({t for row in master["tickers"] for t in row})
    log.info(f"고유 티커 수: {len(all_tickers)}")

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    px_map = fetch_all_prices(all_tickers, today)
    log.info(f"가격 fetch 성공: {len(px_map)}/{len(all_tickers)} 티커")

    px_close_cache = {}
    px_logret = {}
    px_dates_ord = {}
    for t, s in px_map.items():
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="last")]
        px_close_cache[t] = s.to_numpy(dtype=float)
        logret = np.diff(np.log(s.to_numpy(dtype=float)))
        px_logret[t] = logret
        px_dates_ord[t] = np.array([d.toordinal() for d in s.index[1:]])  # logret[i] belongs to date[i+1]

    item_dates = np.array([d.toordinal() for d in master["ISU_DT"]])
    item_tickers = master["tickers"].tolist()

    sig, rho_raw, mom, n_common = _asset_market_features(item_dates, item_tickers, px_logret, px_dates_ord,
                                                          px_close_cache)

    n_missing = np.isnan(sig).any(axis=1).sum()
    log.info(f"180일 시장데이터 부족/미매핑으로 NaN: {n_missing} / {len(master)}")

    # 정렬 sig1<=sig2<=sig3, 대응 rho12/13/23 재배열
    order = np.argsort(sig, axis=1)  # nan-safe: nan은 뒤로 정렬되지 않을 수 있음 -> 이후 dropna 처리
    sig_sorted = np.take_along_axis(sig, order, axis=1)

    pair_idx = {(0, 1): 0, (1, 0): 0, (0, 2): 1, (2, 0): 1, (1, 2): 2, (2, 1): 2}
    rho12 = np.full(len(master), np.nan)
    rho13 = np.full(len(master), np.nan)
    rho23 = np.full(len(master), np.nan)
    for i in range(len(master)):
        if np.isnan(sig[i]).any():
            continue
        o = order[i]
        rho12[i] = rho_raw[i, pair_idx[(o[0], o[1])]]
        rho13[i] = rho_raw[i, pair_idx[(o[0], o[2])]]
        rho23[i] = rho_raw[i, pair_idx[(o[1], o[2])]]

    sig1, sig2, sig3 = sig_sorted[:, 0], sig_sorted[:, 1], sig_sorted[:, 2]
    sig_mean = np.nanmean(sig_sorted, axis=1)
    rho_mean = np.nanmean(np.stack([rho12, rho13, rho23], axis=1), axis=1)

    # sig_eff: 등가중 바스켓 변동성 (원 자산순서 기준 공분산 사용)
    sig_eff = np.full(len(master), np.nan)
    for i in range(len(master)):
        if np.isnan(sig[i]).any():
            continue
        s = sig[i]
        r = rho_raw[i]
        corr_m = np.array([[1, r[0], r[1]], [r[0], 1, r[2]], [r[1], r[2], 1]])
        cov = np.outer(s, s) * corr_m
        w = np.ones(3) / 3
        sig_eff[i] = np.sqrt(w @ cov @ w)

    mom6m = np.nanmean(mom, axis=1)

    market_df = master.copy()
    market_df["sig1"], market_df["sig2"], market_df["sig3"] = sig1, sig2, sig3
    market_df["rho12"], market_df["rho13"], market_df["rho23"] = rho12, rho13, rho23
    market_df["sig_mean"], market_df["rho"], market_df["sig_eff"] = sig_mean, rho_mean, sig_eff
    market_df["mom6m"] = mom6m
    market_df["n_common_days"] = n_common

    # KR NS 금리커브
    log.info("KR FRED 금리커브 fetch + Nelson-Siegel 적합 중...")
    curve_daily = build_ns_curve(master["ISU_DT"].min().strftime("%Y-%m-%d"),
                                  master["ISU_DT"].max().strftime("%Y-%m-%d"))
    curve_daily.to_parquet(C.CACHE_DIR / "kr_ns_curve.parquet")

    beta = curve_daily.reindex(market_df["ISU_DT"].values, method="ffill")
    market_df["ns_b0"] = beta["b0"].to_numpy()
    market_df["ns_b1"] = beta["b1"].to_numpy()
    market_df["ns_b2"] = beta["b2"].to_numpy()
    market_df["ns_lam"] = beta["lam"].to_numpy()

    nodes = np.array(C.NS_TENOR_NODES)
    basis_nodes = _ns_basis(nodes, market_df["ns_lam"].iloc[0])  # lam 고정이므로 basis 고정
    beta_mat = market_df[["ns_b0", "ns_b1", "ns_b2"]].to_numpy()
    u_mat = beta_mat @ basis_nodes.T
    for k in range(len(nodes)):
        market_df[f"u{k}"] = u_mat[:, k]

    market_df["r"] = ns_rate(market_df["ns_b0"].to_numpy(), market_df["ns_b1"].to_numpy(),
                              market_df["ns_b2"].to_numpy(), market_df["ns_lam"].to_numpy(),
                              market_df["tenor"].to_numpy())

    # cpn_spread: §2 이후 계산 (coupon - r)
    market_df["cpn_spread"] = market_df["coupon"] - market_df["r"]

    before = len(market_df)
    market_df = market_df.dropna(subset=["sig1", "sig2", "sig3", "rho12", "rho13", "rho23"]).reset_index(drop=True)
    log.info(f"시장데이터 결측 제거: {before} -> {len(market_df)}")

    market_df.to_parquet(C.CACHE_DIR / "product_market.parquet", index=False)
    log.info(f"saved -> {C.CACHE_DIR / 'product_market.parquet'} ({len(market_df)} rows)")
    return market_df


if __name__ == "__main__":
    build_product_market()
