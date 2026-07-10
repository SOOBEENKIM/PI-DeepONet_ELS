"""[2] Market data: underlying prices (yfinance) + KR rate curve (FRED)  (Section 3).

Produces, per product (as of its issue date):
  sig1<=sig2<=sig3      sorted 180d annualised vols of the 3 underlyings
  rho12,rho13,rho23     pairwise correlations (in sorted order)
  sig_eff               correlation-aware equal-weight basket vol (branch feature)
  u0..u9                Nelson-Siegel yield at NS_NODES
  r                     NS yield at the product tenor (risk-free / discount)
  mom6m                 mean 126d log-return of the underlyings
  vol_source            'yahoo' if all 3 price histories fetched, else 'fallback'

Robustness: Yahoo may be rate-limited (HTTP 429) in some environments. Missing
price history falls back to asset-class constant vol/corr, clearly flagged by
`vol_source`, so the pipeline still runs end-to-end. Swap in real prices by
re-running with prices reachable.
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request

import numpy as np
import pandas as pd

from . import config as C

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Ticker resolution
# ---------------------------------------------------------------------------
def resolve_ticker(name: str):
    """Return (ticker, asset_class) or (None, None)."""
    if not isinstance(name, str):
        return None, None
    raw = name.strip()
    n = raw.replace("지수", "").replace("레버리지", "").strip()
    up = n.upper()
    for k, v in C.INDEX_TICKER.items():
        ku = k.upper()
        if ku == up or ku in up:
            return v, "index"
    if raw in C.STOCK_TICKER:
        return C.STOCK_TICKER[raw], "stock"
    if n in C.STOCK_TICKER:
        return C.STOCK_TICKER[n], "stock"
    return None, None


# ---------------------------------------------------------------------------
# Price fetch (yfinance) with on-disk cache
# ---------------------------------------------------------------------------
def _px_cache_path(ticker: str):
    safe = ticker.replace("^", "_").replace(".", "_")
    return C.CACHE_DIR / f"px_{safe}.parquet"


def fetch_px(ticker: str, start: str = "2008-01-01"):
    """Return a daily close price Series (cached), or None on failure."""
    p = _px_cache_path(ticker)
    if p.exists():
        try:
            return pd.read_parquet(p)["close"]
        except Exception:
            pass
    try:
        import yfinance as yf

        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            return None
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.dropna()
        s.name = "close"
        s.to_frame().to_parquet(p)
        return s
    except Exception as e:  # noqa: BLE001
        print(f"  [px] {ticker} FAILED: {repr(e)[:100]}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# FRED rate curve
# ---------------------------------------------------------------------------
def fetch_fred(series_id: str):
    p = C.CACHE_DIR / f"fred_{series_id}.parquet"
    if p.exists():
        return pd.read_parquet(p)["value"]
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read().decode()
        df = pd.read_csv(io.StringIO(raw))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().set_index("date")
        df.to_parquet(p)
        return df["value"]
    except Exception as e:  # noqa: BLE001
        print(f"  [fred] {series_id} FAILED: {repr(e)[:100]}", file=sys.stderr)
        return None


def _ns_basis(t, lam):
    t = np.asarray(t, float)
    x = t / lam
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(x > 1e-8, (1 - np.exp(-x)) / x, 1.0)
    load1 = term
    load2 = term - np.exp(-x)
    return np.stack([np.ones_like(t), load1, load2], axis=-1)  # (...,3)


def ns_fit(tenors, rates, lam=C.NS_LAMBDA):
    """Fit NS betas from (few) tenor/rate points; exact/LS solve."""
    A = _ns_basis(tenors, lam)              # (m,3)
    beta, *_ = np.linalg.lstsq(A, np.asarray(rates, float), rcond=None)
    return beta


def ns_eval(beta, tenors, lam=C.NS_LAMBDA):
    return _ns_basis(tenors, lam) @ beta


def build_rate_curves():
    """Daily NS beta curve from the 3 FRED KR series (forward-filled)."""
    series = {}
    for key, sid in C.FRED_SERIES.items():
        s = fetch_fred(sid)
        if s is not None:
            series[key] = s / 100.0  # percent -> decimal
    if len(series) < 2:
        print("[market_data] WARNING: <2 FRED series; using flat 3% curve")
        idx = pd.date_range("2008-01-01", "2027-01-01", freq="D")
        beta = pd.DataFrame({"b0": 0.03, "b1": 0.0, "b2": 0.0}, index=idx)
        return beta
    wide = pd.DataFrame(series).sort_index()
    wide = wide.resample("D").ffill().ffill()
    ten = [C.FRED_TENORS[k] for k in wide.columns]
    betas = np.array([ns_fit(ten, row.values) for _, row in wide.iterrows()])
    return pd.DataFrame(betas, index=wide.index, columns=["b0", "b1", "b2"])


# ---------------------------------------------------------------------------
# Per-product assembly
# ---------------------------------------------------------------------------
def _vol_corr(px_list, d0, window, fallback_classes):
    """Return (sigmas[3], corr[3,3], mom6m, all_available)."""
    rets = []
    avail = []
    for px, cls in zip(px_list, fallback_classes):
        if px is not None:
            h = px.loc[:d0].tail(window + 1)
            if len(h) >= 30:
                rets.append(np.log(h).diff().dropna())
                avail.append(True)
                continue
        rets.append(None)
        avail.append(False)

    sig = np.zeros(3)
    mom = []
    for i, (r, cls) in enumerate(zip(rets, fallback_classes)):
        if r is not None:
            sig[i] = r.std() * np.sqrt(TRADING_DAYS)
            mom.append(r.tail(C.MC["mom_window"]).sum())
        else:
            sig[i] = C.FALLBACK_VOL[cls]
    corr = np.eye(3)
    if all(avail):
        aligned = pd.concat(rets, axis=1, join="inner").dropna()
        if len(aligned) >= 30:
            corr = np.clip(aligned.corr().to_numpy(), -0.99, 0.99)
            np.fill_diagonal(corr, 1.0)
        else:
            corr = _fallback_corr()
    else:
        corr = _fallback_corr()
    mom6m = float(np.mean(mom)) if mom else 0.0
    return sig, corr, mom6m, all(avail)


def _fallback_corr():
    c = np.full((3, 3), C.FALLBACK_CORR)
    np.fill_diagonal(c, 1.0)
    return c


def _sort_by_vol(sig, corr):
    order = np.argsort(sig)
    s = sig[order]
    c = corr[np.ix_(order, order)]
    return s, c


def build(vol_source: str = "auto") -> pd.DataFrame:
    pm = pd.read_parquet(C.PRODUCT_MASTER)
    pm["isu_dt"] = pd.to_datetime(pm["isu_dt"])
    print(f"[market_data] products: {len(pm):,}")

    # 1) resolve tickers for every underlying, fetch prices once
    name_to_tc = {}
    for names in pm["udly_names"]:
        for nm in (list(names) if names is not None else []):
            if nm not in name_to_tc:
                name_to_tc[nm] = resolve_ticker(nm)
    unresolved = sorted(n for n, (t, _) in name_to_tc.items() if t is None)
    if unresolved:
        print(f"[market_data] {len(unresolved)} underlyings unresolved -> products dropped. e.g. {unresolved[:8]}")

    px_cache = {}
    if vol_source != "fallback":
        tickers = sorted({t for t, _ in name_to_tc.values() if t})
        print(f"[market_data] fetching {len(tickers)} tickers via yfinance ...")
        for t in tickers:
            px_cache[t] = fetch_px(t)
        n_ok = sum(v is not None for v in px_cache.values())
        print(f"[market_data] price series available: {n_ok}/{len(tickers)}")
        if n_ok == 0 and vol_source == "auto":
            print("[market_data] NOTE: no prices fetched (yahoo unreachable?) -> full fallback vol/corr.")

    # 2) rate curves
    print("[market_data] building FRED NS rate curves ...")
    betas = build_rate_curves()
    bidx = betas.index

    def curve_at(d0):
        i = bidx.searchsorted(pd.Timestamp(d0), side="right") - 1
        i = max(i, 0)
        return betas.iloc[i].to_numpy()

    # 3) per-product features
    rows = []
    nodes = np.array(C.NS_NODES)
    for _, row in pm.iterrows():
        names = list(row["udly_names"]) if row["udly_names"] is not None else []
        if len(names) < 3 or any(name_to_tc.get(nm, (None, None))[0] is None for nm in names[:3]):
            continue
        classes = [name_to_tc[nm][1] for nm in names[:3]]
        pxs = [px_cache.get(name_to_tc[nm][0]) for nm in names[:3]]
        sig, corr, mom6m, all_ok = _vol_corr(pxs, row["isu_dt"], C.MC["vol_window"], classes)
        sig, corr = _sort_by_vol(sig, corr)
        beta = curve_at(row["isu_dt"])
        us = ns_eval(beta, nodes)
        r = float(ns_eval(beta, np.array([row["tenor"]]))[0])
        sig_eff = float(np.sqrt(np.clip((np.outer(sig, sig) * corr).sum() / 9.0, 1e-8, None)))
        rec = {
            "item": row["item"],
            "sig1": sig[0], "sig2": sig[1], "sig3": sig[2],
            "rho12": corr[0, 1], "rho13": corr[0, 2], "rho23": corr[1, 2],
            "sig_eff": sig_eff, "mom6m": mom6m, "r": r,
            "vol_source": "yahoo" if all_ok else "fallback",
        }
        for i in range(len(nodes)):
            rec[f"u{i}"] = float(us[i])
        # Section 2: per-observation NS discount factors exp(-z(t_k) * t_k)
        for k in range(C.N_STRK):
            tk = row.get(f"texer_{k}", np.nan)
            if np.isfinite(tk) and tk > 0:
                zk = float(ns_eval(beta, np.array([tk]))[0])
                rec[f"disc_{k}"] = float(np.exp(-zk * tk))
            else:
                rec[f"disc_{k}"] = 1.0
        rows.append(rec)

    mkt = pd.DataFrame(rows)
    out = pm.merge(mkt, on="item", how="inner")
    out["cpn_spread"] = out["coupon"] - out["r"]
    print(f"[market_data] products with market data: {len(out):,}")
    print("  vol_source:", out["vol_source"].value_counts().to_dict())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol-source", choices=["auto", "yahoo", "fallback"], default="auto")
    args = ap.parse_args()
    out = build(vol_source=args.vol_source)
    out.to_parquet(C.MARKET_MASTER)
    print(f"[market_data] -> {C.MARKET_MASTER}")
    cols = ["item", "sig1", "sig2", "sig3", "rho12", "sig_eff", "r", "vol_source"]
    print(out[cols].head())


if __name__ == "__main__":
    main()
