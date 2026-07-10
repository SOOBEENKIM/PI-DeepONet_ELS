"""[3] Monte-Carlo theoretical value: worst-of stepdown autocall  (Section 4).

Risk-neutral GBM simulated *at the observation dates only* (exact for an
autocall whose payoff depends on the underlyings only on observation dates):
for each interval (t_{j-1}, t_j] the log-return of the 3 assets is multivariate
normal with mean (r-0.5 sigma^2) dt and covariance Sigma dt. This removes the
per-day path loop (steps = M observations, not ~756 days) while staying exact
on the observation grid. Knock-in (v2) is monitored DAILY within each interval
(ki_mode="daily") — a path that touches worst<B on any day is knocked-in, which
is the realistic ELS behaviour and removes the MC>FAIR bias of European KI.
ki_mode="european" keeps the old maturity-only check.

price is per unit notional (S0=1). Cash flows are discounted with per-observation
NS term-structure factors (disc, Section 2) or exp(-r*t) as fallback.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C

STRK = [f"strk_{i}" for i in range(C.N_STRK)]
CPN = [f"cpn_{i}" for i in range(C.N_STRK)]
TEX = [f"texer_{i}" for i in range(C.N_STRK)]


def price_els(strk, cpn, texer, B, knock_in, sigma, corr, r,
              n_paths=100_000, seed=0, ki_mode="daily",
              steps_per_year=252, ki_grace_yrs=0.0, disc=None):
    """Monte-Carlo price of one worst-of stepdown autocall (notional=1).

    ki_mode="daily"     : continuous (daily) knock-in monitoring within each
                          observation interval (a path that touches worst<B any
                          day is knocked-in). This is the realistic ELS behaviour.
    ki_mode="european"  : KI checked only at maturity (old behaviour).
    disc                : optional per-observation discount factors (NS term
                          structure, Section 2). Falls back to exp(-r*t).
    ki_grace_yrs        : KI is monitored only after this time from issue.
    """
    strk = np.asarray(strk, float)
    cpn = np.asarray(cpn, float)
    texer = np.asarray(texer, float)
    sigma = np.asarray(sigma, float)
    M = len(strk)
    t_prev = np.concatenate([[0.0], texer[:-1]])
    dt = np.clip(texer - t_prev, 0.0, None)

    corr = np.asarray(corr, float)
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(corr)
        L = V @ np.diag(np.sqrt(np.clip(w, 1e-8, None)))

    rng = np.random.default_rng(seed)
    logS = np.zeros((n_paths, 3))
    price = np.full(n_paths, np.nan)
    alive = np.ones(n_paths, bool)
    ki_hit = np.zeros(n_paths, bool)

    for j in range(M):
        dtj = dt[j]
        if dtj > 0:
            if ki_mode == "daily":
                nsub = max(1, int(round(dtj * steps_per_year)))
                ddt = dtj / nsub
                sq = np.sqrt(ddt)
                drift = (r - 0.5 * sigma ** 2) * ddt
                monitor = bool(knock_in) and (texer[j] >= ki_grace_yrs)
                for _ in range(nsub):
                    z = rng.standard_normal((n_paths, 3)) @ L.T
                    logS += drift + sigma * sq * z
                    if monitor:
                        worst_now = np.exp(logS).min(axis=1)
                        ki_hit |= alive & (worst_now < B)
            else:  # european: single jump on the observation grid
                z = rng.standard_normal((n_paths, 3)) @ L.T
                logS += (r - 0.5 * sigma ** 2) * dtj + sigma * np.sqrt(dtj) * z

        worst = np.exp(logS).min(axis=1)
        df = float(disc[j]) if disc is not None else np.exp(-r * texer[j])
        if j < M - 1:
            call = alive & (worst >= strk[j])
            price[call] = (1.0 + cpn[j]) * df
            alive &= ~call
        else:  # maturity
            call = alive & (worst >= strk[j])
            price[call] = (1.0 + cpn[j]) * df
            rest = alive & ~call
            if knock_in:
                breach = rest & (ki_hit if ki_mode == "daily" else (worst < B))
                price[breach] = worst[breach] * df
                price[rest & ~breach] = 1.0 * df
            else:
                price[rest] = worst[rest] * df
    return float(np.nanmean(price))


# ---------------------------------------------------------------------------
# BS sanity check (Section 4.2): 1-asset, no-KI, single European digital-ish
# ---------------------------------------------------------------------------
def _bs_selftest():
    """A degenerate 1-observation product priced by MC vs closed form.

    Product: pays 1 at T if S_T >= K else S_T (no KI). Closed form:
      E[e^{-rT}(1{S>=K} + S 1{S<K})] with S lognormal, S0=1.
    """
    from scipy.stats import norm
    r, sig, T, K = 0.03, 0.2, 1.0, 0.9
    # closed form
    d = (np.log(1 / K) + (r - 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    P_above = norm.cdf(d)                        # P(S_T>=K)
    # E[S_T 1{S<K}] e^{-rT}: S_T = e^{(r-.5s^2)T+s sqrt(T) Z}
    d2 = (np.log(1 / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    E_S_below = np.exp(r * T) * norm.cdf(-d2)    # E[S_T 1{S<K}] (undiscounted, since e^{rT}*e^{-.5..})
    cf = np.exp(-r * T) * (P_above + E_S_below)
    # perfectly-correlated equal-vol triple -> worst == single asset -> matches BS
    mc = price_els([K], [0.0], [T], B=0.0, knock_in=0,
                   sigma=[sig, sig, sig], corr=np.ones((3, 3)), r=r,
                   n_paths=400_000, seed=1)
    print(f"[selftest] MC={mc:.5f}  BS={cf:.5f}  diff={abs(mc-cf):.5f}")
    assert abs(mc - cf) < 3e-3, "MC vs BS mismatch"


def _disc_and_grace(row, tex):
    """Per-observation NS discount factors (Section 2) and KI grace (years)."""
    disc = []
    for k, tk in enumerate(tex):
        dk = row.get(f"disc_{k}") if k < C.N_STRK else None
        if dk is not None and np.isfinite(dk):
            disc.append(float(dk))
        else:
            disc.append(float(np.exp(-row["r"] * tk)))
    try:
        kg = float(row.get("kigrc", 0.0))
        grace = kg / 365.25 if np.isfinite(kg) else 0.0
    except (TypeError, ValueError):
        grace = 0.0
    return disc, grace


def _price_one(row, n_paths, seed, ki_mode, steps_per_year):
    strk = list(row["strk_all"]); cpn = list(row["cpn_all"]); tex = list(row["texer_all"])
    corr = np.array([[1, row["rho12"], row["rho13"]],
                     [row["rho12"], 1, row["rho23"]],
                     [row["rho13"], row["rho23"], 1]], float)
    sigma = [row["sig1"], row["sig2"], row["sig3"]]
    disc, grace = _disc_and_grace(row, tex)
    return price_els(strk, cpn, tex, row["B"], int(row["knock_in"]), sigma, corr,
                     row["r"], n_paths=n_paths, seed=seed, ki_mode=ki_mode,
                     steps_per_year=steps_per_year, ki_grace_yrs=grace, disc=disc)


def _price_chunk(args):
    records, n_paths, seed, ki_mode, steps_per_year = args
    out = np.empty(len(records))
    for k, row in enumerate(records):
        out[k] = _price_one(row, n_paths, seed + k, ki_mode, steps_per_year)
    return out


def run_batch_parallel(df, n_paths=None, seed=0, jobs=16, limit=None,
                       ki_mode="daily", steps_per_year=None) -> pd.DataFrame:
    import multiprocessing as mp
    n_paths = n_paths or C.MC["n_paths"]
    steps_per_year = steps_per_year or C.MC["steps_per_year"]
    if limit:
        df = df.head(limit).copy()
    records = df.to_dict("records")
    chunks = [records[i::jobs] for i in range(jobs)]  # round-robin balances tenor mix
    args = [(chunks[i], n_paths, seed + i * 100_000, ki_mode, steps_per_year) for i in range(jobs)]
    t0 = time.time()
    with mp.Pool(jobs) as pool:
        results = pool.map(_price_chunk, args)
    prices = np.empty(len(records))
    for i in range(jobs):
        prices[i::jobs] = results[i]
    out = df.copy()
    out["mc"] = prices
    dt = time.time() - t0
    print(f"[mc_engine] priced {len(df):,} products in {dt:.1f}s "
          f"({jobs} jobs, {n_paths:,} paths, ki_mode={ki_mode}, "
          f"{dt*jobs/len(df):.3f} s/product-core)")
    return out


def run_batch(df, n_paths=None, seed=0, limit=None, ki_mode="daily",
              steps_per_year=None) -> pd.DataFrame:
    n_paths = n_paths or C.MC["n_paths"]
    steps_per_year = steps_per_year or C.MC["steps_per_year"]
    if limit:
        df = df.head(limit).copy()
    prices = np.empty(len(df))
    t0 = time.time()
    per_ms = []
    for k, (_, row) in enumerate(df.iterrows()):
        t1 = time.time()
        prices[k] = _price_one(row, n_paths, seed, ki_mode, steps_per_year)
        per_ms.append((time.time() - t1) * 1000)
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(df)}  avg {np.mean(per_ms):.1f} ms/product")
    out = df.copy()
    out["mc"] = prices
    out["mc_ms"] = per_ms
    print(f"[mc_engine] priced {len(df):,} products in {time.time()-t0:.1f}s "
          f"({np.mean(per_ms):.1f} ms/product, {n_paths:,} paths, ki_mode={ki_mode})")
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=C.MC["n_paths"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--ki-mode", default="daily", choices=["daily", "european"])
    ap.add_argument("--steps-per-year", type=int, default=C.MC["steps_per_year"])
    ap.add_argument("--out", default=None, help="override output parquet path")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    _bs_selftest()
    if args.selftest:
        return
    df = pd.read_parquet(C.MARKET_MASTER)
    kw = dict(n_paths=args.n_paths, ki_mode=args.ki_mode, steps_per_year=args.steps_per_year)
    if args.jobs > 1:
        out = run_batch_parallel(df, jobs=args.jobs, limit=args.limit, **kw)
    else:
        out = run_batch(df, limit=args.limit, **kw)
    dest = args.out or C.MC_MASTER
    out.to_parquet(dest)
    print(f"[mc_engine] -> {dest}")
    d = out["fair"] - out["mc"]
    print(f"  mc: mean={out['mc'].mean():.4f}  fair-mc: mean={d.mean():.4f} std={d.std():.4f}")


if __name__ == "__main__":
    main()
