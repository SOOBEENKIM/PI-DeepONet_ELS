# els_pricing — ELS fair-value fast pricing (independent reimplementation)

From-scratch implementation of the design in `ELS_독립재구현_지시서.md`:
raw DART data → MC theoretical value → DeepONet-Curve surrogate (Stage-1) →
margin MLP (Stage-2) → metrics / speed / case analysis.

## Pipeline / modules

| step | module | output |
|---|---|---|
| [1] | `data_prep.py`      | `data/cache/product_master.parquet` (filter §2.2, schedule vectors, barrier) |
| [2] | `market_data.py`    | `product_market.parquet` (yfinance 180d vol/corr + FRED Nelson-Siegel curve) |
| [3] | `mc_engine.py`      | `product_mc.parquet` (worst-of stepdown autocall MC, 100k paths) |
| [4] | `deeponet.py`       | `product_deeponet.parquet` (Stage-1 surrogate of the MC value) |
| [5] | `margin_mlp.py`     | `product_final.parquet` (recent_margin anchor + resid MLP; Final = MC + margin) |
| [8] | `metrics.py`        | %error / moneyness buckets / calibration / per-stage R² |
| [9] | `speed_benchmark.py`| MC vs DeepONet timing |
| [10]| `case_analysis.py`  | low-price-region error drivers + extreme cases |
| —   | `figures.py`        | `figures/*.png` |
| —   | `run_all.py`        | orchestrates [1]–[8] |

Shared: `config.py` (paths/filters/tickers), `features.py`, `splits.py`
(walk-forward folds, random train/valid).

## How to run (in the container)

```bash
# derived image with yfinance/pyarrow already installed:
docker run --rm -u "$(id -u):$(id -g)" \
  -e OMP_NUM_THREADS=1 -e PYTHONPATH=/w/src -e PROJECT_DIR=/w \
  -v "$PWD":/w -w /w els-hedging-2:full \
  /opt/conda/envs/els_hedging/bin/python -m els_pricing.run_all --jobs 32
```

Individual steps: `python -m els_pricing.data_prep`, `... market_data`,
`... mc_engine --n-paths 100000 --jobs 32`, `... deeponet --loss mse`
(or `--all-losses`), `... margin_mlp`, `... metrics`, `... speed_benchmark`,
`... case_analysis`, `... figures`.

## Key modelling decisions (real-data grounded)

- **Filter** (§2.2): PRODUCT_TYPE=ELS, worst-of (STCK_MTHD=ALL_MIN), 3-star
  (UDRL_CNT=3), STEP, KRW, CPN_YN=0, fair∈[0.70,1.05], tenor∈[0.5,5] →
  **40,226** products. Tenor from MAT_DT−ISU_DT (not DT_DIFF).
- **Barrier**: `BARR_1` is ~92% empty in the raw data, so the loss barrier `B`
  is taken from the maturity strike (Digital_Call_Put row) when BARR_1 is absent.
- **Coupons**: use `PMT_1` (cumulative payout per step), not just ANL_RTRN.
- **MC** (§4): risk-neutral 3-asset GBM. Observation dates are simulated exactly;
  within each interval KI is monitored **daily** (`ki_mode="daily"`, v2 §1) — the
  realistic ELS behaviour, chosen over European KI because it removes the MC>FAIR
  bias. `ki_mode="european"` keeps the fast observation-grid variant. Cash flows
  discounted with **per-observation NS term-structure factors** (v2 §2). Validated
  against Black–Scholes (diff 8e-5). Daily full set (50k paths, 44-core): ~75 min.
- **Market data**: yfinance actually fetched 27/28 index/stock tickers despite
  Yahoo rate-limiting; 39,503/39,611 products use real 180-day vol/corr, 108
  fall back to class-based vol (flagged by `vol_source`).

## Results (v2, OOS walk-forward 60→100%, 15,844 products) — all CPU

- **fair − MC**: **−0.0557** (daily KI) vs **−0.0611** (European KI). Continuous
  daily knock-in monitoring shrinks the gap ~9% (v2 §1); the residual gap is
  dominated by the first cause — implied vol not used. See
  `figures/fair_minus_mc.png` (before/after).
- **Cost-function benchmark** (v2 §4 — standardised target, same folds/seed):

  | loss | R²(mc) | MAPE | bp |
  |------|--------|------|----|
  | MSE  | 0.926  | 0.45%| 44 |
  | **MAE**  | **0.953** | **0.30%** | **29** |
  | MAPE | 0.793  | 0.80%| 77 |

  → With target standardisation + grad-clip, **MSE R² recovered 0.29→0.93**,
  confirming the doc's hypothesis that low MSE-R² was a *training* artifact, not
  a property of the loss. **MAE is still best** and is the default.
- **Stage-1 DeepONet vs MC (MAE)**: R² **0.953**, MAPE **0.30%**, bp 29,
  Spearman **0.979** (v1 was 0.74 — v2 §4 stabilisation is the big win).
- **Final = MC + margin vs FAIR**: R² **0.627**, MAPE **2.28%**, Spearman 0.80.
  The causal `recent_margin` anchor alone gives R² 0.67 / MAPE 2.12%; the surrogate
  is now essentially lossless — `final_fast` (DeepONet+margin) R² **0.625** ≈ Final.
- **Speed** (v2 §3, honest basis): daily-KI MC **6,682 ms/product** vs DeepONet
  **2.65 µs/product** → **~2.5M× faster** *[basis: daily MC]*. (v1's 4,400× was on
  the cheaper observation-grid MC — different engine, stated explicitly.)
- **Low-price region (§10)**: MAPE rises from ~0.17% (ATM) to ~0.71% (deep/low
  price); |error| correlates most with coupon, effective vol, σ₃, cpn_spread.

Discounting uses per-observation NS term-structure factors (v2 §2, `disc_*`).
See `figures/` for calibration, moneyness-error, MC-vs-DeepONet, fair−MC, speed.

## Stage-2 model comparison: MLP vs tree (`margin_gbr.py`)

`margin_gbr.py` is a drop-in alternative to `margin_mlp.py`: same
`compute_recent_margin`, `features.tabular_matrix`, and
`splits.walk_forward_folds` / `train_valid_split`, but Stage-2 residual
prediction uses sklearn `GradientBoostingRegressor` (huber loss) instead of
the MLP — no xgboost dependency. OOS, same folds:

| Stage-2 | R² | MAPE |
|---|---|---|
| MC only | -1.537 | 6.65% |
| MC + recent_margin (no model) | 0.674 | 2.12% |
| Final, MLP (`margin_mlp.py`) | 0.627 | 2.28% |
| Final, tree, 500 est. (fixed) | 0.749 | 1.83% |
| Final, tree, 2000 est. + validation-monitored early stop | 0.737 | 1.86% |

The tree residual model beats both the MLP and the `recent_margin`-only
baseline; the MLP does not (it lands *below* MC+recent, i.e. net negative
contribution — 3 of 4 folds have negative OOS resid R²). At 2000 estimators
with `patience=50` the validation huber loss never actually plateaus (all
4 folds still improving at the cap), and pushing past 500 trees makes fold 3
worse out-of-sample (resid R² **+0.083 → −0.043**) despite validation loss
still falling — a sign the random (non-time-ordered) `train_valid_split`
under-detects overfitting relative to the true walk-forward OOS fold. 500
estimators (itself not an early-stopped optimum, just a smaller cap) is the
better result of the two tree configs tried so far.
