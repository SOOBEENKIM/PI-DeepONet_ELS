# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Fast fair-value pricing surrogate for 3-star worst-of step-down autocallable ELS (Equity-Linked Securities) with
knock-in, built as an independent from-scratch reimplementation of a team pipeline described in internal Notion docs.
The core deliverable is a **theoretical-price (이론가) surrogate**: a PI-DeepONet that reproduces Monte Carlo fair
value at ~µs inference speed instead of ~seconds for full MC. A secondary MLP stage models the residual
issuer/market margin on top of that theoretical price. Per professor/team feedback, **Stage-1 (MC reproduction) is
the headline result; Stage-2/Final are secondary** — don't over-index on Final R² when discussing results.

Read these docs first when resuming work — they are the actual spec and current status, more authoritative than any
summary here:
- `ELS_hedging_V1_현황.md` — current status / where things left off (results, next steps)
- `V1_빌드_지시서.md` — original build spec (pipeline stages 0-8, module layout, filters, feature groups, targets)
- `V1_피드백반영_지시서.md` — feedback-driven revisions (bug fixes, Stage-2 XGBoost→MLP switch, %error framing)
- `V1_발표보완_지시서.md` — presentation-gap-filling spec (signed bias plots, scatter, batch-speed remeasure, tables)

## Environment setup

No `requirements.txt`/`pyproject.toml` exists — install manually per `V1_빌드_지시서.md` §부록:

```bash
pip install numpy pandas pyarrow matplotlib torch yfinance pandas-datareader scipy xgboost tqdm finance-datareader
```

Every run needs these env vars set first (Windows/OpenMP + console encoding):

```bash
set KMP_DUPLICATE_LIB_OK=TRUE & set PYTHONIOENCODING=utf-8
```

(bash equivalent: `export KMP_DUPLICATE_LIB_OK=TRUE; export PYTHONIOENCODING=utf-8` — see `run_feedback_pipeline.sh` /
`run_presentation_pipeline.sh` for the canonical form.)

There is no test suite and no lint config in this repo.

## Running the pipeline

All modules are run with `python -m els_hedging_v1.<module>` from the repo root. Full pipeline, in dependency order:

```bash
python -m els_hedging_v1.data_prep                              # [1] raw 3 CSV -> product_master.parquet
python -m els_hedging_v1.market_data                             # [2] 180d vol/corr + KR NS curve -> product_market.parquet
python -m els_hedging_v1.mc_engine --n-paths 100000 --jobs <N>   # [3] MC fair value -> product_mc.parquet (SLOW)
python -m els_hedging_v1.features                                # [4] recent_margin -> product_features.parquet
python -m els_hedging_v1.datasets                                 # [5] dataset_ml.parquet, dataset_deeponet.parquet
python -m els_hedging_v1.deeponet --all-losses                    # [6] Stage-1 DeepONet (MSE/MAE/MAPE bench)
python -m els_hedging_v1.margin_mlp                                # Stage-2 MLP (margin residual)
python -m els_hedging_v1.benchmark                                  # Stage-2 XGB (comparison only) + direct benches
python -m els_hedging_v1.run_all                                    # [7] walk-forward 4-fold eval, all models
python -m els_hedging_v1.metrics                                    # %error / moneyness / calibration tables
python -m els_hedging_v1.speed_benchmark                            # MC vs DeepONet batch-forward speed
python -m els_hedging_v1.case_analysis                              # low-price segment/driver analysis
python -m els_hedging_v1.eda                                        # filter cascade + EDA figures
python -m els_hedging_v1.figures                                    # the 4-6 team-comparable figures
```

**⚠️ Never re-run `mc_engine` unless explicitly asked.** It's the expensive step (10万 paths × full universe); the
convention is to reuse the cached `data/cache/product_mc.parquet` indefinitely. Two convenience scripts capture the
usual re-run subsets that assume MC output already exists:

- `run_feedback_pipeline.sh` — datasets → deeponet --all-losses → margin_mlp → metrics → speed_benchmark →
  case_analysis → figures (used after changing Stage-1/2 model code, not MC/data prep)
- `run_presentation_pipeline.sh` — metrics → speed_benchmark → figures only (used when only regenerating
  presentation output from existing predictions)

`mc_engine.py` supports `--n-paths`, `--jobs`, `--limit` (for quick smoke runs on a subset), `--out` (checkpoint
filename). It writes incremental checkpoints during long runs.

## Architecture

Pipeline stages, each module reading the previous stage's cached parquet and writing its own to `data/cache/` or
`data/out/`:

```
data/raw (3 DART CSVs)
  -> data_prep.py    [1] filter + schedule-vector parsing -> product_master.parquet
  -> market_data.py  [2] yfinance/FRED fetch, 180d vol/corr, KR Nelson-Siegel curve -> product_market.parquet
  -> mc_engine.py     [3] worst-of GBM Monte Carlo (10万 paths, daily KI watch) -> product_mc.parquet
  -> features.py      [4] recent_margin = causal 90-day mean(fair - MC) -> product_features.parquet
  -> datasets.py       [5] assemble ml/deeponet feature sets -> dataset_ml.parquet, dataset_deeponet.parquet
  -> deeponet.py         Stage-1: DeepONet-Curve (branch=1D-CNN over rate curve+vol/corr, trunk=contract terms)
  -> margin_mlp.py        Stage-2 (default): small MLP on residual = fair - MC - recent_margin
  -> benchmark.py          Stage-2 XGBoost (comparison only, not the reported model) + direct-regression benches
  -> run_all.py          [7] walk-forward 4-fold orchestration across all of the above -> final_comparison_table.csv
  -> metrics.py / speed_benchmark.py / case_analysis.py / eda.py / figures.py   downstream analysis & plots
```

`config.py` is the single source of truth for filters, column groups, and constants — read it before touching any
other module. Key pieces:
- `TICKER_MAP`: Korean underlying-asset name → yfinance ticker (with FDR fallback for `.KS`/`.KQ` via
  `fdr_fallback_code`)
- `BASE_NUM_COLS` / `REG_COLS` / `CAT_COLS`: tabular (Stage-2/direct) feature groups
- `BRANCH_COLS` / `TRUNK_COLS`: DeepONet branch (curve+vol/corr) vs trunk (contract terms) inputs
- `WF_CUM_FRACTIONS`: walk-forward cumulative train fractions `[0.60, 0.70, 0.80, 0.90, 1.00]`
- `TEAM_TARGETS`: the team's reference R²/Spearman bands used to judge "reproduction success"

### Frozen conventions (do not change without explicit instruction)

- Volatility/correlation: **180 trading-day historical**, not implied vol (company data has no IV).
- Discount/drift curve: **KR FRED rates (call/3M/10Y) fit with Nelson-Siegel**, since products are KRW-denominated.
- Filters (`data_prep.py`): `PRODUCT_TYPE=='ELS'`, worst-of, 3-star, `OPT_TYPE=='STEP'`, `CUR_CD=='KRW'`,
  `fair∈[0.70,1.05]`, `tenor∈[0.5,5]` (tenor = `MAT_DT - ISU_DT`, not `DT_DIFF`) — target ≈23,151-23,479 rows;
  don't force-fit the exact count, just log the discrepancy source.
- Stage-2 target is `resid = fair - MC - recent_margin`; Stage-2 inputs are BASE+CAT only (MC, recent_margin, and
  REG excluded from Stage-2 features to avoid leakage/circularity).
- Stage-2 model is **MLP** (`margin_mlp.py`), not XGBoost — switched per professor feedback because trees can't
  extrapolate. `benchmark.py`'s XGBoost Stage-2 is kept only as a comparison baseline, not the reported result.
- DeepONet train/valid split is **random shuffle**; the held-out **test fold stays chronological** (walk-forward,
  no look-ahead into the future).
- Evaluation is **%error-first** (MAPE, bp error, calibration slope/intercept), not just R² — this is a deliberate
  framing choice per feedback ("이론가 중심", show it doesn't produce nonsense prices), reported alongside
  per-stage R² (Stage-1 MC-reproduction / Stage-2 residual / Final) and Spearman rank correlation.

### Data directory

- `data/raw/` — original DART CSVs (gitignored, local only)
- `data/cache/` — all intermediate parquet (product_master/market/mc/features, dataset_ml/deeponet, px_*.parquet
  price cache) — gitignored, reused across runs, **never delete/regenerate `product_mc.parquet` casually**
- `data/out/` — CSV tables and `figures/` PNGs — these ARE tracked/pushed (along with code and the instruction docs)

### Git

- Remote: `github.com/SOOBEENKIM/PI-DeepONet_ELS`. Work happens on branch **`els-hedging-v1`**.
- `main` holds an unrelated prior project (`els_pricing` v2) — do not touch it from this branch's work.
- Only push code, instruction docs, figures, and CSV tables. Raw data and cache/parquet stay local
  (`.gitignore` excludes `data/raw/`, `data/cache/`, `*.parquet`).
