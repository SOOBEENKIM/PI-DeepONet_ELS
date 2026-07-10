# PI-DeepONet_ELS — ELS fair-value fast pricing

Independent reimplementation of an ELS (Equity-Linked Securities) fast-pricing
pipeline: **raw DART data → Monte-Carlo theoretical value → DeepONet-Curve
surrogate (Stage-1) → margin model (Stage-2) → metrics / speed / case analysis.**

The core work lives in [`src/els_pricing/`](src/els_pricing/) — see
[`src/els_pricing/README.md`](src/els_pricing/README.md) for the full pipeline,
modelling decisions, and results. [`src/els_hedging/`](src/els_hedging/) is a
purpose-organized library skeleton for follow-on hedging experiments.

## Pipeline (`els_pricing`)

| step | module | output |
|---|---|---|
| [1] | `data_prep.py`       | product master (filter §2.2, schedule vectors, barrier) |
| [2] | `market_data.py`     | yfinance 180d vol/corr + FRED Nelson-Siegel curve |
| [3] | `mc_engine.py`       | worst-of stepdown autocall MC, 100k paths, daily KI |
| [4] | `deeponet.py`        | Stage-1 DeepONet surrogate of the MC value |
| [5] | `margin_mlp.py`      | Stage-2 `recent_margin` anchor + resid MLP (Final = MC + margin) |
| [8] | `metrics.py`         | %error / moneyness buckets / calibration / per-stage R² |
| [9] | `speed_benchmark.py` | MC vs DeepONet timing |
| [10]| `case_analysis.py`   | low-price-region error drivers + extreme cases |
| —   | `figures.py`         | `figures/*.png` |
| —   | `run_all.py`         | orchestrates [1]–[8] |

## Headline results (v2, OOS walk-forward, 15,844 products — all CPU)

- **Stage-1 DeepONet vs MC**: R² **0.953**, MAPE **0.30%**, Spearman **0.979** (MAE loss, the default).
- **Final = MC + margin vs FAIR**: R² **0.627**, MAPE 2.28% (the causal `recent_margin` anchor alone gives R² 0.67).
- **Speed**: daily-KI MC **6,682 ms/product** vs DeepONet **2.65 µs/product** → **~2.5M× faster**.
- **MC engine**: risk-neutral 3-asset worst-of GBM, **daily** KI monitoring, per-observation NS-curve discounting; validated against Black–Scholes (diff 8e-5).

Full numbers, cost-function benchmark, and low-price-region analysis are in
[`src/els_pricing/README.md`](src/els_pricing/README.md).

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"

# run the whole pipeline
python -m els_pricing.run_all --jobs 32

# or individual steps
python -m els_pricing.data_prep
python -m els_pricing.market_data
python -m els_pricing.mc_engine --n-paths 100000 --jobs 32
python -m els_pricing.deeponet --loss mae
python -m els_pricing.margin_mlp
python -m els_pricing.metrics
python -m els_pricing.speed_benchmark
python -m els_pricing.case_analysis
python -m els_pricing.figures
```

## Repository layout

```text
src/els_pricing/   # ELS fast-pricing pipeline (the core work)
src/els_hedging/   # library skeleton for follow-on hedging experiments
configs/           # experiment configs (YAML)
notebooks/         # exploratory / analysis notebooks
docs/              # reports and notes
requirements.txt
```

Data, parquet caches, generated `figures/`, and `outputs/` are not tracked
(see `.gitignore`).

## Docker (optional)

A CUDA image and helper scripts (`build_image.sh`, `create_container.sh`,
`enter_container.sh`) are provided for a reproducible container. The pipeline
itself runs on CPU. Set `PYTHONPATH=.../src` so `import els_pricing` /
`import els_hedging` resolve, and fill in `.env` (`WANDB_*`) if online logging
is needed.
