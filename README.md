# ELS hedging_2

Workspace for ELS (Equity-Linked Securities) pricing, risk analysis, and hedging
experiments — v2, with a purpose-organized `src` package.

## Layout

```text
ELS hedging_2/
├── src/els_hedging/      # reusable library code, split by purpose
│   ├── data/             # data loading, cleaning, market-data adapters
│   ├── pricing/          # ELS payoff definitions & pricers (MC / PDE / closed-form)
│   ├── simulation/       # underlying-path simulation (GBM, Heston, local-vol, ...)
│   ├── hedging/          # hedging strategies (delta/gamma/vega, deep hedging)
│   ├── models/           # ML/DL models (e.g. deep-hedging networks)
│   ├── evaluation/       # backtests, PnL, risk metrics, reporting
│   ├── utils/            # config, IO, logging, seeding helpers
│   └── smoke_test.py     # environment sanity check
├── notebooks/            # exploratory / analysis notebooks
├── configs/              # experiment configs (YAML)
├── data/                 # raw / interim / processed data (ignored by Docker build)
│   ├── raw/              # immutable source data
│   ├── interim/          # intermediate transforms
│   └── processed/        # model-ready data
├── experiments/          # named experiment snapshots
├── figures/              # generated figures
├── outputs/              # generated outputs (models, tables, logs)
├── docs/                 # reports and notes
├── tests/                # lightweight checks
├── Dockerfile
├── requirements.txt
├── build_image.sh
├── create_container.sh
├── enter_container.sh
└── fix_permissions.sh
```

## Docker

```bash
./build_image.sh        # build the CUDA 11.1 image
./create_container.sh   # create + start the container (mounts this folder)
./enter_container.sh    # open a shell inside it
```

Defaults:

- Image: `soobeenkim-els-hedging-2:cu111`
- Container: `soobeenkim_els_hedging_2`
- Workspace in container: `/workspace/els_hedging_2`

`PYTHONPATH` is set to `.../src`, so `import els_hedging` works anywhere in the container.

## First Check

Inside the container:

```bash
python -m els_hedging.smoke_test
```

## W&B

Fill in `.env` before creating the container if online logging is needed.

```text
WANDB_PROJECT=els-hedging-2
WANDB_ENTITY=
WANDB_MODE=online
WANDB_API_KEY=
```
