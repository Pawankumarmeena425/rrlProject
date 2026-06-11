# RRL Portfolio Optimization Replication

This repository contains a reproducible implementation of:

"An Adaptive Portfolio Trading System: A Risk-Return Portfolio Optimization Using Recurrent Reinforcement Learning with Expected Maximum Drawdown" (Almahdi & Yang, 2017).

## Contents

- `rrl_portfolio/`: package containing data acquisition, portfolio models, objectives, and backtesting utilities.
- `scripts/run_experiments.py`: entrypoint for experiments and figure generation.
- `requirements.txt`: Python dependencies.
- `research_report.md`: replication notes, assumptions, and findings.

## Setup

1. Create a Python environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the experiment pipeline:

```bash
python scripts/run_experiments.py
```

3. Outputs will be saved under `outputs/`.

## Notes

- The implementation downloads historical ETF data automatically from Yahoo Finance.
- The paper's weekly dataset uses SPY, IWD, IWC, DEM, and CLY from 2011-01-01 to 2015-12-31.
- `CLY` is delisted on Yahoo Finance, so the code substitutes a close proxy instrument (`VCLT`) and reports the proxy mapping transparently.
- The model implements Sharpe, Sterling, and Calmar objective variants and a dynamic stop-loss mechanism.

## Reproducibility

The code is modular and designed for extension. See `research_report.md` for methodology, assumptions, and limitations.
