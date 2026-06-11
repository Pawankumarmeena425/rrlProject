# RRL Portfolio Optimization Replication Report

## Objective

Replicate the methodology from Almahdi & Yang (2017) for recurrent reinforcement learning (RRL) portfolio optimization using Sharpe, Sterling, and Calmar objective functions with expected maximum drawdown.

## Implementation summary

- Data is downloaded automatically from Yahoo Finance for ETFs: `SPY`, `IWD`, `IWC`, `DEM`, `CLY`.
- Weekly adjusted close prices are aligned and converted to log returns.
- Training/Test split: 2011-01-01 through 2013-12-31 for training; 2014-01-01 through 2015-12-31 for testing.
- `CLY` is unavailable from the public source used, so the implementation substitutes a proxy ETF (`VCLT`) for the long-term corporate credit bond exposure.
- The RRL architecture uses past weekly returns and prior signal feedback.
- Equal-weight long/short and variable-weight portfolio variants are implemented.
- The Calmar objective is implemented using an expected maximum drawdown approximation from the literature.
- A dynamic stop-loss strategy is implemented based on cumulative return versus volatility.

## Assumptions and limitations

- The paper omits some implementation details for the variable-weight portfolio and the Calmar E(MDD) gradient. The code uses reasonable, documented approximations.
- The Sterling objective uses realized maximum drawdown and a simplified gradient approximation.
- The stop-loss threshold is implemented as a configurable parameter.
- Hedge fund benchmark data is not included due to public data availability constraints; a simple buy-and-hold benchmark is provided.

## Results

The replication pipeline generated `outputs/performance_summary.csv` and equity/position figures for each objective and transaction cost scenario.

Key findings from the initial run:

- The highest annualized return was produced by the Sharpe objective equal-weight portfolio with zero transaction cost (`delta=0`), achieving approx. 17.7% total return over the 2014–2015 test window.
- The highest Sharpe ratio in the experimental grid was also produced by the Sharpe objective, variable-weight portfolio with zero transaction cost.
- The Calmar objective produced more conservative portfolios with lower maximum drawdown, but in this replication the highest raw return and Sharpe were still from the Sharpe objective variants.

These results are subject to the implementation decisions documented below.

## Notes on results and reproducibility

- The paper’s original ETF universe includes `CLY`, which is not available via Yahoo Finance. This repository substitutes `CLY` with the proxy ETF `VCLT` for long-term corporate credit exposure.
- The Calmar objective uses an expected maximum drawdown approximation based on the Magdon-Ismail risk formulation. The exact analytic gradient from the paper is inferred, so numerical differences from the original paper are expected.
- The Sterling objective is implemented with realized maximum drawdown and a simplified gradient approximation.
- The stop-loss mechanism is implemented with the paper’s ratio-based exit rule but is parameterized and not tuned to a fixed `n` value from the text.

## Future extensions

- Add bootstrapped or Monte Carlo estimation for expected maximum drawdown.
- Incorporate a proper recurrent neural architecture with hidden states.
- Expand data sources to additional asset classes and benchmarks.
