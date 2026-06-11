"""RRL portfolio replication package."""
from .data import ASSET_UNIVERSE, download_etf_returns, download_etf_prices, prepare_weekly_returns
from .objectives import (annualized_return, annualized_volatility, calmar_ratio, expected_max_drawdown,
                         max_drawdown, sharpe_ratio, sterling_ratio)
from .rrl import RRLModel
from .backtest import backtest_strategy, simulate_portfolio, strategy_metrics

__all__ = [
    "ASSET_UNIVERSE",
    "download_etf_returns",
    "download_etf_prices",
    "prepare_weekly_returns",
    "annualized_return",
    "annualized_volatility",
    "calmar_ratio",
    "expected_max_drawdown",
    "max_drawdown",
    "sharpe_ratio",
    "sterling_ratio",
    "RRLModel",
    "backtest_strategy",
    "simulate_portfolio",
    "strategy_metrics",
]
