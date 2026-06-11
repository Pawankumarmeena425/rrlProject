from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series | np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return 0.0
    mu = returns.mean()
    sigma = returns.std(ddof=0)
    if sigma <= 0:
        return 0.0
    return mu / sigma


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return 0.0
    wealth = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(wealth)
    drawdowns = (wealth - peak) / peak
    return abs(np.min(drawdowns))


def sterling_ratio(returns: pd.Series | np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    mdd = max_drawdown(returns)
    if mdd <= 0:
        return 0.0
    return returns.mean() / mdd


def expected_max_drawdown(mean_return: float, sigma: float, T: int) -> float:
    if sigma <= 0 or T <= 0:
        return 1e-8
    if mean_return > 0:
        sharpe = max(mean_return / sigma, 1e-8)
        return sigma * (0.63519 + 0.5 * np.log(T) + np.log(sharpe))
    if np.isclose(mean_return, 0.0):
        return 1.2533 * sigma * np.sqrt(T)
    return max(-mean_return * T, 1e-8)


def calmar_ratio(returns: pd.Series | np.ndarray, T: int | None = None) -> float:
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return 0.0
    if T is None:
        T = len(returns)
    mu = returns.mean()
    sigma = returns.std(ddof=0)
    mdd = expected_max_drawdown(mu, sigma, T)
    if mdd <= 0:
        return 0.0
    return mu / mdd


def annualized_return(returns: pd.Series | np.ndarray, periods_per_year: int = 52) -> float:
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return 0.0
    total_growth = np.exp(np.sum(returns))
    return total_growth ** (periods_per_year / len(returns)) - 1


def annualized_volatility(returns: pd.Series | np.ndarray, periods_per_year: int = 52) -> float:
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return 0.0
    return np.std(returns, ddof=0) * np.sqrt(periods_per_year)
