from __future__ import annotations

import numpy as np
import pandas as pd

from .objectives import annualized_return, annualized_volatility, max_drawdown, sharpe_ratio
from .rrl import RRLModel


def strategy_metrics(returns: pd.Series, periods_per_year: int = 52) -> dict[str, float]:
    returns = returns.dropna()
    wealth = np.exp(np.cumsum(returns))
    max_dd = max_drawdown(returns)
    ann_return = annualized_return(returns, periods_per_year=periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year=periods_per_year)
    sharpe = sharpe_ratio(returns) * np.sqrt(periods_per_year)
    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "total_return": float(wealth.iloc[-1] - 1.0) if len(wealth) else 0.0,
    }


def simulate_portfolio(
    model,
    training_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    stop_loss_n: float | None = None,
    retrain_on_stop: bool = False,
    retrain_epochs: int = 10,
    retrain_updates: int | None = None,
) -> pd.DataFrame:
    full_returns = pd.concat([training_returns, test_returns])
    current_model = model
    current_model.train(training_returns, epochs=retrain_epochs, max_updates=retrain_updates, verbose=False)
    positions_prev = np.zeros(model.n_assets, dtype=float)
    prev_raw = np.zeros(model.n_assets, dtype=float)
    records = []
    returns_history: list[float] = []
    for t in range(len(training_returns), len(full_returns)):
        X = current_model._build_features(full_returns, t, prev_raw)
        raw = np.einsum("ij,ij->i", X, current_model.theta)
        signal = current_model._activation(raw)
        positions = current_model._compute_positions(signal)
        period_return = current_model._portfolio_return(
            positions_prev, positions, full_returns.iloc[t].values.astype(float)
        )
        returns_history.append(period_return)
        cumulative_return = np.exp(np.sum(returns_history)) - 1.0
        records.append(
            {
                "date": full_returns.index[t],
                "return": period_return,
                "cumulative_return": cumulative_return,
                **{f"pos_{name}": positions[i] for i, name in enumerate(model.asset_names)},
            }
        )
        if stop_loss_n is not None and len(returns_history) > 1:
            cumret = np.sum(returns_history[:-1])
            vol = np.std(np.asarray(returns_history[:-1]), ddof=0)
            if vol > 0 and cumret / vol <= -stop_loss_n:
                if retrain_on_stop:
                    current_model = RRLModel(
                        asset_names=model.asset_names,
                        M=model.M,
                        mu=model.mu,
                        delta=model.delta,
                        learning_rate=model.learning_rate,
                        objective=model.objective,
                        variable_weight=model.variable_weight,
                    )
                    retrain_returns = full_returns.iloc[: t + 1]
                    current_model.train(retrain_returns, epochs=retrain_epochs, max_updates=retrain_updates, verbose=False)
                positions_prev = np.zeros(model.n_assets, dtype=float)
                prev_raw = np.zeros(model.n_assets, dtype=float)
                continue
        positions_prev = positions
        prev_raw = raw
    result = pd.DataFrame(records).set_index("date")
    return result


def backtest_strategy(returns: pd.DataFrame, positions: pd.DataFrame, delta: float, mu: float) -> pd.Series:
    prev_positions = np.zeros(positions.shape[1], dtype=float)
    returns_out = []
    for date, pos in positions.iterrows():
        asset_returns = returns.loc[date].values.astype(float)
        changes = np.abs(pos.values - prev_positions)
        portfolio_return = mu * (np.dot(prev_positions, asset_returns) - delta * np.sum(changes))
        returns_out.append(portfolio_return)
        prev_positions = pos.values
    return pd.Series(returns_out, index=positions.index)
