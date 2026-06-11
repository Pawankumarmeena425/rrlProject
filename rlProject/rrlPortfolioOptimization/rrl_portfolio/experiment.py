from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .backtest import backtest_strategy, simulate_portfolio, strategy_metrics
from .data import ASSET_UNIVERSE, download_etf_returns
from .rrl import RRLModel
from .objectives import annualized_return, annualized_volatility, max_drawdown, sharpe_ratio

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))


def ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def split_train_test(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = returns.loc["2011-01-01":"2013-12-31"]
    test = returns.loc["2014-01-01":"2015-12-31"]
    return train, test


def run_experiment(
    objective: str,
    variable_weight: bool,
    delta: float,
    stop_loss_n: float | None = None,
    retrain_on_stop: bool = False,
) -> pd.DataFrame:
    returns = download_etf_returns(ASSET_UNIVERSE, "2011-01-01", "2015-12-31")
    train_returns, test_returns = split_train_test(returns)
    model = RRLModel(
        asset_names=ASSET_UNIVERSE,
        M=104,
        mu=100.0,
        delta=delta,
        learning_rate=1e-4,
        objective=objective,
        variable_weight=variable_weight,
    )
    test_scores = simulate_portfolio(
        model=model,
        training_returns=train_returns,
        test_returns=test_returns,
        stop_loss_n=stop_loss_n,
        retrain_on_stop=False,
        retrain_epochs=20,
        retrain_updates=2000,
    )
    metrics = strategy_metrics(test_scores["return"])
    metrics.update({"objective": objective, "variable_weight": variable_weight, "delta": delta, "stop_loss_n": stop_loss_n})
    out_path = ensure_output_dir()
    plot_equity_curve(test_scores["cumulative_return"], objective, variable_weight, delta, stop_loss_n, out_path)
    if variable_weight:
        plot_position_heatmap(test_scores.filter(regex="^pos_"), objective, variable_weight, delta, stop_loss_n, out_path)
    return pd.DataFrame([metrics])


def plot_equity_curve(equity: pd.Series, objective: str, variable_weight: bool, delta: float, stop_loss_n: float | None, output_dir: str) -> None:
    suffix = f"{objective}_{'vw' if variable_weight else 'ew'}_d{int(delta*10000)}"
    if stop_loss_n is not None:
        suffix += f"_stop{stop_loss_n}"
    plt.figure(figsize=(10, 5))
    equity.plot()
    plt.title(f"Equity Curve ({objective.upper()} - {'VW' if variable_weight else 'EW'}) delta={delta}")
    plt.ylabel("Cumulative Return")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"equity_{suffix}.png"), dpi=200)
    plt.close()


def plot_position_heatmap(positions: pd.DataFrame, objective: str, variable_weight: bool, delta: float, stop_loss_n: float | None, output_dir: str) -> None:
    suffix = f"{objective}_vw_d{int(delta*10000)}"
    if stop_loss_n is not None:
        suffix += f"_stop{stop_loss_n}"
    plt.figure(figsize=(12, 6))
    sns.heatmap(positions.T, cmap="coolwarm", center=0)
    plt.title(f"Position Heatmap ({objective.upper()} VW, delta={delta})")
    plt.ylabel("Asset")
    plt.xlabel("Time step")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"positions_{suffix}.png"), dpi=200)
    plt.close()


def run_full_study() -> pd.DataFrame:
    results = []
    for objective in ["sharpe", "sterling", "calmar"]:
        for variable_weight in [False, True]:
            for delta in [0.0, 0.001, 0.0015, 0.002, 0.0025]:
                metrics = run_experiment(objective, variable_weight, delta)
                results.append(metrics)
    table = pd.concat(results, ignore_index=True)
    table.to_csv(os.path.join(ensure_output_dir(), "performance_summary.csv"), index=False)
    return table
