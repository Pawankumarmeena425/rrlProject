import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import pandas as pd

from rrl_portfolio.backtest import simulate_portfolio
from rrl_portfolio.data import download_etf_returns, ASSET_UNIVERSE
from rrl_portfolio.rrl import RRLModel


def make_key_result_plot() -> None:
    output_dir = os.path.join(ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    returns = download_etf_returns(ASSET_UNIVERSE, "2011-01-01", "2015-12-31")
    train = returns.loc["2011-01-01":"2013-12-31"]
    test = returns.loc["2014-01-01":"2015-12-31"]

    scenarios = [
        {"objective": "sharpe", "variable_weight": False, "label": "Sharpe EW", "color": "tab:blue"},
        {"objective": "sharpe", "variable_weight": True, "label": "Sharpe VW", "color": "tab:orange"},
        {"objective": "calmar", "variable_weight": False, "label": "Calmar EW", "color": "tab:green"},
    ]

    plt.figure(figsize=(12, 7))
    for scenario in scenarios:
        model = RRLModel(
            asset_names=ASSET_UNIVERSE,
            M=104,
            mu=100.0,
            delta=0.0,
            learning_rate=1e-4,
            objective=scenario["objective"],
            variable_weight=scenario["variable_weight"],
        )
        result = simulate_portfolio(
            model=model,
            training_returns=train,
            test_returns=test,
            stop_loss_n=None,
            retrain_on_stop=False,
            retrain_epochs=20,
            retrain_updates=2000,
        )
        plt.plot(result.index, result["cumulative_return"], label=scenario["label"], color=scenario["color"], linewidth=2)

    benchmark = test.mean(axis=1)
    benchmark_cum = (benchmark + 1.0).cumprod() - 1.0
    plt.plot(benchmark_cum.index, benchmark_cum.values, label="Equal-weight benchmark", color="tab:red", linestyle="--", linewidth=2)

    plt.title("Key Replication Result: Cumulative Return Comparison")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "key_result_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved key result plot to {fig_path}")


if __name__ == "__main__":
    make_key_result_plot()
