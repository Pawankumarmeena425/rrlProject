import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from rrl_portfolio.backtest import simulate_portfolio
from rrl_portfolio.data import download_etf_returns, ASSET_UNIVERSE
from rrl_portfolio.rrl import RRLModel


def make_all_presentation_graphs() -> None:
    output_dir = os.path.join(ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    returns = download_etf_returns(ASSET_UNIVERSE, "2011-01-01", "2015-12-31")
    train = returns.loc["2011-01-01":"2013-12-31"]
    test = returns.loc["2014-01-01":"2015-12-31"]

    perf_df = pd.read_csv(os.path.join(output_dir, "performance_summary.csv"))

    # Graph 1: Main result - cumulative returns comparison
    print("Creating Graph 1: Cumulative Returns Comparison...")
    scenarios = [
        {"objective": "sharpe", "variable_weight": False, "label": "Sharpe EW", "color": "tab:blue"},
        {"objective": "sharpe", "variable_weight": True, "label": "Sharpe VW", "color": "tab:cyan"},
        {"objective": "calmar", "variable_weight": False, "label": "Calmar EW", "color": "tab:green"},
        {"objective": "calmar", "variable_weight": True, "label": "Calmar VW", "color": "tab:olive"},
    ]

    plt.figure(figsize=(13, 7))
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
        plt.plot(result.index, result["cumulative_return"], label=scenario["label"], color=scenario["color"], linewidth=2.5, marker="o", markersize=3, markevery=5)

    plt.title("RRL Portfolio Strategies: Cumulative Return (2014-2015)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ppt_1_cumulative_returns.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Graph 1 saved")

    # Graph 2: Risk-Return scatter (Sharpe vs Annual Return vs Volatility)
    print("Creating Graph 2: Risk-Return Tradeoff...")
    plt.figure(figsize=(13, 7))
    colors_obj = {"sharpe": "tab:blue", "sterling": "tab:orange", "calmar": "tab:green"}
    markers = {False: "o", True: "s"}
    
    for _, row in perf_df[perf_df["delta"] == 0.0].iterrows():
        obj = row["objective"]
        vw = row["variable_weight"]
        plt.scatter(
            row["annualized_volatility"],
            row["annualized_return"],
            s=300,
            color=colors_obj.get(obj, "gray"),
            marker=markers.get(vw, "o"),
            alpha=0.7,
            edgecolors="black",
            linewidth=1.5,
            label=f"{obj.capitalize()} {'VW' if vw else 'EW'}" if (obj, vw) not in [(prev_obj, prev_vw) for prev_obj, prev_vw in []] else ""
        )
    
    plt.xlabel("Annualized Volatility", fontsize=12)
    plt.ylabel("Annualized Return", fontsize=12)
    plt.title("Risk-Return Tradeoff (δ=0 bps)", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ppt_2_risk_return.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Graph 2 saved")

    # Graph 3: Transaction cost sensitivity (Sharpe objective)
    print("Creating Graph 3: Transaction Cost Sensitivity...")
    plt.figure(figsize=(13, 7))
    sharpe_data = perf_df[perf_df["objective"] == "sharpe"]
    ew_data = sharpe_data[sharpe_data["variable_weight"] == False].sort_values("delta")
    vw_data = sharpe_data[sharpe_data["variable_weight"] == True].sort_values("delta")
    
    plt.plot(ew_data["delta"] * 10000, ew_data["total_return"], marker="o", linewidth=2.5, markersize=8, label="Sharpe EW", color="tab:blue")
    plt.plot(vw_data["delta"] * 10000, vw_data["total_return"], marker="s", linewidth=2.5, markersize=8, label="Sharpe VW", color="tab:cyan")
    
    plt.xlabel("Transaction Cost (basis points)", fontsize=12)
    plt.ylabel("Total Return", fontsize=12)
    plt.title("Transaction Cost Sensitivity: Sharpe Objective", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=11, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ppt_3_txn_cost_sensitivity.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Graph 3 saved")

    # Graph 4: Objective function comparison (Sharpe ratio at zero cost)
    print("Creating Graph 4: Objective Function Comparison...")
    plt.figure(figsize=(13, 7))
    zero_cost = perf_df[perf_df["delta"] == 0.0]
    objectives = zero_cost["objective"].unique()
    x_pos = np.arange(len(objectives))
    
    ew_returns = [zero_cost[(zero_cost["objective"] == obj) & (zero_cost["variable_weight"] == False)]["total_return"].values[0] for obj in objectives]
    vw_returns = [zero_cost[(zero_cost["objective"] == obj) & (zero_cost["variable_weight"] == True)]["total_return"].values[0] for obj in objectives]
    
    width = 0.35
    plt.bar(x_pos - width/2, ew_returns, width, label="Equal-Weight", color="tab:blue", alpha=0.8, edgecolor="black")
    plt.bar(x_pos + width/2, vw_returns, width, label="Variable-Weight", color="tab:cyan", alpha=0.8, edgecolor="black")
    
    plt.xlabel("Objective Function", fontsize=12)
    plt.ylabel("Total Return (2014-2015)", fontsize=12)
    plt.title("Portfolio Performance by Objective Function", fontsize=14, fontweight="bold")
    plt.xticks(x_pos, [obj.capitalize() for obj in objectives], fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ppt_4_objective_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Graph 4 saved")

    # Graph 5: Maximum drawdown comparison
    print("Creating Graph 5: Maximum Drawdown Comparison...")
    plt.figure(figsize=(13, 7))
    zero_cost = perf_df[perf_df["delta"] == 0.0]
    objectives = zero_cost["objective"].unique()
    x_pos = np.arange(len(objectives))
    
    ew_dd = [zero_cost[(zero_cost["objective"] == obj) & (zero_cost["variable_weight"] == False)]["max_drawdown"].values[0] for obj in objectives]
    vw_dd = [zero_cost[(zero_cost["objective"] == obj) & (zero_cost["variable_weight"] == True)]["max_drawdown"].values[0] for obj in objectives]
    
    width = 0.35
    plt.bar(x_pos - width/2, ew_dd, width, label="Equal-Weight", color="tab:red", alpha=0.8, edgecolor="black")
    plt.bar(x_pos + width/2, vw_dd, width, label="Variable-Weight", color="tab:orange", alpha=0.8, edgecolor="black")
    
    plt.xlabel("Objective Function", fontsize=12)
    plt.ylabel("Maximum Drawdown", fontsize=12)
    plt.title("Risk Measure: Maximum Drawdown by Objective", fontsize=14, fontweight="bold")
    plt.xticks(x_pos, [obj.capitalize() for obj in objectives], fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ppt_5_max_drawdown.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Graph 5 saved")

    # Graph 6: Sharpe ratio comparison (best performers)
    print("Creating Graph 6: Sharpe Ratio Comparison...")
    plt.figure(figsize=(13, 7))
    zero_cost = perf_df[perf_df["delta"] == 0.0].sort_values("sharpe_ratio", ascending=False).head(8)
    labels = [f"{row['objective'].capitalize()} {'VW' if row['variable_weight'] else 'EW'}" for _, row in zero_cost.iterrows()]
    
    colors_palette = ["tab:blue", "tab:cyan", "tab:green", "tab:olive", "tab:orange", "tab:red", "tab:purple", "tab:brown"]
    plt.barh(labels, zero_cost["sharpe_ratio"], color=colors_palette[:len(labels)], alpha=0.8, edgecolor="black")
    
    plt.xlabel("Sharpe Ratio", fontsize=12)
    plt.title("Top 8 Strategies by Sharpe Ratio (δ=0)", fontsize=14, fontweight="bold")
    plt.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ppt_6_sharpe_ranking.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Graph 6 saved")

    print("\n" + "=" * 60)
    print("✅ All 6 presentation graphs created successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  1. ppt_1_cumulative_returns.png - Main result comparison")
    print("  2. ppt_2_risk_return.png - Risk-return tradeoff")
    print("  3. ppt_3_txn_cost_sensitivity.png - Transaction cost impact")
    print("  4. ppt_4_objective_comparison.png - Objective function comparison")
    print("  5. ppt_5_max_drawdown.png - Risk measure comparison")
    print("  6. ppt_6_sharpe_ranking.png - Top strategies ranking")


if __name__ == "__main__":
    make_all_presentation_graphs()
