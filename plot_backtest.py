import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("portfolio_data/backtest_timeseries.csv", parse_dates=["Date"])

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(df["Date"], df["OptimisedPortfolio"], label="Optimised portfolio")
ax.plot(df["Date"], df["MarketCapBenchmark"], label="Market-cap-weighted benchmark")
ax.plot(df["Date"], df["FTSE100"], label="FTSE 100 index")

ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)

ax.set_ylabel("Growth of £1 (rebased to backtest start)")
ax.set_title("Backtest performance: all portfolios from the same start point")
ax.legend()
fig.tight_layout()

fig.savefig("portfolio_data/backtest_performance.png", dpi=150)
print("Saved chart to portfolio_data/backtest_performance.png")