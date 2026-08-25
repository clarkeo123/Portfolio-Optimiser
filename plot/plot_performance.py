import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("portfolio_data/performance_timeseries.csv", parse_dates=["Date"])

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(df["Date"], df["OptimisedPortfolio"], label="Optimised portfolio")
ax.plot(df["Date"], df["MarketCapBenchmark"], label="Market-cap-weighted benchmark")
ax.plot(df["Date"], df["FTSE100"], label="FTSE 100 index")

boundary = df.loc[df["Period"] == "Backtest", "Date"].iloc[0]
ax.axvline(boundary, color="grey", linestyle="--", linewidth=1)
ax.text(boundary, ax.get_ylim()[1], "  training | backtest (out-of-sample)",
        va="top", fontsize=8, color="grey")

ax.set_ylabel("Growth of £1")
ax.set_title("Portfolio performance: training period vs. backtest")
ax.legend()
fig.tight_layout()
fig.savefig("portfolio_data/performance.png", dpi=150)
print("Saved chart to portfolio_data/performance.png")