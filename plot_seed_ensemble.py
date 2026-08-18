import pandas as pd
import matplotlib.pyplot as plt


comparison = pd.read_csv(
    "portfolio_data/seed_ensemble_timeseries.csv",
    parse_dates=["Date"]
)

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    comparison["Date"],
    comparison["SeedEnsemble"],
    label="Seed ensemble portfolio"
)

ax.plot(
    comparison["Date"],
    comparison["MarketCapBenchmark"],
    label="Market-cap-weighted benchmark"
)

ax.set_xlabel("Date")
ax.set_ylabel("Growth of £1")
ax.set_title(
    "Seed Ensemble Portfolio vs Market-Cap-Weighted Benchmark"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "portfolio_data/seed_ensemble_vs_benchmark.png",
    dpi=150
)

plt.show()