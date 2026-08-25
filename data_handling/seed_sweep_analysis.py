import pandas as pd
import numpy as np

import argparse
import os

parser = argparse.ArgumentParser(
    description="Analyse seed sweep"
)

df = pd.read_csv("portfolio_data/seed_sweep_results.csv")
backtest = pd.read_csv("portfolio_data/backtest.csv")
assets = pd.read_csv("portfolio_data/assets.csv")
metadata = pd.read_csv("portfolio_data/metadata.csv")
prices = pd.read_csv(
    "portfolio_data/backtest_prices.csv",
    parse_dates=["Date"]
)
backtest_prices = pd.read_csv(
    "portfolio_data/backtest_prices.csv",
    index_col="Date",
    parse_dates=True
)

def calculate_max_drawdown(value_series):
    """
    Calculate maximum drawdown from a cumulative value series.
    """

    running_max = value_series.cummax()

    drawdown = (
        value_series / running_max
    ) - 1.0

    return drawdown.min()

def calculate_portfolio_stats(
    weights,
    prices,
    cash_weight,
    total_cash_return,
    trading_days_per_year=252
):
    """
    Calculate annualised return, volatility and Sharpe ratio
    from a portfolio weight series and price history.
    """

    daily_returns = prices.pct_change().fillna(0.0)

    # Portfolio stock returns
    portfolio_stock_returns = (
        daily_returns
        .mul(weights, axis=1)
        .sum(axis=1)
    )

    # Convert total cash return into daily return
    daily_cash_return = (
        (1.0 + total_cash_return)
        ** (1.0 / max(len(prices) - 1, 1))
        - 1.0
    )

    portfolio_daily_returns = (
        portfolio_stock_returns
        + cash_weight * daily_cash_return
    )

    # Growth of £1
    value_series = (1.0 + portfolio_daily_returns).cumprod()

    # Annualised return
    years = (
        len(prices) - 1
    ) / trading_days_per_year

    annualised_return = (
        value_series.iloc[-1] ** (1.0 / years)
    ) - 1.0

    # Annualised volatility
    volatility = (
        portfolio_daily_returns.std(ddof=1)
        * np.sqrt(trading_days_per_year)
    )

    # Annualised risk-free rate
    annualised_risk_free = (
        (1.0 + total_cash_return) ** (1.0 / years)
    ) - 1.0

    sharpe = (
        (annualised_return - annualised_risk_free)
        / volatility
        if volatility > 0 else 0.0
    )

    max_drawdown = calculate_max_drawdown(value_series)

    return {
        "Return": annualised_return,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "ValueSeries": value_series,
        "DailyReturns": portfolio_daily_returns,
        "MaxDrawdown": max_drawdown
    }

parser.add_argument(
    "--quiet",
    action="store_true",
    help="Suppress detailed diagnostic output"
)

args = parser.parse_args()

# averages portfolio weights across all seeds
weight_columns = [
    column for column in df.columns
    if column.startswith("Weight_")
]

ensemble_weights = df[weight_columns].mean()

if not args.quiet:
    print("\nEnsemble portfolio weights")
    print("--------------------------")

    for column, weight in ensemble_weights.sort_values(ascending=False).items():
        print(f"{column.replace('Weight_', ''):>6}: {weight:8.4%}")

ensemble_net_exposure = ensemble_weights.sum()
ensemble_gross_exposure = ensemble_weights.abs().sum()
ensemble_cash_weight = 1.0 - ensemble_net_exposure

print(
    f"\nEnsemble gross exposure: "
    f"{ensemble_weights.abs().sum():.4f}"
)

print(
    f"Ensemble net exposure: "
    f"{ensemble_weights.sum():.4f}"
)

print(
    f"Ensemble cash weight: "
    f"{ensemble_cash_weight:.4f}"
)

# evaluates ensemble on the backtest data
ensemble_weight_values = ensemble_weights.to_numpy()

backtest_return = df["BacktestReturn"].to_numpy()
backtest_volatility = df["BacktestVolatility"].to_numpy()
backtest_sharpe = df["BacktestSharpe"].to_numpy()

print("\nIndividual seed comparison")
print("--------------------------")
print(f"Mean backtest return:       {backtest_return.mean():.4f}")
print(f"Mean backtest volatility:   {backtest_volatility.mean():.4f}")
print(f"Mean backtest Sharpe:       {backtest_sharpe.mean():.4f}")

backtest_returns = backtest.set_index("YFinanceTicker")["ForwardReturn"]

# converts portfolio tickers to their YFinance tickers
ticker_to_yfinance = dict(
    zip(assets["Ticker"], assets["YFinanceTicker"])
)

ensemble_yfinance_weights = pd.Series(
    {
        ticker_to_yfinance[ticker.replace("Weight_", "")]: weight
        for ticker, weight in ensemble_weights.items()
        if ticker.replace("Weight_", "") in ticker_to_yfinance
    }
)

# keeps only stocks available in both datasets
common_tickers = ensemble_yfinance_weights.index.intersection(
    backtest_returns.index
)

ensemble_yfinance_weights = ensemble_yfinance_weights.loc[common_tickers]
aligned_returns = backtest_returns.loc[common_tickers]

# ensemble vs market-cap benchmark statistics

# re-indexes weights so they align exactly with price columns
aligned_ensemble_weights = (
    ensemble_yfinance_weights
    .reindex(prices.columns.drop("Date"))
    .fillna(0.0)
)

ensemble_cash_weight = (
    1.0 - aligned_ensemble_weights.sum()
)

# market cap benchmark weights
benchmark_weights = (
    assets
    .set_index("YFinanceTicker")["MarketCapWeight"]
    .reindex(prices.columns.drop("Date"))
    .fillna(0.0)
)

benchmark_cash_weight = (
    1.0 - benchmark_weights.sum()
)

# extracts backtest risk-free return
backtest_risk_free_return = float(
    metadata.loc[
        metadata["Parameter"] == "BacktestRiskFreeReturn",
        "Value"
    ].iloc[0]
)

# calculates both portfolios using identical methodology
ensemble_stats = calculate_portfolio_stats(
    weights=aligned_ensemble_weights,
    prices=prices.drop(columns="Date"),
    cash_weight=ensemble_cash_weight,
    total_cash_return=backtest_risk_free_return
)

benchmark_stats = calculate_portfolio_stats(
    weights=benchmark_weights,
    prices=prices.drop(columns="Date"),
    cash_weight=benchmark_cash_weight,
    total_cash_return=backtest_risk_free_return
)

print("\nEnsemble vs Market-Cap Benchmark")
print("--------------------------------")

comparison = pd.DataFrame({
    "Seed Ensemble": [
        ensemble_stats["Return"],
        ensemble_stats["Volatility"],
        ensemble_stats["Sharpe"],
        ensemble_stats["MaxDrawdown"]
    ],
    "Market-Cap Benchmark": [
        benchmark_stats["Return"],
        benchmark_stats["Volatility"],
        benchmark_stats["Sharpe"],
        benchmark_stats["MaxDrawdown"]
    ]
}, index=[
    "Annualised Return",
    "Annualised Volatility",
    "Sharpe Ratio",
    "Maximum Drawdown"
])

print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))

comparison["Difference"] = (
    comparison["Seed Ensemble"]
    - comparison["Market-Cap Benchmark"]
)

print("\nEnsemble advantage vs benchmark")
print("-------------------------------")

print(
    comparison[["Difference"]]
    .to_string(float_format=lambda x: f"{x:.4f}")
)

print(f"\n{len(df)} seeds\n")

for column in ["TrainSharpe", "BacktestReturn", "BacktestVolatility", "BacktestSharpe"]:
    if column not in df.columns or df[column].isna().all():
        continue
    values = df[column].dropna()
    mean = values.mean()
    std = values.std(ddof=1)
    cv = std / abs(mean) if mean != 0 else float("nan")
    print(f"{column:>20}: mean={mean:.4f}  std={std:.4f}  min={values.min():.4f}  max={values.max():.4f}  CV={cv:.2%}")

# checks whether training Sharpe predicts backtest Sharpe
if "TrainSharpe" in df.columns and "BacktestSharpe" in df.columns:
    valid_sharpe = df[["Seed", "TrainSharpe", "BacktestSharpe"]].dropna()

    correlation = valid_sharpe["TrainSharpe"].corr(
        valid_sharpe["BacktestSharpe"]
    )

    print("\nTraining vs backtest Sharpe")
    print("--------------------------")

    print(f"Correlation: {correlation:.4f}")

# checks whether training expected return predicts backtest return
if (
    "TrainExpectedReturn" in df.columns
    and "BacktestReturn" in df.columns
):
    valid_returns = df[
        ["Seed", "TrainExpectedReturn", "BacktestReturn"]
    ].dropna()

    return_correlation = valid_returns["TrainExpectedReturn"].corr(
        valid_returns["BacktestReturn"]
    )

    print("\nTraining vs backtest return")
    print("--------------------------")

    print(f"Correlation: {return_correlation:.4f}")

if not args.quiet:
    print("\nExposure by seed")
    print("----------------")

    for _, row in df.sort_values("GrossExposure").iterrows():
        print(
            f"Seed {int(row['Seed']):2d}: "
            f"Backtest vol={row['BacktestVolatility']:.4f}  "
            f"Gross={row['GrossExposure']:.4f}  "
            f"Net={row['NetExposure']:.4f}"
        )

# checks whether gross exposure predicts backtest performance
if (
    "GrossExposure" in df.columns
    and "BacktestReturn" in df.columns
):
    valid_exposure = df[
        [
            "Seed",
            "GrossExposure",
            "BacktestReturn",
            "BacktestSharpe",
            "BacktestVolatility"
        ]
    ].dropna()

    gross_return_correlation = valid_exposure["GrossExposure"].corr(
        valid_exposure["BacktestReturn"]
    )

    gross_sharpe_correlation = valid_exposure["GrossExposure"].corr(
        valid_exposure["BacktestSharpe"]
    )

    print("\nGross exposure vs backtest performance")
    print("--------------------------------------")
    print(f"Gross exposure vs return: {gross_return_correlation:.4f}")
    print(f"Gross exposure vs Sharpe: {gross_sharpe_correlation:.4f}")
    gross_vol_correlation = valid_exposure["GrossExposure"].corr(
        valid_exposure["BacktestVolatility"]
    )

    print(f"Gross exposure vs volatility: {gross_vol_correlation:.4f}")

price_cols = [
    column for column in prices.columns
    if column != "Date"
]

# maps the seed ensemble weights to Yahoo Finance tickers
weight_by_yf = {}

for column, weight in ensemble_weights.items():
    ticker = column.replace("Weight_", "")

    match = assets.loc[
        assets["Ticker"] == ticker,
        "YFinanceTicker"
    ]

    if not match.empty:
        weight_by_yf[match.iloc[0]] = weight

aligned_ensemble_weights = (
    pd.Series(weight_by_yf, dtype=float)
    .reindex(price_cols)
    .fillna(0.0)
)

stock_returns = prices[price_cols].pct_change().fillna(0.0)

# saves ensemble vs benchmark performance time series

ensemble_value = ensemble_stats["ValueSeries"]
benchmark_value = benchmark_stats["ValueSeries"]

comparison_timeseries = pd.DataFrame({
    "Date": prices["Date"],
    "SeedEnsemble": ensemble_value,
    "MarketCapBenchmark": benchmark_value,
})

comparison_timeseries.to_csv(
    "portfolio_data/seed_ensemble_timeseries.csv",
    index=False
)

print(
    "\nWrote portfolio_data/seed_ensemble_timeseries.csv"
)