import math
import numpy as np
import pandas as pd

BACKTEST_TIMESERIES_PATH = "portfolio_data/backtest_timeseries.csv"
METADATA_PATH = "portfolio_data/metadata.csv"
TRADING_DAYS_PER_YEAR = 252
NUMBER_OF_BOOTSTRAP_SAMPLES = 10000
RANDOM_SEED = 42


def load_daily_log_returns(path):
    prices = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return np.log(prices / prices.shift(1)).dropna()


def annualised_sharpe(log_returns, risk_free_rate_annual, trading_days=TRADING_DAYS_PER_YEAR):
    annualised_return = np.exp(log_returns.mean() * trading_days) - 1.0
    annualised_volatility = log_returns.std(ddof=1) * np.sqrt(trading_days)
    return (annualised_return - risk_free_rate_annual) / annualised_volatility


def paired_t_test(log_returns, column_a, column_b):
    diff = log_returns[column_a] - log_returns[column_b]
    n = len(diff)
    mean_diff = diff.mean()
    standard_error = diff.std(ddof=1) / math.sqrt(n)
    t_statistic = mean_diff / standard_error
    # normal approximation to the two-sided p-value
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_statistic) / math.sqrt(2.0))))
    return mean_diff, t_statistic, p_value


def bootstrap_sharpe_differences(log_returns, risk_free_rate_annual, pairs,
                                  number_of_samples=NUMBER_OF_BOOTSTRAP_SAMPLES, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(log_returns)
    differences = {pair: np.empty(number_of_samples) for pair in pairs}

    for i in range(number_of_samples):
        # same resampled day-indices used across all columns, to preserve pairing
        sample_indices = rng.integers(0, n, size=n)
        resampled = log_returns.iloc[sample_indices]

        sharpes = {
            column: annualised_sharpe(resampled[column], risk_free_rate_annual)
            for column in log_returns.columns
        }
        for (a, b) in pairs:
            differences[(a, b)][i] = sharpes[a] - sharpes[b]

    return differences


def main():
    metadata = pd.read_csv(METADATA_PATH, index_col="Parameter")["Value"]
    total_cash_return = float(metadata["BacktestRiskFreeReturn"])

    log_returns = load_daily_log_returns(BACKTEST_TIMESERIES_PATH)

    # matches the C++ Sharpe calculation's annualisation exactly
    years = len(log_returns) / TRADING_DAYS_PER_YEAR
    risk_free_rate_annual = (1.0 + total_cash_return) ** (1.0 / years) - 1.0

    print("Point-estimate annualised Sharpe ratios")
    print("========================================")
    for column in log_returns.columns:
        print(f"{column:>22}: {annualised_sharpe(log_returns[column], risk_free_rate_annual):.4f}")
    print()

    pairs = [("OptimisedPortfolio", "MarketCapBenchmark"), ("OptimisedPortfolio", "FTSE100")]

    print(f"Paired t-test on daily log returns ({len(log_returns)} observations)")
    print("========================================")
    for a, b in pairs:
        mean_diff, t_stat, p_value = paired_t_test(log_returns, a, b)
        verdict = "not significant at 5%" if p_value > 0.05 else "significant at 5%"
        print(f"{a} vs {b}: mean daily diff = {mean_diff:.6f}, t = {t_stat:.3f}, p = {p_value:.3f}  ({verdict})")
    print()

    print(f"Bootstrap 95% CI on Sharpe ratio differences ({NUMBER_OF_BOOTSTRAP_SAMPLES} resamples)")
    print("========================================")
    differences = bootstrap_sharpe_differences(log_returns, risk_free_rate_annual, pairs)
    for a, b in pairs:
        diff_samples = differences[(a, b)]
        lower, upper = np.percentile(diff_samples, [2.5, 97.5])
        contains_zero = lower <= 0.0 <= upper
        verdict = "includes zero: not distinguishable from no difference" if contains_zero \
            else "excludes zero: likely a real difference"
        print(f"{a} - {b}: point ~ {diff_samples.mean():.4f}, 95% CI [{lower:.4f}, {upper:.4f}]  ({verdict})")


if __name__ == "__main__":
    main()