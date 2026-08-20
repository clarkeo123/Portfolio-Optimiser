#include "Portfolio.hpp"

#include <cmath>
#include <stdexcept>

double calculateExpectedReturn(
    const Portfolio& portfolio,
    const MarketData& marketData
) {
    if (portfolio.weights.size() != marketData.assets.size()) {
        throw std::runtime_error(
            "Number of portfolio weights does not match number of assets."
        );
    }

    for (double weight : portfolio.weights) {
        if (!std::isfinite(weight)) {
            throw std::runtime_error(
                "Portfolio contains a non-finite asset weight."
            );
        }
    }

    if (!std::isfinite(portfolio.cashWeight)) {
        throw std::runtime_error(
            "Portfolio contains a non-finite cash weight."
        );
    }

    double expectedReturn = 0.0;

    for (std::size_t i = 0; i < portfolio.weights.size(); ++i) {
        expectedReturn +=
            portfolio.weights[i]
            * marketData.assets[i].expectedReturn;
    }

    // cash earns the risk-free rate.
    expectedReturn += portfolio.cashWeight * marketData.riskFreeRate;

    return expectedReturn;
}


double calculateVolatility(
    const Portfolio& portfolio,
    const MarketData& marketData
) {
    const std::size_t n = marketData.assets.size();

    if (portfolio.weights.size() != n) {
        throw std::runtime_error(
            "Number of portfolio weights does not match number of assets."
        );
    }

    // converts std::vector<double> to an Eigen vector
    Eigen::VectorXd weights(n);

    for (std::size_t i = 0; i < n; ++i) {
        weights(i) = portfolio.weights[i];
    }

    double variance = weights.transpose() * marketData.covariance * weights;

    if (variance < 0.0) {
        // allows for tiny numerical errors
        if (variance > -1e-12) {
            variance = 0.0;
        } else {
            throw std::runtime_error("Portfolio variance is negative.");
        }
    }

    return std::sqrt(variance);
}

double calculateSharpeRatioFromMetrics(
    double expectedReturn,
    double volatility,
    double riskFreeRate
) {
    if (volatility <= 0.0) {
        return 0.0;
    }

    return (expectedReturn - riskFreeRate) / volatility;
}

Portfolio buildMarketCapWeightedPortfolio(const MarketData& marketData) {
    const std::size_t n = marketData.assets.size();

    Portfolio portfolio;

    portfolio.weights.resize(n);

    for (std::size_t i = 0; i < n; ++i) {
        portfolio.weights[i] = marketData.assets[i].marketCapWeight;
    }

    // market-cap weights are normalised in Python to sum to 1, but cash
    // is recomputed here from the actual total to absorb any tiny
    // floating-point drift rather than assuming it's exactly zero
    double stockTotal = 0.0;

    for (double weight : portfolio.weights) { stockTotal += weight; }

    portfolio.cashWeight = 1.0 - stockTotal;

    portfolio.expectedReturn = calculateExpectedReturn(portfolio, marketData);

    portfolio.volatility = calculateVolatility(portfolio, marketData);

    portfolio.sharpeRatio = calculateSharpeRatioFromMetrics(
        portfolio.expectedReturn,
        portfolio.volatility,
        marketData.riskFreeRate
    );
    return portfolio;
}

double calculatePortfolioForwardReturn(
    const Portfolio& portfolio,
    const MarketData& marketData
) {
    if (!marketData.backtestAvailable) {
        throw std::runtime_error(
            "No backtest data is available - regenerate portfolio_data "
            "with an --end-date earlier than today."
        );
    }

    if (portfolio.weights.size() != marketData.assets.size()) {
        throw std::runtime_error(
            "Number of portfolio weights does not match number of assets."
        );
    }

    for (double weight : portfolio.weights) {
        if (!std::isfinite(weight)) {
            throw std::runtime_error(
                "Portfolio contains a non-finite asset weight."
            );
        }
    }

    if (!std::isfinite(portfolio.cashWeight)) {
        throw std::runtime_error(
            "Portfolio contains a non-finite cash weight."
        );
    }

    double forwardReturn = 0.0;

    for (std::size_t i = 0; i < portfolio.weights.size(); ++i) {
        forwardReturn +=
            portfolio.weights[i] * marketData.assets[i].forwardReturn;
    }

    forwardReturn +=
        portfolio.cashWeight * marketData.backtestRiskFreeReturn;

    return forwardReturn;
}

namespace {

// sample stdev of log returns of a value series, annualised
double annualisedVolatilityFromSeries(
    const std::vector<double>& value,
    int tradingDaysPerYear
) {
    if (value.size() < 3) {
        throw std::runtime_error(
            "At least three values are required to calculate volatility."
        );
    }

    if (tradingDaysPerYear <= 0) {
        throw std::runtime_error(
            "Trading days per year must be positive."
        );
    }

    for (double v : value) {
        if (!std::isfinite(v) || v <= 0.0) {
            throw std::runtime_error(
                "Value series contains a non-positive or non-finite value."
            );
        }
    }

    const std::size_t numberOfReturns = value.size() - 1;

    double mean = 0.0;
    std::vector<double> logReturns(numberOfReturns);

    for (std::size_t t = 0; t < numberOfReturns; ++t) {
        logReturns[t] = std::log(value[t + 1] / value[t]);
        mean += logReturns[t];
    }
    mean /= static_cast<double>(numberOfReturns);

    double variance = 0.0;
    for (double r : logReturns) { variance += (r - mean) * (r - mean); }
    variance /= static_cast<double>(numberOfReturns - 1);

    return std::sqrt(variance) 
        * std::sqrt(static_cast<double>(tradingDaysPerYear));
}

std::vector<double> computeValueSeries(
    const std::vector<double>& weights,
    double cashWeight,
    const Eigen::MatrixXd& prices,
    double totalCashReturn
) {
    if (weights.size() != static_cast<std::size_t>(prices.cols())) {
        throw std::runtime_error(
            "Number of portfolio weights does not match number of price columns."
        );
    }

    if (prices.rows() < 2) {
        throw std::runtime_error(
            "At least two price dates are required."
        );
    }

    if (totalCashReturn <= -1.0 || !std::isfinite(totalCashReturn)) {
        throw std::runtime_error(
            "Invalid total cash return."
        );
    }

    for (Eigen::Index i = 0; i < prices.cols(); ++i) {
        if (!std::isfinite(prices(0, i)) || prices(0, i) <= 0.0) {
            throw std::runtime_error(
                "Initial asset price is non-positive or non-finite."
            );
        }
    }

    const std::size_t numberOfDates = prices.rows();
    const std::size_t n = weights.size();

    double dailyCashLogGrowth =
        std::log(1.0 + totalCashReturn)
        / static_cast<double>(numberOfDates - 1);

    std::vector<double> value(numberOfDates, 0.0);

    for (std::size_t t = 0; t < numberOfDates; ++t) {
        double stockValue = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            stockValue += weights[i] * (prices(t, i) / prices(0, i));
        }
        value[t] = stockValue + cashWeight * std::exp(dailyCashLogGrowth * t);
    }

    return value;
}

} // namespace

std::vector<double> calculateBacktestValueSeries(
    const Portfolio& portfolio,
    const MarketData& marketData
) {
    if (!marketData.backtestAvailable) {
        throw std::runtime_error("No backtest data is available.");
    }

    return computeValueSeries(
        portfolio.weights, portfolio.cashWeight,
        marketData.backtestPrices, marketData.backtestRiskFreeReturn
    );
}

std::vector<double> calculateTrainingValueSeries(
    const Portfolio& portfolio,
    const MarketData& marketData
) {
    double years =
        static_cast<double>(marketData.trainingPrices.rows() - 1)
        / static_cast<double>(marketData.tradingDaysPerYear);

    double totalCashReturn =
        std::pow(1.0 + marketData.riskFreeRate, years) - 1.0;

    return computeValueSeries(
        portfolio.weights, portfolio.cashWeight,
        marketData.trainingPrices, totalCashReturn
    );
}

double calculateBacktestVolatility(
    const Portfolio& portfolio,
    const MarketData& marketData
) {
    return annualisedVolatilityFromSeries(
        calculateBacktestValueSeries(portfolio, marketData),
        marketData.tradingDaysPerYear
    );
}

double calculateBacktestSharpeRatio(
    const Portfolio& portfolio,
    const MarketData& marketData
) {
    double volatility = calculateBacktestVolatility(portfolio, marketData);
    if (volatility <= 0.0) {
        return 0.0;
    }

    double years =
        static_cast<double>(marketData.backtestPrices.rows() - 1)
        / static_cast<double>(marketData.tradingDaysPerYear);

    if (!std::isfinite(years) || years <= 0.0) {
        throw std::runtime_error(
            "Invalid backtest period."
        );
    }

    double totalReturn = calculatePortfolioForwardReturn(portfolio, marketData);

    if (!std::isfinite(totalReturn) || totalReturn <= -1.0) {
        throw std::runtime_error(
            "Invalid portfolio forward return."
        );
    }

    double annualisedReturn =
        std::pow(1.0 + totalReturn, 1.0 / years) - 1.0;

    double annualisedRiskFreeRate =
        std::pow(
            1.0 + marketData.backtestRiskFreeReturn,
            1.0 / years
        ) - 1.0;

    return (annualisedReturn - annualisedRiskFreeRate) / volatility;
}


double calculateFTSEBacktestVolatility(const MarketData& marketData) {
    const auto& p = marketData.ftseBacktestPrices;
    std::vector<double> value(p.size());
    for (int t = 0; t < p.size(); ++t) { value[t] = p(t) / p(0); }

    return annualisedVolatilityFromSeries(value, marketData.tradingDaysPerYear);
}

double calculateFTSEBacktestSharpeRatio(
    const MarketData& marketData
) {
    double volatility = calculateFTSEBacktestVolatility(marketData);
    if (volatility <= 0.0) {
        return 0.0;
    }

    double years =
        static_cast<double>(marketData.ftseBacktestPrices.size() - 1)
        / static_cast<double>(marketData.tradingDaysPerYear);

    if (!std::isfinite(years) || years <= 0.0) {
        throw std::runtime_error(
            "Invalid backtest period."
        );
    }

    double totalReturn = marketData.ftseForwardReturn;

    if (!std::isfinite(totalReturn) || totalReturn <= -1.0) {
        throw std::runtime_error(
            "Invalid FTSE forward return."
        );
    }

    double annualisedReturn =
        std::pow(1.0 + totalReturn, 1.0 / years) - 1.0;

    double annualisedRiskFreeRate =
        std::pow(
            1.0 + marketData.backtestRiskFreeReturn,
            1.0 / years
        ) - 1.0;

    return (annualisedReturn - annualisedRiskFreeRate) / volatility;
}
