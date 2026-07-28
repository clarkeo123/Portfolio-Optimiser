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

    // converts std::vector<double> to an Eigen vector.
    Eigen::VectorXd weights(n);

    for (std::size_t i = 0; i < n; ++i) {
        weights(i) = portfolio.weights[i];
    }

    // portfolio variance:
    //
    //     w^T Sigma w
    //
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


double calculateSharpeRatio(
    const Portfolio& portfolio,
    const MarketData& marketData
)
{
    double volatility = calculateVolatility(portfolio, marketData);

    if (volatility <= 0.0) { return 0.0; }

    double expectedReturn = calculateExpectedReturn(portfolio, marketData);

    return (expectedReturn - marketData.riskFreeRate) / volatility;
}


double calculateFitness(
    const Portfolio& portfolio,
    const MarketData& marketData,
    double returnWeight,
    double volatilityWeight,
    double sharpeWeight
)
{
    double expectedReturn = calculateExpectedReturn(portfolio, marketData);

    double volatility = calculateVolatility(portfolio, marketData);

    double sharpe = calculateSharpeRatio(portfolio, marketData);

    /*
        todo:

            maximise expected return
            minimise volatility
            maximise Sharpe ratio

        Therefore:

            + return
            - volatility
            + Sharpe
    */

    return (
        returnWeight * expectedReturn
        - volatilityWeight * volatility
        + sharpeWeight * sharpe
    );
}