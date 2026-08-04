#ifndef PORTFOLIO_HPP
#define PORTFOLIO_HPP

#include <vector>

#include "MarketData.hpp"

struct Portfolio
{
    // stock weights
    // weights[i] corresponds to assets[i]
    std::vector<double> weights;

    // weight allocated to cash
    double cashWeight;

    // portfolio statistics
    double expectedReturn;
    double volatility;
    double sharpeRatio;

    // cached composite fitness
    double fitness;
};

// calculates portfolio expected return
double calculateExpectedReturn(
    const Portfolio& portfolio,
    const MarketData& marketData
);

// calculates portfolio volatility
double calculateVolatility(
    const Portfolio& portfolio,
    const MarketData& marketData
);

// calculates Sharpe ratio
double calculateSharpeRatio(
    const Portfolio& portfolio,
    const MarketData& marketData
);

// builds the market-cap-weighted benchmark portfolio (the "index" portfolio)
Portfolio buildMarketCapWeightedPortfolio(
    const MarketData& marketData
);

// calculates the total (simple) return a portfolio's stock holdings
// would have produced from the training end date to today
// only meaningful when MarketData::backtestAvailable is true
double calculatePortfolioForwardReturn(
    const Portfolio& portfolio,
    const MarketData& marketData
);

#endif