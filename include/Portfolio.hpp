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

// daily portfolio value path, rebased to 1.0 at the start of the
// backtest window (fixed weights, no rebalancing)
std::vector<double> calculateBacktestValueSeries(
    const Portfolio& portfolio,
    const MarketData& marketData
);

// annualised volatility / Sharpe of a portfolio over the backtest period
double calculateBacktestVolatility(
    const Portfolio& portfolio,
    const MarketData& marketData
);

double calculateBacktestSharpeRatio(
    const Portfolio& portfolio,
    const MarketData& marketData
);

// same, but for the actual FTSE 100 index rather than a constituent portfolio
double calculateFTSEBacktestVolatility(const MarketData& marketData);

double calculateFTSEBacktestSharpeRatio(const MarketData& marketData);

#endif