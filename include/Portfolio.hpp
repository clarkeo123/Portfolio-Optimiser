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


// calculates the multi-objective fitness
double calculateFitness(
    const Portfolio& portfolio,
    const MarketData& marketData,
    double returnWeight,
    double volatilityWeight,
    double sharpeWeight
);

#endif