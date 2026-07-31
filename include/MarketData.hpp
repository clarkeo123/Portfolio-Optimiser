#ifndef MARKET_DATA_HPP
#define MARKET_DATA_HPP

#include <string>
#include <vector>

#include <Eigen/Dense>

struct Asset
{
    std::string company;
    std::string ticker;
    std::string yfinanceTicker;

    double beta;
    double expectedReturn;
    double marketCapWeight;
};

struct MarketData
{
    std::vector<Asset> assets;

    Eigen::MatrixXd covariance;

    double riskFreeRate;
    double marketReturn;

    std::string startDate;
    std::string endDate;

    int years;
    int numberOfObservations;
    int tradingDaysPerYear;
};

MarketData loadMarketData(
    const std::string& dataDirectory
);

#endif