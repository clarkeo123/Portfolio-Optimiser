#include "MarketData.hpp"

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <iostream>

// helper functions

std::vector<std::string> splitCSVLine(const std::string& line) {
    std::vector<std::string> values;

    std::stringstream stream(line);

    std::string value;

    while (std::getline(stream, value, ',')) { values.push_back(value); }

    return values;
}


double stringToDouble(const std::string& value) {
    try {
        return std::stod(value);
    } catch (...) {
        throw std::runtime_error(
            "Could not convert '" + value + "' to a number."
        );
    }
}


int stringToInt(const std::string& value) {
    try {
        return std::stoi(value);
    } catch (...) {
        throw std::runtime_error(
            "Could not convert '" + value + "' to an integer."
        );
    }
}

// loads assets.csv
void loadAssets(const std::string& filename, MarketData& marketData) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open assets file: " + filename);
    }

    std::string line;

    // skips header
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) { continue; }

        std::vector<std::string> values = splitCSVLine(line);

        if (values.size() < 6) {
            throw std::runtime_error("Invalid row in assets.csv: " + line);
        }

        Asset asset;

        asset.company = values[0];
        asset.ticker = values[1];
        asset.yfinanceTicker = values[2];

        asset.beta = stringToDouble(values[3]);

        asset.expectedReturn = stringToDouble(values[4]);

        asset.marketCapWeight = stringToDouble(values[5]);

        marketData.assets.push_back(asset);
    }

    if (marketData.assets.empty()) {
        throw std::runtime_error("No assets were loaded from assets.csv.");
    }
}

// loads covariance.csv
void loadCovariance(const std::string& filename, MarketData& marketData) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open covariance file: " + filename);
    }

    std::string line;

    // first line contains the column tickers
    if (!std::getline(file, line)) {
        throw std::runtime_error("Covariance file is empty.");
    }

    std::vector<std::string> headers = splitCSVLine(line);

    const std::size_t numberOfAssets = marketData.assets.size();

    if (headers.size() != numberOfAssets + 1) {
        throw std::runtime_error(
            "Covariance matrix size does not match the number of assets."
        );
    }

    for (std::size_t i = 0; i < numberOfAssets; ++i) {
        if (headers[i + 1] != marketData.assets[i].yfinanceTicker) {
            throw std::runtime_error(
                "covariance.csv column order does not match assets.csv row "
                "order at position " + std::to_string(i) +
                " (expected '" + marketData.assets[i].yfinanceTicker +
                "', found '" + headers[i + 1] + "')."
            );
        }
    }

    marketData.covariance = Eigen::MatrixXd(numberOfAssets, numberOfAssets);

    std::size_t row = 0;

    while (std::getline(file, line)) {
        if (line.empty()) { continue; }

        std::vector<std::string> values = splitCSVLine(line);

        if (values.size() != numberOfAssets + 1) {
            throw std::runtime_error("Invalid row in covariance.csv.");
        }

        if (row >= numberOfAssets) {
            throw std::runtime_error("Too many rows in covariance.csv.");
        }

        if (values[0] != marketData.assets[row].yfinanceTicker) {
            throw std::runtime_error(
                "covariance.csv row order does not match assets.csv row "
                "order at row " + std::to_string(row) + "."
            );
        }

        for (std::size_t column = 0; column < numberOfAssets; ++column) {
            marketData.covariance(row, column) = 
                stringToDouble(values[column + 1]);
        }

        ++row;
    }

    if (row != numberOfAssets) {
        throw std::runtime_error(
            "Covariance matrix does not contain the expected number of rows."
        );
    }
}

// loads metadata.csv
void loadMetadata(const std::string& filename, MarketData& marketData) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open metadata file: " + filename);
    }

    std::string line;

    // skips header
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()){ continue; }

        std::vector<std::string> values = splitCSVLine(line);

        if (values.size() < 2) { continue; }

        const std::string& parameter = values[0];

        const std::string& value = values[1];

        if (parameter == "RiskFreeRate") {
            marketData.riskFreeRate = stringToDouble(value);
        } else if (parameter == "MarketReturn") {
            marketData.marketReturn = stringToDouble(value);
        } else if (parameter == "StartDate") {
            marketData.startDate = value;
        } else if (parameter == "EndDate") {
            marketData.endDate = value;
        } else if (parameter == "Years") {
            marketData.years = stringToInt(value);
        } else if (parameter == "NumberOfObservations") {
            marketData.numberOfObservations = stringToInt(value);
        } else if (parameter == "TradingDaysPerYear") {
            marketData.tradingDaysPerYear = stringToInt(value);
        } else if (parameter == "BacktestStartDate") {
            marketData.backtestStartDate = value;
        } else if (parameter == "BacktestEndDate") {
            marketData.backtestEndDate = value;
        } else if (parameter == "FTSEForwardReturn") {
            marketData.ftseForwardReturn = stringToDouble(value);
            marketData.backtestAvailable = true;
        } else if (parameter == "BacktestRiskFreeReturn") {
            marketData.backtestRiskFreeReturn = stringToDouble(value);
        }
    }
}

// loads backtest.csv (optional - only present if a backtest period
// was generated). Matches rows to assets by ticker rather than row
// position, since this file may be missing a few stocks relative to
// assets.csv (e.g. ones delisted since the training end date).
void loadBacktestData(const std::string& filename, MarketData& marketData) {
    if (!marketData.backtestAvailable) { return; }

    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr
            << "Warning: backtest metadata was found, but "
            << filename
            << " could not be opened. Disabling backtest.\n";

        marketData.backtestAvailable = false;

        return;
    }

    std::string line;

    // skips header
    std::getline(file, line);

    std::map<std::string, double> forwardReturnsByTicker;

    while (std::getline(file, line)) {
        if (line.empty()) { continue; }

        std::vector<std::string> values = splitCSVLine(line);

        if (values.size() < 2) { continue; }

        forwardReturnsByTicker[values[0]] = stringToDouble(values[1]);
    }

    for (Asset& asset : marketData.assets) {
        auto it = forwardReturnsByTicker.find(asset.yfinanceTicker);

        if (it == forwardReturnsByTicker.end()) {
            throw std::runtime_error(
                "No backtest forward return found for " + asset.ticker
            );
        }

        asset.forwardReturn = it->second;
    }
}

void loadPriceSeries(
    const std::string& filename,
    const MarketData& marketData,
    std::vector<std::string>& dates,
    Eigen::MatrixXd& prices,
    Eigen::VectorXd& ftsePrices
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open price file: " + filename);
    }

    std::string line;
    std::getline(file, line);
    std::vector<std::string> headers = splitCSVLine(line);

    std::map<std::string, std::size_t> columnByTicker;
    for (std::size_t col = 1; col < headers.size(); ++col) {
        columnByTicker[headers[col]] = col;
    }

    const std::size_t n = marketData.assets.size();
    std::vector<std::size_t> assetColumns(n);

    for (std::size_t i = 0; i < n; ++i) {
        auto it = columnByTicker.find(marketData.assets[i].yfinanceTicker);
        if (it == columnByTicker.end()) {
            throw std::runtime_error(
                "No price series for " + marketData.assets[i].ticker
                + " in " + filename
            );
        }
        assetColumns[i] = it->second;
    }

    auto ftseIt = columnByTicker.find("^FTSE");
    if (ftseIt == columnByTicker.end()) {
        throw std::runtime_error("No FTSE 100 column in " + filename);
    }
    std::size_t ftseColumn = ftseIt->second;

    std::vector<std::vector<double>> rows;
    std::vector<double> ftseRows;

    while (std::getline(file, line)) {
        if (line.empty()) { continue; }
        std::vector<std::string> values = splitCSVLine(line);

        dates.push_back(values[0]);

        std::vector<double> row(n);
        for (std::size_t i = 0; i < n; ++i) {
            row[i] = stringToDouble(values[assetColumns[i]]);
        }
        rows.push_back(row);
        ftseRows.push_back(stringToDouble(values[ftseColumn]));
    }

    const std::size_t numberOfDates = rows.size();
    if (numberOfDates < 2) {
        throw std::runtime_error("Not enough dates in " + filename);
    }

    prices = Eigen::MatrixXd(numberOfDates, n);
    ftsePrices = Eigen::VectorXd(numberOfDates);

    for (std::size_t t = 0; t < numberOfDates; ++t) {
        for (std::size_t i = 0; i < n; ++i) { prices(t, i) = rows[t][i]; }
        ftsePrices(t) = ftseRows[t];
    }
}

// loads backtest_prices.csv (optional - only present alongside a
// backtest period). Ticker columns are matched by name rather than
// position, same approach as loadBacktestData.
void loadBacktestPrices(const std::string& filename, MarketData& marketData) {
    if (!marketData.backtestAvailable) { return; }

    loadPriceSeries(
        filename,
        marketData,
        marketData.backtestDates,
        marketData.backtestPrices,
        marketData.ftseBacktestPrices
    );
}

void loadTrainingPrices(const std::string& filename, MarketData& marketData) {
    loadPriceSeries(
        filename,
        marketData,
        marketData.trainingDates,
        marketData.trainingPrices,
        marketData.trainingFtsePrices
    );
}

// validates data
void validateMarketData(const MarketData& marketData) {
    const std::size_t numberOfAssets = marketData.assets.size();

    if (numberOfAssets == 0) {
        throw std::runtime_error("Market data contains no assets.");
    }

    if (
        marketData.covariance.rows() != static_cast<int>(numberOfAssets)
        ||
        marketData.covariance.cols() != static_cast<int>(numberOfAssets)
    ) {
        throw std::runtime_error(
            "Covariance matrix dimensions do not match the number of assets."
        );
    }

    // covariance matrix should be symmetric
    double symmetryError =
        (
            marketData.covariance - marketData.covariance.transpose()
        ).cwiseAbs().maxCoeff();

    if (symmetryError > 1e-10) {
        throw std::runtime_error("Covariance matrix is not symmetric.");
    }
}

// main loader
MarketData loadMarketData(const std::string& dataDirectory) {
    MarketData marketData{};

    loadAssets(dataDirectory + "/assets.csv", marketData);

    loadCovariance(dataDirectory + "/covariance.csv", marketData);

    loadMetadata(dataDirectory + "/metadata.csv", marketData);

    loadBacktestData(dataDirectory + "/backtest.csv", marketData);

    loadBacktestPrices(dataDirectory + "/backtest_prices.csv", marketData);

    loadTrainingPrices(dataDirectory + "/training_prices.csv", marketData);

    validateMarketData(marketData);

    return marketData;
}