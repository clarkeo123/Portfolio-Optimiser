#include "MarketData.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

// helper functions

std::vector<std::string> splitCSVLine(const std::string& line) {
    std::vector<std::string> values;

    std::stringstream stream(line);

    std::string value;

    while (std::getline(stream, value, ',')) {
        values.push_back(value);
    }

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

    // First line contains the column tickers.
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
        }
    }
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

    validateMarketData(marketData);

    return marketData;
}