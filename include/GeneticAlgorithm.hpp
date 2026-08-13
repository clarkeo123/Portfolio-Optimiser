#ifndef GENETIC_ALGORITHM_HPP
#define GENETIC_ALGORITHM_HPP

#include <vector>
#include <random>
#include <string>

#include "MarketData.hpp"
#include "Portfolio.hpp"


struct GASettings
{
    int populationSize;
    int generations;

    double mutationRate;
    double mutationStrength;

    int tournamentSize;

    double eliteFraction;

    // objective weights
    double returnWeight;
    double volatilityWeight;
    double sharpeWeight;

    // small tie-breaking penalty for holding cash
    double cashPenaltyWeight;

    // stock constraints
    double minimumStockWeight;
    double maximumStockWeight;

    unsigned int randomSeed;

    std::string historyFile;
};

struct ObjectiveRanges
{
    double minimumReturn;
    double maximumReturn;

    double minimumVolatility;
    double maximumVolatility;

    double minimumSharpe;
    double maximumSharpe;
};


class GeneticAlgorithm
{
public:

    GeneticAlgorithm(
        const MarketData& marketData,
        const GASettings& settings
    );

    Portfolio run();

private:

    const MarketData& marketData;
    GASettings settings;

    std::mt19937 randomGenerator;

    ObjectiveRanges objectiveRanges;

    int generationsWithoutImprovement = 0;

    Portfolio generateRandomPortfolio();

    void repairPortfolio(
        Portfolio& portfolio
    );

    double evaluateFitness(
        Portfolio& portfolio
    );

    double calculateNormalisedReturn(
        double expectedReturn
    ) const;

    double calculateNormalisedVolatility(
        double volatility
    ) const;

    double calculateNormalisedSharpe(
        double sharpe
    ) const;

    void calculateObjectiveRanges();

    double calculatePopulationDiversity(
        const std::vector<Portfolio>& population
    ) const;

    void saveHistory(
        const std::vector<std::string>& history
    ) const;

    Portfolio tournamentSelection(
        const std::vector<Portfolio>& population
    );

    Portfolio crossover(
        const Portfolio& parent1,
        const Portfolio& parent2
    );

    void mutate(
        Portfolio& portfolio
    );

    bool isValidPortfolio(
        const Portfolio& portfolio
    );

    double randomDouble(
        double minimum,
        double maximum
    );

    int randomInt(
        int minimum,
        int maximum
    );
};

#endif