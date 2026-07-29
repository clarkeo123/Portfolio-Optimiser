#ifndef GENETIC_ALGORITHM_HPP
#define GENETIC_ALGORITHM_HPP

#include <vector>
#include <random>

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

    // Objective weights.
    double returnWeight;
    double volatilityWeight;
    double sharpeWeight;

    // Stock constraints.
    double minimumStockWeight;
    double maximumStockWeight;
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


    Portfolio generateRandomPortfolio();

    void repairPortfolio(
        Portfolio& portfolio
    );

    double evaluateFitness(
        Portfolio& portfolio
    );

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