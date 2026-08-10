#include "GeneticAlgorithm.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

GeneticAlgorithm::GeneticAlgorithm(
    const MarketData& marketData,
    const GASettings& settings
)
    : marketData(marketData),
      settings(settings),
      randomGenerator(settings.randomSeed)
{
    if (settings.populationSize <= 0) {
        throw std::runtime_error("Population size must be positive.");
    }

    if (settings.generations <= 0) {
        throw std::runtime_error("Number of generations must be positive.");
    }

    if (settings.minimumStockWeight > settings.maximumStockWeight) {
        throw std::runtime_error(
            "Minimum stock weight cannot exceed maximum stock weight."
        );
    }

    if (settings.tournamentSize <= 0) {
        throw std::runtime_error("Tournament size must be positive.");
    }

    if (settings.eliteFraction < 0.0 || settings.eliteFraction > 1.0) {
        throw std::invalid_argument("Elite fraction must be between 0 and 1.");
    }
}

// random number helpers

double GeneticAlgorithm::randomDouble(double minimum, double maximum) {
    std::uniform_real_distribution<double> distribution(minimum, maximum);

    return distribution(randomGenerator);
}

int GeneticAlgorithm::randomInt(int minimum, int maximum) {
    std::uniform_int_distribution<int> distribution(minimum, maximum);

    return distribution(randomGenerator);
}

// generate random portfolio

Portfolio GeneticAlgorithm::generateRandomPortfolio() {
    const std::size_t n = marketData.assets.size();

    Portfolio portfolio;

    portfolio.weights.resize(n);

    // starts by assigning every stock a random weight
    // between the user-defined limits

    for (std::size_t i = 0; i < n; ++i) {
        portfolio.weights[i] =
            randomDouble(
                settings.minimumStockWeight,
                settings.maximumStockWeight
            );
    }

    //  cash is initially zero

    //  repairPortfolio() will modify the portfolio so
    //  that all weights sum to exactly 1

    portfolio.cashWeight = 0.0;

    repairPortfolio(portfolio);

    return portfolio;
}

// repair portfolio

void GeneticAlgorithm::repairPortfolio(Portfolio& portfolio) {
    const std::size_t n = portfolio.weights.size();

    // the portfolio must satisfy: sum(stock weights) + cash = 1

    // cash cannot be negative

    // stocks must satisfy: minimum <= weight <= maximum

    // first clamps every stock to its allowed range

    for (double& weight : portfolio.weights) {
        weight = std::clamp(
            weight,
            settings.minimumStockWeight,
            settings.maximumStockWeight
        );
    }

    // calculates the total stock allocation

    double stockTotal = 0.0;

    for (double weight : portfolio.weights) { stockTotal += weight; }

    /*
        if stocks currently account for less than 100%,
        the remaining amount goes to cash

        this ensures cash is never negative at this stage
    */

    if (stockTotal <= 1.0) {
        portfolio.cashWeight = 1.0 - stockTotal;

        return;
    }

    //  repeatedly removes weight from stocks which are above their minimum

    // If stocks use more than 100%, reduce the stock
    // allocations proportionally according to how much
    // each one can be reduced above its minimum.
    double amountToRemove = stockTotal - 1.0;

    double totalAvailableReduction = 0.0;

    for (double weight : portfolio.weights) {
        totalAvailableReduction += weight - settings.minimumStockWeight;
    }

    if (totalAvailableReduction <= 0.0) {
        throw std::runtime_error(
            "Unable to repair portfolio: "
            "stock weights exceed 100% but cannot be reduced."
        );
    }

    for (double& weight : portfolio.weights) {
        double availableReduction = weight - settings.minimumStockWeight;

        double reduction =
            amountToRemove * (availableReduction / totalAvailableReduction);

        weight -= reduction;

        weight = std::clamp(
            weight,
            settings.minimumStockWeight,
            settings.maximumStockWeight
        );
    }

    portfolio.cashWeight = 0.0;

    //  final correction to guarantee:

    //  sum(weights) + cash = 1

    stockTotal = 0.0;

    for (double weight : portfolio.weights) { stockTotal += weight;}

    portfolio.cashWeight = 1.0 - stockTotal;

    if (portfolio.cashWeight < -1e-9) {
        throw std::runtime_error("Portfolio repair produced negative cash.");
    }

    if (std::abs(portfolio.cashWeight) < 1e-12) { portfolio.cashWeight = 0.0; }
}

// validate portfolio

bool GeneticAlgorithm::isValidPortfolio(const Portfolio& portfolio) {
    if (portfolio.weights.size() != marketData.assets.size()) { return false; }

    for (double weight : portfolio.weights) {
        if (
            weight < settings.minimumStockWeight - 1e-9
            ||
            weight > settings.maximumStockWeight + 1e-9
        ) {
            return false;
        }
    }

    // cash cannot be borrowed

    if (portfolio.cashWeight < -1e-9) { return false; }

    double totalWeight = portfolio.cashWeight;

    for (double weight : portfolio.weights) { totalWeight += weight;}

    return std::abs(totalWeight - 1.0) < 1e-8;
}

void GeneticAlgorithm::calculateObjectiveRanges() {
    
    // generates a separate calibration population

    // ranges are fixed before the actual optimisation begins
    // so fitness values are comparable between generations

    const int calibrationPopulationSize =
        std::max(settings.populationSize * 10, 1000);

    objectiveRanges.minimumReturn = std::numeric_limits<double>::max();

    objectiveRanges.maximumReturn = std::numeric_limits<double>::lowest();

    objectiveRanges.minimumVolatility = std::numeric_limits<double>::max();

    objectiveRanges.maximumVolatility = std::numeric_limits<double>::lowest();

    objectiveRanges.minimumSharpe = std::numeric_limits<double>::max();

    objectiveRanges.maximumSharpe = std::numeric_limits<double>::lowest();

    for (int i = 0; i < calibrationPopulationSize; ++i) {
        Portfolio portfolio = generateRandomPortfolio();

        portfolio.expectedReturn =
            calculateExpectedReturn(portfolio, marketData);

        portfolio.volatility = calculateVolatility(portfolio, marketData);

        portfolio.sharpeRatio = calculateSharpeRatio(portfolio, marketData);

        objectiveRanges.minimumReturn =
            std::min(objectiveRanges.minimumReturn, portfolio.expectedReturn);

        objectiveRanges.maximumReturn =
            std::max(objectiveRanges.maximumReturn, portfolio.expectedReturn);

        objectiveRanges.minimumVolatility =
            std::min(objectiveRanges.minimumVolatility, portfolio.volatility);

        objectiveRanges.maximumVolatility =
            std::max(objectiveRanges.maximumVolatility, portfolio.volatility);

        objectiveRanges.minimumSharpe =
            std::min(objectiveRanges.minimumSharpe, portfolio.sharpeRatio);

        objectiveRanges.maximumSharpe =
            std::max(objectiveRanges.maximumSharpe, portfolio.sharpeRatio);
    }

    std::cout << "\nObjective normalisation ranges\n";
    std::cout << "-----------------------------\n";

    std::cout << std::fixed << std::setprecision(4);

    std::cout
        << "Expected return: "
        << objectiveRanges.minimumReturn * 100.0
        << "% to "
        << objectiveRanges.maximumReturn * 100.0
        << "%\n";

    std::cout
        << "Volatility:       "
        << objectiveRanges.minimumVolatility * 100.0
        << "% to "
        << objectiveRanges.maximumVolatility * 100.0
        << "%\n";

    std::cout
        << "Sharpe ratio:     "
        << objectiveRanges.minimumSharpe
        << " to "
        << objectiveRanges.maximumSharpe
        << "\n";
}

double GeneticAlgorithm::calculatePopulationDiversity(
    const std::vector<Portfolio>& population
) const {
    if (population.empty()) {
        return 0.0;
    }

    const std::size_t n = marketData.assets.size();

    std::vector<double> averageWeights(n, 0.0);

    for (const Portfolio& portfolio : population) {
        for (std::size_t i = 0; i < n; ++i) {
            averageWeights[i] += portfolio.weights[i];
        }
    }

    for (double& weight : averageWeights) {
        weight /= static_cast<double>(population.size());
    }

    double totalDistance = 0.0;

    for (const Portfolio& portfolio : population) {
        double squaredDistance = 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            double difference =
                portfolio.weights[i] - averageWeights[i];

            squaredDistance += difference * difference;
        }

        totalDistance += std::sqrt(squaredDistance);
    }

    return totalDistance / static_cast<double>(population.size());
}

double GeneticAlgorithm::calculateNormalisedReturn(
    double expectedReturn
) const {
    double range =
        objectiveRanges.maximumReturn - objectiveRanges.minimumReturn;

    if (std::abs(range) < 1e-12) { return 0.5; }

    return (expectedReturn - objectiveRanges.minimumReturn) / range;
}

double GeneticAlgorithm::calculateNormalisedVolatility(
    double volatility
) const {
    double range =
        objectiveRanges.maximumVolatility - objectiveRanges.minimumVolatility;

    if (std::abs(range) < 1e-12) { return 0.5; }

    return (volatility - objectiveRanges.minimumVolatility) / range;
}

double GeneticAlgorithm::calculateNormalisedSharpe(
    double sharpeRatio
) const{
    double range =
        objectiveRanges.maximumSharpe - objectiveRanges.minimumSharpe;

    if (std::abs(range) < 1e-12) { return 0.5; }

    return (sharpeRatio - objectiveRanges.minimumSharpe) / range;
}

// fitness

double GeneticAlgorithm::evaluateFitness(Portfolio& portfolio)
{
    portfolio.expectedReturn = calculateExpectedReturn(portfolio, marketData);

    portfolio.volatility = calculateVolatility(portfolio, marketData);

    portfolio.sharpeRatio = calculateSharpeRatio(portfolio, marketData);

    const double normalisedReturn =
        calculateNormalisedReturn(portfolio.expectedReturn);

    const double normalisedVolatility =
        calculateNormalisedVolatility(portfolio.volatility);

    const double normalisedSharpe =
        calculateNormalisedSharpe(portfolio.sharpeRatio);

    portfolio.fitness =
        settings.returnWeight * normalisedReturn
        - settings.volatilityWeight * normalisedVolatility
        + settings.sharpeWeight * normalisedSharpe;

    return portfolio.fitness;
}

// tournament selection

Portfolio GeneticAlgorithm::tournamentSelection(
    const std::vector<Portfolio>& population
) {
    int bestIndex = randomInt(0, static_cast<int>(population.size()) - 1);
    double bestFitness = population[bestIndex].fitness;

    for (int i = 1; i < settings.tournamentSize; ++i) {
        int index = randomInt(0, static_cast<int>(population.size()) - 1);

        if (population[index].fitness > bestFitness) {
            bestIndex = index;
            bestFitness = population[index].fitness;
        }
    }

    return population[bestIndex];
}

// crossover

Portfolio GeneticAlgorithm::crossover(
    const Portfolio& parent1,
    const Portfolio& parent2
) {
    Portfolio child;

    const std::size_t n = parent1.weights.size();

    child.weights.resize(n);

    //  for every stock: child = alpha * parent1 + (1-alpha) * parent2

    for (std::size_t i = 0; i < n; ++i) {
        double alpha = randomDouble(0.0, 1.0);

        child.weights[i] = 
            alpha * parent1.weights[i] + (1.0 - alpha) * parent2.weights[i];
    }

    /*
        cash is also inherited

        repairPortfolio() will recalculate the
        final cash allocation to guarantee validity
    */

    child.cashWeight = (parent1.cashWeight + parent2.cashWeight) / 2.0;

    repairPortfolio(child);

    return child;
}

// mutation

void GeneticAlgorithm::mutate(Portfolio& portfolio) {
    const std::size_t n = portfolio.weights.size();

    for (std::size_t i = 0; i < n; ++i) {
        double probability = randomDouble(0.0, 1.0);

        if (probability < settings.mutationRate) {
            // adds a random perturbation

            double mutation =
                randomDouble(
                    -settings.mutationStrength,
                    settings.mutationStrength
                );

            portfolio.weights[i] += mutation;
        }
    }
    // brings the weights into a valid region

    repairPortfolio(portfolio);
}

void GeneticAlgorithm::saveHistory(
    const std::vector<std::string>& history
) const {
    std::ofstream output(settings.historyFile);

    if (!output) {
        std::cerr
            << "Warning: could not write optimisation "
               "history to "
            << settings.historyFile
            << "\n";

        return;
    }

    for (const std::string& line : history) { output << line << '\n'; }
}

// run genetic algorithm

Portfolio GeneticAlgorithm::run(){
    std::cout << "\nCalculating objective ranges...\n";

    calculateObjectiveRanges();

    std::vector<Portfolio> population;

    population.reserve(settings.populationSize);

    // initial population

    for (int i = 0; i < settings.populationSize; ++i) {
        Portfolio portfolio = generateRandomPortfolio();

        evaluateFitness(portfolio);

        population.push_back(portfolio);
    }

    Portfolio bestPortfolio = population[0];

    double bestFitness = bestPortfolio.fitness;

    // stores optimisation history
    
    std::vector<std::string> history;

    history.push_back(
        "Generation,"
        "BestFitness,"
        "ExpectedReturn,"
        "Volatility,"
        "SharpeRatio,"
        "CashWeight,"
        "PopulationDiversity,"
        "GenerationsWithoutImprovement"
    );

    // evolution

    for (int generation = 0; generation < settings.generations; ++generation) {
        std::vector<Portfolio> newPopulation;

        newPopulation.reserve(settings.populationSize);

        // elitism

        int eliteCount =
            static_cast<int>(settings.populationSize * settings.eliteFraction);

        eliteCount = std::max(1, eliteCount);

        // sorts population from best to worst

        std::sort(
            population.begin(),
            population.end(),
            [](const Portfolio& a, const Portfolio& b)
            {
                return a.fitness > b.fitness;
            }
        );

        Portfolio generationBest = population.front();

        double generationBestFitness = generationBest.fitness;

        double populationDiversity = calculatePopulationDiversity(population);

        if (generationBestFitness > bestFitness)
        {
            bestPortfolio = generationBest;

            bestFitness = generationBestFitness;

            generationsWithoutImprovement = 0;
        } else {
            ++generationsWithoutImprovement;
        }


        std::ostringstream historyLine;

        historyLine
            << generation
            << ","
            << std::setprecision(10)
            << bestFitness
            << ","
            << bestPortfolio.expectedReturn
            << ","
            << bestPortfolio.volatility
            << ","
            << bestPortfolio.sharpeRatio
            << ","
            << bestPortfolio.cashWeight
            << ","
            << populationDiversity
            << ","
            << generationsWithoutImprovement;

        history.push_back(historyLine.str());

        // preserve the best portfolios.

        for (int i = 0; i < eliteCount; ++i) {
            newPopulation.push_back(population[i]);
        }

        // generate remaining population

        while (
            static_cast<int>(newPopulation.size()) < settings.populationSize
        ) {
            Portfolio parent1 = tournamentSelection(population);

            Portfolio parent2 = tournamentSelection(population);

            Portfolio child = crossover(parent1, parent2);

            mutate(child);

            evaluateFitness(child);

            newPopulation.push_back(child);
        }

        population = std::move(newPopulation);


        // print progress periodically

        if (generation % 10 == 0 || generation == settings.generations - 1) {
            std::cout
                << "Generation "
                << generation + 1
                << "/"
                << settings.generations
                << "  Fitness: "
                << bestFitness
                << '\n';
        }
    }

    saveHistory(history);

    std::cout
        << "\nOptimisation history saved to: "
        << settings.historyFile
        << "\n";


    return bestPortfolio;
}