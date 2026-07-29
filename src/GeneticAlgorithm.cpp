#include "GeneticAlgorithm.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

GeneticAlgorithm::GeneticAlgorithm(
    const MarketData& marketData,
    const GASettings& settings
)
    : marketData(marketData),
      settings(settings),
      randomGenerator(std::random_device{}())
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

    /*
        stocks currently account for more than 100%

        needs to reduce stock weights until their total is exactly 1

        cash will then be zero
    */

    double excess = stockTotal - 1.0;

    //  repeatedly removes weight from stocks which are above their minimum

    for (std::size_t iteration = 0;
         iteration < n && excess > 1e-12;
         ++iteration)
    {
        double availableReduction = 0.0;

        for (double weight : portfolio.weights) {
            availableReduction += weight - settings.minimumStockWeight;
        }

        if (availableReduction <= 0.0) {
            throw std::runtime_error(
                "Portfolio constraints cannot produce "
                "a fully invested portfolio."
            );
        }

        for (double& weight : portfolio.weights) {
            double available = weight - settings.minimumStockWeight;

            if (available <= 0.0) { continue; }

            double reduction = excess * (available / availableReduction);

            reduction = std::min(reduction, available);

            weight -= reduction;
            excess -= reduction;
        }
    }

    // removes any tiny floating-point error

    if (std::abs(excess) < 1e-10) { excess = 0.0;}

    if (excess > 1e-8) {
        throw std::runtime_error("Unable to repair portfolio weights.");
    }

    portfolio.cashWeight = 0.0;

    //  final correction to guarantee:

    //  sum(weights) + cash = 1

    double finalStockTotal = 0.0;

    for (double weight : portfolio.weights) { finalStockTotal += weight;}

    portfolio.cashWeight = 1.0 - finalStockTotal;

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

// fitness

double GeneticAlgorithm::evaluateFitness(Portfolio& portfolio)
{
    portfolio.expectedReturn = calculateExpectedReturn(portfolio, marketData);

    portfolio.volatility = calculateVolatility(portfolio, marketData);

    portfolio.sharpeRatio = calculateSharpeRatio(portfolio, marketData);

    return calculateFitness(
        portfolio,
        marketData,
        settings.returnWeight,
        settings.volatilityWeight,
        settings.sharpeWeight
    );
}

// tournament selection

Portfolio GeneticAlgorithm::tournamentSelection(
    const std::vector<Portfolio>& population
) {
    int populationSize = static_cast<int>(population.size());

    int bestIndex = -1;

    double bestFitness = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < settings.tournamentSize; ++i) {
        int index = randomInt(0, populationSize - 1);

        Portfolio candidate = population[index];

        double fitness =
            calculateFitness(
                candidate,
                marketData,
                settings.returnWeight,
                settings.volatilityWeight,
                settings.sharpeWeight
            );


        if (fitness > bestFitness) {
            bestFitness = fitness;
            bestIndex = index;
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
            /*
                adds a random perturbation

                mutationStrength controls the size of the mutation
            */

            double mutation =
                randomDouble(
                    -settings.mutationStrength,
                    settings.mutationStrength
                );

            portfolio.weights[i] += mutation;
        }
    }


    /*
        mutation can produce invalid weights

        repair brings the portfolio back into the feasible region
    */

    repairPortfolio(portfolio);
}

// run genetic algorithm

Portfolio GeneticAlgorithm::run(){
    std::vector<Portfolio> population;

    population.reserve(settings.populationSize);

    // initial population

    for (int i = 0; i < settings.populationSize; ++i) {
        Portfolio portfolio = generateRandomPortfolio();

        evaluateFitness(portfolio);

        population.push_back(portfolio);
    }

    Portfolio bestPortfolio = population[0];

    double bestFitness =
        calculateFitness(
            bestPortfolio,
            marketData,
            settings.returnWeight,
            settings.volatilityWeight,
            settings.sharpeWeight
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
            [&](const Portfolio& a,
                const Portfolio& b)
            {
                return calculateFitness(
                    a,
                    marketData,
                    settings.returnWeight,
                    settings.volatilityWeight,
                    settings.sharpeWeight
                )
                >
                calculateFitness(
                    b,
                    marketData,
                    settings.returnWeight,
                    settings.volatilityWeight,
                    settings.sharpeWeight
                );
            }
        );

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

        // track best solution

        for (const Portfolio& portfolio : population) {
            double fitness =
                calculateFitness(
                    portfolio,
                    marketData,
                    settings.returnWeight,
                    settings.volatilityWeight,
                    settings.sharpeWeight
                );

            if (fitness > bestFitness) {
                bestFitness = fitness;

                bestPortfolio = portfolio;
            }
        }


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

    return bestPortfolio;
}