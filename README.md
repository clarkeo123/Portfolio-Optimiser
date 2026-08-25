# Portfolio Optimiser

A C++20 portfolio optimisation project that uses a genetic algorithm to construct long-only or long/short stock portfolios from historical market data. The optimiser estimates expected returns using CAPM, measures risk using covariance-based volatility and evaluates portfolios using a configurable combination of return, volatility and Sharpe ratio objectives.

The repository also includes a Python data pipeline, backtesting tools, seed-sweep analysis, plotting scripts and statistical significance tests.

> **Disclaimer:** This project is for research and educational purposes. It is not financial advice, and historical or simulated performance is not a guarantee of future results.

## Features

- Historical price and market data downloaded through `yfinance`.
- CAPM-based expected returns calculated using estimated stock betas and a market benchmark.
- Covariance-based portfolio volatility.
- Configurable genetic algorithm with tournament selection, crossover, mutation and elitism.
- Stock weight constraints, including optional short positions.
- Cash allocation when the optimiser does not use the full net exposure.
- Repeated runs with different random seeds to reduce sensitivity to a single stochastic solution.
- Fixed-weight out-of-sample backtesting against a market-cap-weighted benchmark.
- Performance plots and statistical tests for comparing the optimised portfolio with benchmarks.

## Project structure

```text
.
├── commands/
│   ├── run_portfolio.sh              # End-to-end workflow
│   └── seed_sweep.sh                 # Runs a fixed set of optimiser seeds
├── data_handling/
│   ├── seed_sweep_analysis.py        # Aggregates and analyses seed results
│   ├── statistical_significance.py   # t-tests and bootstrap Sharpe analysis
│   └── stockinfo.py                  # Downloads and prepares market data
├── include/
│   ├── GeneticAlgorithm.hpp
│   ├── MarketData.hpp
│   └── Portfolio.hpp
├── portfolio_data/                   # Generated inputs and outputs
├── plot/
│   ├── plot_backtest.py              # Plots backtest results
│   ├── plot_performance.py           # Plots portfolio performance
│   └── plot_seed_ensemble.py         # Plots ensemble vs benchmarks
├── src/
│   ├── GeneticAlgorithm.cpp
│   ├── MarketData.cpp
│   ├── Portfolio.cpp
│   └── main.cpp
└── CMakeLists.txt
```

## How it works

The workflow has five main stages.

### 1. Prepares the market data

`stockinfo.py` downloads historical prices for the selected assets and a market benchmark. It creates the files consumed by the C++ optimiser, including:

- `assets.csv`
- `training_prices.csv`
- `covariance.csv`
- `metadata.csv`
- `backtest_prices.csv`
- `backtest.csv`

The training period is used to estimate portfolio characteristics. The later period is reserved for forward evaluation.

### 2. Estimates expected returns

For each stock, the program estimates beta relative to the market:

```math
\beta_i = \frac{\mathrm{Cov}(R_i, R_m)}{\mathrm{Var}(R_m)}
```

The expected annual return is then calculated using CAPM:

$$
E[R_i] = R_f + \beta_i\left(E[R_m] - R_f\right)
$$

where `R_f` is the risk-free rate and `R_m` is the estimated annualised market return.

### 3. Generates and evaluates portfolios

Each candidate portfolio contains:

- A weight for every stock.
- A cash weight.
- Expected return.
- Volatility.
- Sharpe ratio.
- A composite fitness score.

Portfolio volatility is calculated from the covariance matrix:

$$
\sigma_p = \sqrt{\mathbf{w}^{\mathsf{T}}\Sigma\mathbf{w}}
$$

The Sharpe ratio is calculated from expected return, volatility and the risk-free rate:

```math
\mathrm{Sharpe}_p = \frac{E[R_p] - R_f}{\sigma_p}
```

The genetic algorithm combines these metrics into a configurable fitness score. Return and Sharpe are maximised, while volatility is penalised. Objective components are normalised before being combined so that one metric doesn't dominate just because of its numerical scale.

A cash penalty can be used as a tie-breaker to discourage portfolios that leave excessive capital uninvested.

### 4. Evolves the population

The genetic algorithm:

1. Creates an initial population of random portfolios.
2. Repairs portfolios that violate weight constraints or the total-weight constraint.
3. Evaluates fitness.
4. Preserves the strongest individuals through elitism.
5. Selects parents using tournament selection.
6. Creates offspring using crossover.
7. Applies random mutations to portfolio weights.
8. Repeats for the requested number of generations.

Because the search is stochastic, different random seeds can produce different but valid solutions.

### 5. Runs the out-of-sample backtest

After optimisation, the resulting portfolio is evaluated over an out-of-sample period using fixed weights. Periodic rebalancing is in the process of being implemented but is not currently functional.

The backtest compares the optimised portfolio with:

- A market-cap-weighted portfolio benchmark constructed from the selected assets.
- The FTSE 100 index (when running a single seed). This was chosen due to the lack of a usable FTSE total return index on yfinance.

Please note that the benchmark tends to outperform the FTSE 100 index as the price data it uses from yfinance assumes dividends are reinvested. As the FTSE 100 index does not do this it tends to underperform the benchmark by ~3-4% annually. The FTSE 100 index also contains stocks which may have been removed from the optimised portfolio due to a lack of price data. Because of this, the benchmark is the preferred portfolio to compare the optimised portfolio against as they are both calculated using the same price data but with different individual stock weights.

Reported metrics include annualised return, volatility, Sharpe ratio and maximum drawdown.

## Requirements

### Software

- Python 3.9 or later
- A C++ compiler with C++20 support
- CMake 3.23 or later
- Eigen3
- Git (optional but recommended)

### Python packages

Install the required packages in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install numpy pandas matplotlib yfinance
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Building the optimiser

From the repository root:

```bash
cmake -S . -B build
cmake --build build --config Release
```

The executable will be created at:

```text
build/portfolio_optimizer
```

Eigen3 must be discoverable by CMake. If CMake cannot find it, install Eigen through your operating system's package manager or provide the appropriate CMake search path.

## Quick start

The recommended entry point is the end-to-end script:

```bash
commands/run_portfolio.sh
```

Make it executable first if necessary:

```bash
chmod +x commands/run_portfolio.sh
```

The script:

1. Downloads and prepares data.
2. Configures and builds the C++ executable.
3. Runs the optimiser for each random seed.
4. Aggregates the seed results.
5. Generates the seed ensemble time series.
6. Provides a statistical analysis of the results.
7. Creates comparison plots.

The default workflow uses 30 seeds, a population of 1,000 portfolios and 500 generations.

## Running with custom settings

The runner exposes the main experiment parameters:

```bash
commands/run_portfolio.sh \
  --end-date 2025-01-01 \
  --backtest-end-date 2026-01-01 \
  --years 3 \
  --population 1000 \
  --generations 500 \
  --min-weight 0.0 \
  --max-weight 1.0 \
  --return-weight 1.0 \
  --volatility-weight 0.0 \
  --sharpe-weight 1.0 \
  --cash-penalty-weight 5.0 \
  --seeds "1 2 3 4 5" \
  --verbose
```

### Main parameters

| Option | Meaning |
|---|---|
| `--end-date` | End date of the training data, in `YYYY-MM-DD` format |
| `--backtest-end-date` | End date of the out-of-sample backtest |
| `--years` | Number of years of historical training data |
| `--population` | Number of candidate portfolios per generation |
| `--generations` | Number of evolutionary generations |
| `--min-weight` | Minimum permitted weight for each stock |
| `--max-weight` | Maximum permitted weight for each stock |
| `--return-weight` | Relative importance of expected return |
| `--volatility-weight` | Relative penalty applied to volatility |
| `--sharpe-weight` | Relative importance of Sharpe ratio |
| `--cash-penalty-weight` | Penalty for leaving capital in cash |
| `--seeds` | Space-separated list of random seeds |
| `--verbose` | Print detailed seed-sweep diagnostics |

Objective weights are relative rather than percentages. For example, increasing `--sharpe-weight` relative to `--return-weight` makes Sharpe ratio more influential in the composite objective.

## Important constraint behaviour

The portfolio weights are constrained by the configured minimum and maximum stock weights. The portfolio repair step also ensures that the portfolio remains valid under the optimiser's total exposure rules.

Examples:

```text
--min-weight 0.0 --max-weight 0.10
```

creates a long-only portfolio with a maximum 10% allocation to each stock.

```text
--min-weight -0.05 --max-weight 0.10
```

allows short positions down to -5% per stock.

If the sum of stock weights does not equal 100%, the remainder is held as cash.

## Running the C++ program directly

The executable is interactive:

```bash
./build/portfolio_optimizer
```

It asks for:

1. Population size.
2. Number of generations.
3. Minimum stock weight.
4. Maximum stock weight.
5. Expected-return objective weight.
6. Volatility objective weight.
7. Sharpe-ratio objective weight.
8. Cash penalty weight.
9. Random seed.

The same executable can also be driven non-interactively, which is what the shell scripts do:

```bash
printf "1000\n500\n0.0\n1.0\n1.0\n0.0\n1.0\n5.0\n42\n" \
  | ./build/portfolio_optimizer
```

## Seed ensembles

A single genetic algorithm run may find a locally strong solution that depends on its initial population and random mutations. To make the results less dependent on one run, the project supports a seed ensemble.

Run the sweep directly with:

```bash
commands/seed_sweep.sh
```

This writes:

```text
portfolio_data/seed_sweep_results.csv
```

The ensemble analysis averages the portfolio weights across seeds and evaluates the resulting fixed-weight portfolio over the backtest period:

```bash
python3 data_handling/seed_sweep_analysis.py
```

Use `--quiet` to suppress detailed diagnostics:

```bash
python3 data_handling/seed_sweep_analysis.py --quiet
```

## Output files

The main generated outputs are:

| File | Description |
|---|---|
| `portfolio_data/optimisation_history.csv` | Best fitness and optimisation diagnostics by generation |
| `portfolio_data/seed_sweep_results.csv` | Portfolio metrics and weights for each random seed |
| `portfolio_data/seed_ensemble_timeseries.csv` | Ensemble portfolio value compared with the benchmark |
| `portfolio_data/performance_timeseries.csv` | Training/backtest performance data |
| `portfolio_data/backtest_timeseries.csv` | Daily backtest prices or portfolio series used for analysis |
| `portfolio_data/performance.png` | Portfolio performance plot |
| `portfolio_data/backtest_performance.png` | Backtest comparison plot |
| `portfolio_data/seed_ensemble_vs_benchmark.png` | Seed-ensemble vs benchmark plot |

The exact set of generated files may vary with the version of the data pipeline.

## Statistical significance analysis

The project includes a separate script for comparing the optimised portfolio with benchmarks:

```bash
python3 data_handling/statistical_significance.py
```

It reports:

- Annualised Sharpe ratios.
- Paired t-tests on daily log returns.
- Bootstrap confidence intervals for differences in Sharpe ratios.

The bootstrap uses paired resampling of trading days, which preserves the relationship between the portfolio and benchmark observations.

Statistical results should be interpreted cautiously. A favourable historical result can arise from estimation error, model assumptions or market-regime effects.

## Reproducibility

For reproducible experiments:

- Record the training and backtest dates.
- Record all genetic algorithm settings.
- Record the random seeds.
- Keep the generated input data alongside experiment results when appropriate.
- Avoid changing the asset universe or data source without documenting the change.

The optimiser itself is deterministic for a given prepared dataset, configuration and random seed.

## Current limitations

- The current workflow uses fixed portfolio weights throughout the backtest.
- Periodic rebalancing is not yet implemented end-to-end. The `--periodic` option is intentionally rejected by `run_portfolio.sh`.
- CAPM expected returns are model estimates, not forecasts with guaranteed predictive power.
- Historical covariance and returns may be unstable, especially for short training windows.
- Yahoo Finance data availability and metadata quality can vary.
- The optimiser does not model transaction costs, taxes, bid-ask spreads or market impact.
- The genetic algorithm is heuristic and does not guarantee a globally optimal portfolio.
- Backtests are sensitive to the chosen universe, dates and assumptions.

## Suggested workflow for experiments

A typical experiment might look like this:

```bash
# 1. Prepare data and run the default seed ensemble
commands/run_portfolio.sh

# 2. Inspect the generated metrics and plots
ls portfolio_data

# 3. Run additional statistical checks
python3 data_handling/statistical_significance.py

# 4. Experiment with a different objective balance
commands/run_portfolio.sh \
  --return-weight 0.5 \
  --volatility-weight 0.5 \
  --sharpe-weight 1.0 \
  --seeds "1 2 3 4 5 6 7 8 9 10"
```

## License

See [LICENSE](LICENSE) for the project's licence terms.
