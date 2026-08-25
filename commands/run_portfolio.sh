#!/usr/bin/env bash
set -euo pipefail

# One-command runner for the current non-periodic seed-ensemble workflow.
# Periodic rebalancing is intentionally rejected until the optimiser/data
# pipeline has a matching quarterly implementation.

END_DATE="2025-01-01"
BACKTEST_END_DATE="2026-01-01"
YEARS=3
POPULATION=1000
GENERATIONS=500
MIN_WEIGHT=0.0
MAX_WEIGHT=1.0
RETURN_WEIGHT=1.0
VOLATILITY_WEIGHT=0.0
SHARPE_WEIGHT=1.0
CASH_PENALTY_WEIGHT=5.0
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30"
PERIODIC=0
REBALANCE_MONTHS=4
VERBOSE=0

usage() {
  cat <<USAGE
Usage: ./run_portfolio.sh [options]

Options:
  --end-date DATE              Training cutoff, YYYY-MM-DD
  --backtest-end-date DATE     Backtest cutoff, YYYY-MM-DD
  --years N                    Training window in years (default: 3)
  --population N               GA population (default: 1000)
  --generations N              GA generations (default: 500)
  --min-weight X               Minimum stock weight (default: 0.0)
  --max-weight X               Maximum stock weight (default: 1.0)
  --return-weight X            Expected-return objective weight (default: 1.0)
  --volatility-weight X        Volatility objective weight (default: 0.0)
  --sharpe-weight X            Sharpe objective weight (default: 1.0)
  --cash-penalty-weight X      Cash penalty weight (default: 0.0)
  --seeds "1 2 3"              Space-separated optimiser seeds
  --periodic                   Request periodic rebalancing
  --rebalance-months N         Rebalance interval (default: 4)
  --verbose                    Lists full seed sweep analysis
  -h, --help                   Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --end-date) END_DATE="$2"; shift 2;;
    --backtest-end-date) BACKTEST_END_DATE="$2"; shift 2;;
    --years) YEARS="$2"; shift 2;;
    --population) POPULATION="$2"; shift 2;;
    --generations) GENERATIONS="$2"; shift 2;;
    --min-weight) MIN_WEIGHT="$2"; shift 2;;
    --max-weight) MAX_WEIGHT="$2"; shift 2;;
    --return-weight) RETURN_WEIGHT="$2"; shift 2;;
    --volatility-weight) VOLATILITY_WEIGHT="$2"; shift 2;;
    --sharpe-weight) SHARPE_WEIGHT="$2"; shift 2;;
    --cash-penalty-weight) CASH_PENALTY_WEIGHT="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --periodic) PERIODIC=1; shift;;
    --rebalance-months) REBALANCE_MONTHS="$2"; shift 2;;
    --verbose) VERBOSE=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done

if [[ "$PERIODIC" -eq 1 ]]; then
  echo "ERROR: periodic rebalancing is not wired through the current codebase." >&2
  echo "The existing analysis applies one averaged ensemble weight vector to the whole backtest." >&2
  echo "Do not use --periodic until the quarterly optimiser loop and quarterly plotting output are implemented." >&2
  exit 1
fi

if [[ -z "$END_DATE" ]]; then
  END_DATE="$(date +%F)"
fi
if [[ -z "$BACKTEST_END_DATE" ]]; then
  BACKTEST_END_DATE="$(date +%F)"
fi

mkdir -p portfolio_data

echo "[1/5] Generating data"
python3 data_handling/stockinfo.py --end-date "$END_DATE" --backtest-end-date "$BACKTEST_END_DATE" --years "$YEARS"

echo "[2/5] Building optimiser"
cmake -S . -B build
cmake --build build --config Release

echo "[3/5] Running seed ensemble"
rm -f portfolio_data/seed_sweep_results.csv
completed=0
total_seeds=$(awk '{print NF}' <<< "$SEEDS")

for seed in $SEEDS; do
  printf '%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n' \
    "$POPULATION" "$GENERATIONS" "$MIN_WEIGHT" "$MAX_WEIGHT" \
    "$RETURN_WEIGHT" "$VOLATILITY_WEIGHT" "$SHARPE_WEIGHT" \
    "$CASH_PENALTY_WEIGHT" "$seed" \
    | ./build/portfolio_optimizer > /dev/null 2>&1

  completed=$((completed + 1))
  printf "\r  Completed seeds: %d/%d" "$completed" "$total_seeds"
done
printf "\n"

echo "[4/5] Building ensemble time series"
if [ "$VERBOSE" -eq 1 ]; then
    python3 data_handling/seed_sweep_analysis.py
else
    python3 data_handling/seed_sweep_analysis.py --quiet
fi

echo "[5/5] Plotting"
python3 plot/plot_seed_ensemble.py

echo
echo "Complete."
echo "Graph: portfolio_data/seed_ensemble_vs_benchmark.png"
echo "Weights/results: portfolio_data/seed_sweep_results.csv"
echo "Time series: portfolio_data/seed_ensemble_timeseries.csv"
