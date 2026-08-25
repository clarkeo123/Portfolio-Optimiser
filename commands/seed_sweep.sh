set -e

POPULATION=1000
GENERATIONS=500
MIN_WEIGHT=0.0
MAX_WEIGHT=1.0
RETURN_WEIGHT=1.0
VOLATILITY_WEIGHT=0.0
SHARPE_WEIGHT=1.0
CASH_PENALTY_WEIGHT=0.0
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

rm -f portfolio_data/seed_sweep_results.csv

for seed in "${SEEDS[@]}"; do
    echo "Running seed $seed..."
    printf "%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n" \
        "$POPULATION" "$GENERATIONS" "$MIN_WEIGHT" "$MAX_WEIGHT" \
        "$RETURN_WEIGHT" "$VOLATILITY_WEIGHT" "$SHARPE_WEIGHT" \
        "$CASH_PENALTY_WEIGHT" "$seed" \
        | ./build/portfolio_optimizer > /dev/null
done

echo "Done - results in portfolio_data/seed_sweep_results.csv"