set -e

POPULATION=500
GENERATIONS=500
MIN_WEIGHT=-0.05
MAX_WEIGHT=0.10
RETURN_WEIGHT=1.0
VOLATILITY_WEIGHT=1.0
SHARPE_WEIGHT=1.0
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

rm -f portfolio_data/seed_sweep_results.csv

for seed in "${SEEDS[@]}"; do
    echo "Running seed $seed..."
    printf "%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n" \
        "$POPULATION" "$GENERATIONS" "$MIN_WEIGHT" "$MAX_WEIGHT" \
        "$RETURN_WEIGHT" "$VOLATILITY_WEIGHT" "$SHARPE_WEIGHT" "$seed" \
        | ./build/portfolio_optimizer > /dev/null
done

echo "Done - results in portfolio_data/seed_sweep_results.csv"