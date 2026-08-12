import pandas as pd

df = pd.read_csv("portfolio_data/seed_sweep_results.csv")
print(f"{len(df)} seeds\n")

for column in ["TrainSharpe", "BacktestReturn", "BacktestVolatility", "BacktestSharpe"]:
    if column not in df.columns or df[column].isna().all():
        continue
    values = df[column].dropna()
    mean = values.mean()
    std = values.std(ddof=1)
    cv = std / abs(mean) if mean != 0 else float("nan")
    print(f"{column:>20}: mean={mean:.4f}  std={std:.4f}  min={values.min():.4f}  max={values.max():.4f}  CV={cv:.2%}")

# compares the lowest and highest backtest volatility seeds
if "BacktestVolatility" in df.columns:
    valid = df.dropna(subset=["BacktestVolatility"]).copy()

    if len(valid) >= 2:
        low_vol = valid.loc[valid["BacktestVolatility"].idxmin()]
        high_vol = valid.loc[valid["BacktestVolatility"].idxmax()]

        print("\nLowest backtest-volatility seed")
        print("--------------------------------")
        print(
            f"Seed: {int(low_vol['Seed'])}  "
            f"Backtest volatility: {low_vol['BacktestVolatility']:.4f}"
        )

        print("\nHighest backtest-volatility seed")
        print("---------------------------------")
        print(
            f"Seed: {int(high_vol['Seed'])}  "
            f"Backtest volatility: {high_vol['BacktestVolatility']:.4f}"
        )

        weight_columns = [
            column
            for column in df.columns
            if column.startswith("Weight_")
            and column != "Weight_Cash"
        ]

        print("\nLargest weight differences")
        print("--------------------------")

        differences = []

        for column in weight_columns:
            difference = abs(low_vol[column] - high_vol[column])

            differences.append(
                (difference, column, low_vol[column], high_vol[column])
            )

        differences.sort(reverse=True)

        for difference, column, low_weight, high_weight in differences[:15]:
            ticker = column.replace("Weight_", "")

            print(
                f"{ticker:>8}: "
                f"{low_weight * 100:7.2f}% -> "
                f"{high_weight * 100:7.2f}%   "
                f"(difference {difference * 100:6.2f}%)"
            )

print("\nExposure by seed")
print("----------------")

for _, row in df.sort_values("BacktestVolatility").iterrows():
    print(
        f"Seed {int(row['Seed']):2d}: "
        f"Backtest vol={row['BacktestVolatility']:.4f}  "
        f"Gross={row['GrossExposure']:.4f}  "
        f"Net={row['NetExposure']:.4f}"
    )

print("\nTraining metrics for low- and high-exposure seeds")
print("--------------------------------------------------")

for seed in [15, 2]:
    row = df.loc[df["Seed"] == seed].iloc[0]

    print(
        f"Seed {seed}: "
        f"ExpectedReturn={row['TrainExpectedReturn']:.6f}  "
        f"Volatility={row['TrainVolatility']:.6f}  "
        f"Sharpe={row['TrainSharpe']:.6f}  "
        f"Gross={row['GrossExposure']:.4f}  "
        f"Net={row['NetExposure']:.4f}"
    )