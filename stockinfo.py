from datetime import date

import yfinance as yf
import pandas as pd
import math
import numpy as np
import io
import zipfile
import requests
import openpyxl

WIKI_URL = "https://en.wikipedia.org/wiki/FTSE_100_Index"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

ZIP_URL = "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/latest-yield-curve-data.zip"

def get_ftse100_tickers():
    # returns a DataFrame with columns: company, ticker, yfinance_ticker from 
    # FTSE 100 constituents table on Wikipedia
    resp = requests.get(WIKI_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
 
    tables = pd.read_html(io.StringIO(resp.text))
 
    const_table = None
    for t in tables:
        if "Ticker" in t.columns:
            const_table = t
            break
    if const_table is None:
        raise ValueError("Could not find constituents table - Wikipedia page layout may have changed")
 
    const_table = const_table.rename(columns={"Company": "company", "Ticker": "ticker"})
 
    def to_yfinance(ticker):
        ticker = str(ticker).strip().upper()
        ticker = ticker.replace(".", "-")
        return f"{ticker}.L"
 
    const_table["yfinance_ticker"] = const_table["ticker"].apply(to_yfinance)
    return const_table[["company", "ticker", "yfinance_ticker"]]

def download_nominal_workbook(url: str = ZIP_URL):
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    with zf.open("GLC Nominal daily data current month.xlsx") as f:
        data = f.read()
 
    return openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
 
def get_uk_1y_gilt_yield():
    # returns (date, yield_percent) for the most recent 1-year (12-month) 
    # point on the BoE's UK nominal gilt spot curve.

    wb = download_nominal_workbook()
    ws = wb["4. spot curve"]
 
    rows = list(ws.iter_rows(values_only=True))

    months_row = next(r for r in rows if r and r[0] == "years:")
    target_col = None
    for idx, val in enumerate(months_row):
        if val == 1:
            target_col = idx
            break
    if target_col is None:
        raise ValueError("Could not find a 1 year column in the header row")
 
    for row in reversed(rows):
        date_val = row[0]
        if date_val is None or not hasattr(date_val, "year"):
            continue
        yield_val = row[target_col]
        if isinstance(yield_val, (int, float)):
            return date_val.date(), yield_val
 
    raise ValueError("No populated 1-year yield found in the sheet")

FTSE100_INDEX = "^FTSE"


def get_market_data(tickers: list[str], years: int = 5, end_date=None):
    # Returns:
    #   stock_prices
    #   stock_log_returns
    #   market_log_returns
    #   start_date
    #   end_date

    if end_date is None:
        end_date = pd.Timestamp.today().normalize()

    start_date = end_date - pd.DateOffset(years=years)

    download_tickers = list(tickers) + [FTSE100_INDEX]

    data = yf.download(
        download_tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    prices = {}

    for ticker in tickers:
        try:
            close = data[ticker]["Close"]
        except KeyError:
            print(f"Warning: no data found for {ticker}")
            continue

        close = close.dropna()

        if len(close) < 2:
            print(f"Warning: insufficient data for {ticker}")
            continue

        prices[ticker] = close

    stock_prices = pd.DataFrame(prices).sort_index()

    # FTSE 100 index prices.
    try:
        market_prices = (
            data[FTSE100_INDEX]["Close"]
            .dropna()
        )
    except KeyError:
        raise ValueError(
            "Could not download FTSE 100 index data (^FTSE)."
        )

    # Calculate log returns.
    stock_log_returns = np.log(
        stock_prices / stock_prices.shift(1)
    )

    market_log_returns = np.log(
        market_prices / market_prices.shift(1)
    )

    # Align everything by date, but DO NOT remove dates
    # just because one stock is missing.
    combined_returns = pd.concat(
        [
            stock_log_returns,
            market_log_returns.rename(FTSE100_INDEX)
        ],
        axis=1
    )

    stock_log_returns = combined_returns.drop(
        columns=FTSE100_INDEX
    )

    market_log_returns = combined_returns[
        FTSE100_INDEX
    ]

    return (
        stock_prices,
        stock_log_returns,
        market_log_returns,
        start_date,
        end_date
    )

def calculate_market_return(market_log_returns, trading_days=252):
    # annualised historical FTSE 100 return.
    mean_daily_log_return = market_log_returns.mean()

    annualised_log_return = (mean_daily_log_return * trading_days)

    annualised_return = (np.exp(annualised_log_return) - 1)

    return annualised_return


def calculate_betas(
    stock_log_returns,
    market_log_returns
):
    # Calculate beta for each stock:
    #
    # beta = Cov(stock, market) / Var(market)

    betas = {}

    for ticker in stock_log_returns.columns:

        aligned = pd.concat(
            [
                stock_log_returns[ticker],
                market_log_returns
            ],
            axis=1
        ).dropna()

        stock_returns = aligned.iloc[:, 0]
        market_returns = aligned.iloc[:, 1]

        market_variance = market_returns.var(
            ddof=1
        )

        if market_variance <= 0:
            raise ValueError(
                f"Market variance is zero for {ticker}."
            )

        covariance = stock_returns.cov(
            market_returns
        )

        betas[ticker] = (
            covariance / market_variance
        )

    return pd.Series(
        betas,
        name="Beta"
    )

def calculate_capm_returns(betas, risk_free_rate, market_return):
    market_risk_premium = market_return - risk_free_rate

    expected_returns = risk_free_rate + (betas * market_risk_premium)

    expected_returns.name = "ExpectedReturn"

    return expected_returns

def get_market_caps(tickers) -> pd.Series:
    # returns a series of market capitalisations indexed by yfinance

    market_caps = {}

    for ticker in tickers:
        try:
            market_cap = yf.Ticker(ticker).fast_info["market_cap"]
        except Exception:
            market_cap = None

        if not market_cap or market_cap <= 0:
            print(f"Warning: no market cap found for {ticker}")
            continue

        market_caps[ticker] = market_cap

    return pd.Series(market_caps, name="MarketCap")


def calculate_market_cap_weights(market_caps: pd.Series) -> pd.Series:
    # normalises market caps into portfolio weights that sum to 1
    total = market_caps.sum()

    if total <= 0:
        raise ValueError("Total market capitalisation is zero or negative.")

    weights = market_caps / total
    weights.name = "MarketCapWeight"

    return weights

def calculate_covariance_matrix(
    log_returns: pd.DataFrame,
    trading_days: int = 252
) -> pd.DataFrame:
    daily_covariance = log_returns.cov()

    annualised_covariance = daily_covariance * trading_days

    return annualised_covariance

def get_backtest_data(tickers, backtest_start, backtest_end):
    """
    Downloads prices for `tickers` plus the FTSE 100 index between
    backtest_start and backtest_end, and returns the total (simple)
    return each one produced over that window:

        return = final_close / first_close - 1

    Returns (stock_forward_returns: pd.Series, ftse_forward_return: float)
    """
    download_tickers = list(tickers) + [FTSE100_INDEX]

    data = yf.download(
        download_tickers,
        start=backtest_start.strftime("%Y-%m-%d"),
        end=(backtest_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    forward_returns = {}

    for ticker in tickers:
        try:
            close = data[ticker]["Close"].dropna()
        except KeyError:
            print(f"Warning: no backtest data found for {ticker}")
            continue

        if len(close) < 2:
            print(f"Warning: insufficient backtest data for {ticker}")
            continue

        forward_returns[ticker] = close.iloc[-1] / close.iloc[0] - 1

    try:
        ftse_close = data[FTSE100_INDEX]["Close"].dropna()
    except KeyError:
        raise ValueError("Could not download FTSE 100 index backtest data.")

    if len(ftse_close) < 2:
        raise ValueError("Insufficient FTSE 100 index backtest data.")

    ftse_forward_return = ftse_close.iloc[-1] / ftse_close.iloc[0] - 1

    return (
        pd.Series(forward_returns, name="ForwardReturn"),
        ftse_forward_return
    )


def save_backtest(forward_returns: pd.Series, output_path: str):
    backtest_df = forward_returns.rename("ForwardReturn").reset_index()

    backtest_df.columns = ["YFinanceTicker", "ForwardReturn"]

    backtest_df.to_csv(output_path, index=False)

def save_assets(
    tickers_df, betas, expected_returns, market_cap_weights, output_path
):
    assets = tickers_df[["company", "ticker", "yfinance_ticker"]].copy()

    assets = assets.rename(
        columns={
            "company": "Company",
            "ticker": "Ticker",
            "yfinance_ticker": "YFinanceTicker"
        }
    )

    assets["Beta"] = assets["YFinanceTicker"].map(betas)

    assets["ExpectedReturn"] = assets["YFinanceTicker"].map(expected_returns)

    assets["MarketCapWeight"] = (
        assets["YFinanceTicker"].map(market_cap_weights)
    )

    assets = assets.dropna(
        subset=["Beta", "ExpectedReturn", "MarketCapWeight"]
    )

    assets.to_csv(output_path, index=False)

def save_covariance(covariance: pd.DataFrame, output_path: str):
    covariance.to_csv(output_path, index_label="Ticker")

def save_metadata(
    risk_free_rate,
    market_return,
    start_date,
    end_date,
    years,
    number_of_stocks,
    number_of_observations,
    output_path,
    trading_days=252,
    backtest_start_date=None,
    backtest_end_date=None,
    ftse_forward_return=None
):
    parameters = [
        "RiskFreeRate",
        "MarketReturn",
        "StartDate",
        "EndDate",
        "Years",
        "NumberOfStocks",
        "NumberOfObservations",
        "TradingDaysPerYear"
    ]

    values = [
        risk_free_rate,
        market_return,
        start_date,
        end_date,
        years,
        number_of_stocks,
        number_of_observations,
        trading_days
    ]

    # only present when a backtest period was actually generated
    if backtest_start_date is not None:
        parameters += [
            "BacktestStartDate", "BacktestEndDate", "FTSEForwardReturn"
        ]
        values += [backtest_start_date, backtest_end_date, ftse_forward_return]

    metadata = pd.DataFrame({"Parameter": parameters, "Value": values})

    metadata.to_csv(output_path, index=False)

def filter_stocks_by_data_quality(
    stock_log_returns,
    minimum_data_fraction=0.95
):
    """
    Remove stocks which have less than the required fraction
    of available observations.

    For example, 0.95 means a stock must have at least 95%
    of the observations available.
    """

    total_observations = len(stock_log_returns)

    minimum_observations = (
        total_observations
        * minimum_data_fraction
    )

    valid_tickers = []

    for ticker in stock_log_returns.columns:

        available_observations = (
            stock_log_returns[ticker]
            .notna()
            .sum()
        )

        fraction_available = (
            available_observations
            / total_observations
        )

        if available_observations >= minimum_observations:
            valid_tickers.append(ticker)

        else:
            print(
                f"Removing {ticker}: "
                f"{fraction_available:.2%} "
                f"of observations available."
            )

    return stock_log_returns[
        valid_tickers
    ]

if __name__ == "__main__":

    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Generate portfolio optimisation input data."
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help=(
            "Training data cutoff date (YYYY-MM-DD). Stock data used to "
            "build the portfolio is drawn from before this date. Defaults "
            "to today, in which case no backtest period is available."
        ),
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Length of the historical training window, in years.",
    )
    args = parser.parse_args()

    TODAY = pd.Timestamp.today().normalize()

    END_DATE = (
        pd.Timestamp(args.end_date).normalize() if args.end_date else TODAY
    )

    YEARS = args.years
    TRADING_DAYS = 252
    OUTPUT_DIR = "portfolio_data"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # FTSE 100 constituents
    ftse_df = get_ftse100_tickers()

    tickers = ftse_df["yfinance_ticker"].tolist()

    print(f"Found {len(tickers)} FTSE 100 constituents.")

    # risk-free rate
    gilt_date, yield_pct = get_uk_1y_gilt_yield()

    risk_free_rate = yield_pct / 100.0

    print(f"UK 1-year gilt yield ({gilt_date}): {risk_free_rate:.4%}")

    # historical stock + FTSE 100 data
    (
        prices, stock_log_returns, market_log_returns, start_date, end_date
    ) = get_market_data(tickers, years=YEARS, end_date=END_DATE)

    # filters stocks based on historical data availability
    stock_log_returns = filter_stocks_by_data_quality(
        stock_log_returns,
        minimum_data_fraction=0.95
    )

    print(
        f"Retained {len(stock_log_returns.columns)} "
        f"stocks after data-quality filtering."
    )

    print(
        f"Using {len(stock_log_returns.columns)} stocks "
        f"and {len(stock_log_returns)} observations."
    )

    # historical FTSE 100 market return
    market_return = calculate_market_return(
        market_log_returns,
        trading_days=TRADING_DAYS
    )

    print(
        f"Historical FTSE 100 return: "
        f"{market_return:.4%}"
    )

    # stock betas
    betas = calculate_betas(stock_log_returns, market_log_returns)

    # CAPM expected returns
    expected_returns = calculate_capm_returns(
        betas,
        risk_free_rate,
        market_return
    )

    # covariance matrix
    covariance = calculate_covariance_matrix(
        stock_log_returns,
        trading_days=TRADING_DAYS
    )

    # makes sure all datasets contain the same stocks
    common_tickers = (
        stock_log_returns.columns
        .intersection(expected_returns.index)
        .intersection(covariance.index)
    )

    stock_log_returns = stock_log_returns[common_tickers]

    betas = betas[common_tickers]

    expected_returns = expected_returns[common_tickers]

    covariance = covariance.loc[common_tickers, common_tickers]

    ftse_df = ftse_df[ftse_df["yfinance_ticker"].isin(common_tickers)]

    # market capitalisations, for the market-cap-weighted benchmark portfolio
    market_caps = get_market_caps(common_tickers)

    # keep only stocks we could get a market cap for, preserving the
    # existing order explicitly rather than relying on Index.intersection
    common_tickers = [
        ticker for ticker in common_tickers if ticker in market_caps.index
    ]

    stock_log_returns = stock_log_returns[common_tickers]
    betas = betas[common_tickers]
    expected_returns = expected_returns[common_tickers]
    covariance = covariance.loc[common_tickers, common_tickers]
    ftse_df = ftse_df[ftse_df["yfinance_ticker"].isin(common_tickers)]

    market_cap_weights = calculate_market_cap_weights(
        market_caps.loc[common_tickers]
    )

    # backtest: how would this portfolio's stocks have performed from
    # end_date up to today?
    backtest_available = END_DATE < TODAY

    forward_returns = None
    ftse_forward_return = None

    if backtest_available:
        print(
            f"\nRunning backtest from {END_DATE.date()} to {TODAY.date()}..."
        )

        forward_returns, ftse_forward_return = get_backtest_data(
            common_tickers, END_DATE, TODAY
        )

        # some stocks may be missing backtest data (e.g. delisted or
        # newly listed since end_date) - drop them everywhere consistently
        common_tickers = [
            ticker for ticker in common_tickers
            if ticker in forward_returns.index
        ]

        stock_log_returns = stock_log_returns[common_tickers]
        betas = betas[common_tickers]
        expected_returns = expected_returns[common_tickers]
        covariance = covariance.loc[common_tickers, common_tickers]
        ftse_df = ftse_df[ftse_df["yfinance_ticker"].isin(common_tickers)]
        market_cap_weights = market_cap_weights[common_tickers]
        forward_returns = forward_returns[common_tickers]

        print(
            "FTSE 100 index return over backtest period: "
            f"{ftse_forward_return:.2%}"
        )
    else:
        print(
            "\nNo backtest period available (end date is today) - pass "
            "--end-date with an earlier date to enable a backtest."
        )

    # saves data for C++
    save_assets(
        ftse_df,
        betas,
        expected_returns,
        market_cap_weights,
        f"{OUTPUT_DIR}/assets.csv"
    )

    save_covariance(covariance, f"{OUTPUT_DIR}/covariance.csv")

    if backtest_available:
        save_backtest(forward_returns, f"{OUTPUT_DIR}/backtest.csv")

    save_metadata(
        risk_free_rate,
        market_return,
        start_date.date(),
        end_date.date(),
        YEARS,
        len(common_tickers),
        len(stock_log_returns),
        f"{OUTPUT_DIR}/metadata.csv",
        TRADING_DAYS,
        backtest_start_date=END_DATE.date() if backtest_available else None,
        backtest_end_date=TODAY.date() if backtest_available else None,
        ftse_forward_return=(
            ftse_forward_return if backtest_available else None
        )
    )

    print()
    print("Portfolio optimisation data generated.")