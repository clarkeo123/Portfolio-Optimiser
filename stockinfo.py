from datetime import date
from sklearn.covariance import LedoitWolf

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

SONIA_SERIES_CODE = "IUDSOIA"

def get_sonia_data(start_date, end_date):
    """
    Download daily SONIA observations from the Bank of England.

    Returns a DataFrame with:
        Date
        SONIA

    SONIA is expressed as a percentage, e.g. 4.75 means 4.75%.
    """
    url = (
        "https://www.bankofengland.co.uk/boeapps/database/"
        "_iadb-fromshowcolumns.asp"
    )

    params = {
        "csv.x": "yes",
        "Datefrom": (
            pd.Timestamp(start_date) - pd.Timedelta(days=7)
        ).strftime("%d/%b/%Y"),
        "Dateto": pd.Timestamp(end_date).strftime("%d/%b/%Y"),
        "SeriesCodes": SONIA_SERIES_CODE,
        "CSVF": "TN",
        "UsingCodes": "Y",
        "VPD": "N",
        "VFD": "N",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
        headers=HEADERS
    )

    response.raise_for_status()

    data = pd.read_csv(
        io.StringIO(response.text)
    )

    if data.shape[1] < 2:
        raise ValueError(
            "Bank of England SONIA download returned no usable data."
        )

    data.columns = ["Date", "SONIA"]

    data["Date"] = pd.to_datetime(
        data["Date"],
        dayfirst=True,
        errors="coerce"
    )

    data["SONIA"] = pd.to_numeric(
        data["SONIA"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Date", "SONIA"]
    ).sort_values("Date")

    if data.empty:
        raise ValueError(
            "No SONIA observations were returned by the Bank of England."
        )

    return data

def calculate_backtest_risk_free_return(backtest_start, backtest_end):
    # calculates the compounded cash return between two dates using
    # historical daily SONIA observations

    start_date = pd.Timestamp(backtest_start).normalize()
    end_date = pd.Timestamp(backtest_end).normalize()

    if end_date <= start_date:
        raise ValueError(
            "Backtest end date must be after backtest start date."
        )

    sonia = get_sonia_data(start_date, end_date)

    before_start = sonia[sonia["Date"] <= start_date]

    if before_start.empty:
        raise ValueError(
            "No SONIA observation exists on or before the backtest start date."
        )

    # starts with the most recent available SONIA observation
    # on or before the backtest start
    current_rate = before_start.iloc[-1]["SONIA"]
    current_date = start_date

    growth_factor = 1.0

    future_observations = sonia[sonia["Date"] > start_date]

    for _, row in future_observations.iterrows():

        observation_date = row["Date"]

        if observation_date > end_date: break

        days = (observation_date - current_date).days

        if days > 0:
            growth_factor *= (1.0 + (current_rate / 100.0) * (days / 365.0))

        current_date = observation_date
        current_rate = row["SONIA"]

    # accrues from the final SONIA observation through the
    # requested backtest end date
    remaining_days = (end_date - current_date).days

    if remaining_days > 0:
        growth_factor *= (
            1.0
            + (current_rate / 100.0)
            * (remaining_days / 365.0)
        )

    return growth_factor - 1.0

def annualise_return(total_return, start_date, end_date):
    years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0

FTSE100_INDEX = "^FTSE"
FTSE100_FALLBACK = "ISF.L"

def get_market_data(tickers: list[str], years: int = 5, end_date=None):
    # returns:
    #   stock_prices
    #   stock_log_returns
    #   market_log_returns
    #   start_date
    #   end_date

    if end_date is None: end_date = pd.Timestamp.today().normalize()

    start_date = end_date - pd.DateOffset(years=years)

    download_tickers = list(tickers) + [
        FTSE100_INDEX,
        FTSE100_FALLBACK,
    ]

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

    # FTSE 100 market prices
    market_prices = pd.Series(dtype=float)

    try:
        market_prices = data[FTSE100_INDEX]["Close"].dropna()
    except KeyError:
        pass

    if len(market_prices) < 2:
        print(
            f"Warning: {FTSE100_INDEX} unavailable; "
            f"using {FTSE100_FALLBACK} as FTSE 100 proxy."
        )

        try:
            market_prices = data[FTSE100_FALLBACK]["Close"].dropna()
        except KeyError:
            market_prices = pd.Series(dtype=float)

    if len(market_prices) < 2:
        raise ValueError(
            "Could not download either ^FTSE or ISF.L market data."
        )
    
    # calculates log returns
    stock_log_returns = np.log(stock_prices / stock_prices.shift(1))

    market_log_returns = np.log(market_prices / market_prices.shift(1))

    # aligns everything by date, but doesn't remove dates
    # just because one stock is missing
    combined_returns = pd.concat(
        [stock_log_returns, market_log_returns.rename(FTSE100_INDEX)],
        axis=1
    )

    stock_log_returns = combined_returns.drop(columns=FTSE100_INDEX)

    market_log_returns = combined_returns[FTSE100_INDEX]

    return (
        stock_prices,
        stock_log_returns,
        market_log_returns,
        market_prices,
        start_date,
        end_date
    )

def calculate_market_return(market_log_returns, trading_days=252):
    # annualised historical FTSE 100 return.
    mean_daily_log_return = market_log_returns.mean()

    annualised_log_return = (mean_daily_log_return * trading_days)

    annualised_return = (np.exp(annualised_log_return) - 1)

    return annualised_return


def calculate_betas(stock_log_returns, market_log_returns):
    # calculates beta for each stock

    betas = {}

    for ticker in stock_log_returns.columns:

        aligned = pd.concat(
            [stock_log_returns[ticker], market_log_returns],
            axis=1
        ).dropna()

        stock_returns = aligned.iloc[:, 0]
        market_returns = aligned.iloc[:, 1]

        market_variance = market_returns.var(ddof=1)

        if market_variance <= 0:
            raise ValueError(f"Market variance is zero for {ticker}.")

        covariance = stock_returns.cov(market_returns)

        betas[ticker] = (covariance / market_variance)

    return pd.Series(betas, name="Beta")

def calculate_capm_returns(betas, risk_free_rate, market_return):
    market_risk_premium = market_return - risk_free_rate

    expected_returns = risk_free_rate + (betas * market_risk_premium)

    if expected_returns.isna().any():
        missing = expected_returns[expected_returns.isna()].index.tolist()

        raise ValueError(
            f"Expected returns contain NaN values for: {missing}"
        )

    expected_returns.name = "ExpectedReturn"

    return expected_returns

def get_market_caps(tickers, target_date) -> pd.Series:
    # returns a series of market capitalisations indexed by yfinance

    market_caps = {}

    for ticker in tickers:
        try:
            df_hist = yf.Ticker(ticker).history(start=target_date)
            close_price = df_hist["Close"].iloc[0]

            # gets shares outstanding from balance sheet / financials
            shares = yf.Ticker(ticker).get_shares_full(start=target_date)
            if shares is not None and not shares.empty:
                share_count = shares.iloc[0]
            else:
                print(
                    f"Warning: no shares outstanding data found for {ticker} " 
                    f"on {target_date}, using current info instead"
                )
                # falls back to current info if historical count isn't indexed
                share_count = yf.Ticker(ticker).info.get("sharesOutstanding")

            market_cap = close_price * share_count
        except Exception:
            market_cap = None

        if not market_cap or market_cap <= 0:
            print(
                f"Warning: no market cap found for {ticker} on "
                f"{target_date}, using current market cap instead"
            )
            market_cap = yf.Ticker(ticker).fast_info["market_cap"]
            if not market_cap or market_cap <= 0:
                print(f"Warning: no market cap found for {ticker}")
                continue

        market_caps[ticker] = market_cap

    return pd.Series(market_caps, name="MarketCap")


def calculate_market_cap_weights(market_caps: pd.Series) -> pd.Series:
    # normalises market caps into portfolio weights that sum to 1
    total = market_caps.sum()

    if total <= 0:
        raise ValueError("Total market capitalisation is zero or negative")

    weights = market_caps / total
    weights.name = "MarketCapWeight"

    return weights

def calculate_covariance_matrix(
    log_returns: pd.DataFrame,
    trading_days: int = 252,
    shrinkage: float = 0.7
) -> pd.DataFrame:
    daily_covariance = log_returns.cov()

    # converts covariance to correlation
    std = np.sqrt(np.diag(daily_covariance.values))
    correlation = daily_covariance.values / np.outer(std, std)

    # averages off-diagonal correlation
    n = correlation.shape[0]
    average_correlation = (
        correlation.sum() - np.trace(correlation)
    ) / (n * (n - 1))

    # builds constant-correlation target
    target_correlation = np.full(
        correlation.shape,
        average_correlation
    )
    np.fill_diagonal(target_correlation, 1.0)

    # converts target correlation back to covariance
    target_covariance = (
        target_correlation * np.outer(std, std)
    )

    # shrinks the sample covariance toward the target
    shrunk_covariance = (
        (1.0 - shrinkage) * daily_covariance.values
        + shrinkage * target_covariance
    )

    annualised_covariance = (
        pd.DataFrame(
            shrunk_covariance,
            index=daily_covariance.index,
            columns=daily_covariance.columns
        )
        * trading_days
    )

    return annualised_covariance

def get_backtest_data(tickers, backtest_start, backtest_end):
    # rerturns total simple returns across the backtest period
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
    closes = {}

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
        closes[ticker] = close

    try:
        ftse_close = data[FTSE100_INDEX]["Close"].dropna()
    except KeyError:
        raise ValueError("Could not download FTSE 100 index backtest data.")

    if len(ftse_close) < 2:
        raise ValueError("Insufficient FTSE 100 index backtest data.")

    ftse_forward_return = ftse_close.iloc[-1] / ftse_close.iloc[0] - 1
    closes[FTSE100_INDEX] = ftse_close

    backtest_prices = pd.DataFrame(closes).dropna()

    return (
        pd.Series(forward_returns, name="ForwardReturn"),
        ftse_forward_return,
        backtest_prices
    )


def save_backtest(forward_returns: pd.Series, output_path: str):
    backtest_df = forward_returns.rename("ForwardReturn").reset_index()

    backtest_df.columns = ["YFinanceTicker", "ForwardReturn"]

    backtest_df.to_csv(output_path, index=False)

def save_backtest_prices(backtest_prices: pd.DataFrame, output_path: str):
    backtest_prices.to_csv(output_path, index_label="Date")

def save_training_prices(
    stock_prices: pd.DataFrame, market_prices: pd.Series, output_path: str
):
    combined = stock_prices.copy()
    combined[FTSE100_INDEX] = market_prices
    combined = combined.dropna()
    combined.to_csv(output_path, index_label="Date")

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
    ftse_forward_return=None,
    backtest_risk_free_return=None
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
            "BacktestStartDate",
            "BacktestEndDate",
            "FTSEForwardReturn",
            "BacktestRiskFreeReturn"
        ]

        values += [
            backtest_start_date,
            backtest_end_date,
            ftse_forward_return,
            backtest_risk_free_return
        ]

    metadata = pd.DataFrame({"Parameter": parameters, "Value": values})

    metadata.to_csv(output_path, index=False)

def filter_stocks_by_data_quality(
    stock_log_returns,
    minimum_data_fraction=0.95
):
    # removes stocks with insufficient historical data availability

    total_observations = len(stock_log_returns)

    minimum_observations = (total_observations * minimum_data_fraction)

    valid_tickers = []

    for ticker in stock_log_returns.columns:

        available_observations = (stock_log_returns[ticker].notna().sum())

        fraction_available = (available_observations / total_observations)

        if available_observations >= minimum_observations:
            valid_tickers.append(ticker)

        else:
            print(
                f"Removing {ticker}: "
                f"{fraction_available:.2%} "
                f"of observations available."
            )

    return stock_log_returns[valid_tickers]

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
    parser.add_argument(
        "--backtest-end-date",
        type=str,
        default=None,
        help=(
            "Backtest end date (YYYY-MM-DD). Defaults to today."
        ),
    )
    args = parser.parse_args()

    TODAY = pd.Timestamp.today().normalize()

    END_DATE = (
        pd.Timestamp(args.end_date).normalize() if args.end_date else TODAY
    )

    BACKTEST_END_DATE = (
        pd.Timestamp(args.backtest_end_date).normalize()
        if args.backtest_end_date
        else TODAY
    )

    if BACKTEST_END_DATE <= END_DATE:
        raise ValueError(
            "Backtest end date must be after the training end date."
        )

    YEARS = args.years
    TRADING_DAYS = 252
    OUTPUT_DIR = "portfolio_data"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # FTSE 100 constituents
    ftse_df = get_ftse100_tickers()

    tickers = ftse_df["yfinance_ticker"].tolist()

    print(f"Found {len(tickers)} FTSE 100 constituents.")

    # historical stock + FTSE 100 data
    (
        prices, stock_log_returns, market_log_returns, market_prices, start_date, end_date
    ) = get_market_data(tickers, years=YEARS, end_date=END_DATE)

    training_cash_return = calculate_backtest_risk_free_return(
        start_date, end_date
    )
    risk_free_rate = annualise_return(
        training_cash_return, start_date, end_date
    )

    print(
        f"Historical SONIA-derived risk-free rate ({start_date.date()} to "
        f"{end_date.date()}): {risk_free_rate:.4%}"
    )

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
    market_caps = get_market_caps(common_tickers, args.end_date)

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

    # backtest
    backtest_available = BACKTEST_END_DATE > END_DATE

    forward_returns = None
    ftse_forward_return = None
    backtest_risk_free_return = None

    if backtest_available:
        print(
            f"\nRunning backtest from {END_DATE.date()} "
            f"to {BACKTEST_END_DATE.date()}..."
        )

        forward_returns, ftse_forward_return, backtest_prices = get_backtest_data(
            common_tickers, END_DATE, BACKTEST_END_DATE
        )

        backtest_risk_free_return = calculate_backtest_risk_free_return(
            END_DATE,
            BACKTEST_END_DATE
        )

        print(
            "Cash/risk-free return over backtest period: "
            f"{backtest_risk_free_return:.2%}"
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
            "\nNo backtest period available. Pass --end-date and, optionally, "
            "--backtest-end-date to define a backtest period."
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

    save_training_prices(
        prices[common_tickers],
        market_prices,
        f"{OUTPUT_DIR}/training_prices.csv"
    )

    if backtest_available:
        save_backtest(forward_returns, f"{OUTPUT_DIR}/backtest.csv")

        save_backtest_prices(
            backtest_prices,
            f"{OUTPUT_DIR}/backtest_prices.csv"
        )

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
        backtest_end_date=(
            BACKTEST_END_DATE.date() if backtest_available else None
        ),
        ftse_forward_return=(
            ftse_forward_return if backtest_available else None
        ),
        backtest_risk_free_return=(
            backtest_risk_free_return if backtest_available else None
        )
    )

    print()
    print("Portfolio optimisation data generated")

    assets = pd.read_csv("portfolio_data/assets.csv")
    backtest = pd.read_csv("portfolio_data/backtest.csv")

    comparison = assets[["YFinanceTicker", "ExpectedReturn"]].merge(
        backtest[["YFinanceTicker", "ForwardReturn"]],
        on="YFinanceTicker",
        how="inner"
    )

    correlation = comparison["ExpectedReturn"].corr(
        comparison["ForwardReturn"]
    )

    print(f"\nExpectedReturn -> ForwardReturn correlation: {correlation:.4f}")

    beta_comparison = assets[["YFinanceTicker", "Beta"]].merge(
        backtest[["YFinanceTicker", "ForwardReturn"]],
        on="YFinanceTicker",
        how="inner"
    )

    beta_correlation = beta_comparison["Beta"].corr(
        beta_comparison["ForwardReturn"]
    )

    print(f"Beta -> ForwardReturn correlation: {beta_correlation:.4f}")

    rank_correlation = comparison["ExpectedReturn"].corr(
        comparison["ForwardReturn"],
        method="spearman"
    )

    print(f"ExpectedReturn rank -> ForwardReturn rank correlation: {rank_correlation:.4f}")