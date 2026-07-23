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

def get_market_data(tickers: list[str], years: int = 3):
    # returns a prices dataframe and log returns dataframe for the given tickers
    # over the last 'years' years.
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=years)

    data = yf.download(
        tickers,
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
            continue

        close = close.dropna()

        if len(close) < 2:
            continue

        prices[ticker] = close

    prices_df = pd.DataFrame(prices).sort_index()

    # gets log returns
    log_returns = np.log(prices_df / prices_df.shift(1)).dropna(how="all")

    # removes stocks which have no usable return data.
    log_returns = log_returns.dropna(axis=1, how="all")

    log_returns = log_returns.dropna(axis=0, how="any")

    return prices_df, log_returns

def calculate_expected_returns(log_returns: pd.DataFrame,
                               trading_days: int = 252) -> pd.Series:
    mean_daily_log_return = log_returns.mean()

    annualised_return = (
        np.exp(mean_daily_log_return * trading_days) - 1
    )

    return annualised_return

def calculate_covariance_matrix(log_returns: pd.DataFrame,
                                trading_days: int = 252) -> pd.DataFrame:
    daily_covariance = log_returns.cov()

    annualised_covariance = daily_covariance * trading_days

    return annualised_covariance

def save_assets(
    tickers_df: pd.DataFrame,
    expected_returns: pd.Series,
    output_path: str
):
    assets = tickers_df[
        ["company", "yfinance_ticker"]
    ].copy()

    assets = assets.rename(
        columns={
            "company": "Company",
            "yfinance_ticker": "Ticker"
        }
    )

    assets["ExpectedReturn"] = (
        assets["Ticker"]
        .map(expected_returns)
    )

    assets = assets.dropna(subset=["ExpectedReturn"])

    assets.to_csv(
        output_path,
        index=False
    )

def save_covariance(
    covariance: pd.DataFrame,
    output_path: str
):
    covariance.to_csv(
        output_path,
        index_label="Ticker"
    )

def save_metadata(
    risk_free_rate: float,
    start_date,
    end_date,
    years: int,
    output_path: str,
    trading_days: int = 252
):
    metadata = pd.DataFrame({
        "Parameter": [
            "RiskFreeRate",
            "StartDate",
            "EndDate",
            "Years",
            "TradingDaysPerYear",
        ],
        "Value": [
            risk_free_rate,
            start_date,
            end_date,
            years,
            trading_days,
        ]
    })

    metadata.to_csv(
        output_path,
        index=False
    )

if __name__ == "__main__":
    YEARS = 5
    TRADING_DAYS = 252
    OUTPUT_DIR = "portfolio_data"

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # FTSE 100 constituents
    ftse_df = get_ftse100_tickers()

    tickers = ftse_df["yfinance_ticker"].tolist()

    # Risk-free rate
    gilt_date, yield_pct = get_uk_1y_gilt_yield()
    risk_free_rate = yield_pct / 100.0

    print(
        f"UK 1-year gilt yield "
        f"({gilt_date}): {risk_free_rate:.4%}"
    )

    # Historical market data
    prices, log_returns = get_market_data(
        tickers,
        years=YEARS
    )

    print(
        f"Using {len(log_returns.columns)} stocks "
        f"and {len(log_returns)} observations."
    )

    # Expected returns
    expected_returns = calculate_expected_returns(
        log_returns,
        trading_days=TRADING_DAYS
    )

    # Covariance matrix
    covariance = calculate_covariance_matrix(
        log_returns,
        trading_days=TRADING_DAYS
    )

    # Only retain stocks which exist in all datasets.
    common_tickers = (
        expected_returns.index
        .intersection(covariance.index)
    )

    expected_returns = expected_returns.loc[common_tickers]
    covariance = covariance.loc[
        common_tickers,
        common_tickers
    ]

    ftse_df = ftse_df[
        ftse_df["yfinance_ticker"].isin(common_tickers)
    ]

    # saves data for C++
    save_assets(
        ftse_df,
        expected_returns,
        f"{OUTPUT_DIR}/assets.csv"
    )

    save_covariance(
        covariance,
        f"{OUTPUT_DIR}/covariance.csv"
    )

    save_metadata(
        risk_free_rate,
        log_returns.index.min().date(),
        log_returns.index.max().date(),
        YEARS,
        f"{OUTPUT_DIR}/metadata.csv",
        TRADING_DAYS
    )

    print("Portfolio optimisation data generated.")