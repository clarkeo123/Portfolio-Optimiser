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

def get_all_summaries(tickers: list[str]) -> pd.DataFrame:
    # one HTTP-batched call instead of N separate ones
    # group_by="ticker" gives a column MultiIndex: (ticker, field)
    data = yf.download(
        tickers,
        start="2022-01-01",
        end="2025-01-01",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    rows = []
    for tickerName in tickers:
        close = data[tickerName]["Close"].dropna()

        log_returns = np.log(close / close.shift(1)).dropna()
        if log_returns.empty:
            continue

        mean_log_return = log_returns.mean()

        rows.append({
            "Ticker": tickerName,
            "Mean Log Return": mean_log_return,
            "Geometric Mean Log Return": math.exp(mean_log_return) - 1,
            "Log Return Variance": log_returns.var(ddof=1),
            "Expected Return": 0,
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    ftsedf = get_ftse100_tickers()
    date, yield_pct = get_uk_1y_gilt_yield()
    print(f"UK 1-year gilt yield ({date}): {yield_pct:.4f}%")

    summary_df = get_all_summaries(ftsedf["yfinance_ticker"].tolist())
    print(summary_df)