import yfinance as yf
import pandas as pd
import math
import numpy as np
import io
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/FTSE_100_Index"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

def get_ftse100_tickers():
    # returns a DataFrame with columns: company, ticker, yfinance_ticker from FTSE 100 constituents table on Wikipedia
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

    summary_df = get_all_summaries(ftsedf["yfinance_ticker"].tolist())
    print(summary_df)