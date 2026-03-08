"""Fetch S&P 500 tickers and stock data from Yahoo Finance."""

import io

import pandas as pd
import requests


def fetch_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 ticker list from Wikipedia.

    Returns a list of ~503 ticker symbols.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers={"User-Agent": "stock-screener/1.0"}, timeout=10)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    # The 'Symbol' column contains tickers; some have dots (BRK.B) which
    # yfinance expects as dashes (BRK-B).
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers
