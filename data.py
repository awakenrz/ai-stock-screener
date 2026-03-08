"""Fetch S&P 500 tickers and stock data from Yahoo Finance."""

import io

import numpy as np
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


def compute_rsi(prices: pd.Series, period: int = 14) -> float | None:
    """Compute RSI using Wilder smoothing.

    Args:
        prices: Series of closing prices (oldest first).
        period: Lookback period (default 14).

    Returns:
        RSI value (0-100) or None if insufficient data.
    """
    if len(prices) < period + 1:
        return None

    deltas = prices.diff().dropna()

    gains = deltas.where(deltas > 0, 0.0)
    losses = (-deltas).where(deltas < 0, 0.0)

    # First average: simple mean of first `period` values
    avg_gain = gains.iloc[:period].mean()
    avg_loss = losses.iloc[:period].mean()

    # Wilder smoothing for remaining values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
