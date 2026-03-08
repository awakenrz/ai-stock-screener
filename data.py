"""Fetch S&P 500 tickers and stock data from Yahoo Finance."""

import io
import logging
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)


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


def fetch_stock_data(tickers: list[str]) -> pd.DataFrame:
    """Fetch price data and fundamentals for a list of tickers.

    For each ticker, pulls 60 days of OHLCV data and computes
    SMA(20), SMA(50), RSI(14), and volume ratio. Also grabs P/E
    ratio and sector from yfinance fundamentals.

    Args:
        tickers: List of ticker symbols.

    Returns:
        DataFrame with one row per successfully fetched ticker.
    """
    rows = []
    skipped = 0

    for i, symbol in enumerate(tickers):
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="3mo")  # ~60 trading days

            if hist.empty or len(hist) < 20:
                logger.warning(f"Skipping {symbol}: insufficient price data")
                skipped += 1
                continue

            close = hist["Close"]
            volume = hist["Volume"]

            info = stock.info or {}

            row = {
                "ticker": symbol,
                "close": close.iloc[-1],
                "sma_20": close.rolling(20).mean().iloc[-1],
                "sma_50": close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None,
                "rsi": compute_rsi(close),
                "volume": volume.iloc[-1],
                "avg_volume_20": volume.rolling(20).mean().iloc[-1],
                "volume_ratio": volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]
                    if volume.rolling(20).mean().iloc[-1] > 0 else None,
                "pe_ratio": info.get("forwardPE") or info.get("trailingPE"),
                "sector": info.get("sector", "Unknown"),
            }
            rows.append(row)

        except Exception as e:
            logger.warning(f"Skipping {symbol}: {e}")
            skipped += 1

        # Rate limit: sleep between requests (skip on last ticker)
        if i < len(tickers) - 1:
            time.sleep(0.3)

    if skipped > 0:
        logger.info(f"Skipped {skipped}/{len(tickers)} tickers due to errors")

    return pd.DataFrame(rows)
