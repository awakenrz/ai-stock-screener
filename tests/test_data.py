import numpy as np
import pandas as pd

from data import compute_rsi, fetch_sp500_tickers


def test_fetch_sp500_tickers_returns_list_of_strings():
    tickers = fetch_sp500_tickers()
    assert isinstance(tickers, list)
    assert len(tickers) > 400  # S&P 500 has ~503 tickers
    assert all(isinstance(t, str) for t in tickers)


def test_fetch_sp500_tickers_contains_known_stocks():
    tickers = fetch_sp500_tickers()
    # These blue-chips should always be in the S&P 500
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "GOOGL" in tickers


def test_compute_rsi_known_values():
    # 14 gains of +1 each should give RSI = 100 (all gains, no losses)
    prices = pd.Series([float(i) for i in range(15)])  # 0,1,2,...,14
    rsi = compute_rsi(prices, period=14)
    assert rsi == 100.0


def test_compute_rsi_all_losses():
    # 14 losses of -1 each should give RSI = 0
    prices = pd.Series([float(14 - i) for i in range(15)])  # 14,13,...,0
    rsi = compute_rsi(prices, period=14)
    assert rsi == 0.0


def test_compute_rsi_mixed():
    # With mixed gains/losses, RSI should be between 0 and 100
    np.random.seed(42)
    prices = pd.Series(np.cumsum(np.random.randn(60)) + 100)
    rsi = compute_rsi(prices, period=14)
    assert 0.0 <= rsi <= 100.0


def test_compute_rsi_insufficient_data():
    prices = pd.Series([1.0, 2.0, 3.0])  # Only 3 data points, need 15
    rsi = compute_rsi(prices, period=14)
    assert rsi is None
