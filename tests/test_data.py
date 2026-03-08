from data import fetch_sp500_tickers


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
