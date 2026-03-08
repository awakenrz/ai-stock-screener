# tests/test_integration.py
"""Integration test that runs the full pipeline with mocked network calls."""

from unittest.mock import patch, MagicMock
import pandas as pd
from data import fetch_sp500_tickers, fetch_stock_data, compute_rsi
from filters import screen
from sentiment import _parse_sentiment_response
from report import print_report


def test_full_pipeline_with_synthetic_data(capsys):
    """Run the full pipeline with hand-crafted data to verify all modules
    connect properly."""
    # Synthetic stock data that passes all filters
    stock_data = pd.DataFrame([
        {
            "ticker": "FAKE1", "close": 150.0, "sma_20": 145.0,
            "sma_50": 140.0, "rsi": 55.0, "volume": 2_000_000,
            "avg_volume_20": 1_000_000, "volume_ratio": 2.0,
            "pe_ratio": 18.0, "sector": "Technology",
        },
        {
            "ticker": "FAKE2", "close": 50.0, "sma_20": 48.0,
            "sma_50": 45.0, "rsi": 42.0, "volume": 900_000,
            "avg_volume_20": 500_000, "volume_ratio": 1.8,
            "pe_ratio": 12.0, "sector": "Healthcare",
        },
        {
            # This one should be filtered out (P/E too high)
            "ticker": "FAKE3", "close": 300.0, "sma_20": 290.0,
            "sma_50": 280.0, "rsi": 65.0, "volume": 3_000_000,
            "avg_volume_20": 1_500_000, "volume_ratio": 2.0,
            "pe_ratio": 50.0, "sector": "Consumer",
        },
    ])

    # Screen
    screened = screen(stock_data)
    assert len(screened) == 2  # FAKE1 and FAKE2 pass, FAKE3 doesn't

    # Sentiment (synthetic)
    sentiment = {
        "FAKE1": {
            "sentiment_score": 0.65,
            "summary": "Strong momentum on product launch",
            "key_catalyst": "New product launch",
        },
        "FAKE2": {
            "sentiment_score": -0.2,
            "summary": "Mixed signals on clinical trial results",
            "key_catalyst": "Phase 3 trial data",
        },
    }

    # Report
    print_report(screened, sentiment, total_screened=3)
    captured = capsys.readouterr()
    assert "FAKE1" in captured.out
    assert "FAKE2" in captured.out
    assert "FAKE3" not in captured.out
