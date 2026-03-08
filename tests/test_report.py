# tests/test_report.py
import pandas as pd
from report import print_report


def test_print_report_outputs_something(capsys):
    df = pd.DataFrame([{
        "ticker": "AAPL", "close": 187.42, "pe_ratio": 18.3,
        "rsi": 55.2, "volume_ratio": 2.1, "sma_20": 185.0,
        "sma_50": 180.0, "volume": 1_000_000,
        "avg_volume_20": 500_000, "sector": "Technology",
    }])
    sentiment = {
        "AAPL": {
            "sentiment_score": 0.72,
            "summary": "Strong bullish sentiment on AI chip news",
            "key_catalyst": "AI chip announcement",
        }
    }
    print_report(df, sentiment, total_screened=503)
    captured = capsys.readouterr()
    assert "AAPL" in captured.out
    assert "187.42" in captured.out or "187.4" in captured.out


def test_print_report_handles_no_matches(capsys):
    df = pd.DataFrame()
    print_report(df, {}, total_screened=503)
    captured = capsys.readouterr()
    assert "No stocks" in captured.out or "no stocks" in captured.out.lower()


def test_print_report_handles_none_sentiment(capsys):
    df = pd.DataFrame([{
        "ticker": "XYZ", "close": 50.0, "pe_ratio": 12.0,
        "rsi": 45.0, "volume_ratio": 1.8, "sma_20": 48.0,
        "sma_50": 46.0, "volume": 800_000,
        "avg_volume_20": 400_000, "sector": "Healthcare",
    }])
    sentiment = {"XYZ": None}
    print_report(df, sentiment, total_screened=503)
    captured = capsys.readouterr()
    assert "XYZ" in captured.out
    assert "N/A" in captured.out


def test_print_report_plain_fallback(capsys, monkeypatch):
    monkeypatch.setattr("report.HAS_RICH", False)
    df = pd.DataFrame([{
        "ticker": "MSFT", "close": 420.50, "pe_ratio": 22.1,
        "rsi": 48.0, "volume_ratio": 1.7, "sma_20": 418.0,
        "sma_50": 415.0, "volume": 900_000,
        "avg_volume_20": 500_000, "sector": "Technology",
    }])
    sentiment = {
        "MSFT": {
            "sentiment_score": -0.4,
            "summary": "Bearish on cloud revenue miss",
            "key_catalyst": "Cloud revenue miss",
        }
    }
    print_report(df, sentiment, total_screened=503)
    captured = capsys.readouterr()
    assert "MSFT" in captured.out
    assert "420.50" in captured.out
    assert "AI Stock Screener" in captured.out
