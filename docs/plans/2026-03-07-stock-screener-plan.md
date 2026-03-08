# AI Stock Screener Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an AI-powered stock screener that fetches S&P 500 data, applies multi-factor screening filters, runs Claude AI sentiment analysis on headlines, and prints a formatted terminal report.

**Architecture:** Multi-module Python package — `data.py` (fetch), `filters.py` (screen), `sentiment.py` (Claude AI), `report.py` (rich output), `main.py` (orchestrator). Each module exposes 1-2 functions with DataFrame or dict inputs/outputs. No external TA library; RSI computed from scratch.

**Tech Stack:** Python 3.10+, yfinance, pandas, anthropic SDK, python-dotenv, rich

---

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`

**Step 1: Create requirements.txt**

```
yfinance
pandas
anthropic
python-dotenv
rich
```

**Step 2: Create .gitignore**

```
.env
__pycache__/
*.pyc
venv/
.venv/
```

**Step 3: Create .env.example**

```
ANTHROPIC_API_KEY=your_api_key_here
```

**Step 4: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully.

**Step 5: Commit**

```bash
git add requirements.txt .gitignore .env.example
git commit -m "feat: add project scaffolding"
```

---

### Task 2: Data layer — fetch_sp500_tickers

**Files:**
- Create: `data.py`
- Create: `tests/test_data.py`

**Step 1: Write the failing test**

```python
# tests/test_data.py
import pandas as pd
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data'`

**Step 3: Write minimal implementation**

```python
# data.py
"""Fetch S&P 500 tickers and stock data from Yahoo Finance."""

import pandas as pd


def fetch_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 ticker list from Wikipedia.

    Returns a list of ~503 ticker symbols.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    # The 'Symbol' column contains tickers; some have dots (BRK.B) which
    # yfinance expects as dashes (BRK-B).
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add data.py tests/test_data.py
git commit -m "feat: add fetch_sp500_tickers"
```

---

### Task 3: Data layer — compute_rsi helper

**Files:**
- Modify: `data.py`
- Modify: `tests/test_data.py`

**Step 1: Write the failing test**

```python
# Append to tests/test_data.py
import numpy as np
from data import compute_rsi


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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data.py::test_compute_rsi_known_values -v`
Expected: FAIL — `ImportError: cannot import name 'compute_rsi'`

**Step 3: Write minimal implementation**

```python
# Add to data.py, after the existing imports
import numpy as np


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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_data.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add data.py tests/test_data.py
git commit -m "feat: add compute_rsi with Wilder smoothing"
```

---

### Task 4: Data layer — fetch_stock_data

**Files:**
- Modify: `data.py`
- Modify: `tests/test_data.py`

**Step 1: Write the failing test**

```python
# Append to tests/test_data.py
from data import fetch_stock_data


def test_fetch_stock_data_returns_dataframe():
    # Test with just 3 tickers to keep it fast
    df = fetch_stock_data(["AAPL", "MSFT", "GOOGL"])
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0  # At least some should succeed


def test_fetch_stock_data_has_expected_columns():
    df = fetch_stock_data(["AAPL"])
    expected_cols = [
        "ticker", "close", "sma_20", "sma_50", "rsi",
        "volume", "avg_volume_20", "volume_ratio", "pe_ratio", "sector",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_fetch_stock_data_skips_bad_tickers():
    df = fetch_stock_data(["AAPL", "ZZZZZZNOTREAL"])
    assert len(df) == 1  # Only AAPL should be present
    assert df.iloc[0]["ticker"] == "AAPL"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data.py::test_fetch_stock_data_returns_dataframe -v`
Expected: FAIL — `ImportError: cannot import name 'fetch_stock_data'`

**Step 3: Write minimal implementation**

```python
# Add to data.py, after existing code
import time
import logging
import yfinance as yf

logger = logging.getLogger(__name__)


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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_data.py -v`
Expected: All PASS (the network tests may be slow — ~10s)

**Step 5: Commit**

```bash
git add data.py tests/test_data.py
git commit -m "feat: add fetch_stock_data with SMA, RSI, and fundamentals"
```

---

### Task 5: Screening filters

**Files:**
- Create: `filters.py`
- Create: `tests/test_filters.py`

**Step 1: Write the failing test**

```python
# tests/test_filters.py
import pandas as pd
from filters import screen, screen_detail


def _make_stock(
    ticker="TEST", close=100, sma_20=95, sma_50=90,
    rsi=50, volume_ratio=2.0, pe_ratio=15,
):
    """Helper to build a single-row DataFrame with valid defaults."""
    return {
        "ticker": ticker, "close": close, "sma_20": sma_20,
        "sma_50": sma_50, "rsi": rsi, "volume": 1_000_000,
        "avg_volume_20": 500_000, "volume_ratio": volume_ratio,
        "pe_ratio": pe_ratio, "sector": "Technology",
    }


def test_screen_passes_stock_meeting_all_criteria():
    df = pd.DataFrame([_make_stock()])
    result = screen(df)
    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "TEST"


def test_screen_rejects_high_pe():
    df = pd.DataFrame([_make_stock(pe_ratio=50)])
    result = screen(df)
    assert len(result) == 0


def test_screen_rejects_low_pe():
    df = pd.DataFrame([_make_stock(pe_ratio=3)])
    result = screen(df)
    assert len(result) == 0


def test_screen_rejects_overbought_rsi():
    df = pd.DataFrame([_make_stock(rsi=80)])
    result = screen(df)
    assert len(result) == 0


def test_screen_rejects_low_volume():
    df = pd.DataFrame([_make_stock(volume_ratio=1.0)])
    result = screen(df)
    assert len(result) == 0


def test_screen_rejects_below_sma50():
    df = pd.DataFrame([_make_stock(close=85, sma_50=90)])
    result = screen(df)
    assert len(result) == 0


def test_screen_rejects_death_cross():
    # sma_20 < sma_50 = death cross (opposite of golden cross)
    df = pd.DataFrame([_make_stock(sma_20=85, sma_50=90)])
    result = screen(df)
    assert len(result) == 0


def test_screen_rejects_none_pe():
    df = pd.DataFrame([_make_stock(pe_ratio=None)])
    result = screen(df)
    assert len(result) == 0


def test_screen_detail_includes_filter_columns():
    df = pd.DataFrame([_make_stock()])
    detail = screen_detail(df)
    assert "pass_pe" in detail.columns
    assert "pass_rsi" in detail.columns
    assert "pass_volume" in detail.columns
    assert "pass_above_sma50" in detail.columns
    assert "pass_golden_cross" in detail.columns
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_filters.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'filters'`

**Step 3: Write minimal implementation**

```python
# filters.py
"""Multi-factor stock screening filters."""

import pandas as pd


def screen_detail(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all filters and return the DataFrame with boolean filter columns.

    Each filter adds a pass_* column. Useful for inspecting near-misses.
    """
    result = df.copy()

    result["pass_pe"] = result["pe_ratio"].between(5, 25)
    result["pass_rsi"] = result["rsi"].between(30, 70)
    result["pass_volume"] = result["volume_ratio"] >= 1.5
    result["pass_above_sma50"] = result["close"] > result["sma_50"]
    result["pass_golden_cross"] = result["sma_20"] > result["sma_50"]

    return result


def screen(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all filters and return only stocks passing every one.

    Filters:
        - P/E ratio between 5 and 25
        - RSI between 30 and 70
        - Volume >= 1.5x 20-day average
        - Price above 50-day SMA
        - Golden cross (20 SMA > 50 SMA)
    """
    # Drop rows with None in critical columns before filtering
    required_cols = ["pe_ratio", "rsi", "volume_ratio", "close", "sma_50", "sma_20"]
    clean = df.dropna(subset=required_cols)

    detail = screen_detail(clean)
    pass_cols = [c for c in detail.columns if c.startswith("pass_")]
    mask = detail[pass_cols].all(axis=1)
    return detail[mask].drop(columns=pass_cols).reset_index(drop=True)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_filters.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add filters.py tests/test_filters.py
git commit -m "feat: add multi-factor screening filters"
```

---

### Task 6: AI sentiment layer

**Files:**
- Create: `sentiment.py`
- Create: `tests/test_sentiment.py`

**Step 1: Write the failing test**

We test the parsing logic without making real API calls by mocking the Anthropic client.

```python
# tests/test_sentiment.py
from unittest.mock import patch, MagicMock
from sentiment import analyze, _parse_sentiment_response


def test_parse_sentiment_response_valid_json():
    raw = '{"sentiment_score": 0.7, "summary": "Bullish outlook", "key_catalyst": "New product"}'
    result = _parse_sentiment_response(raw)
    assert result["sentiment_score"] == 0.7
    assert result["summary"] == "Bullish outlook"
    assert result["key_catalyst"] == "New product"


def test_parse_sentiment_response_strips_markdown():
    raw = '```json\n{"sentiment_score": -0.3, "summary": "Bearish", "key_catalyst": "Earnings miss"}\n```'
    result = _parse_sentiment_response(raw)
    assert result["sentiment_score"] == -0.3


def test_parse_sentiment_response_invalid():
    result = _parse_sentiment_response("not json at all")
    assert result is None


def test_analyze_returns_dict_with_mock():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = '{"sentiment_score": 0.5, "summary": "Positive", "key_catalyst": "Growth"}'
    mock_client.messages.create.return_value = mock_response

    mock_news = [
        {"title": "Stock surges on earnings beat"},
        {"title": "Company announces new partnership"},
    ]

    with patch("sentiment.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.news = mock_news
        result = analyze(["AAPL"], client=mock_client)

    assert "AAPL" in result
    assert result["AAPL"]["sentiment_score"] == 0.5
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sentiment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentiment'`

**Step 3: Write minimal implementation**

```python
# sentiment.py
"""Claude AI sentiment analysis for stock news headlines."""

import json
import time
import logging

import anthropic
import yfinance as yf

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5"

PROMPT_TEMPLATE = """Analyze these news headlines for {ticker} and assess market sentiment.

Headlines:
{headlines}

Return ONLY a JSON object with these fields:
- "sentiment_score": float from -1.0 (very bearish) to 1.0 (very bullish)
- "summary": one sentence explaining the overall sentiment
- "key_catalyst": the single most impactful headline

Return raw JSON only. No markdown, no explanation."""


def _parse_sentiment_response(raw: str) -> dict | None:
    """Parse Claude's response into a sentiment dict.

    Handles markdown code fences and malformed JSON gracefully.
    """
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse sentiment JSON: {text[:100]}")
        return None


def analyze(
    tickers: list[str],
    client: anthropic.Anthropic | None = None,
) -> dict[str, dict]:
    """Analyze news sentiment for each ticker using Claude Haiku.

    Args:
        tickers: List of ticker symbols to analyze.
        client: Optional Anthropic client (for testing). If None, creates one.

    Returns:
        Dict mapping ticker -> {"sentiment_score", "summary", "key_catalyst"}
        or ticker -> None if analysis failed.
    """
    if client is None:
        client = anthropic.Anthropic()

    results = {}

    for symbol in tickers:
        try:
            stock = yf.Ticker(symbol)
            news = stock.news or []
            headlines = [item.get("title", "") for item in news[:5]]

            if not headlines:
                logger.warning(f"{symbol}: no news headlines found")
                results[symbol] = None
                continue

            headline_text = "\n".join(f"- {h}" for h in headlines)
            prompt = PROMPT_TEMPLATE.format(
                ticker=symbol, headlines=headline_text,
            )

            response = client.messages.create(
                model=MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text
            parsed = _parse_sentiment_response(raw_text)

            if parsed is not None:
                results[symbol] = parsed
            else:
                results[symbol] = None

        except anthropic.RateLimitError:
            logger.warning(f"{symbol}: rate limited, retrying in 2s...")
            time.sleep(2)
            try:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                parsed = _parse_sentiment_response(response.content[0].text)
                results[symbol] = parsed
            except Exception as e:
                logger.error(f"{symbol}: retry failed: {e}")
                results[symbol] = None

        except Exception as e:
            logger.error(f"{symbol}: sentiment analysis failed: {e}")
            results[symbol] = None

    return results
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sentiment.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add sentiment.py tests/test_sentiment.py
git commit -m "feat: add Claude AI sentiment analysis layer"
```

---

### Task 7: Terminal report

**Files:**
- Create: `report.py`
- Create: `tests/test_report.py`

**Step 1: Write the failing test**

```python
# tests/test_report.py
import pandas as pd
from io import StringIO
from unittest.mock import patch
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'report'`

**Step 3: Write minimal implementation**

```python
# report.py
"""Terminal report output using rich tables."""

from datetime import date

import pandas as pd

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _sentiment_color(score: float | None) -> str:
    """Map sentiment score to a color name."""
    if score is None:
        return "dim"
    if score > 0.3:
        return "green"
    if score < -0.3:
        return "red"
    return "yellow"


def _sentiment_label(score: float | None) -> str:
    """Format sentiment score for display."""
    if score is None:
        return "N/A"
    sign = "+" if score > 0 else ""
    return f"{sign}{score:.2f}"


def print_report(
    screened_df: pd.DataFrame,
    sentiment_data: dict,
    total_screened: int = 0,
) -> None:
    """Print screening results to the terminal.

    Uses rich for formatted tables if available, falls back to plain text.

    Args:
        screened_df: DataFrame of stocks that passed all filters.
        sentiment_data: Dict mapping ticker -> sentiment dict or None.
        total_screened: Total number of stocks that were screened.
    """
    if screened_df.empty:
        print(f"\nNo stocks matched all screening criteria today.")
        print("Consider loosening one or more filters (e.g., volume_ratio >= 1.2).\n")
        return

    if HAS_RICH:
        _print_rich(screened_df, sentiment_data, total_screened)
    else:
        _print_plain(screened_df, sentiment_data, total_screened)


def _print_rich(df: pd.DataFrame, sentiment: dict, total: int) -> None:
    """Print report using rich library."""
    console = Console()

    today = date.today().strftime("%Y-%m-%d")
    console.print(
        Panel(
            f"[bold]AI Stock Screener[/bold] - {today} Pre-Market Report",
            style="blue",
        )
    )
    console.print(
        f"\nScreened {total} S&P 500 stocks -> "
        f"[bold green]{len(df)}[/bold green] passed all filters\n"
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("Ticker", style="bold")
    table.add_column("Price", justify="right")
    table.add_column("P/E", justify="right")
    table.add_column("RSI", justify="right")
    table.add_column("Vol Spike", justify="right")
    table.add_column("Sentiment", justify="right")

    for _, row in df.iterrows():
        ticker = row["ticker"]
        sent = sentiment.get(ticker)
        score = sent["sentiment_score"] if sent else None
        color = _sentiment_color(score)

        table.add_row(
            ticker,
            f"{row['close']:.2f}",
            f"{row['pe_ratio']:.1f}",
            f"{row['rsi']:.1f}",
            f"{row['volume_ratio']:.1f}x",
            f"[{color}]{_sentiment_label(score)}[/{color}]",
        )

    console.print(table)

    # AI Commentary
    has_commentary = any(v is not None for v in sentiment.values())
    if has_commentary:
        console.print("\n[bold]AI Commentary:[/bold]")
        for _, row in df.iterrows():
            ticker = row["ticker"]
            sent = sentiment.get(ticker)
            if sent:
                color = _sentiment_color(sent["sentiment_score"])
                console.print(
                    f"  [{color}]{ticker}[/{color}]: {sent['summary']}"
                )
            else:
                console.print(f"  [dim]{ticker}[/dim]: Sentiment unavailable")
        console.print()


def _print_plain(df: pd.DataFrame, sentiment: dict, total: int) -> None:
    """Fallback plain-text report when rich is not installed."""
    today = date.today().strftime("%Y-%m-%d")
    print(f"\nAI Stock Screener - {today} Pre-Market Report")
    print(f"Screened {total} stocks -> {len(df)} passed all filters\n")

    header = f"{'Ticker':<8} {'Price':>8} {'P/E':>6} {'RSI':>6} {'Vol':>6} {'Sent':>8}"
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        ticker = row["ticker"]
        sent = sentiment.get(ticker)
        score = sent["sentiment_score"] if sent else None

        print(
            f"{ticker:<8} {row['close']:>8.2f} {row['pe_ratio']:>6.1f} "
            f"{row['rsi']:>6.1f} {row['volume_ratio']:>5.1f}x "
            f"{_sentiment_label(score):>8}"
        )

    print()
    for _, row in df.iterrows():
        ticker = row["ticker"]
        sent = sentiment.get(ticker)
        if sent:
            print(f"  {ticker}: {sent['summary']}")
    print()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_report.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add report.py tests/test_report.py
git commit -m "feat: add terminal report with rich tables"
```

---

### Task 8: Main orchestrator

**Files:**
- Create: `main.py`

**Step 1: Write the orchestrator**

```python
# main.py
"""AI Stock Screener — main entry point.

Run with: python main.py
"""

import os
import sys
import logging

from dotenv import load_dotenv

from data import fetch_sp500_tickers, fetch_stock_data
from filters import screen
from sentiment import analyze
from report import print_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        print("1. Get a key at https://console.anthropic.com")
        print("2. Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # Step 1: Fetch S&P 500 tickers
    logger.info("Fetching S&P 500 ticker list...")
    tickers = fetch_sp500_tickers()
    logger.info(f"Found {len(tickers)} tickers")

    # Step 2: Fetch stock data
    logger.info("Fetching stock data (this may take a few minutes)...")
    stock_data = fetch_stock_data(tickers)
    logger.info(f"Successfully fetched data for {len(stock_data)} stocks")

    # Step 3: Apply screening filters
    logger.info("Applying screening filters...")
    screened = screen(stock_data)
    logger.info(f"{len(screened)} stocks passed all filters")

    # Step 4: AI sentiment analysis
    if len(screened) > 0:
        logger.info(f"Running AI sentiment analysis on {len(screened)} stocks...")
        sentiment = analyze(screened["ticker"].tolist())
    else:
        sentiment = {}

    # Step 5: Print report
    print_report(screened, sentiment, total_screened=len(tickers))


if __name__ == "__main__":
    main()
```

**Step 2: Smoke-test it manually**

Run: `python main.py`
Expected: Either runs the full pipeline (if .env exists) or prints the missing API key error message.

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add main orchestrator"
```

---

### Task 9: End-to-end smoke test and final cleanup

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_integration.py`

**Step 1: Create tests/__init__.py**

```python
# tests/__init__.py
```

(Empty file — makes tests importable.)

**Step 2: Write integration test (mocked network)**

```python
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
```

**Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/
git commit -m "test: add integration test for full pipeline"
```

---

## Summary

| Task | Module | What it does |
|------|--------|-------------|
| 1 | Scaffolding | requirements.txt, .gitignore, .env.example |
| 2 | data.py | `fetch_sp500_tickers()` — Wikipedia scrape |
| 3 | data.py | `compute_rsi()` — Wilder RSI from scratch |
| 4 | data.py | `fetch_stock_data()` — yfinance OHLCV + fundamentals |
| 5 | filters.py | `screen()` / `screen_detail()` — 5 filters |
| 6 | sentiment.py | `analyze()` — Claude Haiku headline sentiment |
| 7 | report.py | `print_report()` — rich terminal tables |
| 8 | main.py | Orchestrator — wires it all together |
| 9 | tests/ | Integration test with synthetic data |
