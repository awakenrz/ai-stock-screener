# AI Stock Screener — Design Document

**Date:** 2026-03-07
**Goal:** Learning project — build an AI-powered stock screener to understand how yfinance, Claude API, and data pipelines fit together.
**Approach:** Multi-module Python package with terminal output.

## Project Structure

```
ai-stock-screener/
├── .env                    # ANTHROPIC_API_KEY (git-ignored)
├── .gitignore
├── requirements.txt
├── main.py                 # Orchestrator — runs the pipeline
├── data.py                 # Fetches S&P 500 tickers + stock data via yfinance
├── filters.py              # Technical/fundamental screening filters
├── sentiment.py            # Claude API sentiment analysis
└── report.py               # Terminal output formatting (rich tables)
```

## Pipeline Flow

```
main.py
  │
  ├─ data.fetch_tickers()        → list[str]          (503 S&P 500 symbols)
  ├─ data.fetch_stock_data()     → pd.DataFrame       (one row per ticker)
  ├─ filters.screen()            → pd.DataFrame       (stocks passing all filters)
  ├─ sentiment.analyze()         → dict[str, dict]     (sentiment per ticker)
  └─ report.print_report()       → terminal output
```

## Module Designs

### data.py

**`fetch_sp500_tickers() -> list[str]`**
- Scrapes Wikipedia "List of S&P 500 companies" using `pandas.read_html()`.
- Returns ~503 ticker symbols.

**`fetch_stock_data(tickers: list[str]) -> pd.DataFrame`**
- For each ticker, uses `yfinance.Ticker(symbol)` to pull:
  - 60 trading days of OHLCV (enough for 50-day SMA).
  - Fundamentals from `.info`: P/E ratio (`forwardPE` or `trailingPE`), market cap, sector.
- Computes derived columns: 20-day SMA, 50-day SMA, RSI(14), volume ratio (latest / 20-day avg).
- `time.sleep(0.3)` between tickers to avoid Yahoo rate-limiting.
- Returns one row per ticker.

**RSI calculation:** 14-day RSI using Wilder smoothing (avg gain / avg loss). No external TA library.

### filters.py

**`screen(df: pd.DataFrame) -> pd.DataFrame`**
Applies all filters and returns rows where every condition is True:

| Filter         | Condition              | Rationale                                  |
|----------------|------------------------|--------------------------------------------|
| P/E Ratio      | 5 ≤ P/E ≤ 25          | Eliminates no-earnings and extreme overval  |
| RSI            | 30 ≤ RSI ≤ 70          | Avoids overbought/oversold extremes        |
| Volume Spike   | volume ≥ 1.5× 20d avg  | Institutional money flowing in             |
| Price > 50 SMA | close > SMA_50         | Confirms uptrend                           |
| Golden Cross   | SMA_20 > SMA_50        | Bullish crossover signal                   |

Each filter is a separate boolean column so you can inspect pass/fail per stock.

**`screen_detail(df: pd.DataFrame) -> pd.DataFrame`**
Returns the full DataFrame with boolean filter columns (for inspecting near-misses).

### sentiment.py

**`analyze(tickers: list[str]) -> dict[str, dict]`**
- For each ticker, grabs 5 most recent headlines via `yfinance.Ticker(symbol).news`.
- Sends headlines to Claude Haiku (`claude-3-5-haiku-20241022`) with a structured prompt requesting JSON:
  - `sentiment_score`: float, -1.0 (bearish) to 1.0 (bullish)
  - `summary`: one-sentence explanation
  - `key_catalyst`: most important headline
- Parses JSON with `try/except` fallback (strips markdown backticks).
- ~10 API calls/day at ~$0.001/call = ~$0.01/day.
- On API error: retry once with 2s backoff, then fall back to `sentiment_score: None`.

### report.py

**`print_report(screened_df: pd.DataFrame, sentiment_data: dict)`**
- Uses `rich` library to print a formatted table with columns: Ticker, Price, P/E, RSI, Vol Spike, Sentiment.
- Sentiment color-coded: green (> 0.3), yellow (-0.3 to 0.3), red (< -0.3).
- AI commentary section below the table with one-line summaries per stock.
- Falls back to plain text if `rich` is unavailable.
- Shows summary line: "Screened X stocks → Y passed all filters".

## Error Handling

| Failure Mode                 | Response                                             |
|------------------------------|------------------------------------------------------|
| yfinance returns empty data  | Skip ticker, log warning, continue. Report skip count. |
| Claude API error             | Retry once (2s backoff). Fall back to None sentiment. |
| No stocks pass all filters   | Print "No matches" message. Suggest loosening a filter. |
| Network issues               | Top-level try/except with clear error message.       |
| Missing .env / API key       | Check at startup, print help pointing to console.anthropic.com. |

## Dependencies (requirements.txt)

```
yfinance
pandas
anthropic
python-dotenv
rich
```

## Decisions

- **Manual runs only** — no cron or GitHub Actions. Run with `python main.py`.
- **Terminal output only** — no email. Can be added later.
- **No external TA library** — RSI computed manually for learning purposes.
- **Claude Haiku model** — cheapest and fast enough for headline sentiment.
- **S&P 500 universe** — scraped from Wikipedia, not hardcoded.
