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
