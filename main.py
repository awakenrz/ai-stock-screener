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
