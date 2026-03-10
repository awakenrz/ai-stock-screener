# sentiment.py
"""Claude AI sentiment analysis for stock news headlines."""

import json
import os
import sys
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


def _call_claude(client: anthropic.Anthropic, prompt: str) -> str:
    """Send a prompt to Claude and return the raw response text."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _parse_sentiment_response(raw: str) -> dict | None:
    """Parse Claude's response into a sentiment dict.

    Handles markdown code fences and malformed JSON gracefully.
    Returns None if parsing or validation fails.
    """
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse sentiment JSON: %s", text[:100])
        return None

    # Validate required fields and types
    score = data.get("sentiment_score")
    if not isinstance(score, (int, float)) or score < -1.0 or score > 1.0:
        logger.warning(
            "Invalid sentiment_score (must be number in [-1.0, 1.0]): %s", score,
        )
        return None

    if not isinstance(data.get("summary"), str):
        logger.warning("Invalid or missing 'summary': expected string")
        return None

    if not isinstance(data.get("key_catalyst"), str):
        logger.warning("Invalid or missing 'key_catalyst': expected string")
        return None

    return data


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
        client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY"),
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        )

    results = {}
    total = len(tickers)

    for idx, symbol in enumerate(tickers):
        print(f"\r  Sentiment analysis: {idx + 1}/{total} — {symbol}    ", end="", flush=True, file=sys.stderr)

        prompt = ""
        try:
            stock = yf.Ticker(symbol)
            news = stock.news or []
            headlines = [
                item.get("content", {}).get("title", "")
                for item in news[:5]
            ]
            headlines = [h for h in headlines if h]

            if not headlines:
                logger.warning("%s: no news headlines found", symbol)
                results[symbol] = None
                continue

            headline_text = "\n".join(f"- {h}" for h in headlines)
            prompt = PROMPT_TEMPLATE.format(
                ticker=symbol, headlines=headline_text,
            )

            raw_text = _call_claude(client, prompt)
            results[symbol] = _parse_sentiment_response(raw_text)

        except anthropic.RateLimitError as e:
            retry_after = int(e.response.headers.get("retry-after", "30"))
            logger.warning("%s: rate limited, waiting %ds...", symbol, retry_after)
            time.sleep(retry_after)
            if not prompt:
                logger.error("%s: no prompt to retry (error before prompt was built)", symbol)
                results[symbol] = None
                continue
            try:
                raw_text = _call_claude(client, prompt)
                results[symbol] = _parse_sentiment_response(raw_text)
            except Exception as retry_err:
                logger.error("%s: retry failed: %s", symbol, retry_err)
                results[symbol] = None

        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                logger.warning("%s: server error (%d), retrying in 2s...", symbol, e.status_code)
                time.sleep(2)
                if not prompt:
                    logger.error("%s: no prompt to retry (error before prompt was built)", symbol)
                    results[symbol] = None
                    continue
                try:
                    raw_text = _call_claude(client, prompt)
                    results[symbol] = _parse_sentiment_response(raw_text)
                except Exception as retry_err:
                    logger.error("%s: retry failed: %s", symbol, retry_err)
                    results[symbol] = None
            else:
                logger.error("%s: API error (%d): %s", symbol, e.status_code, e)
                results[symbol] = None

        except Exception as e:
            logger.error("%s: sentiment analysis failed: %s", symbol, e)
            results[symbol] = None

        # Throttle between Claude API calls (skip on last ticker)
        if idx < total - 1:
            time.sleep(0.5)

    print(file=sys.stderr)  # newline after progress
    return results
