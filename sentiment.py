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
