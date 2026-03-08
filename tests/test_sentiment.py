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
