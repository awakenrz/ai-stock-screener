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
