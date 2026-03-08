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
