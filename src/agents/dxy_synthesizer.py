"""Synthetic DXY calculation from FX components."""
from __future__ import annotations

import time
from typing import Any, Dict

import pandas as pd
import yfinance as yf

from src.config import load_dxy_weights

_cfg = load_dxy_weights()
DXY_MULTIPLIER = _cfg["multiplier"]
DXY_WEIGHTS: Dict[str, float] = dict(_cfg["weights"])
DXY_TICKERS = list(DXY_WEIGHTS.keys())


def extract_close(df: pd.DataFrame) -> pd.Series:
    """Normalize yfinance Close column (handles MultiIndex / single-column quirks)."""
    if df is None or df.empty:
        raise ValueError("Empty dataframe")
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer ('Close', ticker) or first Close level
        if "Close" in df.columns.get_level_values(0):
            closes = df["Close"]
            if isinstance(closes, pd.DataFrame):
                closes = closes.iloc[:, 0]
            return closes.astype(float)
    if "Close" in df.columns:
        closes = df["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        return closes.astype(float)
    raise KeyError("Close column not found in dataframe")


def _fetch_with_retry(
    ticker: str, period: str = "5d", interval: str = "1h", retries: int = 3
) -> pd.DataFrame:
    """Fetch single ticker with retry logic."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if df is None or df.empty:
                raise ValueError(f"Empty data for {ticker}")
            # Validate Close is readable
            _ = extract_close(df)
            return df
        except Exception as e:
            last_err = e
            print(f"[DXY] Fetch attempt {attempt + 1}/{retries} failed for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch {ticker}") from last_err


def fetch_components() -> Dict[str, pd.DataFrame]:
    """Fetch 5d/1h data for all DXY constituents."""
    data: Dict[str, pd.DataFrame] = {}
    for ticker in DXY_TICKERS:
        print(f"[DXY] Fetching {ticker}...")
        data[ticker] = _fetch_with_retry(ticker)
        time.sleep(0.5)
    return data


def fetch_gold() -> pd.DataFrame:
    """Fetch gold futures data."""
    print("[DXY] Fetching GC=F (Gold)...")
    return _fetch_with_retry("GC=F")


def calculate_synthetic_dxy(data: Dict[str, pd.DataFrame], lookback: int = 0) -> float:
    """
    Calculate synthetic DXY value.
    lookback: 0 = latest, 1 = previous bar, etc.
    """
    product = DXY_MULTIPLIER
    idx = -(lookback + 1)
    for ticker, weight in DXY_WEIGHTS.items():
        closes = extract_close(data[ticker])
        if abs(idx) > len(closes):
            raise IndexError(f"lookback {lookback} out of range for {ticker}")
        close = float(closes.iloc[idx])
        if close <= 0:
            raise ValueError(f"Non-positive close for {ticker}: {close}")
        product *= close**weight
    return product


def get_dxy_bias(current: float, previous: float) -> str:
    """Classify dollar strength trend."""
    change = (current - previous) / previous if previous != 0 else 0
    if change > 0.002:
        return "STRONG_DXY"
    if change > 0.0005:
        return "MODERATE_DXY"
    if change < -0.002:
        return "WEAK_DXY"
    if change < -0.0005:
        return "MODERATE_WEAK_DXY"
    return "NEUTRAL"


def get_dxy_context(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Full DXY analysis context for the oracle."""
    current = calculate_synthetic_dxy(data, lookback=0)
    # Approx 24h ago on 1h bars; clamp if series is shorter
    min_len = min(len(extract_close(df)) for df in data.values())
    lookback_24 = min(24, max(1, min_len - 1))
    prev_24h = calculate_synthetic_dxy(data, lookback=lookback_24)
    prev_1h = calculate_synthetic_dxy(data, lookback=1)

    bias = get_dxy_bias(current, prev_24h)
    hourly_change = (current - prev_1h) / prev_1h if prev_1h != 0 else 0

    return {
        "current_dxy": round(current, 4),
        "prev_24h_dxy": round(prev_24h, 4),
        "hourly_change_pct": round(hourly_change * 100, 4),
        "dxy_bias": bias,
        "score": _bias_to_score(bias),
    }


def _bias_to_score(bias: str) -> int:
    mapping = {
        "STRONG_DXY": -3,
        "MODERATE_DXY": -2,
        "WEAK_DXY": 3,
        "MODERATE_WEAK_DXY": 2,
        "NEUTRAL": 0,
    }
    return mapping.get(bias, 0)
