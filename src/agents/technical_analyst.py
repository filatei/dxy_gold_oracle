"""Technical analysis agent for Gold."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.agents.dxy_synthesizer import extract_close


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1])
    if pd.isna(val):
        return 50.0
    return val


def analyze_technicals(gold_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze gold price action. Returns signal, score, and metadata."""
    closes = extract_close(gold_df)

    if len(closes) < 50:
        return {
            "signal": "INSUFFICIENT_DATA",
            "score": 0,
            "rsi": 50,
            "trend": "flat",
            "price": float(closes.iloc[-1]) if len(closes) else 0.0,
            "ema20": 0.0,
            "ema50": 0.0,
            "recent_high": 0.0,
            "recent_low": 0.0,
            "signals": [],
        }

    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)

    current_price = float(closes.iloc[-1])
    ema20_val = float(ema20.iloc[-1])
    ema50_val = float(ema50.iloc[-1])
    rsi_val = _rsi(closes, 14)

    recent_high = float(closes.tail(20).max())
    recent_low = float(closes.tail(20).min())

    score = 0
    signals = []

    if ema20_val > ema50_val * 1.001:
        signals.append("bullish_trend")
        score += 1
    elif ema20_val < ema50_val * 0.999:
        signals.append("bearish_trend")
        score -= 1
    else:
        signals.append("trendless")

    if rsi_val > 60:
        signals.append("rsi_bullish")
        score += 1
    elif rsi_val < 40:
        signals.append("rsi_bearish")
        score -= 1
    else:
        signals.append("rsi_neutral")

    range_size = recent_high - recent_low
    if range_size > 0:
        position = (current_price - recent_low) / range_size
        if position > 0.8:
            signals.append("near_resistance")
            score -= 1
        elif position < 0.2:
            signals.append("near_support")
            score += 1
        else:
            signals.append("mid_range")

    if current_price > ema20_val * 1.002:
        signals.append("above_ema20")
        score += 1
    elif current_price < ema20_val * 0.998:
        signals.append("below_ema20")
        score -= 1

    if score >= 3:
        signal = "BULLISH_BREAKOUT"
    elif score >= 1:
        signal = "BULLISH"
    elif score <= -3:
        signal = "BEARISH_BREAKDOWN"
    elif score <= -1:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    return {
        "signal": signal,
        "score": score,
        "rsi": round(rsi_val, 2),
        "ema20": round(ema20_val, 2),
        "ema50": round(ema50_val, 2),
        "price": round(current_price, 2),
        "recent_high": round(recent_high, 2),
        "recent_low": round(recent_low, 2),
        "signals": signals,
    }
