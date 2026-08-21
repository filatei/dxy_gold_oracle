"""Macro analysis agent using LLM."""
from __future__ import annotations

from typing import Any, Dict

from src.llm.router import FreeLLMRouter


def _safe_int_score(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def analyze_macro_sentiment(session: str, dxy_bias: str, gold_price: float) -> Dict[str, Any]:
    """Query LLM swarm for macro sentiment."""
    router = FreeLLMRouter()
    result = router.analyze_session_sentiment(session, dxy_bias, gold_price)

    raw_score = _safe_int_score(result.get("score", 0), 0)
    clamped = max(-3, min(3, raw_score))

    sentiment = str(result.get("sentiment", "neutral")).lower().strip()
    if sentiment == "bullish":
        score = max(1, clamped) if clamped != 0 else 1
    elif sentiment == "bearish":
        score = min(-1, clamped) if clamped != 0 else -1
    else:
        score = 0
        sentiment = "neutral"

    return {
        "sentiment": sentiment.upper(),
        "score": score,
        "raw_score": raw_score,
        "key_factors": result.get("key_factors") or [],
        "session_context": result.get("session_context") or "",
    }
