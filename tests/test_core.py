"""Unit tests that run without live API keys or market data."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.dxy_synthesizer import (
    DXY_WEIGHTS,
    calculate_synthetic_dxy,
    extract_close,
    get_dxy_bias,
    get_dxy_context,
)
from src.agents.macro_analyst import analyze_macro_sentiment, _safe_int_score
from src.agents.oracle import GoldOracle
from src.agents.technical_analyst import analyze_technicals
from src.llm.router import FreeLLMRouter
from src.utils.telegram_reporter import TelegramReporter
from src.utils.weekly_accuracy import build_report_body


def _ohlc_df(n: int = 80, start: float = 100.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.2, size=n).cumsum()
    close = start + noise
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(n, 1000),
        }
    )


def test_extract_close_simple():
    df = _ohlc_df(10)
    closes = extract_close(df)
    assert len(closes) == 10
    assert float(closes.iloc[-1]) == pytest.approx(float(df["Close"].iloc[-1]))


def test_extract_close_multiindex():
    df = _ohlc_df(10)
    df.columns = pd.MultiIndex.from_product([df.columns, ["GC=F"]])
    closes = extract_close(df)
    assert len(closes) == 10


def test_get_dxy_bias_thresholds():
    assert get_dxy_bias(100.3, 100.0) == "STRONG_DXY"
    assert get_dxy_bias(100.1, 100.0) == "MODERATE_DXY"
    assert get_dxy_bias(99.7, 100.0) == "WEAK_DXY"
    assert get_dxy_bias(99.9, 100.0) == "MODERATE_WEAK_DXY"
    assert get_dxy_bias(100.0, 100.0) == "NEUTRAL"


def test_calculate_synthetic_dxy_and_context():
    data = {}
    for ticker in DXY_WEIGHTS:
        # Use different seeds so product isn't degenerate
        data[ticker] = _ohlc_df(60, start=1.1 if "EUR" in ticker or "GBP" in ticker else 100.0)
    value = calculate_synthetic_dxy(data, lookback=0)
    assert value > 0
    ctx = get_dxy_context(data)
    assert "dxy_bias" in ctx
    assert "score" in ctx
    assert "current_dxy" in ctx


def test_analyze_technicals_enough_data():
    result = analyze_technicals(_ohlc_df(80, start=2300.0))
    assert result["signal"] in {
        "BULLISH_BREAKOUT",
        "BULLISH",
        "BEARISH_BREAKDOWN",
        "BEARISH",
        "NEUTRAL",
    }
    assert "rsi" in result
    assert result["price"] > 0


def test_analyze_technicals_insufficient_data():
    result = analyze_technicals(_ohlc_df(10))
    assert result["signal"] == "INSUFFICIENT_DATA"
    assert result["score"] == 0


def test_safe_int_score():
    assert _safe_int_score("2") == 2
    assert _safe_int_score(2.6) == 3
    assert _safe_int_score("nope", default=0) == 0


def test_macro_analyst_normalizes_with_mock_llm():
    fake = {
        "sentiment": "bullish",
        "score": "2",
        "key_factors": ["Fed"],
        "session_context": "London open",
    }
    with patch("src.agents.macro_analyst.FreeLLMRouter") as cls:
        cls.return_value.analyze_session_sentiment.return_value = fake
        out = analyze_macro_sentiment("london", "WEAK_DXY", 2300.0)
    assert out["sentiment"] == "BULLISH"
    assert out["score"] >= 1
    assert out["raw_score"] == 2


def test_llm_router_neutral_without_keys(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    router = FreeLLMRouter()
    result = router.query("test")
    assert result["sentiment"] == "neutral"
    assert result["score"] == 0


def test_llm_safe_json_parse_fenced():
    router = FreeLLMRouter()
    parsed = router._safe_json_parse('```json\n{"sentiment":"bearish","score":-2}\n```')
    assert parsed["sentiment"] == "bearish"
    assert parsed["score"] == -2
    assert "key_factors" in parsed


def test_oracle_aggregate_thresholds():
    oracle = GoldOracle("london")
    decision = oracle._aggregate(
        {"score": 3},
        {"signal": "BULLISH"},
        {"score": 1},
        {"score": 1},
    )
    assert decision["direction"] == "BULLISH"
    assert decision["score"] == 6
    assert 0 < decision["confidence"] <= 1


def test_telegram_parse_chat_ids(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1, 2, ,3")
    t = TelegramReporter()
    assert t.chat_ids == ["1", "2", "3"]
    assert t.enabled is True


def test_telegram_disabled_without_secrets(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    t = TelegramReporter()
    assert t.enabled is False
    assert t.send_test_message() is False


def test_weekly_accuracy_report_body():
    body = build_report_body(accurate=3, wrong=1, signal_count=5)
    assert "Win Rate" in body
    assert "75.0%" in body
    assert "Signals Reviewed:** 5" in body


def test_weights_config_loads():
    path = Path(__file__).resolve().parents[1] / "config" / "weights.json"
    data = json.loads(path.read_text())
    assert "multiplier" in data
    assert "EURUSD=X" in data
    assert abs(sum(v for k, v in data.items() if k != "multiplier")) > 0


def test_main_cli_help():
    from src.main import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
