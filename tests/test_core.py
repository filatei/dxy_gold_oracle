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
from src.agents.opinion import (
    aggregate_opinion,
    consensus_note,
    dxy_action_from_bias,
    gold_action_from_direction,
    majority_action,
    normalize_action,
    normalize_confidence,
    parse_llm_vote,
)
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
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    router = FreeLLMRouter()
    result = router.query("test")
    assert result["sentiment"] == "neutral"
    assert result["score"] == 0
    assert router.available_providers() == []


def test_llm_safe_json_parse_fenced():
    router = FreeLLMRouter()
    parsed = router._safe_json_parse('```json\n{"sentiment":"bearish","score":-2}\n```')
    assert parsed["sentiment"] == "bearish"
    assert parsed["score"] == -2
    assert "key_factors" in parsed


def test_deepseek_query_mocked(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    router = FreeLLMRouter()
    fake = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "sentiment": "bullish",
                            "score": 2,
                            "key_factors": ["yields"],
                            "session_context": "risk-on",
                        }
                    )
                }
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = fake
    with patch("src.llm.router.requests.post", return_value=mock_resp) as post:
        result = router.query("test prompt")
    assert result["sentiment"] == "bullish"
    assert result["score"] == 2
    assert post.call_args.args[0] == "https://api.deepseek.com/chat/completions"
    assert "Bearer sk-test" in post.call_args.kwargs["headers"]["Authorization"]
    assert post.call_args.kwargs["json"]["model"] == "deepseek-v4-flash"


def test_query_all_deepseek_direct_skips_openrouter_deepseek(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    router = FreeLLMRouter()
    assert "deepseek" in router.available_providers()
    assert "openrouter_deepseek" not in router.available_providers()
    with patch.object(router, "_deepseek", return_value={"gold_action": "BUY"}):
        with patch.object(router, "_openrouter", return_value={"gold_action": "HOLD"}):
            with patch.object(router, "_openrouter_deepseek") as ods:
                results = router.query_all("p", "s")
    providers = [r["provider"] for r in results]
    assert providers == ["deepseek", "openrouter"]
    ods.assert_not_called()


def test_query_all_openrouter_deepseek_when_no_direct_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    router = FreeLLMRouter()
    assert router.available_providers() == ["openrouter", "openrouter_deepseek"]
    with patch.object(router, "_openrouter", return_value={"gold_action": "HOLD"}):
        with patch.object(
            router, "_openrouter_deepseek", return_value={"gold_action": "BUY"}
        ) as ods:
            results = router.query_all("p", "s")
    assert [r["provider"] for r in results] == ["openrouter", "openrouter_deepseek"]
    ods.assert_called_once()


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


def test_normalize_action_aliases_and_junk():
    assert normalize_action("buy") == "BUY"
    assert normalize_action("LONG") == "BUY"
    assert normalize_action("gold_sell") == "SELL"
    assert normalize_action("NEUTRAL") == "HOLD"
    assert normalize_action("YOLO") == "HOLD"
    assert normalize_action(None) == "HOLD"
    assert normalize_action("<script>alert(1)</script>") == "HOLD"


def test_normalize_confidence_fraction_and_clamp():
    assert normalize_confidence(0.8) == 80
    assert normalize_confidence(72) == 72
    assert normalize_confidence(150) == 100
    assert normalize_confidence(-3) == 0
    assert normalize_confidence("nope", default=50) == 50


def test_majority_action_ties_hold():
    assert majority_action(["BUY", "BUY", "SELL"]) == "BUY"
    assert majority_action(["BUY", "SELL"]) == "HOLD"
    assert majority_action(["BUY", "SELL"], tiebreaker="BUY") == "BUY"
    assert majority_action([]) == "HOLD"


def test_agent_direction_maps_to_actions():
    assert gold_action_from_direction("BULLISH") == "BUY"
    assert gold_action_from_direction("MODERATE_BEARISH") == "SELL"
    assert gold_action_from_direction("NEUTRAL") == "HOLD"
    assert dxy_action_from_bias("WEAK_DXY") == "SELL"
    assert dxy_action_from_bias("STRONG_DXY") == "BUY"


def test_parse_llm_vote_coerces_bad_json_fields():
    vote = parse_llm_vote(
        "groq",
        {"gold_action": "yolo", "dxy_action": 123, "confidence": "0.4", "rationale": "x"},
    )
    assert vote["gold_action"] == "HOLD"
    assert vote["dxy_action"] == "HOLD"
    assert vote["confidence"] == 40
    assert vote["source"] == "groq"


def test_aggregate_opinion_majority_and_consensus_note():
    votes = [
        {
            "source": "groq",
            "gold_action": "BUY",
            "dxy_action": "SELL",
            "confidence": 80,
            "rationale": "Mock vote A.",
        },
        {
            "source": "gemini",
            "gold_action": "BUY",
            "dxy_action": "HOLD",
            "confidence": 60,
            "rationale": "Mock vote B.",
        },
        {
            "source": "openrouter",
            "gold_action": "SELL",
            "dxy_action": "SELL",
            "confidence": 40,
            "rationale": "Mock vote C.",
        },
    ]
    out = aggregate_opinion(
        agent_gold="HOLD",
        agent_dxy="HOLD",
        agent_confidence=50,
        agent_rationale="agent fallback",
        votes=votes,
        keys_configured=3,
    )
    assert out["gold_action"] == "BUY"
    assert out["dxy_action"] == "SELL"
    assert out["confidence"] == 60
    assert "2/3 models lean GOLD BUY" in out["consensus_note"]
    assert "2/3 lean DXY SELL" in out["consensus_note"]
    assert out["rationale"] == "Mock vote A."


def test_aggregate_opinion_no_llms_uses_agent():
    out = aggregate_opinion(
        agent_gold="BUY",
        agent_dxy="SELL",
        agent_confidence=60,
        agent_rationale="DXY is WEAK_DXY.",
        votes=[],
        keys_configured=0,
    )
    assert out["gold_action"] == "BUY"
    assert out["dxy_action"] == "SELL"
    assert out["llm_count"] == 0
    assert "No LLM keys" in out["consensus_note"]


def test_oracle_form_opinion_agent_fallback(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    oracle = GoldOracle("london")
    opinion = oracle._form_opinion(
        {"dxy_bias": "WEAK_DXY"},
        {"price": 0, "signal": "NEUTRAL", "rsi": 50},
        {"regime": "NEUTRAL", "correlation": 0},
        {"sentiment": "NEUTRAL", "session_context": ""},
        {"direction": "BULLISH", "confidence": 0.6, "score": 6},
        "DXY is WEAK_DXY.",
    )
    assert opinion["gold_action"] == "BUY"
    assert opinion["dxy_action"] == "SELL"
    assert opinion["llm_count"] == 0


def test_oracle_form_opinion_majority_from_mocked_llms():
    raw = [
        {
            "provider": "groq",
            "payload": {
                "gold_action": "BUY",
                "dxy_action": "SELL",
                "confidence": 80,
                "rationale": "Mock groq vote.",
            },
        },
        {
            "provider": "gemini",
            "payload": {
                "gold_action": "BUY",
                "dxy_action": "SELL",
                "confidence": 50,
                "rationale": "Mock gemini vote.",
            },
        },
    ]
    oracle = GoldOracle("ny")
    with patch("src.agents.oracle.FreeLLMRouter") as cls:
        inst = cls.return_value
        inst.available_providers.return_value = ["groq", "gemini"]
        inst.analyze_trading_opinion.return_value = raw
        opinion = oracle._form_opinion(
            {"dxy_bias": "NEUTRAL"},
            {},
            {},
            {},
            {"direction": "NEUTRAL", "confidence": 0.1, "score": 0},
            "flat",
        )
    assert opinion["gold_action"] == "BUY"
    assert opinion["dxy_action"] == "SELL"
    assert opinion["llm_count"] == 2
    assert "GOLD BUY" in opinion["consensus_note"]


def test_query_all_only_configured(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    router = FreeLLMRouter()
    with patch.object(router, "_groq", return_value={"gold_action": "HOLD"}):
        with patch.object(router, "_gemini") as gem:
            with patch.object(router, "_deepseek") as ds:
                with patch.object(router, "_openrouter") as opr:
                    with patch.object(router, "_openrouter_deepseek") as ods:
                        results = router.query_all("p", "s")
    assert len(results) == 1
    assert results[0]["provider"] == "groq"
    gem.assert_not_called()
    ds.assert_not_called()
    opr.assert_not_called()
    ods.assert_not_called()


def _sample_report(**opinion_overrides):
    opinion = {
        "gold_action": "BUY",
        "dxy_action": "SELL",
        "confidence": 72,
        "confidence_label": "High",
        "rationale": "Mock rationale for formatter test.",
        "consensus_note": "2/3 models lean GOLD BUY",
    }
    opinion.update(opinion_overrides)
    return {
        "session": "london",
        "timestamp": "2026-08-21T07:00:00",
        "decision": {"direction": "BULLISH", "confidence": 0.6, "score": 6},
        "trade_setup": {"confidence": "60.0%", "rationale": "formatter test rationale"},
        "agents": {
            "dxy": {"dxy_bias": "WEAK_DXY"},
            "tech": {"signal": "BULLISH"},
            "macro": {"sentiment": "BULLISH"},
            "corr": {"regime": "STRONG_INVERSE"},
        },
        "opinion": opinion,
    }


def test_telegram_opinion_block_prominent():
    text = TelegramReporter().format_message(_sample_report())
    assert "GOLD: BUY" in text
    assert "DXY: SELL" in text
    assert text.index("OPINION") < text.index("Direction:")
    assert text.index("GOLD: BUY") < text.index("Direction:")
    assert "🟢" in text
    assert "2/3 models lean GOLD BUY" in text


def test_telegram_invalid_actions_hold_and_escape():
    text = TelegramReporter().format_message(
        _sample_report(
            gold_action="YOLO",
            dxy_action="<script>",
            rationale="a < b & c",
            consensus_note="",
        )
    )
    assert "GOLD: HOLD" in text
    assert "DXY: HOLD" in text
    assert "<script>" not in text
    assert "a &lt; b &amp; c" in text


def test_telegram_plain_opinion_matches_actions():
    plain = TelegramReporter().format_opinion_plain(_sample_report()["opinion"])
    assert "GOLD: BUY" in plain
    assert "DXY: SELL" in plain
    assert "High (72)" in plain


def test_consensus_note_single_model():
    note = consensus_note(
        [{"gold_action": "HOLD", "dxy_action": "HOLD"}],
        "HOLD",
        "HOLD",
    )
    assert note.startswith("1/1 model:")

