"""Main Oracle that aggregates all agent signals."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from src.agents.correlation_engine import analyze_dxy_gold_correlation
from src.agents.dxy_synthesizer import fetch_components, fetch_gold, get_dxy_context
from src.agents.macro_analyst import analyze_macro_sentiment
from src.agents.opinion import (
    aggregate_opinion,
    dxy_action_from_bias,
    gold_action_from_direction,
    parse_llm_vote,
)
from src.agents.technical_analyst import analyze_technicals
from src.llm.router import FreeLLMRouter


class GoldOracle:
    def __init__(self, session: str):
        self.session = session  # "london", "ny", "asia"
        self.context: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """Execute full analysis pipeline."""
        timestamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

        print(f"[Oracle] Starting {self.session.upper()} session analysis...")

        print("[Oracle] Fetching market data...")
        dxy_data = fetch_components()
        gold_data = fetch_gold()

        print("[Oracle] Analyzing DXY components...")
        dxy_ctx = get_dxy_context(dxy_data)
        self.context["dxy"] = dxy_ctx

        print("[Oracle] Running technical analysis on Gold...")
        tech = analyze_technicals(gold_data)
        self.context["tech"] = tech

        print("[Oracle] Computing DXY-Gold correlation...")
        corr = analyze_dxy_gold_correlation(dxy_data, gold_data)
        self.context["corr"] = corr

        print("[Oracle] Querying LLM swarm for macro sentiment...")
        gold_price = float(tech.get("price", 0.0) or 0.0)
        macro = analyze_macro_sentiment(self.session, dxy_ctx["dxy_bias"], gold_price)
        self.context["macro"] = macro

        decision = self._aggregate(dxy_ctx, tech, corr, macro)
        rationale = self._build_rationale(dxy_ctx, tech, corr, macro)
        opinion = self._form_opinion(dxy_ctx, tech, corr, macro, decision, rationale)

        report = {
            "timestamp": timestamp,
            "session": self.session,
            "decision": decision,
            "opinion": opinion,
            "agents": self.context,
            "trade_setup": {
                "recommended_direction": decision["direction"],
                "confidence": f"{decision['confidence'] * 100:.1f}%",
                "score": decision["score"],
                "rationale": rationale,
            },
        }

        print(
            f"[Oracle] Decision: {decision['direction']} "
            f"(confidence: {decision['confidence']:.0%})"
        )
        print(
            f"[Oracle] Opinion: GOLD {opinion['gold_action']} / "
            f"DXY {opinion['dxy_action']} "
            f"({opinion['confidence_label']} {opinion['confidence']}) "
            f"— {opinion['consensus_note']}"
        )
        return report

    def _aggregate(
        self, dxy: Dict, tech: Dict, corr: Dict, macro: Dict
    ) -> Dict[str, Any]:
        """Weighted voting system."""
        score = 0
        score += int(dxy.get("score", 0) or 0)

        tech_score_map = {
            "BULLISH_BREAKOUT": 3,
            "BULLISH": 1,
            "BEARISH_BREAKDOWN": -3,
            "BEARISH": -1,
            "NEUTRAL": 0,
            "INSUFFICIENT_DATA": 0,
        }
        score += tech_score_map.get(tech.get("signal", "NEUTRAL"), 0)
        score += int(corr.get("score", 0) or 0)
        score += int(macro.get("score", 0) or 0)

        if score >= 4:
            direction = "BULLISH"
        elif score >= 2:
            direction = "MODERATE_BULLISH"
        elif score <= -4:
            direction = "BEARISH"
        elif score <= -2:
            direction = "MODERATE_BEARISH"
        else:
            direction = "NEUTRAL"

        confidence = min(abs(score) / 10, 1.0)

        return {
            "direction": direction,
            "confidence": confidence,
            "score": score,
        }

    def _build_rationale(self, dxy: Dict, tech: Dict, corr: Dict, macro: Dict) -> str:
        ema20 = tech.get("ema20", 0) or 0
        ema50 = tech.get("ema50", 0) or 0
        parts = [
            f"DXY is {dxy.get('dxy_bias', 'unknown')} ({dxy.get('hourly_change_pct', 0):.2f}% 1h).",
            f"Gold technicals: {tech.get('signal', 'N/A')} "
            f"(RSI {tech.get('rsi', 'N/A')}, EMA20/50 {'>' if ema20 > ema50 else '<'}).",
            f"DXY-Gold correlation: {corr.get('regime', 'N/A')} ({corr.get('correlation', 0)}).",
            f"Macro/LLM sentiment: {macro.get('sentiment', 'N/A')} "
            f"({macro.get('session_context', '')}).",
        ]
        return " ".join(parts)

    def _form_opinion(
        self, dxy: Dict, tech: Dict, corr: Dict, macro: Dict, decision: Dict, rationale: str
    ) -> Dict[str, Any]:
        """Multi-LLM BUY/SELL/HOLD vote, falling back to agent consensus."""
        agent_gold = gold_action_from_direction(decision.get("direction", "NEUTRAL"))
        agent_dxy = dxy_action_from_bias(dxy.get("dxy_bias", "NEUTRAL"))
        try:
            agent_conf = int(round(min(1.0, abs(float(decision.get("confidence") or 0))) * 100))
        except (TypeError, ValueError):
            agent_conf = 50

        router = FreeLLMRouter()
        keys_configured = len(router.available_providers())
        context = {
            "session": self.session,
            "current_dxy": dxy.get("current_dxy"),
            "dxy_bias": dxy.get("dxy_bias"),
            "hourly_change_pct": dxy.get("hourly_change_pct"),
            "gold_price": tech.get("price"),
            "tech_signal": tech.get("signal"),
            "rsi": tech.get("rsi"),
            "corr_regime": corr.get("regime"),
            "correlation": corr.get("correlation"),
            "macro_sentiment": macro.get("sentiment"),
            "macro_context": macro.get("session_context") or "",
            "direction": decision.get("direction"),
            "score": decision.get("score"),
        }

        raw_votes: list = []
        if keys_configured:
            print(f"[Oracle] Collecting trading votes from {keys_configured} LLM(s)...")
            try:
                raw_votes = router.analyze_trading_opinion(context)
            except Exception as e:
                print(f"[Oracle] LLM opinion swarm failed: {e}")
                raw_votes = []

        votes = [
            parse_llm_vote(item.get("provider", "unknown"), item.get("payload"))
            for item in raw_votes
            if isinstance(item, dict)
        ]

        opinion = aggregate_opinion(
            agent_gold=agent_gold,
            agent_dxy=agent_dxy,
            agent_confidence=agent_conf,
            agent_rationale=rationale,
            votes=votes,
            keys_configured=keys_configured,
        )
        return opinion
