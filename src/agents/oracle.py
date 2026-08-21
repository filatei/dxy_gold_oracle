"""Main Oracle that aggregates all agent signals."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from src.agents.correlation_engine import analyze_dxy_gold_correlation
from src.agents.dxy_synthesizer import fetch_components, fetch_gold, get_dxy_context
from src.agents.macro_analyst import analyze_macro_sentiment
from src.agents.technical_analyst import analyze_technicals


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

        report = {
            "timestamp": timestamp,
            "session": self.session,
            "decision": decision,
            "agents": self.context,
            "trade_setup": {
                "recommended_direction": decision["direction"],
                "confidence": f"{decision['confidence'] * 100:.1f}%",
                "score": decision["score"],
                "rationale": self._build_rationale(dxy_ctx, tech, corr, macro),
            },
        }

        print(
            f"[Oracle] Decision: {decision['direction']} "
            f"(confidence: {decision['confidence']:.0%})"
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
