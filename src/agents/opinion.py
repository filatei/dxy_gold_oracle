"""Trading opinion: BUY/SELL/HOLD for GOLD and DXY, with LLM vote aggregation.

Educational / directional bias only — not financial advice.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ACTIONS = frozenset({"BUY", "SELL", "HOLD"})

_ACTION_ALIASES = {
    "LONG": "BUY",
    "SHORT": "SELL",
    "NEUTRAL": "HOLD",
    "WAIT": "HOLD",
    "FLAT": "HOLD",
    "BULLISH": "BUY",
    "BEARISH": "SELL",
    "MODERATE_BULLISH": "BUY",
    "MODERATE_BEARISH": "SELL",
    "STRONG_BUY": "BUY",
    "STRONG_SELL": "SELL",
}

_GOLD_BULLISH = frozenset({"BULLISH", "MODERATE_BULLISH"})
_GOLD_BEARISH = frozenset({"BEARISH", "MODERATE_BEARISH"})
_DXY_BUY = frozenset({"STRONG_DXY", "MODERATE_DXY"})
_DXY_SELL = frozenset({"WEAK_DXY", "MODERATE_WEAK_DXY"})


def normalize_action(value: Any, default: str = "HOLD") -> str:
    """Coerce LLM/agent output to BUY, SELL, or HOLD. Junk → default."""
    if default not in ACTIONS:
        default = "HOLD"
    if value is None:
        return default
    raw = str(value).upper().strip()
    if not raw:
        return default
    cleaned = (
        raw.replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
    for prefix in ("GOLD_", "XAU_", "DXY_", "USD_", "DOLLAR_"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    for suffix in ("_GOLD", "_XAU", "_DXY", "_USD"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    mapped = _ACTION_ALIASES.get(cleaned, cleaned)
    return mapped if mapped in ACTIONS else default


def normalize_confidence(value: Any, default: int = 50) -> int:
    """Coerce confidence to integer 0–100. Fractions in (0, 1] are treated as 0–100%."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return int(default)
    if 0.0 < n <= 1.0:
        n *= 100.0
    return int(round(max(0.0, min(100.0, n))))


def confidence_label(confidence: int) -> str:
    if confidence >= 70:
        return "High"
    if confidence >= 40:
        return "Med"
    return "Low"


def one_line(text: Any, limit: int = 200) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


def gold_action_from_direction(direction: str) -> str:
    d = str(direction or "").upper().strip()
    if d in _GOLD_BULLISH:
        return "BUY"
    if d in _GOLD_BEARISH:
        return "SELL"
    return "HOLD"


def dxy_action_from_bias(bias: str) -> str:
    b = str(bias or "").upper().strip()
    if b in _DXY_BUY:
        return "BUY"
    if b in _DXY_SELL:
        return "SELL"
    return "HOLD"


def majority_action(
    actions: Sequence[str], tiebreaker: str = "HOLD"
) -> str:
    """Majority vote. BUY vs SELL ties (or any multi-way tie) use tiebreaker, else HOLD."""
    cleaned = [normalize_action(a) for a in actions]
    if not cleaned:
        return normalize_action(tiebreaker)
    counts = Counter(cleaned)
    top_n = max(counts.values())
    tied = [a for a, n in counts.items() if n == top_n]
    if len(tied) == 1:
        return tied[0]
    tb = normalize_action(tiebreaker)
    if tb in tied:
        return tb
    return "HOLD"


def consensus_note(
    votes: Sequence[Mapping[str, Any]],
    gold_action: str,
    dxy_action: str,
    *,
    keys_configured: int = 0,
) -> str:
    n = len(votes)
    if n == 0:
        if keys_configured:
            return "LLM votes unavailable; opinion from agent consensus"
        return "No LLM keys; opinion from agent consensus"
    gold = normalize_action(gold_action)
    dxy = normalize_action(dxy_action)
    if n == 1:
        return f"1/1 model: GOLD {gold} / DXY {dxy}"
    g = sum(1 for v in votes if normalize_action(v.get("gold_action")) == gold)
    d = sum(1 for v in votes if normalize_action(v.get("dxy_action")) == dxy)
    g_verb = "agree" if g == n else "lean"
    d_verb = "agree" if d == n else "lean"
    return f"{g}/{n} models {g_verb} GOLD {gold}; {d}/{n} {d_verb} DXY {dxy}"


def parse_llm_vote(provider: str, payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    data = payload if isinstance(payload, Mapping) else {}
    return {
        "source": str(provider or "unknown"),
        "gold_action": normalize_action(data.get("gold_action")),
        "dxy_action": normalize_action(data.get("dxy_action")),
        "confidence": normalize_confidence(data.get("confidence"), default=50),
        "rationale": one_line(data.get("rationale") or "", 200),
    }


def aggregate_opinion(
    *,
    agent_gold: str,
    agent_dxy: str,
    agent_confidence: int,
    agent_rationale: str,
    votes: Iterable[Mapping[str, Any]],
    keys_configured: int = 0,
) -> Dict[str, Any]:
    """Combine LLM votes with an agent fallback. Always returns validated enums."""
    parsed: List[Dict[str, Any]] = [dict(v) for v in votes]
    agent_gold = normalize_action(agent_gold)
    agent_dxy = normalize_action(agent_dxy)
    agent_confidence = normalize_confidence(agent_confidence, default=50)
    rationale = one_line(agent_rationale, 200)

    if parsed:
        gold = majority_action(
            [v["gold_action"] for v in parsed], tiebreaker=agent_gold
        )
        dxy = majority_action(
            [v["dxy_action"] for v in parsed], tiebreaker=agent_dxy
        )
        conf = int(round(sum(int(v["confidence"]) for v in parsed) / len(parsed)))
        conf = normalize_confidence(conf, default=agent_confidence)
        rationale = next(
            (
                v["rationale"]
                for v in parsed
                if v["gold_action"] == gold and v["rationale"]
            ),
            parsed[0].get("rationale") or rationale,
        )
    else:
        gold = agent_gold
        dxy = agent_dxy
        conf = agent_confidence

    return {
        "gold_action": normalize_action(gold),
        "dxy_action": normalize_action(dxy),
        "confidence": conf,
        "confidence_label": confidence_label(conf),
        "rationale": one_line(rationale, 200),
        "consensus_note": consensus_note(
            parsed, gold, dxy, keys_configured=keys_configured
        ),
        "llm_count": len(parsed),
        "votes": parsed,
        "agent_vote": {
            "gold_action": agent_gold,
            "dxy_action": agent_dxy,
            "confidence": agent_confidence,
        },
    }
