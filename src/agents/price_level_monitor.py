"""XAU/USD short-window move + key-level Telegram alerts.

Uses gold futures (GC=F) as the project's XAUUSD proxy (same as session oracle).
Fires when price has moved unusually in a short window AND interacts with a
round psychological level ($50 steps by default). State/cooldown lives under
data/ so GitHub Actions can commit it between runs (like telegram_offset.json).

Defaults (config/xau_price_alerts.json; overridable via env):
  - 15m bars, window = 4 bars ≈ 1 hour
  - unusual move ≥ 0.5% over that window (gold ~2–3× DXY's 0.2% "strong" day move)
  - key levels every $50; "at level" within $3 or crossed in-window
  - cooldown 120 minutes per (level, direction)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agents.dxy_synthesizer import _fetch_with_retry, extract_close
from src.config import CONFIG_DIR, DATA_DIR, ensure_data_dir

ALERTS_CONFIG_PATH = CONFIG_DIR / "xau_price_alerts.json"
STATE_PATH = DATA_DIR / "xau_price_alert_state.json"

# Env overrides (optional)
_ENV_MOVE_PCT = "XAU_ALERT_MOVE_PCT"
_ENV_WINDOW_BARS = "XAU_ALERT_WINDOW_BARS"
_ENV_LEVEL_STEP = "XAU_ALERT_LEVEL_STEP"
_ENV_PROXIMITY = "XAU_ALERT_LEVEL_PROXIMITY"
_ENV_COOLDOWN = "XAU_ALERT_COOLDOWN_MINUTES"


@dataclass(frozen=True)
class AlertConfig:
    ticker: str
    period: str
    interval: str
    move_window_bars: int
    move_pct_threshold: float
    level_step_usd: float
    level_proximity_usd: float
    cooldown_minutes: int
    levels_around_price: int


@dataclass(frozen=True)
class PriceAlert:
    kind: str  # "key_level_move"
    direction: str  # "up" | "down"
    level: float
    price: float
    window_move_pct: float
    window_bars: int
    interval: str
    reason: str

    @property
    def dedupe_key(self) -> str:
        return f"level:{int(self.level)}:{self.direction}"


def load_alert_config(path: Path | None = None) -> AlertConfig:
    cfg_path = path or ALERTS_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    move_pct = float(os.getenv(_ENV_MOVE_PCT) or raw.get("move_pct_threshold", 0.5))
    window_bars = int(os.getenv(_ENV_WINDOW_BARS) or raw.get("move_window_bars", 4))
    level_step = float(os.getenv(_ENV_LEVEL_STEP) or raw.get("level_step_usd", 50))
    proximity = float(os.getenv(_ENV_PROXIMITY) or raw.get("level_proximity_usd", 3.0))
    cooldown = int(os.getenv(_ENV_COOLDOWN) or raw.get("cooldown_minutes", 120))

    return AlertConfig(
        ticker=str(raw.get("ticker") or "GC=F"),
        period=str(raw.get("period") or "5d"),
        interval=str(raw.get("interval") or "15m"),
        move_window_bars=max(1, window_bars),
        move_pct_threshold=max(0.01, move_pct),
        level_step_usd=max(1.0, level_step),
        level_proximity_usd=max(0.0, proximity),
        cooldown_minutes=max(1, cooldown),
        levels_around_price=max(2, int(raw.get("levels_around_price", 12))),
    )


def round_key_levels(
    price: float,
    step: float,
    around: int,
) -> List[float]:
    """Generate psychological round levels near price (e.g. …2600, 2650, 2700…)."""
    if price <= 0 or step <= 0:
        return []
    center = round(price / step) * step
    half = around // 2
    levels = [center + (i - half) * step for i in range(around + 1)]
    # Deduplicate while preserving order
    seen = set()
    out: List[float] = []
    for lvl in levels:
        key = round(lvl, 2)
        if key in seen or key <= 0:
            continue
        seen.add(key)
        out.append(float(key))
    return out


def _crossed_or_near(
    prev: float,
    curr: float,
    level: float,
    proximity: float,
) -> bool:
    """True if price is within proximity of level, or crossed it this bar."""
    if abs(curr - level) <= proximity:
        return True
    lo, hi = (prev, curr) if prev <= curr else (curr, prev)
    return lo <= level <= hi


def detect_alerts(
    closes: List[float],
    cfg: AlertConfig,
) -> List[PriceAlert]:
    """Detect unusual short-window moves that interact with key levels."""
    need = cfg.move_window_bars + 1
    if len(closes) < need:
        return []

    curr = float(closes[-1])
    start = float(closes[-(cfg.move_window_bars + 1)])
    if start <= 0:
        return []

    move_pct = ((curr - start) / start) * 100.0
    if abs(move_pct) < cfg.move_pct_threshold:
        return []

    direction = "up" if move_pct > 0 else "down"
    prev_bar = float(closes[-2]) if len(closes) >= 2 else start
    levels = round_key_levels(curr, cfg.level_step_usd, cfg.levels_around_price)

    # Prefer the level closest to current price among those touched in-window
    window_low = min(closes[-need:])
    window_high = max(closes[-need:])
    candidates: List[Tuple[float, float]] = []  # (distance, level)

    for level in levels:
        in_window_range = window_low - cfg.level_proximity_usd <= level <= (
            window_high + cfg.level_proximity_usd
        )
        near_now = _crossed_or_near(prev_bar, curr, level, cfg.level_proximity_usd)
        if not (in_window_range or near_now):
            continue
        # Require the move path to have engaged the level (cross or proximity)
        engaged = near_now or (
            min(start, curr) - cfg.level_proximity_usd
            <= level
            <= max(start, curr) + cfg.level_proximity_usd
        )
        if not engaged:
            continue
        candidates.append((abs(curr - level), level))

    if not candidates:
        return []

    candidates.sort(key=lambda t: t[0])
    level = candidates[0][1]
    verb = "risen" if direction == "up" else "fallen"
    reason = (
        f"Gold {verb} {abs(move_pct):.2f}% over ~{cfg.move_window_bars}×{cfg.interval} "
        f"and interacted with key level {level:.0f}"
    )
    return [
        PriceAlert(
            kind="key_level_move",
            direction=direction,
            level=level,
            price=round(curr, 2),
            window_move_pct=round(move_pct, 4),
            window_bars=cfg.move_window_bars,
            interval=cfg.interval,
            reason=reason,
        )
    ]


def load_alert_state(path: Path | None = None) -> Dict[str, Any]:
    state_path = path or STATE_PATH
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"last_alerts": {}}
        if not isinstance(data.get("last_alerts"), dict):
            data["last_alerts"] = {}
        return data
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return {"last_alerts": {}}


def save_alert_state(state: Dict[str, Any], path: Path | None = None) -> Path:
    ensure_data_dir()
    state_path = path or STATE_PATH
    payload = {
        "last_alerts": dict(state.get("last_alerts") or {}),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return state_path


def filter_cooldown(
    alerts: List[PriceAlert],
    state: Dict[str, Any],
    cooldown_minutes: int,
    now: Optional[datetime] = None,
) -> List[PriceAlert]:
    """Drop alerts still inside the per-(level, direction) cooldown window."""
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    last = state.get("last_alerts") or {}
    kept: List[PriceAlert] = []
    for alert in alerts:
        raw_ts = last.get(alert.dedupe_key)
        if raw_ts:
            try:
                sent_at = datetime.fromisoformat(str(raw_ts).replace("Z", ""))
                if sent_at.tzinfo is not None:
                    sent_at = sent_at.replace(tzinfo=None)
                if now - sent_at < timedelta(minutes=cooldown_minutes):
                    print(
                        f"[PriceAlerts] Cooldown skip {alert.dedupe_key} "
                        f"(last={raw_ts})"
                    )
                    continue
            except (TypeError, ValueError):
                pass
        kept.append(alert)
    return kept


def mark_alerts_sent(
    state: Dict[str, Any],
    alerts: List[PriceAlert],
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    last = dict(state.get("last_alerts") or {})
    ts = now.isoformat()
    for alert in alerts:
        last[alert.dedupe_key] = ts
    return {"last_alerts": last}


def alert_to_dict(alert: PriceAlert) -> Dict[str, Any]:
    return {
        "kind": alert.kind,
        "direction": alert.direction,
        "level": alert.level,
        "price": alert.price,
        "window_move_pct": alert.window_move_pct,
        "window_bars": alert.window_bars,
        "interval": alert.interval,
        "dedupe_key": alert.dedupe_key,
        "reason": alert.reason,
        "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
    }


def fetch_gold_intraday(cfg: AlertConfig):
    """Fetch intraday gold bars for alert detection."""
    print(
        f"[PriceAlerts] Fetching {cfg.ticker} "
        f"period={cfg.period} interval={cfg.interval}..."
    )
    return _fetch_with_retry(
        cfg.ticker,
        period=cfg.period,
        interval=cfg.interval,
    )


def run_price_alerts(*, dry_run: bool = False) -> Dict[str, Any]:
    """Fetch gold, detect alerts, apply cooldown, optionally send Telegram."""
    from src.utils.telegram_reporter import TelegramReporter

    cfg = load_alert_config()
    result: Dict[str, Any] = {
        "alerts": [],
        "sent": [],
        "skipped_cooldown": 0,
        "dry_run": dry_run,
        "config": {
            "ticker": cfg.ticker,
            "interval": cfg.interval,
            "move_window_bars": cfg.move_window_bars,
            "move_pct_threshold": cfg.move_pct_threshold,
            "level_step_usd": cfg.level_step_usd,
            "level_proximity_usd": cfg.level_proximity_usd,
            "cooldown_minutes": cfg.cooldown_minutes,
        },
    }

    try:
        df = fetch_gold_intraday(cfg)
        closes = [float(x) for x in extract_close(df).tolist()]
    except Exception as e:
        print(f"[PriceAlerts] Fetch failed: {e}")
        result["error"] = str(e)
        return result

    if not closes:
        print("[PriceAlerts] No close data.")
        result["error"] = "empty_closes"
        return result

    result["price"] = round(closes[-1], 2)
    detected = detect_alerts(closes, cfg)
    state = load_alert_state()
    before_cd = len(detected)
    actionable = filter_cooldown(detected, state, cfg.cooldown_minutes)
    result["skipped_cooldown"] = before_cd - len(actionable)
    result["alerts"] = [alert_to_dict(a) for a in actionable]

    if not actionable:
        print("[PriceAlerts] No actionable alerts.")
        return result

    telegram = TelegramReporter()
    sent_keys: List[str] = []
    for alert in actionable:
        payload = alert_to_dict(alert)
        print(f"[PriceAlerts] Alert: {alert.reason}")
        if dry_run:
            print("[PriceAlerts] Dry-run — not sending Telegram.")
            print(telegram.format_price_alert(payload))
            sent_keys.append(alert.dedupe_key)
            continue
        ids = telegram.send_price_alert(payload)
        if ids:
            sent_keys.append(alert.dedupe_key)
        elif not telegram.enabled:
            print("[PriceAlerts] Telegram not configured; alert not sent.")
            break

    if sent_keys and not dry_run:
        state = mark_alerts_sent(
            state, [a for a in actionable if a.dedupe_key in sent_keys]
        )
        path = save_alert_state(state)
        print(f"[PriceAlerts] State saved: {path}")

    result["sent"] = sent_keys
    return result
