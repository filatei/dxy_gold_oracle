"""Telegram reporter with inline keyboard feedback."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import requests

from src.agents.opinion import (
    ACTIONS,
    confidence_label,
    normalize_action,
    normalize_confidence,
    one_line,
)

_ACTION_EMOJI = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}


def _html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


class TelegramReporter:
    def __init__(self):
        self.token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        self.chat_ids = self._parse_chat_ids(os.getenv("TELEGRAM_CHAT_ID", ""))
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else None
        self.enabled = bool(self.token and self.chat_ids)

    def _parse_chat_ids(self, raw: str) -> List[str]:
        if not raw:
            return []
        return [cid.strip() for cid in raw.split(",") if cid.strip()]

    def send_report(self, report: dict, issue_number: Optional[int] = None) -> Dict[str, int]:
        """Send report to all configured chats. Returns {chat_id: message_id}."""
        if not self.enabled:
            print("[Telegram] Not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
            return {}

        text = self.format_message(report)
        keyboard = self._build_keyboard(issue_number) if issue_number else None

        results: Dict[str, int] = {}
        for chat_id in self.chat_ids:
            try:
                payload = {
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                }
                if keyboard:
                    payload["reply_markup"] = json.dumps(keyboard)

                resp = requests.post(
                    f"{self.base_url}/sendMessage",
                    json=payload,
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                if not data.get("ok"):
                    print(f"[Telegram] API error for {chat_id}: {data}")
                    continue
                results[chat_id] = data["result"]["message_id"]
                print(
                    f"[Telegram] Report sent to chat {chat_id}, "
                    f"msg_id={data['result']['message_id']}"
                )
            except Exception as e:
                print(f"[Telegram] Failed to send to {chat_id}: {e}")

        return results

    def _build_keyboard(self, issue_number: int) -> dict:
        return {
            "inline_keyboard": [
                [
                    {"text": "Accurate", "callback_data": f"fb:a:{issue_number}"},
                    {"text": "Wrong", "callback_data": f"fb:w:{issue_number}"},
                ]
            ]
        }

    def format_message(self, report: dict) -> str:
        """Public formatter used by send_report and dry-run console output."""
        session = str(report.get("session") or "session").upper()
        decision = report.get("decision") or {}
        setup = report.get("trade_setup") or {}
        opinion_block = self.format_opinion_block(report.get("opinion") or {})

        direction = str(decision.get("direction") or "NEUTRAL")
        direction_emoji = {
            "BULLISH": "🟢",
            "MODERATE_BULLISH": "🟢",
            "BEARISH": "🔴",
            "MODERATE_BEARISH": "🔴",
            "NEUTRAL": "⚪",
        }.get(direction, "⚪")

        try:
            conf = float(decision.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        confidence_bar = self._confidence_bar(conf)
        setup_conf = _html_escape(str(setup.get("confidence") or f"{conf * 100:.1f}%"))
        rationale = _html_escape(one_line(setup.get("rationale") or "N/A", 350))
        score = _html_escape(str(decision.get("score", "n/a")))
        agents = report.get("agents") if isinstance(report.get("agents"), dict) else {}
        agent_lines = self._agent_signals(agents)

        return f"""<b>DXY-GOLD ORACLE — {session}</b>

{opinion_block}

<b>Direction:</b> {direction_emoji} <code>{_html_escape(direction)}</code>
<b>Confidence:</b> {setup_conf} {confidence_bar}
<b>Score:</b> <code>{score}/10</code>

<b>Agent Signals:</b>
{agent_lines}

<b>Rationale:</b>
<i>{rationale}</i>

⏱ <i>{_html_escape(str(report.get("timestamp", ""))[:19])} UTC</i>
🤖 <i>Automated via GitHub Actions · not financial advice</i>"""

    def format_opinion_block(self, opinion: dict) -> str:
        """HTML opinion block. Invalid LLM enums are coerced so send cannot crash."""
        gold = normalize_action((opinion or {}).get("gold_action"))
        dxy = normalize_action((opinion or {}).get("dxy_action"))
        if gold not in ACTIONS:
            gold = "HOLD"
        if dxy not in ACTIONS:
            dxy = "HOLD"
        conf = normalize_confidence((opinion or {}).get("confidence"), default=0)
        label = str((opinion or {}).get("confidence_label") or confidence_label(conf))
        note = _html_escape(one_line((opinion or {}).get("consensus_note") or "", 180))
        rationale = _html_escape(one_line((opinion or {}).get("rationale") or "", 200))
        bar = self._confidence_bar(conf / 100.0)

        lines = [
            "📊 <b>OPINION</b>",
            f"{_ACTION_EMOJI[gold]} <b>GOLD: {gold}</b>",
            f"{_ACTION_EMOJI[dxy]} <b>DXY: {dxy}</b>",
            f"<b>Confidence:</b> {label} ({conf}) {bar}",
        ]
        if note:
            lines.append(f"<i>{note}</i>")
        if rationale:
            lines.append(f"<i>{rationale}</i>")
        return "\n".join(lines)

    def format_opinion_plain(self, opinion: dict) -> str:
        """Plaintext twin of the Telegram opinion block (dry-run console)."""
        gold = normalize_action((opinion or {}).get("gold_action"))
        dxy = normalize_action((opinion or {}).get("dxy_action"))
        conf = normalize_confidence((opinion or {}).get("confidence"), default=0)
        label = str((opinion or {}).get("confidence_label") or confidence_label(conf))
        note = one_line((opinion or {}).get("consensus_note") or "", 180)
        rationale = one_line((opinion or {}).get("rationale") or "", 200)
        lines = [
            "OPINION",
            f"  {_ACTION_EMOJI[gold]} GOLD: {gold}",
            f"  {_ACTION_EMOJI[dxy]} DXY: {dxy}",
            f"  Confidence: {label} ({conf})",
        ]
        if note:
            lines.append(f"  {note}")
        if rationale:
            lines.append(f"  {rationale}")
        return "\n".join(lines)

    def _confidence_bar(self, conf: float) -> str:
        try:
            filled = max(0, min(10, int(float(conf) * 10)))
        except (TypeError, ValueError):
            filled = 0
        return "█" * filled + "░" * (10 - filled)

    def _agent_signals(self, agents: dict) -> str:
        lines = []
        mapping = {
            "dxy": ("DXY", "dxy_bias"),
            "tech": ("Tech", "signal"),
            "macro": ("Macro", "sentiment"),
            "corr": ("Corr", "regime"),
        }
        for key, (label, subkey) in mapping.items():
            val = agents.get(key, {})
            if isinstance(val, dict):
                display = val.get(subkey, str(val))
            else:
                display = str(val)
            lines.append(f"  {label}: <code>{_html_escape(str(display))}</code>")
        return "\n".join(lines) if lines else "  <i>No agent data</i>"

    def send_test_message(self) -> bool:
        if not self.enabled:
            return False
        text = (
            "<b>DXY-Gold Oracle</b>\n\n"
            "Your Telegram integration is working correctly!\n"
            "You will receive 3 reports daily:\n"
            "• 07:00 UTC (London)\n"
            "• 13:30 UTC (New York)\n"
            "• 22:00 UTC (Asia)\n\n"
            "Each report leads with a multi-LLM <b>BUY / SELL / HOLD</b> "
            "opinion for GOLD and DXY."
        )
        ok_any = False
        for chat_id in self.chat_ids:
            try:
                resp = requests.post(
                    f"{self.base_url}/sendMessage",
                    json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                    timeout=10,
                )
                resp.raise_for_status()
                if resp.json().get("ok"):
                    ok_any = True
                    print(f"[Telegram] Test message sent to {chat_id}")
                else:
                    print(f"[Telegram] Test API error for {chat_id}: {resp.text[:200]}")
            except Exception as e:
                print(f"[Telegram] Test message failed for {chat_id}: {e}")
        return ok_any
