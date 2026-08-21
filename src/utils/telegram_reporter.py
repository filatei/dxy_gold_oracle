"""Telegram reporter with inline keyboard feedback."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import requests


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

        text = self._format_message(report)
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

    def _format_message(self, report: dict) -> str:
        session = report["session"].upper()
        decision = report["decision"]
        setup = report["trade_setup"]

        direction_emoji = {
            "BULLISH": "🟢",
            "MODERATE_BULLISH": "🟢",
            "BEARISH": "🔴",
            "MODERATE_BEARISH": "🔴",
            "NEUTRAL": "⚪",
        }.get(decision["direction"], "⚪")

        confidence_bar = self._confidence_bar(decision.get("confidence", 0))
        rationale = (setup.get("rationale") or "N/A")[:350]

        return f"""<b>DXY-GOLD ORACLE — {session}</b>

<b>Direction:</b> {direction_emoji} <code>{decision['direction']}</code>
<b>Confidence:</b> {setup['confidence']} {confidence_bar}
<b>Score:</b> <code>{decision['score']}/10</code>

<b>Agent Signals:</b>
{self._agent_signals(report.get('agents', {}))}

<b>Rationale:</b>
<i>{rationale}</i>

⏱ <i>{str(report.get('timestamp', ''))[:19]} UTC</i>
🤖 <i>Automated via GitHub Actions</i>"""

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
            lines.append(f"  {label}: <code>{display}</code>")
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
            "• 22:00 UTC (Asia)"
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
