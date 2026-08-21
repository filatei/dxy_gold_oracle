"""Telegram feedback collector with GitHub issue commenting."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests

from src.config import DATA_DIR, ensure_data_dir


class FeedbackCollector:
    def __init__(self):
        self.token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else None
        self.gh_token = (os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN") or "").strip()
        self.repo = (os.getenv("GITHUB_REPOSITORY") or "").strip()
        self.enabled = bool(self.token and self.gh_token and self.repo)
        ensure_data_dir()
        self.offset_file = DATA_DIR / "telegram_offset.json"

    def run(self):
        if not self.enabled:
            print(
                "[FeedbackCollector] Missing TELEGRAM_BOT_TOKEN, "
                "GH_TOKEN/GITHUB_TOKEN, or GITHUB_REPOSITORY"
            )
            return

        offset = self._load_offset()
        print(f"[FeedbackCollector] Starting from offset: {offset}")

        updates = self._get_updates(offset)
        if not updates:
            print("[FeedbackCollector] No new updates.")
            return

        processed = 0
        new_offset = offset

        for update in updates:
            update_id = update["update_id"]
            new_offset = update_id + 1

            if "callback_query" not in update:
                continue

            cq = update["callback_query"]
            callback_id = cq["id"]
            data = cq.get("data", "")
            user = cq.get("from", {})

            if not data.startswith("fb:"):
                continue

            parts = data.split(":")
            if len(parts) != 3:
                continue

            verdict = parts[1]  # 'a' or 'w'
            issue_number = parts[2]
            if verdict not in {"a", "w"} or not re.fullmatch(r"\d+", issue_number):
                print(f"[FeedbackCollector] Ignoring invalid callback: {data}")
                continue

            self._answer_callback(callback_id, verdict)
            self._post_feedback_comment(issue_number, verdict, user)
            processed += 1

        self._save_offset(new_offset)
        print(
            f"[FeedbackCollector] Processed {processed} feedback items. "
            f"New offset: {new_offset}"
        )

    def _get_updates(self, offset: int) -> List[dict]:
        try:
            resp = requests.get(
                f"{self.base_url}/getUpdates",
                params={
                    "offset": offset,
                    "limit": 100,
                    "allowed_updates": json.dumps(["callback_query"]),
                },
                timeout=30,
            )
            result = resp.json()
            if not result.get("ok"):
                print(f"[FeedbackCollector] Telegram API error: {result}")
                return []
            return result.get("result", [])
        except Exception as e:
            print(f"[FeedbackCollector] getUpdates failed: {e}")
            return []

    def _answer_callback(self, callback_id: str, verdict: str):
        msg = (
            "Marked as accurate. Thanks!"
            if verdict == "a"
            else "Marked as wrong. Noted for improvement."
        )
        try:
            requests.post(
                f"{self.base_url}/answerCallbackQuery",
                json={
                    "callback_query_id": callback_id,
                    "text": msg,
                    "show_alert": False,
                },
                timeout=10,
            )
        except Exception as e:
            print(f"[FeedbackCollector] answerCallback failed: {e}")

    def _post_feedback_comment(self, issue_number: str, verdict: str, user: dict):
        emoji = "👍" if verdict == "a" else "👎"
        label = "Accurate" if verdict == "a" else "Wrong"
        username = user.get("username") or f"ID:{user.get('id', 'unknown')}"
        first_name = user.get("first_name", "")
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

        body = f"""## {emoji} User Feedback

**Verdict:** {label}  
**User:** @{username} ({first_name})  
**Time:** {now} UTC

> This feedback is tracked for model accuracy improvement.
"""
        try:
            resp = requests.post(
                f"https://api.github.com/repos/{self.repo}/issues/{issue_number}/comments",
                headers={
                    "Authorization": f"Bearer {self.gh_token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                json={"body": body},
                timeout=15,
            )
            if resp.status_code == 201:
                print(f"[FeedbackCollector] Comment posted to issue #{issue_number}")
            else:
                print(
                    f"[FeedbackCollector] Failed to comment on #{issue_number}: "
                    f"{resp.status_code} {resp.text[:200]}"
                )
        except Exception as e:
            print(f"[FeedbackCollector] GitHub API error: {e}")

    def _load_offset(self) -> int:
        try:
            with open(self.offset_file, "r", encoding="utf-8") as f:
                return int(json.load(f).get("offset", 0))
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
            return 0

    def _save_offset(self, offset: int):
        ensure_data_dir()
        with open(self.offset_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "offset": int(offset),
                    "updated_at": datetime.now(timezone.utc)
                    .replace(tzinfo=None)
                    .isoformat(),
                },
                f,
                indent=2,
            )
