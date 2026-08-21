"""Weekly accuracy report from GitHub issue feedback comments."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def fetch_accuracy_stats(
    repo: str, token: str, days: int = 7
) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    """Return (accurate, wrong, signal_count, issues)."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    url = f"https://api.github.com/repos/{repo}/issues"
    params = {
        "labels": "oracle-signal",
        "since": since,
        "state": "all",
        "per_page": 100,
    }
    resp = requests.get(url, headers=_headers(token), params=params, timeout=30)
    resp.raise_for_status()
    issues = resp.json()
    if not isinstance(issues, list):
        print(f"[Accuracy] Unexpected API response: {issues}")
        return 0, 0, 0, []

    accurate = 0
    wrong = 0
    for issue in issues:
        comments_url = issue.get("comments_url")
        if not comments_url:
            continue
        cr = requests.get(comments_url, headers=_headers(token), timeout=30)
        cr.raise_for_status()
        comments = cr.json()
        if not isinstance(comments, list):
            continue
        for c in comments:
            body = c.get("body", "") or ""
            if "👍" in body and "Accurate" in body:
                accurate += 1
            elif "👎" in body and "Wrong" in body:
                wrong += 1

    return accurate, wrong, len(issues), issues


def build_report_body(accurate: int, wrong: int, signal_count: int) -> str:
    total = accurate + wrong
    rate = (accurate / total * 100) if total else 0.0
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    return "\n".join(
        [
            "# Weekly Oracle Accuracy Report",
            "",
            "**Period:** Last 7 days",
            f"**Signals Reviewed:** {signal_count}",
            f"**Feedback Received:** {total}",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Accurate | {accurate} |",
            f"| Wrong | {wrong} |",
            f"| **Win Rate** | **{rate:.1f}%** |",
            "",
            f"_Generated at {now} UTC_",
        ]
    )


def publish_accuracy_report(
    repo: Optional[str] = None, token: Optional[str] = None
) -> None:
    repo = (repo or os.getenv("GITHUB_REPOSITORY") or os.getenv("REPO") or "").strip()
    token = (token or os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN") or "").strip()
    if not repo or not token:
        raise SystemExit("Missing GITHUB_REPOSITORY/REPO or GH_TOKEN/GITHUB_TOKEN")

    accurate, wrong, signal_count, _ = fetch_accuracy_stats(repo, token)
    report_body = build_report_body(accurate, wrong, signal_count)
    total = accurate + wrong
    rate = (accurate / total * 100) if total else 0.0

    url = f"https://api.github.com/repos/{repo}/issues"
    search = requests.get(
        url,
        headers=_headers(token),
        params={"labels": "accuracy-report", "state": "open", "per_page": 5},
        timeout=30,
    )
    search.raise_for_status()
    existing = search.json()

    if isinstance(existing, list) and existing:
        issue_num = existing[0]["number"]
        patch = requests.patch(
            f"{url}/{issue_num}",
            headers=_headers(token),
            json={"body": report_body},
            timeout=30,
        )
        if patch.status_code == 422:
            patch = requests.patch(
                f"{url}/{issue_num}",
                headers=_headers(token),
                json={"body": report_body, "title": existing[0].get("title")},
                timeout=30,
            )
        patch.raise_for_status()
        print(f"Updated accuracy report in issue #{issue_num}")
    else:
        payload = {
            "title": (
                f"Weekly Accuracy Report — "
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
            ),
            "body": report_body,
            "labels": ["accuracy-report", "weekly"],
        }
        create = requests.post(url, headers=_headers(token), json=payload, timeout=30)
        if create.status_code == 422:
            payload.pop("labels", None)
            create = requests.post(url, headers=_headers(token), json=payload, timeout=30)
        create.raise_for_status()
        print(f"Created accuracy report: {create.json().get('html_url', 'N/A')}")

    print(f"\nWeekly Accuracy: {accurate}/{total} ({rate:.1f}%)")


if __name__ == "__main__":
    publish_accuracy_report()
