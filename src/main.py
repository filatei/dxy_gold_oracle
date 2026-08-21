"""Entry point for DXY-Gold Oracle.

Run from the repository root:

    python -m src.main --session london
    python -m src.main --session london --dry-run
    python -m src.main --test-telegram
    python -m src.main --collect-feedback
    python -m src.main --weekly-accuracy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path when invoked as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from src.agents.feedback_collector import FeedbackCollector
from src.agents.oracle import GoldOracle
from src.utils.github_reporter import GitHubReporter
from src.utils.telegram_reporter import TelegramReporter
from src.utils.weekly_accuracy import publish_accuracy_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DXY-Gold Oracle")
    parser.add_argument(
        "--session",
        choices=["london", "ny", "asia"],
        help="Trading session to analyze",
    )
    parser.add_argument(
        "--collect-feedback",
        action="store_true",
        help="Poll Telegram for user feedback",
    )
    parser.add_argument(
        "--test-telegram",
        action="store_true",
        help="Send test Telegram message",
    )
    parser.add_argument(
        "--weekly-accuracy",
        action="store_true",
        help="Generate weekly accuracy GitHub issue",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip GitHub Issues and Telegram publishing",
    )
    args = parser.parse_args(argv)

    if args.test_telegram:
        t = TelegramReporter()
        if t.send_test_message():
            print("Telegram test message sent successfully!")
            return 0
        print("Telegram not configured. Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return 1

    if args.collect_feedback:
        FeedbackCollector().run()
        return 0

    if args.weekly_accuracy:
        publish_accuracy_report()
        return 0

    if not args.session:
        parser.error(
            "--session is required unless using --collect-feedback, "
            "--test-telegram, or --weekly-accuracy"
        )

    dry_run = args.dry_run or os.getenv("ORACLE_DRY_RUN", "").strip() in {
        "1",
        "true",
        "yes",
    }

    print(f"Gold Oracle — {args.session.upper()} session")
    print("=" * 50)

    oracle = GoldOracle(session=args.session)
    report = oracle.run()

    print("\n" + "=" * 50)
    print("REPORT:")
    print(json.dumps(report, indent=2, default=str))
    print("=" * 50)

    issue_number = None
    if not dry_run:
        try:
            issue_number = GitHubReporter().publish(report)
        except Exception as e:
            print(f"[GitHub Reporter] Warning: {e}")

        try:
            TelegramReporter().send_report(report, issue_number=issue_number)
        except Exception as e:
            print(f"[Telegram Reporter] Warning: {e}")
    else:
        print("[Dry-run] Skipped GitHub Issue and Telegram publish.")

    artifact = {**report, "github_issue": issue_number, "dry_run": dry_run}
    artifact_path = _ROOT / f"report_{args.session}.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, default=str)
    print(f"Artifact saved: {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
