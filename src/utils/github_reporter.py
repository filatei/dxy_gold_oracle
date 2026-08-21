"""GitHub Issues reporter for audit trail."""
from __future__ import annotations

import os
from typing import List, Optional

import requests


class GitHubReporter:
    def __init__(self):
        self.token = (os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN") or "").strip()
        self.repo = (os.getenv("GITHUB_REPOSITORY") or "").strip()
        self.api = "https://api.github.com"
        self.enabled = bool(self.token and self.repo)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def publish(self, report: dict) -> Optional[int]:
        """Create GitHub Issue and return issue number."""
        if not self.enabled:
            print("[GitHub] Not configured (missing GH_TOKEN/GITHUB_TOKEN or GITHUB_REPOSITORY).")
            return None

        session = report["session"].upper()
        decision = report["decision"]
        setup = report["trade_setup"]

        title = f"[{session}] Gold Direction: {decision['direction']}"

        body = f"""## Oracle Decision for {session} Session

**Timestamp:** {report['timestamp']} UTC  
**Direction:** `{decision['direction']}`  
**Confidence:** {setup['confidence']}  
**Aggregate Score:** {decision['score']}/10

### Agents Consensus
| Agent | Signal | Detail |
|-------|--------|--------|
| DXY Synthesizer | {report['agents'].get('dxy', {}).get('dxy_bias', 'N/A')} | DXY={report['agents'].get('dxy', {}).get('current_dxy', 'N/A')} |
| Technical | {report['agents'].get('tech', {}).get('signal', 'N/A')} | RSI={report['agents'].get('tech', {}).get('rsi', 'N/A')} |
| Macro/LLM | {report['agents'].get('macro', {}).get('sentiment', 'N/A')} | Score={report['agents'].get('macro', {}).get('raw_score', 'N/A')} |
| Correlation | {report['agents'].get('corr', {}).get('regime', 'N/A')} | r={report['agents'].get('corr', {}).get('correlation', 'N/A')} |

### Rationale
{setup['rationale']}

---
*Automated by DXY-Gold Oracle via GitHub Actions*
"""

        labels = self._normalize_labels(
            [
                "oracle-signal",
                report["session"],
                decision["direction"].lower().replace("/", "-"),
            ]
        )

        try:
            resp = requests.post(
                f"{self.api}/repos/{self.repo}/issues",
                headers=self._headers(),
                json={"title": title, "body": body, "labels": labels},
                timeout=20,
            )
            if resp.status_code == 422:
                # Labels may not exist yet — retry without them
                print(f"[GitHub] Label create failed ({resp.text[:200]}); retrying without labels")
                resp = requests.post(
                    f"{self.api}/repos/{self.repo}/issues",
                    headers=self._headers(),
                    json={"title": title, "body": body},
                    timeout=20,
                )
            resp.raise_for_status()
            data = resp.json()
            issue_number = data["number"]
            print(f"[GitHub] Issue #{issue_number} created: {data.get('html_url')}")
            return issue_number
        except Exception as e:
            print(f"[GitHub] Failed to create issue: {e}")
            return None

    def _normalize_labels(self, labels: List[str]) -> List[str]:
        cleaned = []
        for label in labels:
            label = (label or "").strip().lower()
            if label and label not in cleaned:
                cleaned.append(label[:50])
        return cleaned
