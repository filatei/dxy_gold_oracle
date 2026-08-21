# DXY-Gold Oracle

Automated gold (XAU/USD) **directional** analysis using a **synthetic DXY** built from FX components (EURUSD, USDJPY, GBPUSD, USDCAD, USDSEK, USDCHF). Designed to run on **GitHub Actions** with optional **Telegram** alerts and inline feedback.

> Educational / directional bias only — **not financial advice** and not a trade-execution system.

---

## What it does

Three times per trading day the Oracle:

1. Fetches DXY constituents and gold futures (`yfinance`)
2. Synthesizes a geometric DXY proxy from basket weights
3. Runs technicals (EMA/RSI), DXY–gold correlation, and macro sentiment via free LLMs
4. Aggregates a direction + confidence score
5. Opens a GitHub Issue (audit trail) and optionally alerts Telegram with Accurate/Wrong buttons
6. Collects Telegram feedback into issue comments for weekly accuracy reporting

### Session schedule (UTC)

| Session | Cron | Notes |
|---------|------|--------|
| London | `0 7 * * 1-5` | Before London cash open |
| New York | `30 13 * * 1-5` | Before NY cash open |
| Asia | `0 22 * * 1-5` | Ahead of Tokyo session |

---

## GitHub Secrets (for forks)

Forks do **not** inherit secrets. After forking, add these under  
**Settings → Secrets and variables → Actions → New repository secret** on **the fork** (not the upstream repo). Also enable **Actions** on the fork if prompted.

Exact names must match (code reads `GEMINI_API_KEY` and `OPENROUTER_API_KEY` — not `GOOGLE_*` / `OR_*`).

| Secret name | Required? | Where to get it | Workflows that need it |
|-------------|-----------|-----------------|------------------------|
| `TELEGRAM_BOT_TOKEN` | Optional (needed for Telegram alerts / feedback) | [@BotFather](https://t.me/botfather) | `london-session`, `ny-session`, `asia-session`, `telegram-feedback`, `test-telegram` |
| `TELEGRAM_CHAT_ID` | Optional (with bot token) | [@userinfobot](https://t.me/userinfobot); groups: `getUpdates` after adding the bot. Multiple IDs: `123,-100456` | `london-session`, `ny-session`, `asia-session`, `test-telegram` |
| `GROQ_API_KEY` | Optional (recommended; first LLM in cascade) | [console.groq.com/keys](https://console.groq.com/keys) | `london-session`, `ny-session`, `asia-session` |
| `GEMINI_API_KEY` | Optional (LLM fallback #2) | [Google AI Studio](https://aistudio.google.com/app/apikey) | `london-session`, `ny-session`, `asia-session` |
| `OPENROUTER_API_KEY` | Optional (LLM fallback #3) | [openrouter.ai/keys](https://openrouter.ai/keys) | `london-session`, `ny-session`, `asia-session` |

**Not a user secret:** `GITHUB_TOKEN` is injected automatically by Actions (mapped to `GH_TOKEN` in workflows). It powers checkout (private repos), Issue creation, feedback comments, and weekly accuracy. `ci.yml` needs no user secrets.

### Private repo + Actions permissions

This repository is **private**. Workflows declare:

```yaml
permissions:
  contents: read   # required for actions/checkout on private repos
  issues: write    # session / weekly workflows that open Issues
```

If you set a `permissions:` block, any omitted scope defaults to **none**. Granting only `issues: write` (without `contents: read`) makes checkout fail with `remote: Repository not found` — GitHub hides private repos from tokens that lack Contents access.

`telegram-feedback.yml` uses `contents: write` so it can commit the Telegram offset file.


**Minimal fork setup**

| Goal | Secrets to add |
|------|----------------|
| Session runs + GitHub Issues only | none (Issues work via `GITHUB_TOKEN`); macro stays neutral without LLM keys |
| Session runs + LLM macro | at least one of `GROQ_API_KEY` / `GEMINI_API_KEY` / `OPENROUTER_API_KEY` |
| Telegram alerts + Accurate/Wrong buttons | `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` |
| Poll button feedback into Issues | `TELEGRAM_BOT_TOKEN` (plus Issues permission via `GITHUB_TOKEN`) |

Optional **Variables** (or env), not secrets: `GROQ_MODEL`, `GEMINI_MODEL`, `OPENROUTER_MODEL`.  
Local-only (see `.env.example`): `GH_TOKEN`, `GITHUB_REPOSITORY`, `ORACLE_DRY_RUN`.

Do **not** commit API keys — use repository Secrets for Actions and `.env` locally.

---

## GitHub Actions setup

1. Fork or clone (`filatei/dxy_gold_oracle`), then add the secrets above in **your** repo.
2. Open **Actions** and enable workflows if prompted.
3. Run **Test Telegram Integration** (manual) if you configured Telegram.
4. Run **London Session Oracle** (manual) once, then confirm:
   - a new Issue (label `oracle-signal` if labels exist)
   - a Telegram message with feedback buttons (if Telegram secrets are set)

Scheduled runs only fire after Actions are enabled.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Checkout: `remote: Repository not found` / `fatal: repository '…' not found` | Private repo + workflow `permissions` missing `contents: read` (token cannot see the repo). Also happens if Actions are disabled or secrets live on upstream while the run is on a fork. | Add `contents: read` (and `issues: write` where Issues are created). Enable Actions on **the fork**. Set secrets on **the fork**. Never hardcode another repo URL in `actions/checkout` — use the default `${{ github.repository }}`. |
| Issues not created | Missing `issues: write` or Actions disabled | Add permission; confirm workflow ran |
| Telegram skipped / no alerts | Missing `TELEGRAM_*` secrets on this repo | Add secrets on the repo that runs the workflow |

### Workflows

| Workflow | Trigger |
|----------|---------|
| `london-session.yml` / `ny-session.yml` / `asia-session.yml` | Cron + manual |
| `telegram-feedback.yml` | Every 20m Mon–Fri 06–23 UTC |
| `weekly-accuracy.yml` | Sunday 18:00 UTC |
| `test-telegram.yml` | Manual |
| `ci.yml` | Push / PR — unit tests (no live keys) |

---

## Local setup

```bash
git clone git@github.com:filatei/dxy_gold_oracle.git
cd dxy_gold_oracle

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your keys
```

### CLI

```bash
# Analysis only (no Issue / Telegram)
python -m src.main --session london --dry-run

# Full local publish (needs GH_TOKEN + GITHUB_REPOSITORY + Telegram)
export GH_TOKEN=ghp_...
export GITHUB_REPOSITORY=filatei/dxy_gold_oracle
python -m src.main --session london

# Telegram smoke test
python -m src.main --test-telegram

# Poll Telegram callbacks → comment on Issues
python -m src.main --collect-feedback

# Weekly accuracy Issue
python -m src.main --weekly-accuracy

# Unit tests (mocked / no API keys)
pytest -q
```

`ORACLE_DRY_RUN=1` in `.env` is equivalent to `--dry-run`.

---

## Architecture

```
Session cron → GoldOracle.run()
                 ├─ DXY synthesizer (Yahoo)
                 ├─ Technical analyst (EMA/RSI)
                 ├─ Correlation engine
                 └─ Macro analyst (Groq → Gemini → OpenRouter)
                        ↓
                 Aggregate decision
                        ├─ GitHub Issue
                        └─ Telegram (+ feedback buttons)
                               ↓
                        Feedback collector → Issue comment
```

### Scoring (summary)

| Agent | Typical contribution |
|-------|----------------------|
| DXY | STRONG_DXY (−3) … WEAK_DXY (+3) |
| Technical | BREAKDOWN (−3) … BREAKOUT (+3) |
| Correlation | POSITIVE (−1) … STRONG_INVERSE (+2) |
| Macro/LLM | −3 … +3 |

Thresholds: ≥4 BULLISH, 2–3 MODERATE_BULLISH, ≤−4 BEARISH, −3…−2 MODERATE_BEARISH, else NEUTRAL.  
Confidence = `min(|score| / 10, 1.0)`.

---

## Layout

```
dxy_gold_oracle/
├── .github/workflows/     # session + feedback + CI
├── config/weights.json    # DXY basket weights
├── data/                  # runtime state (telegram offset); gitignored contents
├── src/
│   ├── main.py            # CLI entry: python -m src.main
│   ├── config.py
│   ├── agents/            # oracle + specialists
│   ├── llm/router.py
│   └── utils/             # GitHub / Telegram / weekly accuracy
├── tests/                 # offline unit tests
├── .env.example
├── requirements.txt
└── README.md
```

---

## Notes & limits

- Synthetic DXY is an approximation of ICE DXY (1973 base / vendor feeds differ).
- `yfinance` data is delayed; suitable for pre-session bias, not scalping.
- LLMs can hallucinate; JSON schema + validation reduce (not eliminate) bad output.
- Missing LLM keys → macro agent returns neutral (other agents still run).
- Creating Issues with custom labels fails until labels exist; the reporter retries without labels.

---

## License

MIT — use at your own risk.
