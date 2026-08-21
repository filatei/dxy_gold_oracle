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

## Repository secrets

In GitHub: **Settings → Secrets and variables → Actions**

| Secret | Required | How to get it |
|--------|----------|----------------|
| `TELEGRAM_BOT_TOKEN` | For alerts | [@BotFather](https://t.me/botfather) |
| `TELEGRAM_CHAT_ID` | For alerts | [@userinfobot](https://t.me/userinfobot); groups: `getUpdates` after adding the bot. Multiple IDs: `123,-100456` |
| `GROQ_API_KEY` | Recommended | [console.groq.com](https://console.groq.com/keys) |
| `GEMINI_API_KEY` | Fallback | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `OPENROUTER_API_KEY` | Fallback | [openrouter.ai](https://openrouter.ai/keys) |

`GITHUB_TOKEN` is injected automatically by Actions. Do **not** put API keys in the repo — use `.env` locally (see `.env.example`).

Optional model overrides (repo Variables or env): `GROQ_MODEL`, `GEMINI_MODEL`, `OPENROUTER_MODEL`.

---

## GitHub Actions setup

1. Push this repo (or use the published `filatei/dxy_gold_oracle`).
2. Add the secrets above.
3. Open **Actions** and enable workflows if prompted.
4. Run **Test Telegram Integration** (manual) to verify the bot.
5. Run **London Session Oracle** (manual) once, then confirm:
   - a new Issue (label `oracle-signal` if labels exist)
   - a Telegram message with feedback buttons

Scheduled runs only fire after Actions are enabled and secrets are present. Forks must re-add secrets.

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
