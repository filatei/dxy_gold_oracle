"""Free LLM Router with cascading fallback."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

import requests


# OpenRouter DeepSeek model used only when DEEPSEEK_API_KEY is unset
# (avoids double-counting the same model when direct DeepSeek is configured).
_OPENROUTER_DEEPSEEK_MODEL = "deepseek/deepseek-chat"


class FreeLLMRouter:
    """Routes prompts to available free LLM APIs with cascading fallback."""

    def __init__(self, timeout: int = 30):
        self.keys = {
            "groq": os.getenv("GROQ_API_KEY", "").strip() or None,
            "gemini": os.getenv("GEMINI_API_KEY", "").strip() or None,
            "deepseek": os.getenv("DEEPSEEK_API_KEY", "").strip() or None,
            "openrouter": os.getenv("OPENROUTER_API_KEY", "").strip() or None,
        }
        self.timeout = timeout
        # Prefer env overrides so Actions/local can pin without code edits
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        # deepseek-v4-flash is the current cheap default; deepseek-chat retired 2026-07-24
        self.deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.openrouter_model = os.getenv(
            "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"
        )
        self.openrouter_deepseek_model = os.getenv(
            "OPENROUTER_DEEPSEEK_MODEL", _OPENROUTER_DEEPSEEK_MODEL
        )

    def available_providers(self) -> List[str]:
        """Providers that will vote in query_all (includes OpenRouter-DeepSeek when eligible)."""
        return [name for name, _ in self._vote_providers()]

    def query(self, prompt: str, system: str = "You are a macro forex analyst.") -> Dict[str, Any]:
        """Try Groq -> Gemini -> DeepSeek -> OpenRouter. Return parsed JSON or neutral fallback."""
        for name, fn in self._cascade_providers():
            if not self.keys.get(name):
                continue
            try:
                return fn(prompt, system)
            except Exception as e:
                print(f"[LLM] {name} failed: {e}")

        print("[LLM] All providers exhausted. Returning neutral.")
        return {
            "sentiment": "neutral",
            "score": 0,
            "key_factors": ["No LLM available"],
            "session_context": "LLM providers unavailable; macro score neutral.",
        }

    def query_all(self, prompt: str, system: str = "You are a macro forex analyst.") -> List[Dict[str, Any]]:
        """Query every configured provider independently. Failures are skipped, not cascaded."""
        results: List[Dict[str, Any]] = []
        for name, fn in self._vote_providers():
            try:
                payload = fn(prompt, system)
                results.append({"provider": name, "payload": payload})
            except Exception as e:
                print(f"[LLM] {name} vote failed: {e}")
        return results

    def _cascade_providers(self) -> Tuple[Tuple[str, Any], ...]:
        """Ordered cascade for single-response query()."""
        return (
            ("groq", self._groq),
            ("gemini", self._gemini),
            ("deepseek", self._deepseek),
            ("openrouter", self._openrouter),
        )

    def _vote_providers(self) -> List[Tuple[str, Any]]:
        """Providers used for multi-LLM voting.

        Direct DeepSeek wins when DEEPSEEK_API_KEY is set. If only OpenRouter is
        configured, also cast a DeepSeek vote via OpenRouter (no double-count).
        """
        providers: List[Tuple[str, Any]] = []
        for name, fn in (
            ("groq", self._groq),
            ("gemini", self._gemini),
            ("deepseek", self._deepseek),
            ("openrouter", self._openrouter),
        ):
            if self.keys.get(name):
                providers.append((name, fn))
        if self.keys.get("openrouter") and not self.keys.get("deepseek"):
            providers.append(("openrouter_deepseek", self._openrouter_deepseek))
        return providers

    def _groq(self, prompt: str, system: str) -> Dict[str, Any]:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.keys['groq']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "max_tokens": 512,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return self._safe_json_parse(content)

    def _gemini(self, prompt: str, system: str) -> Dict[str, Any]:
        key = self.keys["gemini"]
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent?key={key}"
        )
        payload = {
            "contents": [{"parts": [{"text": f"{system}\n\n{prompt}"}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2,
                "maxOutputTokens": 512,
            },
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return self._safe_json_parse(text)

    def _deepseek(self, prompt: str, system: str) -> Dict[str, Any]:
        """Direct DeepSeek OpenAI-compatible chat API (requires DEEPSEEK_API_KEY)."""
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.keys['deepseek']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.deepseek_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "max_tokens": 512,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return self._safe_json_parse(content)

    def _openrouter(self, prompt: str, system: str) -> Dict[str, Any]:
        return self._openrouter_chat(prompt, system, self.openrouter_model)

    def _openrouter_deepseek(self, prompt: str, system: str) -> Dict[str, Any]:
        """DeepSeek via OpenRouter when DEEPSEEK_API_KEY is not set."""
        return self._openrouter_chat(prompt, system, self.openrouter_deepseek_model)

    def _openrouter_chat(self, prompt: str, system: str, model: str) -> Dict[str, Any]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.keys['openrouter']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/filatei/dxy_gold_oracle",
            "X-Title": "DXY-Gold Oracle",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return self._safe_json_parse(content)

    def _safe_json_parse(self, text: str) -> Dict[str, Any]:
        """Extract JSON from markdown code blocks or raw text."""
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise
            parsed = json.loads(match.group(0))

        if not isinstance(parsed, dict):
            raise ValueError("LLM response is not a JSON object")

        parsed.setdefault("sentiment", "neutral")
        parsed.setdefault("score", 0)
        parsed.setdefault("key_factors", [])
        parsed.setdefault("session_context", "")
        return parsed

    def analyze_session_sentiment(
        self, session: str, dxy_bias: str, gold_price: float
    ) -> Dict[str, Any]:
        """Macro analysis tailored to the trading session."""
        prompt = f"""Analyze the current macro environment for Gold (XAU/USD) ahead of the {session.upper()} trading session.

Context:
- Synthetic DXY bias: {dxy_bias}
- Current gold reference: ~{gold_price:.2f}

Consider these factors and rank their importance:
1. DXY trend and US Treasury yield direction
2. Recent Fed communications or macro data
3. Geopolitical risk events
4. Asian/European equity market sentiment (relevant to session)
5. Safe-haven demand signals

Return ONLY a JSON object with this exact schema:
{{
  "sentiment": "bullish" | "bearish" | "neutral",
  "score": <integer from -3 to +3>,
  "key_factors": ["factor 1", "factor 2", "factor 3"],
  "session_context": "<one sentence on why this matters for {session.upper()}>"
}}
"""
        return self.query(
            prompt,
            system="You are a senior macro forex and commodities analyst. Be concise and factual.",
        )

    def analyze_trading_opinion(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Ask each configured LLM to vote BUY/SELL/HOLD on gold and DXY.

        Returns a list of {provider, payload} dicts. Empty if no keys or all calls fail.
        One key is enough; all configured providers are used when present.
        """
        prompt = f"""You are voting on a pre-session directional bias for educational research (not trade execution or financial advice).

Session: {context.get("session", "unknown")}
Synthetic DXY: {context.get("current_dxy", "n/a")} ({context.get("dxy_bias", "n/a")}, {context.get("hourly_change_pct", "n/a")}% 1h)
Gold: ~{context.get("gold_price", "n/a")} | technicals {context.get("tech_signal", "n/a")} | RSI {context.get("rsi", "n/a")}
DXY–Gold correlation: {context.get("corr_regime", "n/a")} (r={context.get("correlation", "n/a")})
Macro sentiment: {context.get("macro_sentiment", "n/a")} — {context.get("macro_context", "")}
Agent aggregate: {context.get("direction", "n/a")} (score {context.get("score", "n/a")})

Return ONLY a JSON object with this exact schema:
{{
  "gold_action": "BUY" | "SELL" | "HOLD",
  "dxy_action": "BUY" | "SELL" | "HOLD",
  "confidence": <integer from 0 to 100>,
  "rationale": "<one sentence, max 25 words>"
}}
"""
        return self.query_all(
            prompt,
            system=(
                "You are a senior macro forex and commodities analyst. "
                "Vote BUY/SELL/HOLD only. Be concise. Return valid JSON."
            ),
        )
