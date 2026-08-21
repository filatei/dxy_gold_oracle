"""Free LLM Router with cascading fallback."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

import requests


class FreeLLMRouter:
    """Routes prompts to available free LLM APIs with cascading fallback."""

    def __init__(self, timeout: int = 30):
        self.keys = {
            "groq": os.getenv("GROQ_API_KEY", "").strip() or None,
            "gemini": os.getenv("GEMINI_API_KEY", "").strip() or None,
            "openrouter": os.getenv("OPENROUTER_API_KEY", "").strip() or None,
        }
        self.timeout = timeout
        # Prefer env overrides so Actions/local can pin without code edits
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.openrouter_model = os.getenv(
            "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"
        )

    def query(self, prompt: str, system: str = "You are a macro forex analyst.") -> Dict[str, Any]:
        """Try Groq -> Gemini -> OpenRouter. Return parsed JSON or neutral fallback."""
        providers = [
            ("groq", self._groq),
            ("gemini", self._gemini),
            ("openrouter", self._openrouter),
        ]
        for name, fn in providers:
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

    def _openrouter(self, prompt: str, system: str) -> Dict[str, Any]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.keys['openrouter']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/filatei/dxy_gold_oracle",
            "X-Title": "DXY-Gold Oracle",
        }
        payload = {
            "model": self.openrouter_model,
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
