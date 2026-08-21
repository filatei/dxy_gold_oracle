"""Shared configuration helpers."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
WEIGHTS_PATH = CONFIG_DIR / "weights.json"


@lru_cache(maxsize=1)
def load_dxy_weights() -> Dict[str, Any]:
    """Load DXY basket weights without mutating the source file contents in memory."""
    with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    multiplier = float(raw.get("multiplier", 50.14348112))
    weights = {k: float(v) for k, v in raw.items() if k != "multiplier"}
    return {"multiplier": multiplier, "weights": weights}


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
