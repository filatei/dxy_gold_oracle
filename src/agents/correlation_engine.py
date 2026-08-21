"""DXY-Gold correlation analysis."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.agents.dxy_synthesizer import calculate_synthetic_dxy, extract_close


def analyze_dxy_gold_correlation(
    dxy_data: Dict[str, pd.DataFrame], gold_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate rolling correlation between synthetic DXY and Gold.
    Returns correlation regime and score.
    """
    dxy_series = []
    gold_series = []

    gold_closes = extract_close(gold_df)
    max_lookback = min(50, len(gold_closes) - 1)

    for i in range(max_lookback, -1, -1):
        try:
            dxy_val = calculate_synthetic_dxy(dxy_data, lookback=i)
            gold_val = float(gold_closes.iloc[-(i + 1)])
            dxy_series.append(dxy_val)
            gold_series.append(gold_val)
        except Exception as e:
            print(f"[Corr] Skipping lookback {i}: {e}")
            continue

    if len(dxy_series) < 10:
        return {"regime": "UNKNOWN", "score": 0, "correlation": 0.0, "sample_size": len(dxy_series)}

    dxy_arr = np.array(dxy_series, dtype=float)
    gold_arr = np.array(gold_series, dtype=float)

    corr = float(np.corrcoef(dxy_arr, gold_arr)[0, 1])
    if np.isnan(corr):
        corr = 0.0

    # Inverse correlation supports gold when DXY moves opposite; score is a soft bias
    # toward trusting the classic DXY↔gold inverse relationship.
    if corr < -0.7:
        regime = "STRONG_INVERSE"
        score = 2
    elif corr < -0.3:
        regime = "WEAK_INVERSE"
        score = 1
    elif corr > 0.3:
        regime = "POSITIVE"
        score = -1
    else:
        regime = "NEUTRAL"
        score = 0

    return {
        "regime": regime,
        "score": score,
        "correlation": round(corr, 3),
        "sample_size": len(dxy_series),
    }
