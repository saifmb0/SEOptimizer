import logging
import math
from typing import Dict, List, Tuple, Optional

import numpy as np


def raw_volume_proxy(keyword: str, freq: int, is_question: bool) -> float:
    base = 1.0 + math.log1p(max(1, freq))  # grows slowly
    if is_question:
        base *= 1.2
    # small boost for longer long-tails
    base *= (1.0 + 0.05 * max(0, len(keyword.split()) - 2))
    return float(base)


def raw_difficulty_proxy(keyword: str, total_results: Optional[int] = None) -> float:
    if total_results:
        return float(math.log10(max(10, total_results)))  # ~1..10
    length_penalty = 1.0 if len(keyword.split()) <= 2 else 0.6
    head_terms = any(h in keyword for h in ["best", "top", "review", "compare", "pricing"])  # slightly harder
    term_penalty = 1.2 if head_terms else 1.0
    return float(length_penalty * term_penalty)


def business_relevance(intent: str, goals: str) -> float:
    g = goals.lower()
    # Match sales/revenue/lead (catches 'leads', 'lead generation', etc.)
    if any(k in g for k in ["sales", "revenue", "lead"]):
        if intent in ("transactional", "commercial"):
            return 1.0
    # Match traffic/brand awareness goals
    if any(k in g for k in ["traffic", "brand", "awareness", "exposure"]):
        if intent in ("informational", "navigational"):
            return 0.8
    # Default fallback
    return 0.6


def compute_metrics(
    keywords: List[str],
    clusters: Dict[str, List[str]],
    intents: Dict[str, str],
    freq: Dict[str, int],
    questions: set,
    provider: str,
    serp_total_results: Optional[int] = None,
) -> Dict[str, Dict]:
    # Raw proxies
    v_raw = {k: raw_volume_proxy(k, freq.get(k, 1), k in questions) for k in keywords}
    d_raw = {k: raw_difficulty_proxy(k, serp_total_results) for k in keywords}

    # Normalize to 0..1
    def normalize(d: Dict[str, float]) -> Dict[str, float]:
        vals = np.array(list(d.values()), dtype=float)
        if len(vals) == 0:
            vals = np.array([1.0])
        vmin, vmax = float(vals.min()), float(vals.max())
        denom = (vmax - vmin) if vmax > vmin else 1.0
        return {k: float((d[k] - vmin) / denom) for k in d}

    v_norm = normalize(v_raw)
    d_norm = normalize(d_raw)

    estimated = True  # no paid SERP data by default; Gemini expansions still mark as estimated

    metrics: Dict[str, Dict] = {}
    for k in keywords:
        metrics[k] = {
            "search_volume": float(max(0.0, min(1.0, v_norm[k]))),
            "difficulty": float(max(0.0, min(1.0, d_norm[k]))),
            "estimated": bool(estimated),
        }
    return metrics


def opportunity_scores(metrics: Dict[str, Dict], intents: Dict[str, str], goals: str) -> Dict[str, float]:
    scores = {}
    for k, m in metrics.items():
        br = business_relevance(intents.get(k, "informational"), goals)  # 0.6..1.0
        v = m.get("search_volume", 0.0)
        d = m.get("difficulty", 0.5)
        score = v * (1.0 - d) * br
        scores[k] = float(max(0.0, min(1.0, score)))
    return scores
