import logging
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import stats


def raw_volume_proxy(
    keyword: str, 
    freq: int, 
    is_question: bool,
    is_validated: bool = False,
) -> float:
    """
    Calculate raw volume proxy for a keyword.
    
    Args:
        keyword: The keyword string
        freq: Document frequency count
        is_question: Whether keyword is a question
        is_validated: Whether keyword was validated via autocomplete (2x boost)
        
    Returns:
        Raw volume score (not normalized)
    """
    base = 1.0 + math.log1p(max(1, freq))  # grows slowly
    if is_question:
        base *= 1.2
    # small boost for longer long-tails
    base *= (1.0 + 0.05 * max(0, len(keyword.split()) - 2))
    # 2x multiplier for autocomplete-validated keywords
    if is_validated:
        base *= 2.0
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
    serp_total_results: Optional[Dict[str, int]] = None,
    validated_keywords: Optional[Dict[str, bool]] = None,
) -> Dict[str, Dict]:
    """
    Compute SEO metrics for keywords.
    
    Args:
        keywords: List of keywords to score
        clusters: Cluster assignments
        intents: Intent classifications
        freq: Document frequency counts
        questions: Set of question-style keywords
        provider: SERP provider used
        serp_total_results: Optional dict of keyword -> total SERP results
        validated_keywords: Optional dict of keyword -> autocomplete validation
        
    Returns:
        Dict mapping keyword to metrics dict
    """
    validated = validated_keywords or {}
    serp_results = serp_total_results or {}
    
    # Raw proxies with autocomplete validation boost
    v_raw = {
        k: raw_volume_proxy(k, freq.get(k, 1), k in questions, validated.get(k, False)) 
        for k in keywords
    }
    d_raw = {k: raw_difficulty_proxy(k, serp_results.get(k)) for k in keywords}

    # Percentile ranking normalization (preserves mid-tier keyword value)
    def normalize_percentile(d: Dict[str, float]) -> Dict[str, float]:
        """Normalize using percentile ranking instead of min-max."""
        if not d:
            return {}
        keys = list(d.keys())
        vals = np.array([d[k] for k in keys], dtype=float)
        
        if len(vals) <= 1:
            return {k: 0.5 for k in keys}
        
        # Use percentile ranking: each value gets its percentile position
        percentiles = stats.rankdata(vals, method='average') / len(vals)
        return {k: float(p) for k, p in zip(keys, percentiles)}
    
    # Legacy min-max normalization for difficulty (competitive metric)
    def normalize_minmax(d: Dict[str, float]) -> Dict[str, float]:
        vals = np.array(list(d.values()), dtype=float)
        if len(vals) == 0:
            return {}
        vmin, vmax = float(vals.min()), float(vals.max())
        denom = (vmax - vmin) if vmax > vmin else 1.0
        return {k: float((d[k] - vmin) / denom) for k in d}

    # Use percentile for volume (preserves long-tail value)
    v_norm = normalize_percentile(v_raw)
    # Use min-max for difficulty (competitive comparison)
    d_norm = normalize_minmax(d_raw)

    # Mark as estimated unless we have real SERP data
    has_real_data = bool(serp_results)

    metrics: Dict[str, Dict] = {}
    for k in keywords:
        metrics[k] = {
            "search_volume": float(max(0.0, min(1.0, v_norm.get(k, 0.5)))),
            "difficulty": float(max(0.0, min(1.0, d_norm.get(k, 0.5)))),
            "estimated": not has_real_data,
            "validated": validated.get(k, False),
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
