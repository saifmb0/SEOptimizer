import logging
import math
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from scipy import stats


# =============================================================================
# Commercial Intent Indicators (CPC Proxy Heuristics)
# =============================================================================

# High-value commercial keywords indicating purchase/lead intent
# These typically have higher CPC in paid search
COMMERCIAL_TRIGGERS = {
    "quote", "quotes", "price", "prices", "pricing", "cost", "costs",
    "rates", "estimate", "estimates", "estimation",
    "companies", "company", "contractors", "contractor", "services",
    "hire", "hiring", "agency", "agencies", "firm", "firms",
    "consultation", "consult", "consultant", "consultants",
    "package", "packages", "plan", "plans",
}

# Transactional triggers (highest commercial value)
TRANSACTIONAL_TRIGGERS = {
    "buy", "purchase", "order", "book", "booking",
    "get quote", "request quote", "free quote",
    "near me", "in my area", "local",
    "for sale", "deals", "discount", "offer",
}

# B2B/Enterprise indicators (high-value leads)
B2B_TRIGGERS = {
    "enterprise", "business", "commercial", "corporate", "b2b",
    "wholesale", "bulk", "industrial", "professional",
    "solutions", "platform", "software", "system",
}

# UAE/Gulf-specific commercial terms
UAE_COMMERCIAL_TRIGGERS = {
    "fit out", "fitout", "turnkey", "mep", "hvac",
    "renovation", "construction", "contracting",
    "interior design", "landscaping",
    "license", "approval", "permit",
    "villa", "warehouse", "office",
}


# =============================================================================
# SERP Feature CTR Adjustments
# =============================================================================
# SERP features that reduce organic CTR for traditional blue links
# Based on industry research on click distribution with various SERP features

# CTR reduction factors (0.0 = no reduction, 1.0 = complete CTR loss)
SERP_FEATURE_CTR_IMPACT = {
    # Knowledge panels/instant answers (major CTR reduction)
    "knowledge_panel": 0.30,    # Definitional queries satisfied in SERP
    "featured_snippet": 0.25,   # Answer displayed, fewer clicks needed
    "instant_answer": 0.35,     # Calculator, weather, time, etc.
    "local_pack": 0.20,         # Map pack takes clicks for local queries
    
    # Rich results (moderate CTR reduction)
    "shopping_results": 0.15,   # Product carousel above organic
    "video_carousel": 0.10,     # Video results for how-to queries
    "image_pack": 0.08,         # Image results for visual queries
    "people_also_ask": 0.05,    # PAA expands but can drive clicks too
    
    # Ads (significant CTR reduction for commercial queries)
    "top_ads": 0.20,            # Google Ads above organic
    "bottom_ads": 0.05,         # Ads below organic (minor impact)
    "shopping_ads": 0.18,       # Product listing ads
}

# Keyword patterns that likely trigger SERP features
SERP_FEATURE_TRIGGERS = {
    # Knowledge panel triggers (definitions, facts)
    "knowledge_panel": [
        "what is", "what are", "define", "definition", "meaning",
        "who is", "who was", "when did", "when was", "where is",
        "capital of", "population of", "height of", "age of",
    ],
    # Featured snippet triggers (instructional/listicle)
    "featured_snippet": [
        "how to", "how do", "steps to", "ways to", "tips for",
        "guide to", "tutorial", "best way", "top 10", "list of",
        "checklist", "template", "example", "examples of",
    ],
    # Instant answer triggers (calculations, conversions)
    "instant_answer": [
        "calculator", "convert", "conversion", "time in", "weather in",
        "exchange rate", "translate", "timer", "countdown",
        "sunrise", "sunset", "timezone", "distance from",
    ],
    # Local pack triggers
    "local_pack": [
        "near me", "in dubai", "in abu dhabi", "in sharjah",
        "nearby", "closest", "local", "around me",
        "open now", "hours", "directions to", "address",
    ],
    # Video carousel triggers
    "video_carousel": [
        "how to", "tutorial", "diy", "review", "unboxing",
        "walkthrough", "demonstration", "explained", "course",
    ],
    # Shopping results triggers
    "shopping_results": [
        "buy", "price", "cheap", "discount", "deal", "sale",
        "best price", "where to buy", "online", "order",
        "for sale", "cost of", "prices for",
    ],
}


def estimate_serp_features(keyword: str, intent: str = "informational") -> List[str]:
    """
    Estimate which SERP features are likely to appear for a keyword.
    
    This is a heuristic-based estimation since we don't have live SERP data.
    Used to adjust CTR expectations for opportunity scoring.
    
    Args:
        keyword: The keyword to analyze
        intent: Pre-classified intent
        
    Returns:
        List of likely SERP features for this keyword
    """
    kw_lower = keyword.lower()
    features = []
    
    # Check each feature's triggers
    for feature, triggers in SERP_FEATURE_TRIGGERS.items():
        for trigger in triggers:
            if trigger in kw_lower:
                features.append(feature)
                break
    
    # Intent-based feature estimation
    if intent == "informational":
        if "featured_snippet" not in features:
            # Informational queries often get PAA
            features.append("people_also_ask")
    
    if intent in ("transactional", "commercial"):
        if "top_ads" not in features:
            features.append("top_ads")  # Commercial queries attract ads
    
    if intent == "local":
        if "local_pack" not in features:
            features.append("local_pack")
    
    return features


def ctr_potential(
    keyword: str,
    intent: str = "informational",
    serp_features: Optional[List[str]] = None,
) -> float:
    """
    Calculate CTR potential score accounting for SERP feature competition.
    
    A score of 1.0 means maximum organic CTR potential (no SERP features).
    Lower scores indicate reduced organic CTR due to SERP features.
    
    This helps prioritize keywords where organic rankings will actually
    drive traffic, rather than keywords dominated by SERP features.
    
    Args:
        keyword: The keyword to evaluate
        intent: Pre-classified intent
        serp_features: Optional list of known SERP features (auto-estimated if None)
        
    Returns:
        CTR potential score (0.0 - 1.0)
        - 1.0: Maximum CTR potential (no SERP features)
        - 0.7-0.9: Good CTR potential (minor SERP features)
        - 0.5-0.7: Moderate CTR potential (some features)
        - 0.3-0.5: Reduced CTR potential (many features)
        - <0.3: Low CTR potential (dominated by features)
    """
    if serp_features is None:
        serp_features = estimate_serp_features(keyword, intent)
    
    # Start with 100% CTR potential
    ctr = 1.0
    
    # Reduce for each SERP feature (diminishing returns)
    for feature in serp_features:
        impact = SERP_FEATURE_CTR_IMPACT.get(feature, 0.05)
        # Apply reduction (multiplicative, not additive)
        ctr *= (1.0 - impact)
    
    # Floor at 0.2 (there's always some organic CTR)
    return float(max(0.2, min(1.0, ctr)))


def raw_volume_proxy(
    keyword: str, 
    freq: int, 
    is_question: bool,
    is_validated: bool = False,
    total_docs: int = 0,
) -> float:
    """
    Calculate raw volume proxy for a keyword.
    
    Args:
        keyword: The keyword string
        freq: Document frequency count
        is_question: Whether keyword is a question
        is_validated: Whether keyword was validated via autocomplete (2x boost)
        total_docs: Total documents in corpus (for universal term detection)
        
    Returns:
        Raw volume score (not normalized)
    """
    # ORYX Fix: Detect universal terms (appear in almost every document)
    # These are likely stopwords, navigation artifacts, or scraping noise
    # Example: "en ae" appearing in every page due to language toggle
    if total_docs > 0 and freq >= total_docs * 0.8:  # Appears in 80%+ of docs
        # Penalize heavily - likely not a real keyword
        logging.debug(f"Universal term detected: '{keyword}' (freq={freq}/{total_docs})")
        return 0.1  # Minimal score
    
    base = 1.0 + math.log1p(max(1, freq))  # grows slowly
    if is_question:
        base *= 1.2
    # small boost for longer long-tails
    base *= (1.0 + 0.05 * max(0, len(keyword.split()) - 2))
    # 2x multiplier for autocomplete-validated keywords
    if is_validated:
        base *= 2.0
    return float(base)


def raw_difficulty_proxy(
    keyword: str, 
    total_results: Optional[int] = None,
    doc_freq: int = 0,
    total_docs: int = 0,
) -> float:
    """
    Calculate raw difficulty proxy for a keyword.
    
    Args:
        keyword: The keyword string
        total_results: Optional SERP total results count
        doc_freq: Document frequency in corpus
        total_docs: Total documents in corpus (for universal term detection)
        
    Returns:
        Raw difficulty score
    """
    # ORYX Fix: Universal terms get max difficulty (they're noise, not keywords)
    if total_docs > 0 and doc_freq >= total_docs * 0.8:
        return 10.0  # Max difficulty - penalize heavily
    
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


def commercial_value(
    keyword: str,
    intent: str = "informational",
    geo: str = "global",
    niche: Optional[str] = None,
) -> float:
    """
    Calculate commercial value score (CPC proxy heuristic).
    
    Since we don't have paid API access to CPC data, we use trigger-word
    heuristics to estimate the commercial value of a keyword.
    
    High commercial value keywords are prioritized for lead generation
    over traffic-focused informational keywords.
    
    Args:
        keyword: The keyword to evaluate
        intent: Pre-classified intent ('transactional', 'commercial', etc.)
        geo: Geographic target (enables locale-specific triggers)
        niche: Optional niche for specialized triggers
        
    Returns:
        Commercial value score (0.0 - 1.0)
        - 0.0-0.3: Low commercial value (informational)
        - 0.3-0.6: Medium commercial value (research phase)
        - 0.6-0.8: High commercial value (comparison/evaluation)
        - 0.8-1.0: Very high commercial value (ready to convert)
    """
    kw_lower = keyword.lower()
    tokens = set(kw_lower.split())
    
    score = 0.0
    
    # Base score from intent
    intent_scores = {
        "transactional": 0.6,
        "commercial": 0.4,
        "comparative": 0.35,
        "local": 0.3,
        "complex_research": 0.2,
        "direct_answer": 0.1,
        "informational": 0.1,
        "navigational": 0.0,
    }
    score += intent_scores.get(intent, 0.1)
    
    # Transactional trigger boost (highest value)
    for trigger in TRANSACTIONAL_TRIGGERS:
        if trigger in kw_lower:
            score += 0.3
            break
    
    # Commercial trigger boost
    commercial_matches = len(tokens & COMMERCIAL_TRIGGERS)
    score += min(0.2, commercial_matches * 0.1)
    
    # B2B/Enterprise trigger boost
    b2b_matches = len(tokens & B2B_TRIGGERS)
    score += min(0.15, b2b_matches * 0.1)
    
    # UAE/Gulf-specific commercial triggers
    if geo.lower() in ("ae", "sa", "qa", "kw", "bh", "om"):
        for trigger in UAE_COMMERCIAL_TRIGGERS:
            if trigger in kw_lower:
                score += 0.15
                break
    
    # Contracting niche boost for specific service terms
    if niche and niche.lower() in ("contracting", "construction"):
        construction_terms = {"villa", "warehouse", "fit out", "renovation", "mep"}
        for term in construction_terms:
            if term in kw_lower:
                score += 0.1
                break
    
    # Cap at 1.0
    return float(min(1.0, max(0.0, score)))


def compute_metrics(
    keywords: List[str],
    clusters: Dict[str, List[str]],
    intents: Dict[str, str],
    freq: Dict[str, int],
    questions: set,
    provider: str,
    serp_total_results: Optional[Dict[str, int]] = None,
    validated_keywords: Optional[Dict[str, bool]] = None,
    total_docs: int = 0,
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
        total_docs: Total documents in corpus (for universal term detection)
        
    Returns:
        Dict mapping keyword to metrics dict
    """
    validated = validated_keywords or {}
    serp_results = serp_total_results or {}
    
    # Raw proxies with autocomplete validation boost and universal term detection
    v_raw = {
        k: raw_volume_proxy(
            k, freq.get(k, 1), k in questions, 
            validated.get(k, False), total_docs
        ) 
        for k in keywords
    }
    d_raw = {
        k: raw_difficulty_proxy(
            k, serp_results.get(k),
            doc_freq=freq.get(k, 0), total_docs=total_docs
        ) 
        for k in keywords
    }

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
        intent = intents.get(k, "informational")
        ctr = ctr_potential(k, intent)
        serp_features = estimate_serp_features(k, intent)
        
        metrics[k] = {
            "search_volume": float(max(0.0, min(1.0, v_norm.get(k, 0.5)))),
            "difficulty": float(max(0.0, min(1.0, d_norm.get(k, 0.5)))),
            "ctr_potential": ctr,
            "serp_features": serp_features,
            "estimated": not has_real_data,
            "validated": validated.get(k, False),
        }
    return metrics


def opportunity_scores(
    metrics: Dict[str, Dict],
    intents: Dict[str, str],
    goals: str,
    geo: str = "global",
    niche: Optional[str] = None,
    use_ctr_adjustment: bool = True,
    commercial_weight: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate opportunity scores for keywords.
    
    Formula: search_volume * ctr_potential * (1 - difficulty) * (business_relevance + commercial_boost)
    
    CTR potential accounts for SERP features that reduce organic click-through.
    For lead-generation goals, commercial_value provides additional boost
    to high-value transactional keywords.
    
    Args:
        metrics: Dict of keyword -> metrics dict
        intents: Dict of keyword -> intent classification
        goals: Business goals string ('traffic', 'leads', etc.)
        geo: Geographic target for locale-specific scoring
        niche: Optional niche for specialized scoring
        use_ctr_adjustment: Whether to apply SERP feature CTR adjustment (default: True)
        commercial_weight: Weight for commercial value boost (0.0-1.0, default: 0.5)
        
    Returns:
        Dict mapping keyword -> opportunity score (0.0 - 1.0)
    """
    scores = {}
    goals_lower = goals.lower()
    
    # Determine if we should prioritize commercial value
    prioritize_leads = any(k in goals_lower for k in ["lead", "sales", "revenue", "conversion"])
    
    for k, m in metrics.items():
        intent = intents.get(k, "informational")
        br = business_relevance(intent, goals)  # 0.6..1.0
        v = m.get("search_volume", 0.0)
        d = m.get("difficulty", 0.5)
        ctr = m.get("ctr_potential", 1.0) if use_ctr_adjustment else 1.0
        
        # Base score formula with CTR adjustment
        # CTR potential reduces score for keywords where SERP features dominate
        base_score = v * ctr * (1.0 - d) * br
        
        # Add commercial boost for lead-focused goals
        if prioritize_leads:
            cv = commercial_value(k, intent, geo, niche)
            # Commercial boost scaled by weight (default 50% boost at max)
            commercial_boost = cv * commercial_weight
            score = base_score + (base_score * commercial_boost)
        else:
            score = base_score
        
        scores[k] = float(max(0.0, min(1.0, score)))
    
    return scores
