import logging
import math
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter

import numpy as np
from scipy import stats


# =============================================================================
# Perplexity / Naturalness Scoring (Week 4)
# =============================================================================
# Uses sentence-transformers to score how "natural" a keyword phrase sounds.
# Low perplexity = natural phrase, High perplexity = nonsense/artifact

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
    # Lazy load the model
    _perplexity_model = None
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    _perplexity_model = None


def _get_perplexity_model():
    """Lazy load the sentence transformer model."""
    global _perplexity_model
    if _perplexity_model is None and HAS_SENTENCE_TRANSFORMERS:
        try:
            _perplexity_model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.debug("Loaded sentence-transformers model for perplexity scoring")
        except Exception as e:
            logging.warning(f"Failed to load sentence-transformers model: {e}")
    return _perplexity_model


# Reference phrases for naturalness comparison
NATURAL_REFERENCE_PHRASES = [
    "best contractors in dubai",
    "villa renovation cost",
    "how to hire a contractor",
    "construction company near me",
    "warehouse building services",
    "professional renovation services",
    "where to find contractors",
    "commercial construction dubai",
]

# Clearly unnatural/garbage phrases for comparison
UNNATURAL_REFERENCE_PHRASES = [
    "template owners developers facility",
    "en ae ar login menu",
    "cookie privacy terms login",
    "vs or and the for",
    "home about contact menu footer",
]


def calculate_naturalness_score(keyword: str) -> float:
    """
    Calculate how "natural" a keyword phrase sounds using embeddings.
    
    Uses sentence-transformers to compute similarity to natural vs unnatural
    reference phrases. Higher score = more natural.
    
    Args:
        keyword: The keyword to score
        
    Returns:
        Float between 0-1 where 1 = highly natural, 0 = nonsense
    """
    model = _get_perplexity_model()
    if model is None:
        return 0.5  # Default to neutral if model unavailable
    
    try:
        # Encode the keyword
        kw_embedding = model.encode(keyword, convert_to_tensor=True)
        
        # Encode reference phrases
        natural_embeddings = model.encode(NATURAL_REFERENCE_PHRASES, convert_to_tensor=True)
        unnatural_embeddings = model.encode(UNNATURAL_REFERENCE_PHRASES, convert_to_tensor=True)
        
        # Calculate similarity to natural phrases
        natural_sims = util.cos_sim(kw_embedding, natural_embeddings)[0]
        avg_natural_sim = float(natural_sims.mean())
        
        # Calculate similarity to unnatural phrases
        unnatural_sims = util.cos_sim(kw_embedding, unnatural_embeddings)[0]
        avg_unnatural_sim = float(unnatural_sims.mean())
        
        # Score = natural_sim - unnatural_sim, normalized to 0-1
        # If more similar to natural, score is high
        raw_score = (avg_natural_sim - avg_unnatural_sim + 1) / 2
        
        return max(0.0, min(1.0, raw_score))
        
    except Exception as e:
        logging.debug(f"Naturalness scoring error for '{keyword}': {e}")
        return 0.5


def batch_naturalness_scores(keywords: List[str]) -> Dict[str, float]:
    """
    Calculate naturalness scores for a batch of keywords.
    
    More efficient than calling calculate_naturalness_score repeatedly.
    
    Args:
        keywords: List of keywords to score
        
    Returns:
        Dict mapping keyword to naturalness score
    """
    if not keywords:
        return {}
    
    model = _get_perplexity_model()
    if model is None:
        return {kw: 0.5 for kw in keywords}
    
    try:
        # Encode all at once for efficiency
        kw_embeddings = model.encode(keywords, convert_to_tensor=True)
        natural_embeddings = model.encode(NATURAL_REFERENCE_PHRASES, convert_to_tensor=True)
        unnatural_embeddings = model.encode(UNNATURAL_REFERENCE_PHRASES, convert_to_tensor=True)
        
        scores = {}
        for i, kw in enumerate(keywords):
            natural_sims = util.cos_sim(kw_embeddings[i], natural_embeddings)[0]
            unnatural_sims = util.cos_sim(kw_embeddings[i], unnatural_embeddings)[0]
            
            raw_score = (float(natural_sims.mean()) - float(unnatural_sims.mean()) + 1) / 2
            scores[kw] = max(0.0, min(1.0, raw_score))
        
        return scores
        
    except Exception as e:
        logging.debug(f"Batch naturalness scoring error: {e}")
        return {kw: 0.5 for kw in keywords}


# =============================================================================
# Universal Term Penalty (Week 4)
# =============================================================================
# Detect terms that appear in >80% of documents (navigation, boilerplate)
# and penalize keywords containing them.

# Default universal terms (common boilerplate)
DEFAULT_UNIVERSAL_TERMS = frozenset({
    "login", "sign in", "sign up", "register", "password",
    "cookie", "cookies", "privacy", "privacy policy", "terms",
    "terms of service", "copyright", "all rights reserved",
    "home", "about", "contact", "menu", "footer", "header",
    "subscribe", "newsletter", "follow us", "share",
})


def detect_universal_terms(
    documents: List[str],
    threshold: float = 0.8,
    min_doc_count: int = 3,
) -> Set[str]:
    """
    Detect terms that appear in a high percentage of documents.
    
    Terms appearing in >threshold% of documents are likely navigation,
    boilerplate, or other non-distinctive content.
    
    Args:
        documents: List of document texts
        threshold: Minimum document frequency to be considered universal
        min_doc_count: Minimum documents required for meaningful analysis
        
    Returns:
        Set of universal terms
    """
    if len(documents) < min_doc_count:
        return DEFAULT_UNIVERSAL_TERMS
    
    # Tokenize and count document frequency
    term_doc_counts: Counter = Counter()
    
    for doc in documents:
        # Get unique terms in this document
        doc_terms = set(doc.lower().split())
        for term in doc_terms:
            term_doc_counts[term] += 1
    
    # Find terms appearing in >threshold% of documents
    universal = set()
    for term, count in term_doc_counts.items():
        doc_freq = count / len(documents)
        if doc_freq >= threshold:
            universal.add(term)
    
    # Add default universal terms
    universal.update(DEFAULT_UNIVERSAL_TERMS)
    
    logging.debug(f"Detected {len(universal)} universal terms (threshold={threshold})")
    return universal


def calculate_universal_term_penalty(
    keyword: str,
    universal_terms: Set[str],
    penalty_per_term: float = 0.3,
    max_penalty: float = 0.9,
) -> float:
    """
    Calculate penalty for keywords containing universal terms.
    
    Keywords with navigation/boilerplate terms get penalized
    to reduce their opportunity score.
    
    Args:
        keyword: The keyword to check
        universal_terms: Set of universal terms to check against
        penalty_per_term: Penalty for each universal term found
        max_penalty: Maximum total penalty (cap)
        
    Returns:
        Float between 0-max_penalty representing the penalty
    """
    kw_tokens = set(keyword.lower().split())
    
    # Count how many universal terms are in the keyword
    universal_count = len(kw_tokens & universal_terms)
    
    if universal_count == 0:
        return 0.0
    
    # Calculate penalty (capped at max_penalty)
    penalty = min(universal_count * penalty_per_term, max_penalty)
    
    logging.debug(f"Universal term penalty for '{keyword}': {penalty:.2f} ({universal_count} terms)")
    return penalty


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
    documents: Optional[List[str]] = None,
    use_naturalness: bool = True,
    use_universal_penalty: bool = True,
) -> Dict[str, float]:
    """
    Calculate opportunity scores for keywords with quality filtering.
    
    Formula: (search_volume * ctr * (1 - difficulty) * relevance) * naturalness - universal_penalty
    
    Week 4 additions:
    - Naturalness scoring via sentence-transformers (penalizes nonsense)
    - Universal term penalty (penalizes boilerplate/navigation terms)
    
    Args:
        metrics: Dict of keyword -> metrics dict
        intents: Dict of keyword -> intent classification
        goals: Business goals string ('traffic', 'leads', etc.)
        geo: Geographic target for locale-specific scoring
        niche: Optional niche for specialized scoring
        use_ctr_adjustment: Whether to apply SERP feature CTR adjustment
        commercial_weight: Weight for commercial value boost (0.0-1.0)
        documents: Optional documents for detecting universal terms
        use_naturalness: Whether to apply naturalness scoring
        use_universal_penalty: Whether to apply universal term penalty
        
    Returns:
        Dict mapping keyword -> opportunity score (0.0 - 1.0)
    """
    scores = {}
    goals_lower = goals.lower()
    
    # Determine if we should prioritize commercial value
    prioritize_leads = any(k in goals_lower for k in ["lead", "sales", "revenue", "conversion"])
    
    # Detect universal terms if documents provided
    universal_terms = set()
    if use_universal_penalty and documents:
        universal_terms = detect_universal_terms(documents)
    elif use_universal_penalty:
        universal_terms = DEFAULT_UNIVERSAL_TERMS
    
    # Calculate naturalness scores in batch for efficiency
    keywords = list(metrics.keys())
    naturalness_scores = {}
    if use_naturalness and HAS_SENTENCE_TRANSFORMERS:
        naturalness_scores = batch_naturalness_scores(keywords)
    
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
        
        # Apply naturalness scoring (Week 4)
        # Low naturalness = likely garbage, reduce score
        if use_naturalness and k in naturalness_scores:
            naturalness = naturalness_scores[k]
            # Naturalness below 0.4 gets heavily penalized
            if naturalness < 0.4:
                score *= naturalness  # Strong reduction
            elif naturalness < 0.6:
                score *= (0.5 + naturalness)  # Moderate reduction
        
        # Apply universal term penalty (Week 4)
        if use_universal_penalty and universal_terms:
            penalty = calculate_universal_term_penalty(k, universal_terms)
            score = score * (1 - penalty)
        
        scores[k] = float(max(0.0, min(1.0, score)))
    
    return scores
