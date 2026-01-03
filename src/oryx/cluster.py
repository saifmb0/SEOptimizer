import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    logging.debug(
        "sentence-transformers not installed. Install with: pip install oryx[ml]. "
        "Falling back to TF-IDF vectorization."
    )

# =============================================================================
# Embedding Model Configuration
# =============================================================================

# English-optimized models
# multi-qa-MiniLM-L6-cos-v1: Optimized for question/answer retrieval (best for GEO/SGE)
# all-MiniLM-L6-v2: General purpose semantic similarity
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Multilingual models for Arabic/non-English markets
# paraphrase-multilingual-MiniLM-L12-v2: Best for Arabic/English bilingual clustering
# distiluse-base-multilingual-cased-v2: Alternative multilingual model
MULTILINGUAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MULTILINGUAL_FALLBACK_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"

# Language-to-model mapping
LANGUAGE_MODEL_MAP = {
    "en": DEFAULT_EMBEDDING_MODEL,
    "ar": MULTILINGUAL_EMBEDDING_MODEL,
    "ar-en": MULTILINGUAL_EMBEDDING_MODEL,
    "ae": MULTILINGUAL_EMBEDDING_MODEL,
    "uae": MULTILINGUAL_EMBEDDING_MODEL,
    "multilingual": MULTILINGUAL_EMBEDDING_MODEL,
}

# Geo-to-model mapping (for automatic model selection based on target market)
GEO_MODEL_MAP = {
    "ae": MULTILINGUAL_EMBEDDING_MODEL,
    "sa": MULTILINGUAL_EMBEDDING_MODEL,  # Saudi Arabia
    "qa": MULTILINGUAL_EMBEDDING_MODEL,  # Qatar
    "kw": MULTILINGUAL_EMBEDDING_MODEL,  # Kuwait
    "bh": MULTILINGUAL_EMBEDDING_MODEL,  # Bahrain
    "om": MULTILINGUAL_EMBEDDING_MODEL,  # Oman
    "eg": MULTILINGUAL_EMBEDDING_MODEL,  # Egypt
    "jo": MULTILINGUAL_EMBEDDING_MODEL,  # Jordan
    "lb": MULTILINGUAL_EMBEDDING_MODEL,  # Lebanon
}


def get_embedding_model_for_locale(
    language: str = "en",
    geo: str = "global",
    model_override: Optional[str] = None,
) -> str:
    """
    Select the appropriate embedding model based on language and geo target.
    
    Args:
        language: Language code ('en', 'ar', 'ar-en', etc.)
        geo: Geographic target ('global', 'ae', 'us', etc.)
        model_override: Optional explicit model name to use
        
    Returns:
        Model name string for SentenceTransformer
    """
    if model_override:
        return model_override
    
    # Check language first
    lang_lower = language.lower().strip()
    if lang_lower in LANGUAGE_MODEL_MAP:
        return LANGUAGE_MODEL_MAP[lang_lower]
    
    # Check geo for Arabic-speaking regions
    geo_lower = geo.lower().strip()
    if geo_lower in GEO_MODEL_MAP:
        return GEO_MODEL_MAP[geo_lower]
    
    # Default to English Q&A model
    return DEFAULT_EMBEDDING_MODEL


# Default intent rules (can be overridden via config)
DEFAULT_INTENT_RULES = {
    "informational": ["who", "what", "why", "how", "guide", "tutorial", "tips", "checklist", "template"],
    "commercial": ["best", "top", "review", "compare", "vs", "alternatives", "pricing"],
    "transactional": ["buy", "discount", "coupon", "deal", "near me"],
    "navigational": [],
}


from dataclasses import dataclass
from typing import Tuple


@dataclass
class IntentResult:
    """Result of intent classification with confidence score."""
    intent: str
    confidence: float  # 0.0 to 1.0
    
    def __iter__(self):
        """Allow tuple unpacking: intent, confidence = infer_intent_with_confidence(...)"""
        return iter((self.intent, self.confidence))


def infer_intent(
    keyword: str, 
    competitors: List[str], 
    intent_rules: Optional[Dict[str, List[str]]] = None
) -> str:
    """
    Infer search intent for a keyword.
    
    Args:
        keyword: The keyword to classify
        competitors: List of competitor domains
        intent_rules: Optional custom intent rules dict (from config)
        
    Returns:
        Intent string: informational, commercial, transactional, or navigational
    """
    result = infer_intent_with_confidence(keyword, competitors, intent_rules)
    return result.intent


def infer_intent_with_confidence(
    keyword: str, 
    competitors: List[str], 
    intent_rules: Optional[Dict[str, List[str]]] = None
) -> IntentResult:
    """
    Infer search intent for a keyword with confidence score.
    
    Confidence scoring:
    - 1.0: Exact match with intent trigger word
    - 0.5: Partial/contextual match
    - 0.1: Default/fallback classification
    
    Args:
        keyword: The keyword to classify
        competitors: List of competitor domains
        intent_rules: Optional custom intent rules dict (from config)
        
    Returns:
        IntentResult with intent string and confidence score
    """
    rules = intent_rules or DEFAULT_INTENT_RULES
    k = keyword.lower()
    tokens = set(k.split())
    
    # Check for exact matches (high confidence)
    for intent, words in rules.items():
        for w in words:
            if " " in w:
                # Multi-word phrase match
                if w in k:
                    return IntentResult(intent=intent, confidence=1.0)
            else:
                # Single word exact match
                if w in tokens:
                    return IntentResult(intent=intent, confidence=1.0)
    
    # Check for partial/contextual matches (medium confidence)
    # Commercial signals without exact match
    commercial_signals = {"review", "vs", "versus", "alternative", "option", "recommend"}
    if tokens & commercial_signals:
        return IntentResult(intent="commercial", confidence=0.5)
    
    # Transactional signals without exact match  
    transactional_signals = {"order", "purchase", "get", "find", "hire", "book"}
    if tokens & transactional_signals:
        return IntentResult(intent="transactional", confidence=0.5)
    
    # Informational signals without exact match
    informational_signals = {"learn", "understand", "mean", "difference", "explain"}
    if tokens & informational_signals:
        return IntentResult(intent="informational", confidence=0.5)
    
    # Navigational: contains competitor brand tokens
    for c in competitors:
        token = c.split(".")[0].lower()
        if token and token in k:
            return IntentResult(intent="navigational", confidence=0.8)
    
    # Default fallback (low confidence - ambiguous intent)
    return IntentResult(intent="informational", confidence=0.1)


def vectorize_keywords(
    keywords: List[str],
    model_name: Optional[str] = None,
    language: str = "en",
    geo: str = "global",
):
    """
    Vectorize keywords for clustering with multilingual support.
    
    Uses sentence-transformers if available. Automatically selects
    multilingual model for Arabic/Gulf markets.
    
    Args:
        keywords: List of keywords to vectorize
        model_name: Optional specific model name override
        language: Language code for automatic model selection
        geo: Geographic target for automatic model selection
        
    Returns:
        numpy array of embeddings
    """
    if HAS_ST and keywords:
        # Select model based on locale
        model_to_use = get_embedding_model_for_locale(language, geo, model_name)
        
        try:
            model = SentenceTransformer(model_to_use)
            X = model.encode(keywords, show_progress_bar=False, normalize_embeddings=True)
            logging.debug(f"Vectorized {len(keywords)} keywords with {model_to_use}")
            return np.array(X)
        except Exception as e:
            logging.debug(f"Failed to load {model_to_use}: {e}, trying fallback")
            # Try appropriate fallback based on locale
            fallback = MULTILINGUAL_FALLBACK_MODEL if geo.lower() in GEO_MODEL_MAP else FALLBACK_EMBEDDING_MODEL
            try:
                model = SentenceTransformer(fallback)
                X = model.encode(keywords, show_progress_bar=False, normalize_embeddings=True)
                return np.array(X)
            except Exception:
                pass
    
    # Fallback to TF-IDF (works for any language but less semantic)
    # Note: Using None for stop_words to support multilingual
    vec = TfidfVectorizer(stop_words=None if geo.lower() in GEO_MODEL_MAP else "english")
    X = vec.fit_transform(keywords)
    return X.toarray()


def choose_k(n_keywords: int, max_clusters: int) -> int:
    """Simple heuristic for choosing K when silhouette is disabled."""
    if n_keywords < 12:
        return max(1, min(4, n_keywords))
    k = min(max_clusters, n_keywords // 10 + 1)
    k = max(6, min(12, k))
    return k


def choose_k_silhouette(
    X: np.ndarray, 
    k_min: int = 4, 
    k_max: int = 15, 
    random_state: int = 42
) -> int:
    """
    Choose optimal K using Silhouette Score.
    
    Runs KMeans for each K in range and picks the one with highest silhouette score.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        k_min: Minimum number of clusters to try
        k_max: Maximum number of clusters to try
        random_state: Random seed for reproducibility
        
    Returns:
        Optimal number of clusters
    """
    from sklearn.metrics import silhouette_score
    
    n_samples = X.shape[0]
    
    # Adjust range based on sample size
    k_min = max(2, k_min)  # Need at least 2 clusters for silhouette
    k_max = min(k_max, n_samples - 1)  # Can't have more clusters than samples
    
    if k_max <= k_min:
        return k_min
    
    best_k = k_min
    best_score = -1
    
    for k in range(k_min, k_max + 1):
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(X)
            
            # Silhouette requires at least 2 clusters with samples
            if len(set(labels)) < 2:
                continue
                
            score = silhouette_score(X, labels)
            logging.debug(f"Silhouette score for K={k}: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            logging.debug(f"Silhouette failed for K={k}: {e}")
            continue
    
    logging.debug(f"Optimal K={best_k} with silhouette score={best_score:.4f}")
    return best_k


def cluster_keywords(
    keywords: List[str], 
    max_clusters: int = 8, 
    random_state: int = 42,
    use_silhouette: bool = False,
    silhouette_k_range: Tuple[int, int] = (4, 15),
):
    """
    Cluster keywords into semantic groups.
    
    Args:
        keywords: List of keywords to cluster
        max_clusters: Maximum number of clusters (used if silhouette disabled)
        random_state: Random seed for reproducibility
        use_silhouette: If True, use silhouette score to find optimal K
        silhouette_k_range: (min_k, max_k) range for silhouette search
        
    Returns:
        Dict mapping cluster names to lists of keywords
    """
    if not keywords:
        return {}
    X = vectorize_keywords(keywords)
    if X.shape[0] < 6:
        # fallback: token overlap grouping by first token
        clusters = {}
        for kw in keywords:
            group = kw.split()[0]
            clusters.setdefault(group, []).append(kw)
        return clusters
    
    # Choose K using silhouette score or heuristic
    if use_silhouette and X.shape[0] >= silhouette_k_range[0]:
        k = choose_k_silhouette(
            X, 
            k_min=silhouette_k_range[0], 
            k_max=min(silhouette_k_range[1], max_clusters),
            random_state=random_state,
        )
    else:
        k = choose_k(len(keywords), max_clusters)
    
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    
    # Group keywords by cluster label
    raw_clusters = {}
    for kw, lbl in zip(keywords, labels):
        raw_clusters.setdefault(lbl, []).append(kw)
    
    # Generate descriptive names for each cluster
    clusters = {}
    for lbl, kws in raw_clusters.items():
        cluster_name = _generate_cluster_name(kws)
        # Handle name collisions by appending label
        if cluster_name in clusters:
            cluster_name = f"{cluster_name}-{lbl}"
        clusters[cluster_name] = kws
    
    return clusters


def _generate_cluster_name(keywords: List[str]) -> str:
    """
    Generate a descriptive name for a cluster based on most frequent n-grams.
    
    Finds the most common 1-gram or 2-gram among the cluster's keywords
    and uses it as the cluster name.
    
    Args:
        keywords: List of keywords in the cluster
        
    Returns:
        Descriptive cluster name (e.g., "cluster-villa", "cluster-renovation")
    """
    from collections import Counter
    
    if not keywords:
        return "cluster-misc"
    
    # Count all 1-grams and 2-grams
    ngram_counts: Counter = Counter()
    
    for kw in keywords:
        tokens = kw.lower().split()
        
        # Count 1-grams (single words)
        for token in tokens:
            # Skip very short tokens and common stopwords
            if len(token) > 2 and token not in {"the", "and", "for", "with", "how", "what", "best", "top"}:
                ngram_counts[token] += 1
        
        # Count 2-grams
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            ngram_counts[bigram] += 1
    
    if not ngram_counts:
        return "cluster-misc"
    
    # Get the most common n-gram
    most_common = ngram_counts.most_common(1)[0][0]
    
    # Clean up the name (replace spaces with hyphens)
    cluster_name = f"cluster-{most_common.replace(' ', '-')}"
    
    return cluster_name


def pick_pillar_per_cluster(clusters: Dict[str, List[str]], freq: Dict[str, int]) -> Dict[str, str]:
    pillars = {}
    for c, kws in clusters.items():
        # highest frequency as pillar
        sorted_kws = sorted(kws, key=lambda k: (-freq.get(k, 0), len(k)))
        if sorted_kws:
            pillars[c] = sorted_kws[0]
    return pillars


def extract_entities_from_keywords(keywords: List[str]) -> Dict[str, int]:
    """
    Extract entities (noun phrases/proper nouns) from keywords.
    
    Uses simple heuristics to identify entities:
    - Capitalized words (proper nouns)
    - Multi-word phrases that appear frequently
    - Domain-specific terms
    
    Args:
        keywords: List of keywords to analyze
        
    Returns:
        Dict mapping entity to frequency count
    """
    entities: Dict[str, int] = {}
    
    # Common stop words to exclude
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "just",
        "and", "but", "if", "or", "because", "until", "while", "about",
        "against", "what", "which", "who", "whom", "this", "that", "these",
        "those", "am", "its", "it", "best", "top", "vs", "review", "guide",
    }
    
    for kw in keywords:
        # Split into tokens and extract potential entities
        tokens = kw.lower().split()
        
        # Single-word entities (2+ chars, not stop words)
        for token in tokens:
            if len(token) >= 2 and token not in stop_words:
                entities[token] = entities.get(token, 0) + 1
        
        # Bigrams as potential compound entities
        if len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                if tokens[i] not in stop_words and tokens[i + 1] not in stop_words:
                    bigram = f"{tokens[i]} {tokens[i + 1]}"
                    entities[bigram] = entities.get(bigram, 0) + 1
    
    return entities


def check_entity_co_occurrence(
    clusters: Dict[str, List[str]],
    min_entity_density: float = 0.5,
    min_unique_entities: int = 3,
) -> Dict[str, Dict]:
    """
    Check entity co-occurrence within clusters to identify thin content risk.
    
    Clusters with low entity density or few unique entities are flagged
    as "Thin Content Risk" because they may lack topical depth.
    
    Args:
        clusters: Dict mapping cluster names to lists of keywords
        min_entity_density: Minimum ratio of unique entities to keywords (0.0-1.0)
        min_unique_entities: Minimum number of unique entities required
        
    Returns:
        Dict mapping cluster name to analysis results:
        {
            "entities": {entity: count},
            "entity_count": int,
            "keyword_count": int,
            "entity_density": float,
            "thin_content_risk": bool,
            "risk_reasons": List[str],
        }
    """
    results = {}
    
    for cluster_name, keywords in clusters.items():
        entities = extract_entities_from_keywords(keywords)
        
        # Filter to significant entities (appear in 2+ keywords or are bigrams)
        significant_entities = {
            e: c for e, c in entities.items() 
            if c >= 2 or " " in e
        }
        
        keyword_count = len(keywords)
        entity_count = len(significant_entities)
        entity_density = entity_count / keyword_count if keyword_count > 0 else 0.0
        
        # Determine thin content risk
        risk_reasons = []
        
        if entity_count < min_unique_entities:
            risk_reasons.append(
                f"Low entity diversity: {entity_count} entities (min: {min_unique_entities})"
            )
        
        if entity_density < min_entity_density:
            risk_reasons.append(
                f"Low entity density: {entity_density:.2f} (min: {min_entity_density})"
            )
        
        # Check for entity overlap (semantic coherence)
        if entity_count > 0:
            # Calculate average entity frequency
            avg_freq = sum(significant_entities.values()) / entity_count
            if avg_freq < 1.5 and keyword_count >= 5:
                risk_reasons.append(
                    f"Low entity coherence: entities appear in avg {avg_freq:.1f} keywords"
                )
        
        thin_content_risk = len(risk_reasons) > 0
        
        results[cluster_name] = {
            "entities": significant_entities,
            "entity_count": entity_count,
            "keyword_count": keyword_count,
            "entity_density": round(entity_density, 3),
            "thin_content_risk": thin_content_risk,
            "risk_reasons": risk_reasons,
        }
        
        if thin_content_risk:
            logging.debug(
                f"Thin Content Risk in {cluster_name}: {', '.join(risk_reasons)}"
            )
    
    return results


def get_cluster_entity_summary(
    clusters: Dict[str, List[str]],
    min_entity_density: float = 0.5,
    min_unique_entities: int = 3,
) -> Tuple[Dict[str, Dict], List[str], List[str]]:
    """
    Get entity analysis summary with lists of healthy and at-risk clusters.
    
    Args:
        clusters: Dict mapping cluster names to lists of keywords
        min_entity_density: Minimum ratio of unique entities to keywords
        min_unique_entities: Minimum number of unique entities required
        
    Returns:
        Tuple of:
        - Full analysis results dict
        - List of healthy cluster names
        - List of at-risk cluster names
    """
    analysis = check_entity_co_occurrence(
        clusters, 
        min_entity_density=min_entity_density,
        min_unique_entities=min_unique_entities,
    )
    
    healthy = [name for name, data in analysis.items() if not data["thin_content_risk"]]
    at_risk = [name for name, data in analysis.items() if data["thin_content_risk"]]
    
    return analysis, healthy, at_risk
