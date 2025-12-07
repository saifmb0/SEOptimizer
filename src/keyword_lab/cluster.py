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
        "sentence-transformers not installed. Install with: pip install keyword-lab[ml]. "
        "Falling back to TF-IDF vectorization."
    )

# Recommended models for different use cases
# multi-qa-MiniLM-L6-cos-v1: Optimized for question/answer retrieval (best for GEO/SGE)
# all-MiniLM-L6-v2: General purpose semantic similarity
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default intent rules (can be overridden via config)
DEFAULT_INTENT_RULES = {
    "informational": ["who", "what", "why", "how", "guide", "tutorial", "tips", "checklist", "template"],
    "commercial": ["best", "top", "review", "compare", "vs", "alternatives", "pricing"],
    "transactional": ["buy", "discount", "coupon", "deal", "near me"],
    "navigational": [],
}


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
    rules = intent_rules or DEFAULT_INTENT_RULES
    k = keyword.lower()
    tokens = set(k.split())
    for intent, words in rules.items():
        for w in words:
            if " " in w:
                if w in k:
                    return intent
            else:
                if w in tokens:
                    return intent
    # navigational if contains competitor brand tokens
    for c in competitors:
        token = c.split(".")[0].lower()
        if token and token in k:
            return "navigational"
    return "informational"


def vectorize_keywords(keywords: List[str], model_name: Optional[str] = None):
    """
    Vectorize keywords for clustering.
    
    Uses sentence-transformers if available, with multi-qa-MiniLM-L6-cos-v1
    as the default model (optimized for Q&A retrieval, ideal for GEO/SGE).
    Falls back to TF-IDF if sentence-transformers not installed.
    
    Args:
        keywords: List of keywords to vectorize
        model_name: Optional specific model name override
        
    Returns:
        numpy array of embeddings
    """
    if HAS_ST and keywords:
        # Use Q&A-optimized model for better GEO alignment
        model_to_use = model_name or DEFAULT_EMBEDDING_MODEL
        try:
            model = SentenceTransformer(model_to_use)
            X = model.encode(keywords, show_progress_bar=False, normalize_embeddings=True)
            logging.debug(f"Vectorized {len(keywords)} keywords with {model_to_use}")
            return np.array(X)
        except Exception as e:
            logging.debug(f"Failed to load {model_to_use}: {e}, trying fallback")
            try:
                model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)
                X = model.encode(keywords, show_progress_bar=False, normalize_embeddings=True)
                return np.array(X)
            except Exception:
                pass
    # Fallback to TF-IDF
    vec = TfidfVectorizer(stop_words="english")
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
    clusters = {}
    for kw, lbl in zip(keywords, labels):
        clusters.setdefault(f"cluster-{lbl}", []).append(kw)
    return clusters


def pick_pillar_per_cluster(clusters: Dict[str, List[str]], freq: Dict[str, int]) -> Dict[str, str]:
    pillars = {}
    for c, kws in clusters.items():
        # highest frequency as pillar
        sorted_kws = sorted(kws, key=lambda k: (-freq.get(k, 0), len(k)))
        if sorted_kws:
            pillars[c] = sorted_kws[0]
    return pillars
