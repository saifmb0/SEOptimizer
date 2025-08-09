import logging
from typing import List, Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    HAS_ST = False


INTENT_RULES = {
    "informational": ["who", "what", "why", "how", "guide", "tutorial", "tips", "checklist", "template"],
    "commercial": ["best", "top", "review", "compare", "vs", "alternatives", "pricing"],
    "transactional": ["buy", "discount", "coupon", "deal", "near me"],
    "navigational": [],
}


def infer_intent(keyword: str, competitors: List[str]) -> str:
    k = keyword.lower()
    tokens = set(k.split())
    for intent, words in INTENT_RULES.items():
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


def vectorize_keywords(keywords: List[str]):
    if HAS_ST and keywords:
        try:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            X = model.encode(keywords, show_progress_bar=False, normalize_embeddings=True)
            return np.array(X)
        except Exception:
            pass
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(keywords)
    return X.toarray()


def choose_k(n_keywords: int, max_clusters: int) -> int:
    if n_keywords < 12:
        return max(1, min(4, n_keywords))
    k = min(max_clusters, n_keywords // 10 + 1)
    k = max(6, min(12, k))
    return k


def cluster_keywords(keywords: List[str], max_clusters: int = 8, random_state: int = 42):
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
