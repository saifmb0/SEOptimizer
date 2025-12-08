import logging
import re
from collections import Counter
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Import multilingual stopwords
from .stopwords import get_stopwords, EN_STOPWORDS, AR_STOPWORDS


# Unicode-aware text cleaning regex
# Supports Arabic script, Latin characters, and other Unicode letters
# \w with re.UNICODE matches Unicode word characters (letters, digits, underscore)
CLEAN_RE = re.compile(r"[^\w\s]+", re.UNICODE)

# Regex to remove digits (optional, configurable)
DIGITS_RE = re.compile(r"\d+")

# =============================================================================
# ORYX Stop Pattern Filtering
# =============================================================================
# These patterns catch scraping artifacts like language toggles, navigation
# elements, and ISO codes that should never appear in keyword output.

# Pattern to detect keywords ending with ISO/navigation artifacts
# Matches: "keyword en", "keyword ae", "keyword ar", "keyword uae"
# Does NOT match: "contractors in uae", "companies in dubai"
ARTIFACT_SUFFIX_PATTERN = re.compile(
    r"^(?:.*\s)?(en|ae|ar|uae)$|"  # Ends with bare ISO code
    r"^(en|ae|ar|uae)\s",  # Starts with bare ISO code
    re.IGNORECASE
)

# Pattern to detect pure navigation/UI garbage
NAVIGATION_GARBAGE_PATTERN = re.compile(
    r"^(en|ae|ar|uae|login|sign in|sign up|register|menu|home|contact|about|"
    r"privacy policy|terms of service|cookie|cookies|subscribe|newsletter)$|"
    r"^(login|sign in|sign up|register)\s|"  # Starts with auth terms
    r"\s(login|sign in|sign up|register)$|"  # Ends with auth terms
    r"\b(en ae|ae en|ar en|uae en|en uae|ar ae)\b|"  # Language switcher combos
    r"^(vs|for|near me|best|how|what|where|when|why|which)\s+(en|ae|ar|uae)(\s|$)|"  # Prefix + ISO
    r"\s(en|ae|ar)\s(ae|en|ar)(\s|$)",  # Consecutive ISO codes
    re.IGNORECASE
)

# Minimum substantive words required (excluding stopwords and artifacts)
ARTIFACT_TOKENS = frozenset({"en", "ae", "ar", "uae", "vs", "for", "near", "me", "best", "how", "what", "where", "when", "why", "which"})


def is_scraping_artifact(keyword: str) -> bool:
    """
    Detect if a keyword is a scraping artifact that should be filtered out.
    
    Catches:
    - Language toggle artifacts: "en ae", "keyword ae", "ae en"
    - Navigation elements: "login", "sign up", "menu"
    - Franken-keywords: "vs en ae", "near me uae en"
    
    Args:
        keyword: The keyword to check
        
    Returns:
        True if the keyword is an artifact and should be filtered
    """
    kw = keyword.strip().lower()
    
    # Quick check for navigation garbage
    if NAVIGATION_GARBAGE_PATTERN.search(kw):
        return True
    
    # Check for artifact suffix (but allow "in uae", "in dubai" patterns)
    if ARTIFACT_SUFFIX_PATTERN.match(kw):
        # Allow if preceded by preposition (legitimate geo-modifier)
        words = kw.split()
        if len(words) >= 2 and words[-2] in {"in", "from", "to", "for", "of"}:
            return False
        return True
    
    # Check if keyword has enough substantive content
    words = kw.split()
    substantive_words = [w for w in words if w not in ARTIFACT_TOKENS and len(w) > 2]
    
    # Require at least 1 substantive word for 2-word phrases, 2 for longer
    min_required = 1 if len(words) <= 2 else 2
    if len(substantive_words) < min_required:
        return True
    
    return False


def filter_scraping_artifacts(keywords: List[str]) -> List[str]:
    """
    Filter out scraping artifacts from a list of keywords.
    
    Args:
        keywords: List of keyword candidates
        
    Returns:
        Filtered list with artifacts removed
    """
    original_count = len(keywords)
    filtered = [kw for kw in keywords if not is_scraping_artifact(kw)]
    removed_count = original_count - len(filtered)
    
    if removed_count > 0:
        logging.info(f"Filtered {removed_count} scraping artifacts from {original_count} keywords")
    
    return filtered


def clean_text(text: str, remove_digits: bool = False, language: str = "en") -> str:
    """
    Clean text for NLP processing with multilingual support.
    
    Args:
        text: Input text to clean
        remove_digits: If True, remove numeric characters
        language: Language code ('en', 'ar', etc.) - currently informational
        
    Returns:
        Cleaned text with non-word characters removed
    """
    # Lowercase (safe for Arabic - doesn't have case, so no-op)
    text = text.lower()
    
    # Remove non-word characters (preserves Arabic/Unicode letters)
    text = CLEAN_RE.sub(" ", text)
    
    # Optionally remove digits
    if remove_digits:
        text = DIGITS_RE.sub(" ", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def tokenize(text: str, language: str = "en") -> List[str]:
    """
    Tokenize text with language-aware stopword removal.
    
    Args:
        text: Text to tokenize
        language: Language code ('en', 'ar', 'ar-en' for bilingual)
        
    Returns:
        List of tokens with stopwords removed
    """
    stopwords = get_stopwords(language)
    return [t for t in text.split() if t and t not in stopwords]


def ngram_counts(texts: List[str], ngram_range=(2, 3), min_df: int = 2) -> pd.DataFrame:
    if not texts:
        return pd.DataFrame(columns=["ngram", "count"])    
    n_docs = len(texts)
    eff_min_df = min_df if n_docs >= min_df else max(1, n_docs)
    try:
        vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=eff_min_df, stop_words="english")
        X = vectorizer.fit_transform(texts)
    except ValueError:
        # Fallback for very small corpora
        vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=1, stop_words="english")
        X = vectorizer.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    ngrams = vectorizer.get_feature_names_out()
    df = pd.DataFrame({"ngram": ngrams, "count": counts})
    df = df.sort_values("count", ascending=False)
    return df


QUESTION_PREFIXES = [
    "how", "what", "best", "vs", "for", "near me", "beginner", "advanced", "guide", "checklist", "template", "why"
]

# Alias for external access (can be overridden via config)
DEFAULT_QUESTION_PREFIXES = QUESTION_PREFIXES.copy()


def generate_questions(phrases: Iterable[str], top_n: int = 50, prefixes: Optional[List[str]] = None) -> List[str]:
    """
    Generate question-style keywords from phrases.
    
    Args:
        phrases: Source phrases to expand
        top_n: Maximum number of phrases to process
        prefixes: Optional custom prefixes (from config), defaults to QUESTION_PREFIXES
        
    Returns:
        List of generated question keywords
    """
    question_prefixes = prefixes if prefixes is not None else QUESTION_PREFIXES
    qs = []
    for p in list(phrases)[:top_n]:
        if len(p.split()) < 2:
            continue
        for pref in question_prefixes:
            q = f"{pref} {p}".strip()
            if len(q.split()) >= 2:
                qs.append(q)
    return qs


def tfidf_top_terms_per_doc(texts: List[str], ngram_range=(2, 3), top_k: int = 10) -> List[str]:
    if not texts:
        return []
    vec = TfidfVectorizer(ngram_range=ngram_range, stop_words="english")
    X = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()
    tops: List[str] = []
    for i in range(X.shape[0]):
        row = X.getrow(i).toarray().ravel()
        idx = np.argsort(-row)[:top_k]
        tops.extend([terms[j] for j in idx if row[j] > 0])
    return list(dict.fromkeys(tops))


def seed_expansions(seed: str, audience: Optional[str] = None) -> List[str]:
    s = clean_text(seed)
    aud = clean_text(audience or "")
    patterns = [
        "how to {s}",
        "what is {s}",
        "best {s}",
        "top {s}",
        "{s} guide",
        "{s} tutorial",
        "{s} checklist",
        "{s} template",
        "{s} vs alternatives",
        "compare {s}",
        "{s} pricing",
        "buy {s}",
        "where to buy {s}",
        "{s} near me",
        "beginner {s} guide",
        "advanced {s}",
    ]
    if aud:
        patterns.extend([
            "best {s} for {a}",
            "{s} for {a}",
            "how to {s} for {a}",
        ])
    cands = []
    for p in patterns:
        q = p.format(s=s, a=aud).strip()
        if len(q.split()) >= 2:
            cands.append(q)
    return list(dict.fromkeys(cands))


def generate_candidates(
    docs: List[Dict], 
    ngram_min_df: int = 2, 
    top_terms_per_doc: int = 10,
    question_prefixes: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate keyword candidates from documents.
    
    Args:
        docs: List of document dicts with 'text' field
        ngram_min_df: Minimum document frequency for ngrams
        top_terms_per_doc: Number of top TF-IDF terms per document
        question_prefixes: Optional custom question prefixes (from config)
        
    Returns:
        List of candidate keywords (filtered for scraping artifacts)
    """
    cleaned_docs = [clean_text(d.get("text", "")) for d in docs]
    cleaned_docs = [t for t in cleaned_docs if t]
    counts_df = ngram_counts(cleaned_docs, min_df=ngram_min_df)
    ngram_list = counts_df["ngram"].tolist()
    questions = generate_questions(ngram_list, top_n=min(50, len(ngram_list)), prefixes=question_prefixes)
    tfidf_terms = tfidf_top_terms_per_doc(cleaned_docs, top_k=top_terms_per_doc)
    cands = list(dict.fromkeys([*ngram_list, *questions, *tfidf_terms]))
    cands = [c.strip().lower() for c in cands if len(c.split()) >= 2]
    
    # ORYX: Filter out scraping artifacts (language toggles, navigation, ISO codes)
    cands = filter_scraping_artifacts(cands)
    
    return cands
