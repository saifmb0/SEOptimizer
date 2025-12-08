import logging
import re
from collections import Counter
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Import multilingual stopwords
from .stopwords import get_stopwords, EN_STOPWORDS, AR_STOPWORDS

# =============================================================================
# Sentence Tokenization (Strict Boundary Detection)
# =============================================================================
# Use NLTK's sentence tokenizer to prevent merging of unrelated content.
# Falls back to simple splitting if NLTK punkt is unavailable.

try:
    import nltk
    # Try to use punkt tokenizer
    try:
        nltk.data.find('tokenizers/punkt')
        HAS_NLTK_PUNKT = True
    except LookupError:
        # Try to download punkt
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            HAS_NLTK_PUNKT = True
        except Exception:
            HAS_NLTK_PUNKT = False
    
    if HAS_NLTK_PUNKT:
        from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
except ImportError:
    HAS_NLTK_PUNKT = False
    _nltk_sent_tokenize = None


def sent_tokenize(text: str, language: str = "en") -> List[str]:
    """
    Tokenize text into sentences with strict boundary detection.
    
    Uses NLTK's punkt tokenizer for proper sentence boundary detection.
    This ensures "Contact Us" and "Privacy Policy" never merge with
    content paragraphs.
    
    Falls back to newline + punctuation splitting if NLTK is unavailable.
    
    Args:
        text: Text to segment into sentences
        language: Language code ('en', 'ar', etc.)
        
    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []
    
    # First, respect explicit newlines as hard boundaries
    # (inserted by scrape.py between block elements)
    lines = text.split('\n')
    
    sentences = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Use NLTK for sentence tokenization within lines
        if HAS_NLTK_PUNKT and _nltk_sent_tokenize:
            try:
                # Map language codes to NLTK language names
                nltk_lang_map = {
                    'en': 'english',
                    'ar': 'english',  # Arabic uses English tokenizer (best available)
                    'de': 'german',
                    'es': 'spanish',
                    'fr': 'french',
                }
                nltk_lang = nltk_lang_map.get(language, 'english')
                line_sents = _nltk_sent_tokenize(line, language=nltk_lang)
                sentences.extend(line_sents)
            except Exception:
                # Fallback on any error
                sentences.append(line)
        else:
            # Fallback: split on sentence-ending punctuation
            # This is less accurate but better than nothing
            fallback_sents = re.split(r'(?<=[.!?])\s+', line)
            sentences.extend([s.strip() for s in fallback_sents if s.strip()])
    
    return sentences


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

# =============================================================================
# Semantic Compatibility Rules for Question Generation
# =============================================================================
# These rules prevent logically nonsensical combinations like:
# - "where to buy contracting company" (you hire, not buy companies)
# - "near me near me warehouse" (duplicate modifiers)

# Service-oriented terms that don't work with "buy" prefixes
SERVICE_TERMS = frozenset({
    "company", "companies", "contractor", "contractors", 
    "service", "services", "agency", "agencies",
    "firm", "firms", "consultant", "consultants",
    "provider", "providers", "specialist", "specialists",
})

# Prefixes that imply purchasing a product (not hiring a service)
BUY_PREFIXES = frozenset({
    "buy", "where to buy", "purchase", "order", "shop for",
})

# Local modifiers that shouldn't be duplicated
LOCAL_MODIFIERS = frozenset({
    "near me", "nearby", "local", "in my area",
})


def generate_questions(phrases: Iterable[str], top_n: int = 50, prefixes: Optional[List[str]] = None) -> List[str]:
    """
    Generate question-style keywords from phrases with semantic logic gates.
    
    Applies compatibility rules to prevent nonsensical combinations:
    - No "buy" prefixes for service companies (you hire, not buy)
    - No duplicate local modifiers ("near me near me")
    - No redundant question prefixes
    
    Args:
        phrases: Source phrases to expand
        top_n: Maximum number of phrases to process
        prefixes: Optional custom prefixes (from config), defaults to QUESTION_PREFIXES
        
    Returns:
        List of generated question keywords (semantically valid only)
    """
    question_prefixes = prefixes if prefixes is not None else QUESTION_PREFIXES
    qs = []
    
    for p in list(phrases)[:top_n]:
        if len(p.split()) < 2:
            continue
        
        p_lower = p.lower()
        p_tokens = set(p_lower.split())
        
        # Detect if phrase is about services (not products)
        is_service_term = bool(p_tokens & SERVICE_TERMS)
        
        # Detect if phrase already has a local modifier
        has_local_modifier = any(mod in p_lower for mod in LOCAL_MODIFIERS)
        
        for pref in question_prefixes:
            pref_lower = pref.lower()
            
            # =================================================================
            # RULE 1: Skip "buy" prefixes for service companies
            # =================================================================
            # "where to buy contracting company" is nonsensical
            # Users HIRE contractors, they don't BUY the company
            if pref_lower in BUY_PREFIXES and is_service_term:
                continue
            
            # =================================================================
            # RULE 2: Skip local modifiers if phrase already has one
            # =================================================================
            # Prevents "near me warehouse near me" duplication
            if pref_lower in LOCAL_MODIFIERS and has_local_modifier:
                continue
            
            # =================================================================
            # RULE 3: Skip if prefix is already in the phrase
            # =================================================================
            # Prevents "best best contractors" or "how to how to"
            if pref_lower in p_lower:
                continue
            
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
    """
    Generate seed-based keyword expansions with semantic logic gates.
    
    Applies same compatibility rules as generate_questions:
    - No "buy" patterns for service-oriented seeds
    - Contextually appropriate expansions only
    
    Args:
        seed: The seed topic/keyword
        audience: Optional target audience for additional expansions
        
    Returns:
        List of expanded keywords
    """
    s = clean_text(seed)
    aud = clean_text(audience or "")
    s_lower = s.lower()
    s_tokens = set(s_lower.split())
    
    # Detect if seed is about services (not products)
    is_service_seed = bool(s_tokens & SERVICE_TERMS)
    
    # Base patterns that work for all seeds
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
        "{s} near me",
        "beginner {s} guide",
        "advanced {s}",
    ]
    
    # Only add "buy" patterns for product-oriented seeds
    if not is_service_seed:
        patterns.extend([
            "buy {s}",
            "where to buy {s}",
        ])
    else:
        # For service seeds, add "hire" patterns instead
        patterns.extend([
            "hire {s}",
            "find {s}",
            "{s} quotes",
            "{s} cost",
        ])
    
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
    language: str = "en",
) -> List[str]:
    """
    Generate keyword candidates with strict sentence boundary detection.
    
    Uses NLTK sentence tokenization to ensure proper boundaries:
    - "Contact Us" never merges with content paragraphs
    - Footer links stay isolated from main content
    - Navigation elements don't contaminate keywords
    
    The key insight: n-grams should NEVER cross sentence/line boundaries.
    
    Args:
        docs: List of document dicts with 'text' field
        ngram_min_df: Minimum document frequency for ngrams
        top_terms_per_doc: Number of top TF-IDF terms per document
        question_prefixes: Optional custom question prefixes (from config)
        language: Language code for sentence tokenization
        
    Returns:
        List of candidate keywords (filtered for scraping artifacts)
    """
    # ==========================================================================
    # Step 1: Sentence Tokenization with Strict Boundaries
    # ==========================================================================
    # Use NLTK sent_tokenize for proper sentence boundary detection.
    # This prevents merging of navigation elements with content.
    
    all_sentences = []
    for d in docs:
        raw_text = d.get("text", "")
        
        # Use sentence tokenizer (respects newlines as hard boundaries)
        sentences = sent_tokenize(raw_text, language=language)
        
        for sentence in sentences:
            # Skip short navigation noise (e.g., "Home", "About", "Login")
            # Require at least 3 words to be a meaningful content line
            if len(sentence.split()) < 3:
                continue
            cleaned = clean_text(sentence)
            if cleaned and len(cleaned.split()) >= 2:
                all_sentences.append(cleaned)
    
    # ==========================================================================
    # Step 2: Pass sentences (not whole docs) to vectorizer
    # ==========================================================================
    # CountVectorizer now sees each sentence as a separate "document"
    # This prevents n-grams like "owners developers facility" from forming
    # when those words came from different sections
    
    counts_df = ngram_counts(all_sentences, min_df=ngram_min_df)
    ngram_list = counts_df["ngram"].tolist()
    
    # ==========================================================================
    # Step 3: Smart Question Generation
    # ==========================================================================
    # Filter artifacts BEFORE generating questions to prevent
    # "where to buy en ae" type nonsense
    
    questions = []
    if ngram_list:
        clean_ngrams = filter_scraping_artifacts(ngram_list)
        questions = generate_questions(
            clean_ngrams, 
            top_n=min(50, len(clean_ngrams)), 
            prefixes=question_prefixes
        )
    
    # ==========================================================================
    # Step 4: TF-IDF on reconstructed docs with sentence boundaries
    # ==========================================================================
    # Reassemble cleaned sentences per doc for TF-IDF context
    # This preserves document-level importance while respecting boundaries
    
    reconstructed_docs = []
    for d in docs:
        raw_text = d.get("text", "")
        sentences = sent_tokenize(raw_text, language=language)
        clean_sents = [clean_text(s) for s in sentences if len(s.split()) >= 3]
        if clean_sents:
            reconstructed_docs.append(" ".join(clean_sents))
    
    tfidf_terms = tfidf_top_terms_per_doc(reconstructed_docs, top_k=top_terms_per_doc)
    
    # ==========================================================================
    # Step 5: Combine and filter
    # ==========================================================================
    cands = list(dict.fromkeys([*ngram_list, *questions, *tfidf_terms]))
    cands = [c.strip().lower() for c in cands if len(c.split()) >= 2]
    
    # Final artifact filter
    cands = filter_scraping_artifacts(cands)
    
    return cands
