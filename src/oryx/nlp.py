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


# =============================================================================
# SpaCy Integration for Grammatical Validation (Week 2)
# =============================================================================
# Uses SpaCy for Part-of-Speech tagging and Named Entity Recognition
# to filter out grammatically invalid or nonsensical keywords.

try:
    import spacy
    try:
        _nlp_spacy = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        HAS_SPACY = True
    except OSError:
        # Model not installed
        HAS_SPACY = False
        _nlp_spacy = None
except ImportError:
    HAS_SPACY = False
    _nlp_spacy = None


# POS tags that should not start a keyword phrase
INVALID_START_POS = frozenset({
    "CC",    # Conjunction (and, or, but)
    "IN",    # Preposition (in, on, at, for, with) - unless idiom
    "DT",    # Determiner (the, a, an)
    "TO",    # "to" as particle
    "EX",    # Existential there
    "WDT",   # Wh-determiner (which, that)
    "WP",    # Wh-pronoun (who, what)
    "WP$",   # Possessive wh-pronoun
    "WRB",   # Wh-adverb (where, when, why, how) - OK at start actually
})

# Exceptions: These starting words are OK even if POS says otherwise
VALID_START_EXCEPTIONS = frozenset({
    "how", "what", "where", "when", "why", "which", "who",  # Question words
    "best", "top",  # Comparison words (NOT "vs" - it needs following content)
    "for",  # "for beginners", "for professionals"
})

# Words that CANNOT start a keyword - always grammatically invalid
# "vs villa construction" is broken - "vs" requires a preceding noun
INVALID_START_PREFIXES = frozenset({
    "vs", "versus", "or", "and", "but", "nor", "yet",
})

# Words that must be followed by meaningful content (not just geo codes)
REQUIRES_CONTENT_AFTER = frozenset({
    "vs", "versus", "or", "and",
})

# Maximum noun-only sequence length before rejection
MAX_NOUN_CLUSTER_SIZE = 3

# Minimum meaningful words (non-stopwords, non-geo) for phrases starting with certain words
MIN_MEANINGFUL_WORDS_FOR_VS = 2  # "vs" needs at least 2 meaningful words after it
GEO_TOKENS = frozenset({"uae", "dubai", "abu", "dhabi", "sharjah", "ae", "en", "ar", "ajman", "fujairah"})

# Common stopwords that don't count as meaningful
COMMON_STOPWORDS = frozenset({"the", "a", "an", "is", "are", "was", "were", "be", "been", "being"})

# =============================================================================
# Negative Sentiment Shield
# =============================================================================
# Terms that indicate negative/harmful content. Keywords containing these
# should be filtered to prevent ranking for complaints, scams, etc.
NEGATIVE_SENTIMENT_TERMS = frozenset({
    "scam", "scams", "scammer", "scammers",
    "fraud", "fraudulent", "frauds",
    "complaint", "complaints",
    "fake", "fakes",
    "illegal", "illegally",
    "fine", "fines", "fined",  # Legal penalties
    "violation", "violations",
    "lawsuit", "lawsuits", "sued",
    "scandal", "scandals",
    "ripoff", "rip-off", "rip off",
    "warning", "warnings",
    "avoid", "avoided",
    "worst", "terrible", "horrible",
    "failed", "failure", "failures",
    "bankrupt", "bankruptcy",
})


def contains_negative_sentiment(keyword: str) -> bool:
    """
    Check if keyword contains negative sentiment terms.
    
    These terms indicate the searcher may be looking for complaints,
    scams, or negative reviews - not ideal for a business to rank for.
    
    Args:
        keyword: The keyword to check
        
    Returns:
        True if the keyword contains negative terms
    """
    words = set(keyword.lower().split())
    return bool(words & NEGATIVE_SENTIMENT_TERMS)


def is_grammatically_valid(keyword: str) -> bool:
    """
    Check if a keyword phrase is grammatically valid.
    
    Uses SpaCy POS tagging to filter out nonsensical combinations:
    - Rejects keywords starting with conjunctions (and, or, but, vs)
    - Rejects keywords starting with prepositions (unless idiom)
    - Rejects pure noun clusters with >3 nouns (Franken-keywords)
    
    Args:
        keyword: The keyword phrase to validate
        
    Returns:
        True if grammatically valid, False otherwise
    """
    keyword = keyword.strip().lower()
    if not keyword or len(keyword.split()) < 2:
        return True  # Single words pass through
    
    first_word = keyword.split()[0]
    
    # =================================================================
    # Rule 0: Hard-reject invalid starting prefixes
    # =================================================================
    # "vs villa construction" is broken - requires preceding noun
    # "or villa construction" is broken - requires preceding clause
    if first_word in INVALID_START_PREFIXES:
        logging.debug(f"Grammar filter: '{keyword}' starts with invalid prefix '{first_word}'")
        return False
    
    # Check for valid start exceptions first
    if first_word in VALID_START_EXCEPTIONS:
        return True
    
    if not HAS_SPACY or not _nlp_spacy:
        # If SpaCy not available, allow all (graceful degradation)
        return True
    
    # =================================================================
    # Rule 1: Check for words that require meaningful content after
    # =================================================================
    # "vs managers uae" -> "vs" needs real content, not just 1 word + geo
    if first_word in REQUIRES_CONTENT_AFTER:
        words = keyword.split()
        meaningful_words = [w for w in words[1:] if w not in GEO_TOKENS and w not in COMMON_STOPWORDS and len(w) > 2]
        # "vs" needs at least 2 meaningful words to form a valid comparison
        if len(meaningful_words) < MIN_MEANINGFUL_WORDS_FOR_VS:
            logging.debug(f"Grammar filter: '{keyword}' - '{first_word}' needs {MIN_MEANINGFUL_WORDS_FOR_VS}+ meaningful words, got {len(meaningful_words)}")
            return False
    
    # Process with SpaCy
    try:
        doc = _nlp_spacy(keyword)
        
        if len(doc) == 0:
            return True
        
        # =================================================================
        # Rule 1: Check starting POS
        # =================================================================
        first_token = doc[0]
        if first_token.pos_ in {"CCONJ", "SCONJ"}:  # Conjunctions
            logging.debug(f"Grammar filter: '{keyword}' starts with conjunction")
            return False
        
        # Check for invalid starting tag (more granular)
        if first_token.tag_ in INVALID_START_POS and first_token.text not in VALID_START_EXCEPTIONS:
            # Allow "vs" and similar if they have meaningful content (checked above)
            if first_token.text not in REQUIRES_CONTENT_AFTER:
                logging.debug(f"Grammar filter: '{keyword}' starts with {first_token.tag_}")
                return False
        
        # =================================================================
        # Rule 2: Check for noun clusters (Franken-keywords)
        # =================================================================
        # Count nouns more aggressively - SpaCy sometimes mistags
        noun_count = 0
        noun_tags = {"NN", "NNS", "NNP", "NNPS"}
        
        for token in doc:
            # Count as noun if POS is NOUN/PROPN OR if tag is noun-like
            # This catches cases where SpaCy mistags (e.g., "facility" as VERB)
            if token.pos_ in {"NOUN", "PROPN"} or token.tag_ in noun_tags:
                noun_count += 1
            # Also count words that LOOK like nouns (lowercase, >3 chars, no special chars)
            elif token.text.islower() and len(token.text) > 3 and token.text.isalpha():
                # Check if it could be a noun (not clearly a verb/adj)
                if token.pos_ not in {"VERB", "ADJ", "ADV", "ADP", "DET", "PRON"}:
                    noun_count += 1
        
        # If phrase has >3 noun-like words and no clear structure, reject
        has_verb = any(t.pos_ == "VERB" and t.text not in {"be", "is", "are"} for t in doc)
        has_adj = any(t.pos_ == "ADJ" for t in doc)
        has_adp = any(t.pos_ == "ADP" and t.text in {"in", "for", "to", "of", "with"} for t in doc)
        
        # Pure noun cluster with >3 nouns and no modifiers = Franken-keyword
        if noun_count > MAX_NOUN_CLUSTER_SIZE and not (has_verb or has_adj or has_adp):
            # Exception: phrases starting with comparison words are OK
            if first_word not in {"vs", "versus", "best", "top", "compare"}:
                logging.debug(f"Grammar filter: '{keyword}' is a noun cluster ({noun_count} noun-like words)")
                return False
        
        # Also reject if ALL words are potentially nouns (no structure at all)
        # But only if phrase doesn't start with a known pattern word
        pattern_starters = {"vs", "versus", "best", "top", "compare", "how", "what", "where", "for"}
        total_content_words = sum(1 for t in doc if not t.is_space and not t.is_punct)
        if total_content_words > 3 and noun_count >= total_content_words - 1:
            # Almost all words are nouns - likely Franken-keyword
            if not (has_verb or has_adj or has_adp) and first_word not in pattern_starters:
                logging.debug(f"Grammar filter: '{keyword}' has {noun_count}/{total_content_words} noun-like words")
                return False
        
        return True
        
    except Exception as e:
        logging.debug(f"SpaCy error on '{keyword}': {e}")
        return True  # On error, allow the keyword


def has_conflicting_entities(keyword: str) -> bool:
    """
    Check if a keyword contains conflicting Named Entities.
    
    Rejects keywords that mix incompatible geographic entities:
    - "Dubai Abu Dhabi" (two cities in one phrase without connector)
    - "UAE Saudi Arabia" (two countries)
    
    Allows:
    - "Dubai to Abu Dhabi" (connector present)
    - "Dubai vs Abu Dhabi" (comparison)
    
    Args:
        keyword: The keyword phrase to check
        
    Returns:
        True if entities conflict, False otherwise
    """
    if not HAS_SPACY or not _nlp_spacy:
        return False
    
    # Connectors that make multi-entity phrases valid
    valid_connectors = {"to", "vs", "versus", "or", "and", "from"}
    
    # UAE city/emirate names
    uae_cities = {"dubai", "abu dhabi", "sharjah", "ajman", "fujairah", 
                  "ras al khaimah", "umm al quwain", "al ain"}
    
    keyword_lower = keyword.lower()
    
    # Quick check: how many UAE cities are mentioned?
    cities_found = [city for city in uae_cities if city in keyword_lower]
    
    if len(cities_found) > 1:
        # Check if a connector is present
        has_connector = any(f" {conn} " in keyword_lower for conn in valid_connectors)
        if not has_connector:
            logging.debug(f"Entity conflict: '{keyword}' has multiple cities without connector")
            return True
    
    return False


def filter_grammatically_invalid(keywords: List[str]) -> List[str]:
    """
    Filter out grammatically invalid keywords.
    
    Applies SpaCy-based grammar rules:
    - POS filtering (no starting conjunctions/prepositions)
    - Noun cluster detection (max 3 consecutive nouns)
    - Entity cohesion check (no conflicting geo entities)
    
    Args:
        keywords: List of keyword candidates
        
    Returns:
        Filtered list with invalid keywords removed
    """
    if not HAS_SPACY:
        logging.debug("SpaCy not available - skipping grammar validation")
        return keywords
    
    original_count = len(keywords)
    
    filtered = []
    for kw in keywords:
        if is_grammatically_valid(kw) and not has_conflicting_entities(kw):
            filtered.append(kw)
    
    removed_count = original_count - len(filtered)
    if removed_count > 0:
        logging.info(f"Grammar filter: Removed {removed_count} invalid keywords")
    
    return filtered


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
    "builder", "builders", "construction", "contracting",
    "renovation", "maintenance", "repair", "installation",
})

# Product-oriented terms that work with "buy" prefixes
PRODUCT_TERMS = frozenset({
    "equipment", "material", "materials", "tools", "supplies",
    "machine", "machines", "product", "products", "item", "items",
    "software", "hardware", "device", "devices",
})

# Prefixes that imply purchasing a product (not hiring a service)
BUY_PREFIXES = frozenset({
    "buy", "where to buy", "purchase", "order", "shop for",
    "cheap", "discount", "for sale", "price of",
})

# Prefixes that imply hiring a service
HIRE_PREFIXES = frozenset({
    "hire", "find", "get quotes", "quotes for", "cost of",
    "best", "top rated", "recommended",
})

# Local modifiers that shouldn't be duplicated
LOCAL_MODIFIERS = frozenset({
    "near me", "nearby", "local", "in my area",
})


class SeedType:
    """Classification of seed topics for semantic logic gates."""
    SERVICE = "service"
    PRODUCT = "product"
    UNKNOWN = "unknown"


def classify_seed_type(seed: str) -> str:
    """
    Classify a seed topic as Service, Product, or Unknown.
    
    This determines which keyword templates are appropriate:
    - SERVICE: Use "hire", "quotes", "cost of". Ban "buy", "cheap".
    - PRODUCT: Use "buy", "price", "reviews". 
    - UNKNOWN: Use general templates.
    
    Args:
        seed: The seed topic/keyword
        
    Returns:
        SeedType constant (SERVICE, PRODUCT, or UNKNOWN)
    """
    seed_lower = seed.lower()
    seed_tokens = set(seed_lower.split())
    
    # Check for service indicators
    if seed_tokens & SERVICE_TERMS:
        return SeedType.SERVICE
    
    # Check for product indicators
    if seed_tokens & PRODUCT_TERMS:
        return SeedType.PRODUCT
    
    # Check for service industry keywords
    service_keywords = {
        "contracting", "consulting", "design", "architecture", "engineering",
        "cleaning", "landscaping", "plumbing", "electrical", "painting",
        "flooring", "roofing", "hvac", "moving", "legal", "accounting",
    }
    if seed_tokens & service_keywords:
        return SeedType.SERVICE
    
    # Check for product keywords
    product_keywords = {
        "coffee", "beans", "furniture", "clothes", "electronics",
        "appliances", "parts", "accessories",
    }
    if seed_tokens & product_keywords:
        return SeedType.PRODUCT
    
    return SeedType.UNKNOWN


def fix_near_me_position(keyword: str) -> Optional[str]:
    """
    Fix 'near me' positioning - must be suffix only.
    
    'near me' at start or middle is grammatically wrong and indicates
    a scraping artifact or template error.
    
    Args:
        keyword: The keyword to check/fix
        
    Returns:
        Fixed keyword (near me moved to end) or None if unfixable
        
    Examples:
        "near me contractors" -> "contractors near me"
        "contractors near me" -> "contractors near me" (unchanged)
        "near me near me" -> None (garbage)
    """
    keyword_lower = keyword.lower().strip()
    
    # Check for multiple "near me" (garbage)
    if keyword_lower.count("near me") > 1:
        return None
    
    # If "near me" is not present, return as-is
    if "near me" not in keyword_lower:
        return keyword
    
    # If already at the end, it's fine
    if keyword_lower.endswith("near me"):
        return keyword
    
    # "near me" is at start or middle - fix it
    # Remove "near me" and add to end
    fixed = keyword_lower.replace("near me", "").strip()
    if not fixed or len(fixed.split()) < 1:
        return None  # Nothing left after removing "near me"
    
    return f"{fixed} near me"


def is_near_me_valid(keyword: str) -> bool:
    """
    Check if 'near me' usage is valid (suffix only).
    
    Args:
        keyword: The keyword to check
        
    Returns:
        True if valid, False if 'near me' is in wrong position
    """
    keyword_lower = keyword.lower().strip()
    
    # No "near me" = valid
    if "near me" not in keyword_lower:
        return True
    
    # Multiple "near me" = invalid
    if keyword_lower.count("near me") > 1:
        return False
    
    # Must be at the end
    return keyword_lower.endswith("near me")


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
            
            # =================================================================
            # RULE 4: Handle "near me" positioning
            # =================================================================
            # "near me" must be a SUFFIX, not a prefix
            # Transform "near me contractors" -> "contractors near me"
            if pref_lower == "near me":
                # Add as suffix instead of prefix
                q = f"{p} near me".strip()
            else:
                q = f"{pref} {p}".strip()
            
            if len(q.split()) >= 2:
                qs.append(q)
    
    # Post-process: filter out any keywords with invalid near-me positioning
    qs = [q for q in qs if is_near_me_valid(q)]
    
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
    
    Uses classify_seed_type to determine appropriate templates:
    - SERVICE seeds: "hire", "quotes", "cost of" (no "buy")
    - PRODUCT seeds: "buy", "price", "reviews"
    - UNKNOWN: General templates
    
    Args:
        seed: The seed topic/keyword
        audience: Optional target audience for additional expansions
        
    Returns:
        List of expanded keywords
    """
    s = clean_text(seed)
    aud = clean_text(audience or "")
    
    # Classify the seed type
    seed_type = classify_seed_type(seed)
    
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
        "{s} near me",  # "near me" as suffix is correct
        "beginner {s} guide",
        "advanced {s}",
    ]
    
    # Add type-specific patterns
    if seed_type == SeedType.SERVICE:
        # Service seeds: hire/find patterns, NO buy patterns
        patterns.extend([
            "hire {s}",
            "find {s}",
            "{s} quotes",
            "{s} cost",
            "{s} companies",
            "trusted {s}",
            "reliable {s}",
            "{s} reviews",
        ])
    elif seed_type == SeedType.PRODUCT:
        # Product seeds: buy/price patterns
        patterns.extend([
            "buy {s}",
            "where to buy {s}",
            "{s} price",
            "cheap {s}",
            "{s} for sale",
            "{s} reviews",
            "{s} comparison",
        ])
    else:
        # Unknown: add both but prioritize general patterns
        patterns.extend([
            "{s} reviews",
            "{s} comparison",
            "find {s}",
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
    
    # Filter scraping artifacts
    cands = filter_scraping_artifacts(cands)
    
    # ==========================================================================
    # Step 6: Grammatical Validation (Week 2 - Grammar Police)
    # ==========================================================================
    # Apply SpaCy-based grammar rules to filter nonsensical keywords
    cands = filter_grammatically_invalid(cands)
    
    # ==========================================================================
    # Step 7: Negative Sentiment Shield
    # ==========================================================================
    # Filter keywords containing terms like "scam", "fraud", "complaint"
    # to prevent ranking for negative/harmful searches
    cands_before = len(cands)
    cands = [c for c in cands if not contains_negative_sentiment(c)]
    filtered = cands_before - len(cands)
    if filtered > 0:
        logging.info(f"Negative sentiment filter: Removed {filtered} keywords")
    
    # ==========================================================================
    # Step 8: Near Me Position Fix (Week 3 - Semantic Logic Gates)
    # ==========================================================================
    # Fix or reject keywords with "near me" in wrong position
    fixed_cands = []
    for c in cands:
        if is_near_me_valid(c):
            fixed_cands.append(c)
        else:
            # Try to fix the position
            fixed = fix_near_me_position(c)
            if fixed:
                fixed_cands.append(fixed)
    
    return list(dict.fromkeys(fixed_cands))
