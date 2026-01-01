"""
Linguistic validation module for ORYX.

Provides POS-based filtering to eliminate "garbage" keywords that lack
proper linguistic structure (e.g., "Monday Saturday", "step by step").

Uses spaCy for efficient POS tagging with disabled heavy pipes for speed.
"""

import logging
import re
from typing import Optional, Set

# Lazy load spacy to avoid import errors if not installed
_spacy = None
_nlp = None


def _get_spacy():
    """Lazy load spacy module."""
    global _spacy
    if _spacy is None:
        try:
            import spacy
            _spacy = spacy
        except ImportError:
            logging.warning(
                "spacy not installed. Install with: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            )
            _spacy = False
    return _spacy if _spacy else None


def _get_nlp():
    """Lazy load the spaCy NLP model (singleton)."""
    global _nlp
    if _nlp is None:
        spacy = _get_spacy()
        if spacy:
            try:
                # Load with disabled heavy pipes for speed
                _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
                logging.debug("Loaded spaCy en_core_web_sm model for linguistic validation")
            except OSError:
                logging.warning(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Download with: python -m spacy download en_core_web_sm"
                )
                _nlp = False
    return _nlp if _nlp else None


# =============================================================================
# Pattern Definitions for Garbage Detection
# =============================================================================

# Pattern 1: Temporal patterns (day names, months)
_TEMPORAL_PATTERN = re.compile(
    r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december)(\s+(monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|january|february|march|april|may|june|july|"
    r"august|september|october|november|december))*$",
    re.IGNORECASE
)

# Pattern 2: Navigational garbage phrases
_NAVIGATIONAL_GARBAGE = frozenset({
    "step by step",
    "read more",
    "click here",
    "learn more",
    "see more",
    "view more",
    "show more",
    "get started",
    "sign up",
    "log in",
    "login",
    "signup",
    "subscribe",
    "download now",
    "buy now",
    "add to cart",
    "checkout",
    "continue reading",
    "next page",
    "previous page",
})

# Pattern 3: Prepositions that shouldn't end phrases
_ENDING_PREPOSITIONS = frozenset({
    "of", "to", "for", "with", "in", "on", "at", "by", "from", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "without",
})


class LinguisticsValidator:
    """
    Singleton class for linguistic validation of keyword phrases.
    
    Uses spaCy POS tagging to identify garbage keywords that lack
    proper linguistic structure.
    
    Usage:
        validator = LinguisticsValidator()
        if validator.is_valid_phrase("construction cost"):
            # Valid keyword
        if not validator.is_valid_phrase("Monday Saturday"):
            # Garbage - reject
    """
    
    _instance: Optional["LinguisticsValidator"] = None
    
    def __new__(cls) -> "LinguisticsValidator":
        """Singleton pattern - only one instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the validator (only once due to singleton)."""
        if self._initialized:
            return
        self._initialized = True
        self._nlp = _get_nlp()
    
    def is_valid_phrase(self, text: str) -> bool:
        """
        Check if a phrase is linguistically valid.
        
        Returns False if the text:
        - Is purely stopwords, verbs, or adjectives
        - Lacks at least one NOUN or PROPN
        - Matches temporal patterns (day names, months)
        - Matches navigational garbage phrases
        - Ends with a preposition (fragment)
        
        Args:
            text: The keyword phrase to validate
            
        Returns:
            True if valid, False if garbage
        """
        if not text or not text.strip():
            return False
        
        text = text.strip().lower()
        
        # Pattern 1: Reject pure temporal patterns
        if _TEMPORAL_PATTERN.match(text):
            logging.debug(f"Rejected temporal pattern: '{text}'")
            return False
        
        # Pattern 2: Reject navigational garbage
        if text in _NAVIGATIONAL_GARBAGE:
            logging.debug(f"Rejected navigational garbage: '{text}'")
            return False
        
        # Pattern 3: Reject fragments ending with prepositions
        words = text.split()
        if words and words[-1] in _ENDING_PREPOSITIONS:
            logging.debug(f"Rejected fragment ending with preposition: '{text}'")
            return False
        
        # POS-based validation (requires spaCy)
        if self._nlp:
            return self._validate_pos(text)
        
        # Fallback: basic heuristic if spaCy not available
        return self._fallback_validate(text)
    
    def _validate_pos(self, text: str) -> bool:
        """
        Validate using spaCy POS tagging.
        
        Requires at least one NOUN or PROPN to be valid.
        Rejects phrases that are purely stopwords, verbs, or adjectives.
        """
        doc = self._nlp(text)
        
        pos_tags = {token.pos_ for token in doc}
        
        # Must contain at least one noun or proper noun
        has_noun = bool(pos_tags & {"NOUN", "PROPN"})
        
        if not has_noun:
            # Check if it's purely stopwords, verbs, or adjectives
            bad_only_tags = {"VERB", "ADJ", "ADV", "DET", "ADP", "CCONJ", "SCONJ", "PART", "INTJ", "PUNCT", "SPACE", "X"}
            
            # Get all tokens that aren't stopwords
            non_stop_tokens = [t for t in doc if not t.is_stop]
            
            if not non_stop_tokens:
                logging.debug(f"Rejected purely stopwords: '{text}'")
                return False
            
            non_stop_pos = {t.pos_ for t in non_stop_tokens}
            
            # If all non-stopword tokens are verbs/adjectives/etc., reject
            if non_stop_pos.issubset(bad_only_tags):
                logging.debug(f"Rejected bad POS pattern: '{text}' (tags: {non_stop_pos})")
                return False
            
            logging.debug(f"Rejected no NOUN/PROPN: '{text}' (tags: {pos_tags})")
            return False
        
        return True
    
    def _fallback_validate(self, text: str) -> bool:
        """
        Fallback validation when spaCy is not available.
        
        Uses simple heuristics to catch obvious garbage.
        """
        words = text.split()
        
        # Single word: allow it (can't do much without POS)
        if len(words) == 1:
            return True
        
        # Very short phrases of only common words: suspicious
        common_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                        "be", "been", "being", "have", "has", "had", "do", "does", "did",
                        "will", "would", "could", "should", "may", "might", "must", "can"}
        
        if all(w in common_words for w in words):
            logging.debug(f"Rejected all common words: '{text}'")
            return False
        
        return True
    
    def get_pos_tags(self, text: str) -> list:
        """
        Get POS tags for a phrase (for debugging).
        
        Args:
            text: The phrase to analyze
            
        Returns:
            List of (token, POS tag) tuples
        """
        if not self._nlp:
            return []
        
        doc = self._nlp(text)
        return [(token.text, token.pos_) for token in doc]


# Convenience function for direct usage
def is_valid_phrase(text: str) -> bool:
    """
    Check if a phrase is linguistically valid.
    
    Convenience wrapper around LinguisticsValidator.is_valid_phrase().
    
    Args:
        text: The keyword phrase to validate
        
    Returns:
        True if valid, False if garbage
    """
    validator = LinguisticsValidator()
    return validator.is_valid_phrase(text)
