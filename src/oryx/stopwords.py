"""
Multilingual Stopwords Module for ORYX.

Provides stopword lists for multiple languages to support
internationalized keyword extraction and NLP processing.

Supported languages:
- English (en)
- Arabic (ar)
- Bilingual Arabic-English (ar-en) for UAE market
"""

from typing import Set, Optional


# =============================================================================
# English Stopwords
# Based on NLTK's English stopwords list
# =============================================================================
EN_STOPWORDS: Set[str] = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
    "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn",
    "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
}


# =============================================================================
# Arabic Stopwords
# Comprehensive list for Modern Standard Arabic and Gulf Arabic variants
# Includes common particles, prepositions, pronouns, and conjunctions
# =============================================================================
AR_STOPWORDS: Set[str] = {
    # Definite article
    "ال",
    
    # Pronouns (personal)
    "أنا", "انا", "نحن", "أنت", "انت", "أنتم", "انتم", "أنتن", "انتن",
    "هو", "هي", "هم", "هن", "هما",
    
    # Demonstratives
    "هذا", "هذه", "هذان", "هاتان", "هؤلاء", "ذلك", "تلك", "ذاك",
    "أولئك", "اولئك",
    
    # Relative pronouns
    "الذي", "التي", "الذين", "اللذين", "اللواتي", "اللاتي", "ما", "من",
    
    # Prepositions
    "في", "على", "إلى", "الى", "من", "عن", "مع", "ب", "ل", "ك",
    "بين", "فوق", "تحت", "أمام", "امام", "خلف", "حول", "عند", "لدى",
    "منذ", "خلال", "ضد", "نحو", "حتى",
    
    # Conjunctions
    "و", "أو", "او", "ثم", "لكن", "لأن", "لان", "إذا", "اذا",
    "إن", "ان", "أن", "لو", "بل", "حيث", "كي", "لما", "إذ", "اذ",
    "ف", "فإن", "فان",
    
    # Adverbs
    "أين", "اين", "كيف", "متى", "لماذا", "ماذا", "أي", "اي",
    "هنا", "هناك", "الآن", "الان", "قد", "سوف", "لن", "لم", "ليس",
    "كان", "كانت", "كانوا", "يكون", "تكون", "أصبح", "اصبح",
    
    # Common verbs (auxiliary/modal)
    "هل", "قال", "قالت", "يقول", "تقول", "كل", "بعض", "جميع",
    "كثير", "قليل", "أكثر", "اكثر", "أقل", "اقل",
    
    # Quantifiers and numbers
    "واحد", "اثنان", "ثلاثة", "أربعة", "اربعة", "خمسة",
    "عدة", "بضع", "معظم", "أغلب", "اغلب",
    
    # Time-related
    "اليوم", "أمس", "امس", "غدا", "غداً", "الأمس", "الامس",
    "صباح", "مساء", "ليل", "نهار",
    
    # Other common function words
    "ذات", "ذو", "ذي", "غير", "سوى", "مثل", "نفس", "عين",
    "أحد", "احد", "آخر", "اخر", "أول", "اول", "آخرون", "اخرون",
    "كلا", "كلتا", "أيضا", "ايضا", "أيضاً", "فقط", "حتما", "ربما",
    "إنما", "انما", "كأن", "كأنما", "لعل", "ليت",
    
    # Gulf Arabic common additions
    "شو", "ليش", "وين", "كيفك", "هاي", "هاد", "هادي",
}


# =============================================================================
# UAE-Specific Terms (NOT stopwords - for reference)
# These are important for the contracting/construction niche
# =============================================================================
UAE_IMPORTANT_TERMS: Set[str] = {
    # Government entities
    "بلدية", "دبي", "أبوظبي", "ابوظبي", "الشارقة", "عجمان", "رأس الخيمة",
    "الفجيرة", "أم القيوين",
    
    # Construction terms
    "مقاولات", "بناء", "تشطيب", "ترميم", "صيانة", "تصميم",
    "مساطحة", "عقد", "رخصة", "تصريح",
    
    # Real estate
    "فيلا", "شقة", "برج", "مبنى", "عمارة", "أرض",
}


def get_stopwords(language: str = "en") -> Set[str]:
    """
    Get stopwords for a specific language.
    
    Args:
        language: Language code:
            - 'en': English only
            - 'ar': Arabic only  
            - 'ar-en' or 'bilingual': Combined Arabic and English
            - 'ae': UAE market (bilingual + common transliterations)
            
    Returns:
        Set of stopwords for the specified language
    """
    language = language.lower().strip()
    
    if language == "en":
        return EN_STOPWORDS.copy()
    
    elif language == "ar":
        return AR_STOPWORDS.copy()
    
    elif language in ("ar-en", "bilingual", "ae", "uae"):
        # Combined for bilingual UAE market
        combined = EN_STOPWORDS | AR_STOPWORDS
        # Add common transliterations used in UAE English
        combined.update({
            "yani", "yalla", "habibi", "inshallah", "mashallah",
            "khalas", "mafi", "shway",
        })
        return combined
    
    else:
        # Default to English for unknown languages
        return EN_STOPWORDS.copy()


def get_stopwords_for_vectorizer(language: str = "en") -> list:
    """
    Get stopwords as a list for sklearn vectorizers.
    
    Args:
        language: Language code
        
    Returns:
        List of stopwords suitable for sklearn vectorizers
    """
    return list(get_stopwords(language))
