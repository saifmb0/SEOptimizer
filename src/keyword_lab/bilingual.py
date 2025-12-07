"""
Bilingual Arabic-English processing for UAE/Gulf markets.

Provides Arabic-English translation mappings, transliteration,
and bilingual keyword expansion for the Abu Dhabi contracting sector.

ORYX Edition: Focused on Gulf Arabic dialect and construction terminology.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import re


# =============================================================================
# Arabic-English Construction/Contracting Terms
# =============================================================================

# Core construction terms (English -> Arabic)
CONSTRUCTION_TERMS_AR = {
    # Building types
    "villa": "فيلا",
    "apartment": "شقة",
    "building": "مبنى",
    "tower": "برج",
    "warehouse": "مستودع",
    "factory": "مصنع",
    "office": "مكتب",
    "shop": "محل",
    "showroom": "صالة عرض",
    "mall": "مول",
    
    # Rooms and spaces
    "kitchen": "مطبخ",
    "bathroom": "حمام",
    "bedroom": "غرفة نوم",
    "living room": "غرفة معيشة",
    "majlis": "مجلس",
    "dining room": "غرفة طعام",
    "balcony": "شرفة",
    "terrace": "تراس",
    "garage": "كراج",
    "parking": "موقف سيارات",
    "garden": "حديقة",
    "pool": "مسبح",
    "swimming pool": "مسبح",
    
    # Construction services
    "renovation": "تجديد",
    "construction": "بناء",
    "maintenance": "صيانة",
    "repair": "إصلاح",
    "installation": "تركيب",
    "fit out": "تشطيب",
    "fitout": "تشطيب",
    "finishing": "تشطيبات",
    "decoration": "ديكور",
    "design": "تصميم",
    "interior design": "تصميم داخلي",
    "landscaping": "تنسيق حدائق",
    "extension": "توسعة",
    "addition": "ملحق",
    
    # Trades
    "painting": "دهان",
    "plumbing": "سباكة",
    "electrical": "كهرباء",
    "carpentry": "نجارة",
    "flooring": "أرضيات",
    "tiling": "بلاط",
    "gypsum": "جبس",
    "false ceiling": "سقف مستعار",
    "air conditioning": "تكييف",
    "hvac": "تكييف وتبريد",
    "aluminium": "ألمنيوم",
    "glass": "زجاج",
    "doors": "أبواب",
    "windows": "نوافذ",
    "curtains": "ستائر",
    "blinds": "ستائر",
    "furniture": "أثاث",
    "lighting": "إضاءة",
    "waterproofing": "عزل مائي",
    "insulation": "عزل",
    
    # Business terms
    "contractor": "مقاول",
    "company": "شركة",
    "engineer": "مهندس",
    "technician": "فني",
    "worker": "عامل",
    "quotation": "عرض سعر",
    "estimate": "تقدير",
    "price": "سعر",
    "cost": "تكلفة",
    "cheap": "رخيص",
    "affordable": "بأسعار معقولة",
    "best": "أفضل",
    "quality": "جودة",
    "professional": "محترف",
    "licensed": "مرخص",
    "approved": "معتمد",
    
    # Permits and legal
    "permit": "تصريح",
    "license": "رخصة",
    "approval": "موافقة",
    "noc": "شهادة عدم ممانعة",
    "contract": "عقد",
    "agreement": "اتفاقية",
}

# Reverse mapping (Arabic -> English)
CONSTRUCTION_TERMS_EN = {v: k for k, v in CONSTRUCTION_TERMS_AR.items()}

# Gulf Arabic dialect variations (common in UAE/Gulf spoken Arabic)
GULF_DIALECT_VARIANTS = {
    # Standard Arabic -> Gulf Arabic pronunciation/spelling
    "شقة": ["شقه"],
    "غرفة": ["غرفه", "اوضة"],
    "تجديد": ["تجديد", "رينوفيشن"],
    "صيانة": ["صيانه", "مينتننس"],
    "تشطيب": ["تشطيب", "فت اوت"],
    "مقاول": ["مقاول", "كونتراكتر"],
    "دهان": ["دهان", "صبغ", "بوية"],
    "سباكة": ["سباكه", "بلمبنق"],
    "كهرباء": ["كهربا", "كهربه"],
    "تكييف": ["تكييف", "اي سي", "ايه سي"],
}

# Common Arabic transliterations (Arabizi/Franco-Arab)
ARABIZI_MAPPINGS = {
    # Numbers used as Arabic letters
    "3": "ع",  # 'ayn
    "7": "ح",  # ha
    "5": "خ",  # kha
    "2": "ء",  # hamza/alef
    "6": "ط",  # ta
    "9": "ص",  # sad
    "8": "ق",  # qaf
    
    # Common transliterations
    "villa": ["فيلا", "vila", "veela"],
    "majlis": ["مجلس", "majles"],
    "tashteeb": ["تشطيب"],
    "tajdeed": ["تجديد"],
    "seyaana": ["صيانة", "siana", "seyana"],
    "mokawel": ["مقاول", "muqawil"],
    "mhandis": ["مهندس", "mohandis", "muhandis"],
}


# =============================================================================
# Location Terms (Arabic)
# =============================================================================

LOCATION_TERMS_AR = {
    # Emirates
    "abu dhabi": "أبوظبي",
    "dubai": "دبي",
    "sharjah": "الشارقة",
    "ajman": "عجمان",
    "ras al khaimah": "رأس الخيمة",
    "fujairah": "الفجيرة",
    "umm al quwain": "أم القيوين",
    "al ain": "العين",
    
    # Common location words
    "area": "منطقة",
    "district": "حي",
    "island": "جزيرة",
    "industrial": "صناعية",
    "commercial": "تجارية",
    "residential": "سكنية",
    "near": "قريب من",
    "in": "في",
}


# =============================================================================
# Intent Keywords (Arabic)
# =============================================================================

INTENT_KEYWORDS_AR = {
    # Informational intent
    "informational": [
        "ما هو", "ما هي", "كيف", "لماذا", "متى", "أين",
        "معلومات عن", "دليل", "شرح", "تعريف",
    ],
    # Commercial intent
    "commercial": [
        "أفضل", "أحسن", "مقارنة", "الفرق بين", "أنواع",
        "مراجعة", "تقييم", "أسعار",
    ],
    # Transactional intent
    "transactional": [
        "شراء", "طلب", "حجز", "اتصل", "رقم", "هاتف",
        "عرض سعر", "تواصل", "احجز", "اطلب",
    ],
    # Local intent
    "local": [
        "قريب مني", "في", "بـ", "منطقة",
    ],
}


# =============================================================================
# Bilingual Processing Classes
# =============================================================================

@dataclass
class BilingualKeyword:
    """Represents a keyword with both English and Arabic variants."""
    english: str
    arabic: Optional[str] = None
    transliteration: Optional[str] = None
    dialect_variants: List[str] = field(default_factory=list)
    is_location: bool = False
    is_construction: bool = False


def get_arabic_equivalent(english_term: str) -> Optional[str]:
    """
    Get Arabic equivalent for an English construction term.
    
    Args:
        english_term: English term to translate
        
    Returns:
        Arabic translation or None if not found
    """
    return CONSTRUCTION_TERMS_AR.get(english_term.lower())


def get_english_equivalent(arabic_term: str) -> Optional[str]:
    """
    Get English equivalent for an Arabic construction term.
    
    Args:
        arabic_term: Arabic term to translate
        
    Returns:
        English translation or None if not found
    """
    return CONSTRUCTION_TERMS_EN.get(arabic_term)


def detect_arabic(text: str) -> bool:
    """
    Detect if text contains Arabic characters.
    
    Args:
        text: Text to analyze
        
    Returns:
        True if text contains Arabic characters
    """
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
    return bool(arabic_pattern.search(text))


def detect_arabizi(text: str) -> bool:
    """
    Detect if text appears to be Arabizi (Arabic written in Latin script).
    
    Common patterns: numbers like 3, 7, 5, 2 used as Arabic letters
    
    Args:
        text: Text to analyze
        
    Returns:
        True if text appears to be Arabizi
    """
    # Check for characteristic Arabizi number usage
    arabizi_pattern = re.compile(r'\b\w*[37529]\w*\b', re.IGNORECASE)
    return bool(arabizi_pattern.search(text))


def expand_bilingual(keyword: str) -> List[str]:
    """
    Expand a keyword with bilingual variants.
    
    Args:
        keyword: English keyword to expand
        
    Returns:
        List of variants including Arabic translations
    """
    variants = [keyword]
    kw_lower = keyword.lower()
    
    # Check for construction terms
    for eng_term, ar_term in CONSTRUCTION_TERMS_AR.items():
        if eng_term in kw_lower:
            # Add Arabic variant
            ar_variant = kw_lower.replace(eng_term, ar_term)
            variants.append(ar_variant)
            
            # Add Gulf dialect variants
            if ar_term in GULF_DIALECT_VARIANTS:
                for dialect in GULF_DIALECT_VARIANTS[ar_term]:
                    dialect_variant = kw_lower.replace(eng_term, dialect)
                    variants.append(dialect_variant)
            break
    
    # Check for location terms
    for eng_loc, ar_loc in LOCATION_TERMS_AR.items():
        if eng_loc in kw_lower:
            ar_variant = kw_lower.replace(eng_loc, ar_loc)
            variants.append(ar_variant)
    
    return list(set(variants))


def classify_arabic_intent(arabic_text: str) -> Optional[str]:
    """
    Classify intent from Arabic text.
    
    Args:
        arabic_text: Arabic text to analyze
        
    Returns:
        Intent classification or None
    """
    for intent, keywords in INTENT_KEYWORDS_AR.items():
        for kw in keywords:
            if kw in arabic_text:
                return intent
    return None


def normalize_gulf_arabic(text: str) -> str:
    """
    Normalize Gulf Arabic text for consistent matching.
    
    Handles common variations:
    - ة vs ه at end of words
    - ي vs ى at end of words
    - Hamza variations
    - Tashkeel (diacritics) removal
    
    Args:
        text: Arabic text to normalize
        
    Returns:
        Normalized text
    """
    # Remove tashkeel (diacritics)
    tashkeel = re.compile(r'[\u064B-\u0652\u0670]')
    text = tashkeel.sub('', text)
    
    # Normalize ta marbuta (ة -> ه at end)
    text = re.sub(r'ة\b', 'ه', text)
    
    # Normalize alef variations
    text = re.sub(r'[إأآا]', 'ا', text)
    
    # Normalize ya variations
    text = re.sub(r'ى\b', 'ي', text)
    
    return text


def generate_bilingual_keywords(
    base_keyword: str,
    include_arabic: bool = True,
    include_dialects: bool = False,
    include_locations: bool = True,
) -> List[BilingualKeyword]:
    """
    Generate comprehensive bilingual keyword variants.
    
    Args:
        base_keyword: Base English keyword
        include_arabic: Include Modern Standard Arabic
        include_dialects: Include Gulf dialect variants
        include_locations: Include location modifiers
        
    Returns:
        List of BilingualKeyword objects
    """
    results = []
    kw_lower = base_keyword.lower()
    
    # Start with base keyword
    base = BilingualKeyword(english=base_keyword)
    
    # Check if it's a construction term
    for eng_term, ar_term in CONSTRUCTION_TERMS_AR.items():
        if eng_term in kw_lower:
            base.arabic = kw_lower.replace(eng_term, ar_term)
            base.is_construction = True
            
            if include_dialects and ar_term in GULF_DIALECT_VARIANTS:
                base.dialect_variants = GULF_DIALECT_VARIANTS[ar_term]
            break
    
    results.append(base)
    
    # Generate location variants
    if include_locations:
        key_locations = ["abu dhabi", "dubai", "sharjah", "al ain"]
        for loc in key_locations:
            loc_kw = BilingualKeyword(
                english=f"{base_keyword} {loc}",
                is_location=True,
                is_construction=base.is_construction,
            )
            if include_arabic and loc in LOCATION_TERMS_AR:
                loc_kw.arabic = f"{base.arabic or base_keyword} {LOCATION_TERMS_AR[loc]}"
            results.append(loc_kw)
    
    return results


# =============================================================================
# Bilingual Search Query Analysis
# =============================================================================

def analyze_bilingual_query(query: str) -> Dict:
    """
    Analyze a search query for bilingual content.
    
    Args:
        query: Search query to analyze
        
    Returns:
        Analysis results including language, intent, and suggestions
    """
    is_arabic = detect_arabic(query)
    is_arabizi = detect_arabizi(query) if not is_arabic else False
    
    result = {
        "original": query,
        "primary_language": "arabic" if is_arabic else "english",
        "is_arabizi": is_arabizi,
        "arabic_intent": None,
        "english_equivalent": None,
        "arabic_equivalent": None,
        "construction_terms": [],
        "location_terms": [],
    }
    
    if is_arabic:
        # Analyze Arabic query
        result["arabic_intent"] = classify_arabic_intent(query)
        
        # Try to find English equivalent
        normalized = normalize_gulf_arabic(query)
        for ar, en in CONSTRUCTION_TERMS_EN.items():
            if ar in normalized:
                result["construction_terms"].append({"arabic": ar, "english": en})
                if not result["english_equivalent"]:
                    result["english_equivalent"] = query.replace(ar, en)
    else:
        # Analyze English query
        query_lower = query.lower()
        
        for en, ar in CONSTRUCTION_TERMS_AR.items():
            if en in query_lower:
                result["construction_terms"].append({"english": en, "arabic": ar})
                if not result["arabic_equivalent"]:
                    result["arabic_equivalent"] = query_lower.replace(en, ar)
        
        for en, ar in LOCATION_TERMS_AR.items():
            if en in query_lower:
                result["location_terms"].append({"english": en, "arabic": ar})
    
    return result
