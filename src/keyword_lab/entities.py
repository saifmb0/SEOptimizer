"""
GEO-specific entity extraction for UAE/Gulf markets.

Extracts local entities (Emirates, cities, districts, landmarks) from keywords
to enable location-based content targeting and local SEO optimization.
"""
from typing import Dict, List, Optional, Set, Tuple
import re


# =============================================================================
# UAE Geographic Entities
# =============================================================================

# Emirates (top-level administrative divisions)
UAE_EMIRATES = {
    "dubai": {"name": "Dubai", "ar": "دبي", "iso": "AE-DU"},
    "abu dhabi": {"name": "Abu Dhabi", "ar": "أبوظبي", "iso": "AE-AZ"},
    "sharjah": {"name": "Sharjah", "ar": "الشارقة", "iso": "AE-SH"},
    "ajman": {"name": "Ajman", "ar": "عجمان", "iso": "AE-AJ"},
    "ras al khaimah": {"name": "Ras Al Khaimah", "ar": "رأس الخيمة", "iso": "AE-RK"},
    "fujairah": {"name": "Fujairah", "ar": "الفجيرة", "iso": "AE-FU"},
    "umm al quwain": {"name": "Umm Al Quwain", "ar": "أم القيوين", "iso": "AE-UQ"},
}

# Alternative spellings and abbreviations
UAE_EMIRATES_ALIASES = {
    "dxb": "dubai",
    "auh": "abu dhabi",
    "shj": "sharjah",
    "rak": "ras al khaimah",
    "rasalkhaimah": "ras al khaimah",
    "uaq": "umm al quwain",
    "ummul quwain": "umm al quwain",
}

# Major areas/districts within Emirates
UAE_DISTRICTS = {
    # Dubai districts
    "dubai marina": {"emirate": "dubai", "type": "residential"},
    "business bay": {"emirate": "dubai", "type": "commercial"},
    "downtown dubai": {"emirate": "dubai", "type": "mixed"},
    "jbr": {"emirate": "dubai", "type": "residential", "full": "Jumeirah Beach Residence"},
    "jumeirah": {"emirate": "dubai", "type": "residential"},
    "jumeirah beach residence": {"emirate": "dubai", "type": "residential"},
    "palm jumeirah": {"emirate": "dubai", "type": "residential"},
    "deira": {"emirate": "dubai", "type": "commercial"},
    "bur dubai": {"emirate": "dubai", "type": "commercial"},
    "al quoz": {"emirate": "dubai", "type": "industrial"},
    "al barsha": {"emirate": "dubai", "type": "residential"},
    "jlt": {"emirate": "dubai", "type": "mixed", "full": "Jumeirah Lake Towers"},
    "jumeirah lake towers": {"emirate": "dubai", "type": "mixed"},
    "difc": {"emirate": "dubai", "type": "commercial", "full": "Dubai International Financial Centre"},
    "jvc": {"emirate": "dubai", "type": "residential", "full": "Jumeirah Village Circle"},
    "silicon oasis": {"emirate": "dubai", "type": "mixed"},
    "motor city": {"emirate": "dubai", "type": "residential"},
    "sports city": {"emirate": "dubai", "type": "residential"},
    "discovery gardens": {"emirate": "dubai", "type": "residential"},
    "international city": {"emirate": "dubai", "type": "residential"},
    "mirdif": {"emirate": "dubai", "type": "residential"},
    "karama": {"emirate": "dubai", "type": "residential"},
    "tecom": {"emirate": "dubai", "type": "commercial"},
    "media city": {"emirate": "dubai", "type": "commercial"},
    "internet city": {"emirate": "dubai", "type": "commercial"},
    
    # Abu Dhabi districts  
    "al reem island": {"emirate": "abu dhabi", "type": "residential"},
    "yas island": {"emirate": "abu dhabi", "type": "mixed"},
    "saadiyat island": {"emirate": "abu dhabi", "type": "residential"},
    "mussafah": {"emirate": "abu dhabi", "type": "industrial"},
    "khalifa city": {"emirate": "abu dhabi", "type": "residential"},
    "al ain": {"emirate": "abu dhabi", "type": "city"},
    "corniche": {"emirate": "abu dhabi", "type": "commercial"},
    
    # Sharjah districts
    "al nahda": {"emirate": "sharjah", "type": "residential"},
    "al khan": {"emirate": "sharjah", "type": "residential"},
    "al majaz": {"emirate": "sharjah", "type": "commercial"},
    "industrial area": {"emirate": "sharjah", "type": "industrial"},
}

# Landmarks and points of interest
UAE_LANDMARKS = {
    "burj khalifa": {"emirate": "dubai", "type": "landmark"},
    "dubai mall": {"emirate": "dubai", "type": "commercial"},
    "mall of the emirates": {"emirate": "dubai", "type": "commercial"},
    "ibn battuta": {"emirate": "dubai", "type": "commercial"},
    "deira city centre": {"emirate": "dubai", "type": "commercial"},
    "dragon mart": {"emirate": "dubai", "type": "commercial"},
    "expo 2020": {"emirate": "dubai", "type": "event"},
    "dubai airport": {"emirate": "dubai", "type": "transport"},
    "jebel ali": {"emirate": "dubai", "type": "industrial"},
    "etihad towers": {"emirate": "abu dhabi", "type": "landmark"},
    "sheikh zayed mosque": {"emirate": "abu dhabi", "type": "landmark"},
    "louvre abu dhabi": {"emirate": "abu dhabi", "type": "landmark"},
}

# Free zones (important for B2B/contracting)
UAE_FREE_ZONES = {
    "jafza": {"emirate": "dubai", "full": "Jebel Ali Free Zone", "type": "logistics"},
    "jebel ali free zone": {"emirate": "dubai", "type": "logistics"},
    "dmcc": {"emirate": "dubai", "full": "Dubai Multi Commodities Centre", "type": "trading"},
    "dafza": {"emirate": "dubai", "full": "Dubai Airport Free Zone", "type": "logistics"},
    "tecom": {"emirate": "dubai", "type": "tech"},
    "dso": {"emirate": "dubai", "full": "Dubai Silicon Oasis", "type": "tech"},
    "kizad": {"emirate": "abu dhabi", "full": "Khalifa Industrial Zone", "type": "industrial"},
    "masdar city": {"emirate": "abu dhabi", "type": "green"},
    "saif zone": {"emirate": "sharjah", "full": "Sharjah Airport International Free Zone", "type": "logistics"},
    "hamriyah free zone": {"emirate": "sharjah", "type": "industrial"},
}


# =============================================================================
# Entity Extraction Functions
# =============================================================================

def extract_entities(keyword: str, geo: str = "ae") -> Dict[str, any]:
    """
    Extract geographic entities from a keyword.
    
    Args:
        keyword: The keyword to analyze
        geo: Geographic context (default: "ae" for UAE)
        
    Returns:
        Dict with extracted entities:
        {
            "emirate": Optional[str],
            "district": Optional[str],
            "landmark": Optional[str],
            "free_zone": Optional[str],
            "location_type": Optional[str],  # residential/commercial/industrial
            "is_local": bool,
        }
    """
    if geo.lower() != "ae":
        return {"is_local": False}
    
    kw_lower = keyword.lower()
    result = {
        "emirate": None,
        "district": None,
        "landmark": None,
        "free_zone": None,
        "location_type": None,
        "is_local": False,
    }
    
    # Check for emirate mentions (including aliases)
    for alias, emirate_key in UAE_EMIRATES_ALIASES.items():
        if alias in kw_lower:
            result["emirate"] = UAE_EMIRATES[emirate_key]["name"]
            result["is_local"] = True
            break
    
    if not result["emirate"]:
        for emirate_key, emirate_data in UAE_EMIRATES.items():
            if emirate_key in kw_lower:
                result["emirate"] = emirate_data["name"]
                result["is_local"] = True
                break
    
    # Check for district mentions
    for district_key, district_data in UAE_DISTRICTS.items():
        if district_key in kw_lower:
            result["district"] = district_key.title()
            result["emirate"] = result["emirate"] or UAE_EMIRATES[district_data["emirate"]]["name"]
            result["location_type"] = district_data.get("type")
            result["is_local"] = True
            break
    
    # Check for landmark mentions
    for landmark_key, landmark_data in UAE_LANDMARKS.items():
        if landmark_key in kw_lower:
            result["landmark"] = landmark_key.title()
            result["emirate"] = result["emirate"] or UAE_EMIRATES[landmark_data["emirate"]]["name"]
            result["is_local"] = True
            break
    
    # Check for free zone mentions
    for fz_key, fz_data in UAE_FREE_ZONES.items():
        if fz_key in kw_lower:
            result["free_zone"] = fz_data.get("full", fz_key.upper())
            result["emirate"] = result["emirate"] or UAE_EMIRATES[fz_data["emirate"]]["name"]
            result["location_type"] = fz_data.get("type")
            result["is_local"] = True
            break
    
    return result


def get_location_variations(
    base_keyword: str,
    include_emirates: bool = True,
    include_districts: bool = False,
    primary_emirate: str = "dubai",
) -> List[str]:
    """
    Generate location-based keyword variations.
    
    Useful for expanding a service keyword to target multiple locations.
    
    Args:
        base_keyword: The base keyword (e.g., "villa renovation")
        include_emirates: Include emirate-level variations
        include_districts: Include district-level variations
        primary_emirate: The primary emirate to focus on
        
    Returns:
        List of keyword variations with location modifiers
        
    Example:
        >>> get_location_variations("villa renovation", include_emirates=True)
        ["villa renovation dubai", "villa renovation abu dhabi", ...]
    """
    variations = [base_keyword]
    
    if include_emirates:
        for emirate_key, emirate_data in UAE_EMIRATES.items():
            variations.append(f"{base_keyword} {emirate_key}")
            variations.append(f"{base_keyword} in {emirate_key}")
    
    if include_districts and primary_emirate:
        for district_key, district_data in UAE_DISTRICTS.items():
            if district_data["emirate"] == primary_emirate:
                variations.append(f"{base_keyword} {district_key}")
    
    return variations


def enrich_keywords_with_entities(
    keywords: List[str],
    geo: str = "ae",
) -> List[Dict]:
    """
    Enrich a list of keywords with geographic entity information.
    
    Args:
        keywords: List of keywords to enrich
        geo: Geographic context
        
    Returns:
        List of dicts with keyword and entity data
    """
    enriched = []
    for kw in keywords:
        entities = extract_entities(kw, geo)
        enriched.append({
            "keyword": kw,
            **entities,
        })
    return enriched


def get_entity_clusters(
    keywords: List[str],
    geo: str = "ae",
) -> Dict[str, List[str]]:
    """
    Group keywords by their primary geographic entity.
    
    Useful for creating location-based content silos.
    
    Args:
        keywords: List of keywords to cluster
        geo: Geographic context
        
    Returns:
        Dict mapping location to list of keywords
    """
    clusters = {"global": []}
    
    for kw in keywords:
        entities = extract_entities(kw, geo)
        if entities["is_local"]:
            location_key = entities["district"] or entities["emirate"] or "UAE"
            if location_key not in clusters:
                clusters[location_key] = []
            clusters[location_key].append(kw)
        else:
            clusters["global"].append(kw)
    
    return clusters


# =============================================================================
# GCC Region Support (Future expansion)
# =============================================================================

GCC_COUNTRIES = {
    "ae": {"name": "United Arab Emirates", "ar": "الإمارات", "currency": "AED"},
    "sa": {"name": "Saudi Arabia", "ar": "السعودية", "currency": "SAR"},
    "qa": {"name": "Qatar", "ar": "قطر", "currency": "QAR"},
    "kw": {"name": "Kuwait", "ar": "الكويت", "currency": "KWD"},
    "bh": {"name": "Bahrain", "ar": "البحرين", "currency": "BHD"},
    "om": {"name": "Oman", "ar": "عُمان", "currency": "OMR"},
}

# Saudi Arabia major cities (for future expansion)
SA_CITIES = {
    "riyadh": {"ar": "الرياض", "type": "capital"},
    "jeddah": {"ar": "جدة", "type": "commercial"},
    "dammam": {"ar": "الدمام", "type": "industrial"},
    "mecca": {"ar": "مكة", "type": "religious"},
    "medina": {"ar": "المدينة", "type": "religious"},
    "neom": {"ar": "نيوم", "type": "megaproject"},
}
