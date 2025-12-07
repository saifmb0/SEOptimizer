"""
GEO-specific entity extraction for UAE/Gulf markets.

Extracts local entities (Emirates, cities, districts, landmarks) from keywords
to enable location-based content targeting and local SEO optimization.

ORYX Edition: Enhanced for Abu Dhabi contracting sector with hyper-local terms.
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
    # Abu Dhabi variations
    "abudhabi": "abu dhabi",
    "abu-dhabi": "abu dhabi",
    "ad": "abu dhabi",
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
    
    # ==========================================================================
    # ABU DHABI DISTRICTS (Expanded for ORYX - Contracting Sector)
    # ==========================================================================
    
    # Abu Dhabi Island (Core)
    "al reem island": {"emirate": "abu dhabi", "type": "residential"},
    "yas island": {"emirate": "abu dhabi", "type": "mixed"},
    "saadiyat island": {"emirate": "abu dhabi", "type": "residential"},
    "corniche": {"emirate": "abu dhabi", "type": "commercial"},
    "khalidiya": {"emirate": "abu dhabi", "type": "residential"},
    "al khalidiyah": {"emirate": "abu dhabi", "type": "residential"},
    "al markaziyah": {"emirate": "abu dhabi", "type": "commercial"},
    "tourist club area": {"emirate": "abu dhabi", "type": "commercial"},
    "al zahiyah": {"emirate": "abu dhabi", "type": "commercial"},
    "al danah": {"emirate": "abu dhabi", "type": "commercial"},
    "al maryah island": {"emirate": "abu dhabi", "type": "commercial"},
    "al bateen": {"emirate": "abu dhabi", "type": "residential"},
    "al mushrif": {"emirate": "abu dhabi", "type": "residential"},
    "al karamah": {"emirate": "abu dhabi", "type": "residential"},
    "al manhal": {"emirate": "abu dhabi", "type": "residential"},
    "al nahyan": {"emirate": "abu dhabi", "type": "residential"},
    "al rowdah": {"emirate": "abu dhabi", "type": "residential"},
    "al muroor": {"emirate": "abu dhabi", "type": "mixed"},
    
    # Mainland Abu Dhabi
    "khalifa city": {"emirate": "abu dhabi", "type": "residential"},
    "khalifa city a": {"emirate": "abu dhabi", "type": "residential"},
    "khalifa city b": {"emirate": "abu dhabi", "type": "residential"},
    "mohamed bin zayed city": {"emirate": "abu dhabi", "type": "residential"},
    "mbz city": {"emirate": "abu dhabi", "type": "residential", "full": "Mohamed Bin Zayed City"},
    "al shamkha": {"emirate": "abu dhabi", "type": "residential"},
    "baniyas": {"emirate": "abu dhabi", "type": "residential"},
    "al reef": {"emirate": "abu dhabi", "type": "residential"},
    "al ghadeer": {"emirate": "abu dhabi", "type": "residential"},
    "masdar city": {"emirate": "abu dhabi", "type": "mixed"},
    "al raha": {"emirate": "abu dhabi", "type": "residential"},
    "al raha beach": {"emirate": "abu dhabi", "type": "residential"},
    "al raha gardens": {"emirate": "abu dhabi", "type": "residential"},
    
    # Industrial Areas (Critical for Contracting)
    "mussafah": {"emirate": "abu dhabi", "type": "industrial"},
    "mussaffah": {"emirate": "abu dhabi", "type": "industrial"},  # Common misspelling
    "musaffah": {"emirate": "abu dhabi", "type": "industrial"},   # Another variant
    "icad": {"emirate": "abu dhabi", "type": "industrial", "full": "Industrial City of Abu Dhabi"},
    "icad 1": {"emirate": "abu dhabi", "type": "industrial"},
    "icad 2": {"emirate": "abu dhabi", "type": "industrial"},
    "icad 3": {"emirate": "abu dhabi", "type": "industrial"},
    
    # Western Region (Al Dhafra)
    "al dhafra": {"emirate": "abu dhabi", "type": "region"},
    "madinat zayed": {"emirate": "abu dhabi", "type": "residential"},
    "liwa": {"emirate": "abu dhabi", "type": "residential"},
    "ghayathi": {"emirate": "abu dhabi", "type": "residential"},
    "ruwais": {"emirate": "abu dhabi", "type": "industrial"},
    "mirfa": {"emirate": "abu dhabi", "type": "residential"},
    
    # Al Ain (Garden City)
    "al ain": {"emirate": "abu dhabi", "type": "city"},
    "al jimi": {"emirate": "abu dhabi", "type": "residential", "city": "al ain"},
    "al mutarad": {"emirate": "abu dhabi", "type": "residential", "city": "al ain"},
    "al muwaiji": {"emirate": "abu dhabi", "type": "residential", "city": "al ain"},
    "al tawia": {"emirate": "abu dhabi", "type": "residential", "city": "al ain"},
    "al khabisi": {"emirate": "abu dhabi", "type": "residential", "city": "al ain"},
    "al sarooj": {"emirate": "abu dhabi", "type": "residential", "city": "al ain"},
    "al jahili": {"emirate": "abu dhabi", "type": "residential", "city": "al ain"},
    "zakher": {"emirate": "abu dhabi", "type": "residential", "city": "al ain"},
    "al hili": {"emirate": "abu dhabi", "type": "industrial", "city": "al ain"},
    
    # Sharjah districts
    "al nahda": {"emirate": "sharjah", "type": "residential"},
    "al khan": {"emirate": "sharjah", "type": "residential"},
    "al majaz": {"emirate": "sharjah", "type": "commercial"},
    "industrial area": {"emirate": "sharjah", "type": "industrial"},
}

# Landmarks and points of interest
UAE_LANDMARKS = {
    # Dubai landmarks
    "burj khalifa": {"emirate": "dubai", "type": "landmark"},
    "dubai mall": {"emirate": "dubai", "type": "commercial"},
    "mall of the emirates": {"emirate": "dubai", "type": "commercial"},
    "ibn battuta": {"emirate": "dubai", "type": "commercial"},
    "deira city centre": {"emirate": "dubai", "type": "commercial"},
    "dragon mart": {"emirate": "dubai", "type": "commercial"},
    "expo 2020": {"emirate": "dubai", "type": "event"},
    "dubai airport": {"emirate": "dubai", "type": "transport"},
    "jebel ali": {"emirate": "dubai", "type": "industrial"},
    
    # Abu Dhabi landmarks (Expanded)
    "etihad towers": {"emirate": "abu dhabi", "type": "landmark"},
    "sheikh zayed mosque": {"emirate": "abu dhabi", "type": "landmark"},
    "sheikh zayed grand mosque": {"emirate": "abu dhabi", "type": "landmark"},
    "louvre abu dhabi": {"emirate": "abu dhabi", "type": "landmark"},
    "qasr al watan": {"emirate": "abu dhabi", "type": "landmark"},
    "emirates palace": {"emirate": "abu dhabi", "type": "landmark"},
    "yas marina circuit": {"emirate": "abu dhabi", "type": "entertainment"},
    "ferrari world": {"emirate": "abu dhabi", "type": "entertainment"},
    "yas waterworld": {"emirate": "abu dhabi", "type": "entertainment"},
    "warner bros world": {"emirate": "abu dhabi", "type": "entertainment"},
    "abu dhabi mall": {"emirate": "abu dhabi", "type": "commercial"},
    "marina mall": {"emirate": "abu dhabi", "type": "commercial"},
    "yas mall": {"emirate": "abu dhabi", "type": "commercial"},
    "al wahda mall": {"emirate": "abu dhabi", "type": "commercial"},
    "dalma mall": {"emirate": "abu dhabi", "type": "commercial"},
    "world trade center abu dhabi": {"emirate": "abu dhabi", "type": "commercial"},
    "adnec": {"emirate": "abu dhabi", "type": "events", "full": "Abu Dhabi National Exhibition Centre"},
    "abu dhabi airport": {"emirate": "abu dhabi", "type": "transport"},
    "zayed international airport": {"emirate": "abu dhabi", "type": "transport"},
}

# Free zones (important for B2B/contracting)
UAE_FREE_ZONES = {
    # Dubai free zones
    "jafza": {"emirate": "dubai", "full": "Jebel Ali Free Zone", "type": "logistics"},
    "jebel ali free zone": {"emirate": "dubai", "type": "logistics"},
    "dmcc": {"emirate": "dubai", "full": "Dubai Multi Commodities Centre", "type": "trading"},
    "dafza": {"emirate": "dubai", "full": "Dubai Airport Free Zone", "type": "logistics"},
    "tecom": {"emirate": "dubai", "type": "tech"},
    "dso": {"emirate": "dubai", "full": "Dubai Silicon Oasis", "type": "tech"},
    
    # Abu Dhabi free zones (Expanded for ORYX)
    "kizad": {"emirate": "abu dhabi", "full": "Khalifa Industrial Zone Abu Dhabi", "type": "industrial"},
    "khalifa industrial zone": {"emirate": "abu dhabi", "type": "industrial"},
    "masdar city": {"emirate": "abu dhabi", "type": "green"},
    "masdar free zone": {"emirate": "abu dhabi", "type": "green"},
    "adgm": {"emirate": "abu dhabi", "full": "Abu Dhabi Global Market", "type": "financial"},
    "abu dhabi global market": {"emirate": "abu dhabi", "type": "financial"},
    "twofour54": {"emirate": "abu dhabi", "type": "media"},
    "hub71": {"emirate": "abu dhabi", "type": "tech"},
    "zonesCorp": {"emirate": "abu dhabi", "type": "industrial"},
    "icad free zone": {"emirate": "abu dhabi", "type": "industrial"},
    "al ain industrial city": {"emirate": "abu dhabi", "type": "industrial"},
    
    # Sharjah free zones
    "saif zone": {"emirate": "sharjah", "full": "Sharjah Airport International Free Zone", "type": "logistics"},
    "hamriyah free zone": {"emirate": "sharjah", "type": "industrial"},
}


# =============================================================================
# ABU DHABI CONTRACTING-SPECIFIC TERMS (ORYX Enhancement)
# =============================================================================

# Government entities and regulators critical for contracting
UAE_GOVERNMENT_ENTITIES = {
    # Abu Dhabi specific (critical for contracting)
    "tamm": {"emirate": "abu dhabi", "type": "government", "full": "TAMM Services Portal", "ar": "تم"},
    "adm": {"emirate": "abu dhabi", "type": "government", "full": "Abu Dhabi Municipality"},
    "abu dhabi municipality": {"emirate": "abu dhabi", "type": "government"},
    "ded": {"emirate": "abu dhabi", "type": "government", "full": "Department of Economic Development"},
    "ded abu dhabi": {"emirate": "abu dhabi", "type": "government"},
    "added": {"emirate": "abu dhabi", "type": "government", "full": "Abu Dhabi Dept of Economic Development"},
    "dmt": {"emirate": "abu dhabi", "type": "government", "full": "Department of Municipalities and Transport"},
    "ead": {"emirate": "abu dhabi", "type": "government", "full": "Environment Agency Abu Dhabi"},
    "adda": {"emirate": "abu dhabi", "type": "government", "full": "Abu Dhabi Digital Authority"},
    "iec": {"emirate": "abu dhabi", "type": "government", "full": "Integrated Energy Company"},
    "tabreed": {"emirate": "abu dhabi", "type": "utility"},
    "addc": {"emirate": "abu dhabi", "type": "utility", "full": "Abu Dhabi Distribution Company"},
    "adwea": {"emirate": "abu dhabi", "type": "utility", "full": "Abu Dhabi Water & Electricity Authority"},
    "ewec": {"emirate": "abu dhabi", "type": "utility", "full": "Emirates Water and Electricity Company"},
    "transad": {"emirate": "abu dhabi", "type": "government", "full": "Abu Dhabi Transport"},
    "adha": {"emirate": "abu dhabi", "type": "government", "full": "Abu Dhabi Housing Authority"},
    "musanada": {"emirate": "abu dhabi", "type": "government", "full": "Abu Dhabi General Services Company"},
    "aldar": {"emirate": "abu dhabi", "type": "developer"},
    "modon": {"emirate": "abu dhabi", "type": "developer"},
    
    # Dubai equivalents for comparison
    "rera": {"emirate": "dubai", "type": "government", "full": "Real Estate Regulatory Agency"},
    "dubai land": {"emirate": "dubai", "type": "government", "full": "Dubai Land Department"},
    "dewa": {"emirate": "dubai", "type": "utility", "full": "Dubai Electricity and Water Authority"},
    "rta": {"emirate": "dubai", "type": "government", "full": "Roads and Transport Authority"},
    "dubai municipality": {"emirate": "dubai", "type": "government"},
}

# Abu Dhabi building/sustainability regulations and certifications
UAE_CERTIFICATIONS = {
    # Abu Dhabi specific (critical for contracting compliance)
    "estidama": {"emirate": "abu dhabi", "type": "certification", "ar": "استدامة"},
    "pearl rating": {"emirate": "abu dhabi", "type": "certification"},
    "pearl 1": {"emirate": "abu dhabi", "type": "certification"},
    "pearl 2": {"emirate": "abu dhabi", "type": "certification"},
    "pearl 3": {"emirate": "abu dhabi", "type": "certification"},
    "pearl 4": {"emirate": "abu dhabi", "type": "certification"},
    "pearl 5": {"emirate": "abu dhabi", "type": "certification"},
    "pvrs": {"emirate": "abu dhabi", "type": "certification", "full": "Pearl Villa Rating System"},
    "pcrs": {"emirate": "abu dhabi", "type": "certification", "full": "Pearl Community Rating System"},
    "pbrs": {"emirate": "abu dhabi", "type": "certification", "full": "Pearl Building Rating System"},
    "upc": {"emirate": "abu dhabi", "type": "regulation", "full": "Urban Planning Council"},
    
    # General UAE/international certifications
    "leed": {"type": "certification", "full": "Leadership in Energy and Environmental Design"},
    "breeam": {"type": "certification"},
    "green building": {"type": "certification"},
    "iso 9001": {"type": "certification"},
    "iso 14001": {"type": "certification"},
    "ohsas 18001": {"type": "certification"},
}

# Abu Dhabi legal/property terms for contracting
UAE_LEGAL_TERMS = {
    # Property ownership types (critical for renovation/construction)
    "musataha": {"type": "ownership", "ar": "مساطحة", "desc": "Long-term lease (up to 50 years)"},
    "usufruct": {"type": "ownership", "ar": "حق الانتفاع", "desc": "Right to use property"},
    "freehold": {"type": "ownership", "ar": "التملك الحر"},
    "leasehold": {"type": "ownership", "ar": "الإيجار"},
    
    # Rental/tenancy (Abu Dhabi specific)
    "tawtheeq": {"emirate": "abu dhabi", "type": "document", "ar": "توثيق", "full": "Tenancy Contract Registration"},
    "ejari": {"emirate": "dubai", "type": "document", "ar": "إيجاري", "full": "Rental Contract Registration"},
    
    # Permits and licenses
    "noc": {"type": "permit", "full": "No Objection Certificate"},
    "building permit": {"type": "permit"},
    "civil defense approval": {"type": "permit"},
    "completion certificate": {"type": "permit"},
    "occupancy permit": {"type": "permit"},
    "trade license": {"type": "license"},
    "contractor license": {"type": "license"},
    "sira": {"emirate": "abu dhabi", "type": "permit", "full": "Security Industry Regulatory Agency"},
}

# Common construction/contracting terms in UAE market
UAE_CONSTRUCTION_TERMS = {
    # Project types
    "villa renovation": {"type": "service"},
    "fit out": {"type": "service"},
    "fitout": {"type": "service"},
    "turnkey": {"type": "service"},
    "mep": {"type": "trade", "full": "Mechanical, Electrical, Plumbing"},
    "hvac": {"type": "trade", "full": "Heating, Ventilation, Air Conditioning"},
    "false ceiling": {"type": "trade"},
    "gypsum": {"type": "material"},
    "joinery": {"type": "trade"},
    "flooring": {"type": "trade"},
    "tiling": {"type": "trade"},
    "painting": {"type": "trade"},
    "waterproofing": {"type": "trade"},
    "landscaping": {"type": "trade"},
    "hardscaping": {"type": "trade"},
    "swimming pool": {"type": "trade"},
    "facade": {"type": "trade"},
    "cladding": {"type": "trade"},
    "aluminium": {"type": "material"},
    "aluminum": {"type": "material"},  # US spelling
    "steel structure": {"type": "trade"},
    "concrete": {"type": "material"},
    "scaffolding": {"type": "equipment"},
    "demolition": {"type": "service"},
    "excavation": {"type": "service"},
    "foundations": {"type": "trade"},
    "boundary wall": {"type": "trade"},
    "pergola": {"type": "trade"},
    "majlis": {"type": "space", "ar": "مجلس", "desc": "Traditional sitting room"},
    "mulhaq": {"type": "space", "ar": "ملحق", "desc": "Extension/annex"},
}


# =============================================================================
# Entity Extraction Functions
# =============================================================================

def extract_entities(keyword: str, geo: str = "ae") -> Dict[str, any]:
    """
    Extract geographic and domain-specific entities from a keyword.
    
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
            "government_entity": Optional[str],
            "certification": Optional[str],
            "legal_term": Optional[str],
            "construction_term": Optional[str],
            "location_type": Optional[str],  # residential/commercial/industrial
            "is_local": bool,
            "is_contracting": bool,  # Related to construction/contracting
        }
    """
    if geo.lower() != "ae":
        return {"is_local": False, "is_contracting": False}
    
    kw_lower = keyword.lower()
    result = {
        "emirate": None,
        "district": None,
        "landmark": None,
        "free_zone": None,
        "government_entity": None,
        "certification": None,
        "legal_term": None,
        "construction_term": None,
        "location_type": None,
        "is_local": False,
        "is_contracting": False,
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
            if "emirate" in fz_data:
                result["emirate"] = result["emirate"] or UAE_EMIRATES[fz_data["emirate"]]["name"]
            result["location_type"] = fz_data.get("type")
            result["is_local"] = True
            break
    
    # Check for government entity mentions (ORYX enhancement)
    for gov_key, gov_data in UAE_GOVERNMENT_ENTITIES.items():
        if gov_key in kw_lower:
            result["government_entity"] = gov_data.get("full", gov_key.upper())
            if "emirate" in gov_data:
                result["emirate"] = result["emirate"] or UAE_EMIRATES[gov_data["emirate"]]["name"]
            result["is_local"] = True
            result["is_contracting"] = True
            break
    
    # Check for certifications (ORYX enhancement)
    for cert_key, cert_data in UAE_CERTIFICATIONS.items():
        if cert_key in kw_lower:
            result["certification"] = cert_data.get("full", cert_key.upper())
            if "emirate" in cert_data:
                result["emirate"] = result["emirate"] or UAE_EMIRATES[cert_data["emirate"]]["name"]
            result["is_contracting"] = True
            break
    
    # Check for legal terms (ORYX enhancement)
    for legal_key, legal_data in UAE_LEGAL_TERMS.items():
        if legal_key in kw_lower:
            result["legal_term"] = legal_data.get("full", legal_key.title())
            if "emirate" in legal_data:
                result["emirate"] = result["emirate"] or UAE_EMIRATES[legal_data["emirate"]]["name"]
            result["is_local"] = True
            result["is_contracting"] = True
            break
    
    # Check for construction terms (ORYX enhancement)
    for const_key, const_data in UAE_CONSTRUCTION_TERMS.items():
        if const_key in kw_lower:
            result["construction_term"] = const_key.title()
            result["is_contracting"] = True
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
