"""
Content gap analysis for SEO optimization.

Identifies content gaps by comparing target keywords against
existing content coverage, enabling strategic content planning.
"""
from typing import Dict, List, Optional, Set, Tuple
import logging
from collections import defaultdict


# =============================================================================
# Content Gap Categories
# =============================================================================

# Standard content gap categories for prioritization
CONTENT_GAP_CATEGORIES = {
    "missing_topic": {
        "priority": 1,
        "description": "No content exists for this topic cluster",
        "action": "Create new pillar content",
    },
    "thin_content": {
        "priority": 2,
        "description": "Content exists but is too short or shallow",
        "action": "Expand and deepen existing content",
    },
    "outdated_content": {
        "priority": 3,
        "description": "Content exists but is outdated",
        "action": "Update and refresh content",
    },
    "missing_format": {
        "priority": 4,
        "description": "Topic covered but missing key format (FAQ, how-to, etc.)",
        "action": "Add complementary content format",
    },
    "missing_location": {
        "priority": 5,
        "description": "Topic covered but missing location-specific variant",
        "action": "Create location-targeted content",
    },
    "competitive_gap": {
        "priority": 6,
        "description": "Competitors rank for this but we don't",
        "action": "Create competing content with differentiation",
    },
}


# Content format patterns
CONTENT_FORMATS = {
    "how_to": {
        "patterns": ["how to", "guide to", "steps to", "tutorial"],
        "word_count_min": 1500,
    },
    "listicle": {
        "patterns": ["top", "best", "list of", "types of", "examples of"],
        "word_count_min": 1200,
    },
    "comparison": {
        "patterns": ["vs", "versus", "compare", "difference between", "or"],
        "word_count_min": 1000,
    },
    "faq": {
        "patterns": ["what is", "what are", "why", "when", "where"],
        "word_count_min": 800,
    },
    "case_study": {
        "patterns": ["case study", "project", "portfolio", "example"],
        "word_count_min": 1000,
    },
    "pricing": {
        "patterns": ["cost", "price", "rates", "quote", "estimate"],
        "word_count_min": 800,
    },
    "local": {
        "patterns": ["in dubai", "in abu dhabi", "near me", "in uae"],
        "word_count_min": 800,
    },
}


# =============================================================================
# Content Gap Analysis Functions
# =============================================================================

def identify_content_format(keyword: str) -> Optional[str]:
    """
    Identify the expected content format for a keyword.
    
    Args:
        keyword: The keyword to analyze
        
    Returns:
        Content format type or None
    """
    kw_lower = keyword.lower()
    
    for format_type, config in CONTENT_FORMATS.items():
        for pattern in config["patterns"]:
            if pattern in kw_lower:
                return format_type
    
    return None


def analyze_content_gaps(
    target_keywords: List[str],
    existing_content: Optional[List[Dict]] = None,
    competitor_keywords: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Analyze content gaps between target keywords and existing content.
    
    Args:
        target_keywords: Keywords we want to rank for
        existing_content: List of dicts with {'url', 'title', 'keywords', 'word_count'}
        competitor_keywords: Keywords competitors rank for
        
    Returns:
        List of content gap opportunities with recommendations
    """
    gaps = []
    existing = existing_content or []
    competitor_kws = set(k.lower() for k in (competitor_keywords or []))
    
    # Build index of existing content
    covered_keywords = set()
    content_by_keyword: Dict[str, Dict] = {}
    
    for content in existing:
        for kw in content.get("keywords", []):
            covered_keywords.add(kw.lower())
            content_by_keyword[kw.lower()] = content
    
    for kw in target_keywords:
        kw_lower = kw.lower()
        expected_format = identify_content_format(kw)
        
        # Check if completely missing
        if kw_lower not in covered_keywords:
            gap_type = "missing_topic"
            
            # Check if competitors have it
            if kw_lower in competitor_kws:
                gap_type = "competitive_gap"
            
            gaps.append({
                "keyword": kw,
                "gap_type": gap_type,
                "priority": CONTENT_GAP_CATEGORIES[gap_type]["priority"],
                "action": CONTENT_GAP_CATEGORIES[gap_type]["action"],
                "expected_format": expected_format,
                "recommended_word_count": CONTENT_FORMATS.get(expected_format, {}).get("word_count_min", 1000),
                "existing_url": None,
            })
        else:
            # Check for thin content
            content = content_by_keyword[kw_lower]
            word_count = content.get("word_count", 0)
            min_words = CONTENT_FORMATS.get(expected_format, {}).get("word_count_min", 800)
            
            if word_count < min_words:
                gaps.append({
                    "keyword": kw,
                    "gap_type": "thin_content",
                    "priority": CONTENT_GAP_CATEGORIES["thin_content"]["priority"],
                    "action": f"Expand from {word_count} to {min_words}+ words",
                    "expected_format": expected_format,
                    "recommended_word_count": min_words,
                    "existing_url": content.get("url"),
                    "current_word_count": word_count,
                })
    
    # Sort by priority
    gaps.sort(key=lambda g: (g["priority"], -len(g["keyword"])))
    
    return gaps


def generate_content_calendar(
    gaps: List[Dict],
    posts_per_week: int = 2,
    weeks: int = 12,
) -> List[Dict]:
    """
    Generate a content calendar from gap analysis.
    
    Args:
        gaps: Content gaps from analyze_content_gaps()
        posts_per_week: Number of posts to plan per week
        weeks: Number of weeks to plan for
        
    Returns:
        List of content calendar entries
    """
    calendar = []
    total_slots = posts_per_week * weeks
    
    for i, gap in enumerate(gaps[:total_slots]):
        week_num = (i // posts_per_week) + 1
        day_in_week = (i % posts_per_week) + 1
        
        calendar.append({
            "week": week_num,
            "slot": day_in_week,
            "keyword": gap["keyword"],
            "gap_type": gap["gap_type"],
            "action": gap["action"],
            "format": gap.get("expected_format", "article"),
            "target_word_count": gap.get("recommended_word_count", 1000),
            "priority": gap["priority"],
        })
    
    return calendar


def cluster_gaps_by_topic(
    gaps: List[Dict],
    clusters: Dict[str, List[str]],
) -> Dict[str, List[Dict]]:
    """
    Group content gaps by their keyword cluster.
    
    Helps identify which topic clusters need the most content.
    
    Args:
        gaps: Content gaps from analyze_content_gaps()
        clusters: Keyword clusters from clustering
        
    Returns:
        Dict mapping cluster name to list of gaps
    """
    # Build reverse index: keyword -> cluster
    keyword_to_cluster = {}
    for cluster_name, keywords in clusters.items():
        for kw in keywords:
            keyword_to_cluster[kw.lower()] = cluster_name
    
    # Group gaps by cluster
    clustered_gaps = defaultdict(list)
    for gap in gaps:
        cluster = keyword_to_cluster.get(gap["keyword"].lower(), "unclustered")
        clustered_gaps[cluster].append(gap)
    
    return dict(clustered_gaps)


# =============================================================================
# Location-Based Content Gap Analysis
# =============================================================================

def analyze_location_gaps(
    keywords: List[str],
    existing_locations: Optional[List[str]] = None,
    target_locations: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Analyze location-based content gaps.
    
    Identifies which location variants are missing for service keywords.
    
    Args:
        keywords: Base service keywords
        existing_locations: Locations already covered in content
        target_locations: Locations we want to target
        
    Returns:
        List of location gap opportunities
    """
    from .entities import UAE_EMIRATES
    
    # Default to major UAE Emirates if no target specified
    if target_locations is None:
        target_locations = list(UAE_EMIRATES.keys())
    
    existing = set(loc.lower() for loc in (existing_locations or []))
    gaps = []
    
    for kw in keywords:
        kw_lower = kw.lower()
        
        # Skip if keyword already has a location
        has_location = any(loc in kw_lower for loc in target_locations)
        if has_location:
            continue
        
        # Find missing location variants
        for location in target_locations:
            location_variant = f"{kw} {location}"
            if location not in existing:
                gaps.append({
                    "base_keyword": kw,
                    "location": location,
                    "keyword_variant": location_variant,
                    "gap_type": "missing_location",
                    "priority": CONTENT_GAP_CATEGORIES["missing_location"]["priority"],
                })
    
    return gaps


def prioritize_gaps_for_leads(
    gaps: List[Dict],
    commercial_value_fn=None,
) -> List[Dict]:
    """
    Re-prioritize content gaps for lead generation.
    
    Boosts priority for high commercial value keywords.
    
    Args:
        gaps: Content gaps from analysis
        commercial_value_fn: Function to compute commercial value
        
    Returns:
        Re-prioritized list of gaps
    """
    if commercial_value_fn is None:
        from .metrics import commercial_value
        commercial_value_fn = commercial_value
    
    for gap in gaps:
        kw = gap["keyword"]
        cv = commercial_value_fn(kw)
        
        # Adjust priority based on commercial value
        # High CV keywords get priority boost (lower number = higher priority)
        if cv > 0.7:
            gap["priority"] = max(1, gap["priority"] - 2)
            gap["commercial_value"] = "high"
        elif cv > 0.4:
            gap["priority"] = max(1, gap["priority"] - 1)
            gap["commercial_value"] = "medium"
        else:
            gap["commercial_value"] = "low"
    
    gaps.sort(key=lambda g: (g["priority"], -len(g["keyword"])))
    return gaps
