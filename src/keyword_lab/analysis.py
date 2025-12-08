"""Competitor Gap Intelligence Analysis.

Identifies keyword opportunities where competitors are weak or absent.
Highlights keywords with high opportunity + low difficulty + competitor missing.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple


def identify_competitor_gaps(
    keywords: List[Dict],
    competitor_rankings: Optional[Dict[str, Set[str]]] = None,
    opportunity_threshold: float = 0.6,
    difficulty_threshold: float = 0.5,
) -> List[Dict]:
    """
    Identify keywords where competitors are weak or missing.
    
    Finds "gap" keywords that represent opportunities:
    - High opportunity score (above threshold)
    - Low difficulty (below threshold)
    - Competitor not ranking (if competitor data provided)
    
    Args:
        keywords: List of keyword dicts with opportunity_score, difficulty
        competitor_rankings: Optional dict mapping competitor name -> set of keywords they rank for
        opportunity_threshold: Minimum opportunity score (0.0-1.0)
        difficulty_threshold: Maximum difficulty (0.0-1.0)
        
    Returns:
        List of gap keyword dicts with additional 'gap_reason' field
    """
    gaps = []
    
    # Flatten competitor keywords for quick lookup
    all_competitor_keywords: Set[str] = set()
    if competitor_rankings:
        for comp_kws in competitor_rankings.values():
            all_competitor_keywords.update(kw.lower() for kw in comp_kws)
    
    for kw_dict in keywords:
        kw = kw_dict.get("keyword", "").lower()
        opp = float(kw_dict.get("opportunity_score", 0))
        diff = float(kw_dict.get("difficulty", 1.0))
        
        # Check if this is a gap opportunity
        reasons = []
        
        if opp >= opportunity_threshold:
            reasons.append(f"high_opportunity ({opp:.2f})")
        else:
            continue  # Must meet opportunity threshold
            
        if diff <= difficulty_threshold:
            reasons.append(f"low_difficulty ({diff:.2f})")
        
        if competitor_rankings and kw not in all_competitor_keywords:
            reasons.append("competitor_missing")
        
        if len(reasons) >= 2:  # At least 2 positive signals
            gap_kw = kw_dict.copy()
            gap_kw["gap_reason"] = ", ".join(reasons)
            gap_kw["gap_score"] = calculate_gap_score(opp, diff, kw not in all_competitor_keywords)
            gaps.append(gap_kw)
    
    # Sort by gap score descending
    gaps = sorted(gaps, key=lambda x: x.get("gap_score", 0), reverse=True)
    
    logging.info(f"Identified {len(gaps)} competitor gap opportunities")
    return gaps


def calculate_gap_score(
    opportunity: float,
    difficulty: float,
    competitor_missing: bool,
) -> float:
    """
    Calculate a composite gap score for prioritization.
    
    Formula: opportunity * (1 - difficulty) * competitor_boost
    
    Args:
        opportunity: Opportunity score (0-1)
        difficulty: Difficulty score (0-1)
        competitor_missing: Whether competitors are missing from this keyword
        
    Returns:
        Gap score (0.0 - 1.5+)
    """
    base_score = opportunity * (1 - difficulty)
    
    # Boost by 50% if competitors are missing
    if competitor_missing:
        return base_score * 1.5
    
    return base_score


def analyze_competitor_overlap(
    our_keywords: List[Dict],
    competitor_rankings: Dict[str, Set[str]],
) -> Dict[str, Dict]:
    """
    Analyze keyword overlap with competitors.
    
    Returns overlap metrics for each competitor:
    - Keywords they rank for that we target
    - Keywords unique to them
    - Keywords unique to us
    
    Args:
        our_keywords: Our keyword list
        competitor_rankings: Dict mapping competitor -> their keywords
        
    Returns:
        Dict mapping competitor name -> overlap analysis
    """
    our_kw_set = {kw.get("keyword", "").lower() for kw in our_keywords}
    
    analysis = {}
    for competitor, their_kws in competitor_rankings.items():
        their_kws_lower = {kw.lower() for kw in their_kws}
        
        overlap = our_kw_set & their_kws_lower
        unique_to_us = our_kw_set - their_kws_lower
        unique_to_them = their_kws_lower - our_kw_set
        
        analysis[competitor] = {
            "overlap_count": len(overlap),
            "overlap_keywords": sorted(list(overlap))[:50],  # Limit for readability
            "unique_to_us_count": len(unique_to_us),
            "unique_to_them_count": len(unique_to_them),
            "unique_to_them_sample": sorted(list(unique_to_them))[:20],
            "overlap_percentage": len(overlap) / len(our_kw_set) * 100 if our_kw_set else 0,
        }
    
    return analysis


def prioritize_quick_wins(
    keywords: List[Dict],
    max_difficulty: float = 0.3,
    min_opportunity: float = 0.5,
    limit: int = 20,
) -> List[Dict]:
    """
    Find "quick win" keywords - high value, low effort.
    
    These are ideal targets for immediate content creation:
    - Low difficulty (easy to rank)
    - Decent opportunity (worth the effort)
    - Validated by autocomplete (real search volume)
    
    Args:
        keywords: List of keyword dicts
        max_difficulty: Maximum difficulty threshold
        min_opportunity: Minimum opportunity threshold
        limit: Max number of quick wins to return
        
    Returns:
        List of quick win keywords sorted by opportunity
    """
    quick_wins = []
    
    for kw in keywords:
        diff = float(kw.get("difficulty", 1.0))
        opp = float(kw.get("opportunity_score", 0))
        validated = kw.get("validated", False)
        
        if diff <= max_difficulty and opp >= min_opportunity:
            qw = kw.copy()
            # Boost validated keywords in sort
            qw["quick_win_score"] = opp * (1.2 if validated else 1.0)
            quick_wins.append(qw)
    
    # Sort by quick win score
    quick_wins = sorted(quick_wins, key=lambda x: x.get("quick_win_score", 0), reverse=True)
    
    logging.info(f"Found {len(quick_wins[:limit])} quick win opportunities")
    return quick_wins[:limit]


def generate_gap_report(
    keywords: List[Dict],
    competitor_rankings: Optional[Dict[str, Set[str]]] = None,
) -> Dict:
    """
    Generate a comprehensive competitor gap intelligence report.
    
    Args:
        keywords: List of keyword dicts
        competitor_rankings: Optional competitor keyword data
        
    Returns:
        Dict containing gap analysis, quick wins, and recommendations
    """
    report = {
        "total_keywords": len(keywords),
        "gap_opportunities": [],
        "quick_wins": [],
        "competitor_analysis": {},
        "recommendations": [],
    }
    
    # Find gap opportunities
    gaps = identify_competitor_gaps(keywords, competitor_rankings)
    report["gap_opportunities"] = gaps[:30]  # Top 30 gaps
    
    # Find quick wins
    quick_wins = prioritize_quick_wins(keywords)
    report["quick_wins"] = quick_wins
    
    # Competitor overlap if data provided
    if competitor_rankings:
        report["competitor_analysis"] = analyze_competitor_overlap(keywords, competitor_rankings)
    
    # Generate recommendations
    if gaps:
        report["recommendations"].append(
            f"Found {len(gaps)} gap opportunities where competitors are weak. "
            "Prioritize content creation for these keywords."
        )
    
    if quick_wins:
        top_cluster = quick_wins[0].get("cluster", "general") if quick_wins else "general"
        report["recommendations"].append(
            f"Top quick-win cluster: '{top_cluster}'. "
            "Create pillar content for this topic first."
        )
    
    # Identify thin clusters
    from collections import Counter
    cluster_counts = Counter(kw.get("cluster", "unknown") for kw in keywords)
    thin_clusters = [c for c, count in cluster_counts.items() if count < 3]
    if thin_clusters:
        report["recommendations"].append(
            f"Thin clusters detected: {', '.join(thin_clusters[:5])}. "
            "Consider merging or expanding these topics."
        )
    
    return report
