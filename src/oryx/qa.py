"""
QA Validation Module for ORYX.

Provides quality assurance functions to validate and clean keyword clusters:
- Discard clusters with fewer than minimum keywords
- Remove keywords exceeding word count limits
- Validate keyword quality metrics
- Generate QA reports

This module ensures output quality by applying configurable validation rules.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any


# Default QA thresholds
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_MAX_WORD_COUNT = 6
DEFAULT_MIN_WORD_COUNT = 2
DEFAULT_MIN_OPPORTUNITY_SCORE = 0.0
DEFAULT_MIN_RELATIVE_INTEREST = 0.0


def validate_keyword_length(
    keyword: str,
    min_words: int = DEFAULT_MIN_WORD_COUNT,
    max_words: int = DEFAULT_MAX_WORD_COUNT,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a keyword's word count.
    
    Args:
        keyword: The keyword to validate
        min_words: Minimum word count allowed
        max_words: Maximum word count allowed
        
    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    word_count = len(keyword.split())
    
    if word_count < min_words:
        return False, f"Too few words: {word_count} < {min_words}"
    
    if word_count > max_words:
        return False, f"Too many words: {word_count} > {max_words}"
    
    return True, None


def validate_cluster_size(
    keywords: List[str],
    min_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a cluster has minimum keyword count.
    
    Args:
        keywords: List of keywords in the cluster
        min_size: Minimum number of keywords required
        
    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    if len(keywords) < min_size:
        return False, f"Cluster too small: {len(keywords)} < {min_size}"
    
    return True, None


def filter_keywords_by_length(
    keywords: List[str],
    min_words: int = DEFAULT_MIN_WORD_COUNT,
    max_words: int = DEFAULT_MAX_WORD_COUNT,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Filter keywords by word count.
    
    Args:
        keywords: List of keywords to filter
        min_words: Minimum word count allowed
        max_words: Maximum word count allowed
        
    Returns:
        Tuple of (valid_keywords, rejected_with_reasons)
    """
    valid = []
    rejected = []
    
    for kw in keywords:
        is_valid, reason = validate_keyword_length(kw, min_words, max_words)
        if is_valid:
            valid.append(kw)
        else:
            rejected.append({"keyword": kw, "reason": reason})
    
    return valid, rejected


def filter_clusters_by_size(
    clusters: Dict[str, List[str]],
    min_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Filter clusters by minimum keyword count.
    
    Args:
        clusters: Dict mapping cluster names to keyword lists
        min_size: Minimum number of keywords per cluster
        
    Returns:
        Tuple of (valid_clusters, rejected_clusters)
    """
    valid = {}
    rejected = {}
    
    for name, keywords in clusters.items():
        is_valid, reason = validate_cluster_size(keywords, min_size)
        if is_valid:
            valid[name] = keywords
        else:
            rejected[name] = keywords
            logging.debug(f"Rejected cluster '{name}': {reason}")
    
    return valid, rejected


def validate_pipeline_output(
    items: List[Dict],
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    max_word_count: int = DEFAULT_MAX_WORD_COUNT,
    min_word_count: int = DEFAULT_MIN_WORD_COUNT,
    min_opportunity_score: float = DEFAULT_MIN_OPPORTUNITY_SCORE,
    min_relative_interest: float = DEFAULT_MIN_RELATIVE_INTEREST,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Validate and filter pipeline output items.
    
    Applies QA validation rules to pipeline output:
    1. Remove keywords exceeding word count limits
    2. Remove clusters with fewer than minimum keywords
    3. Remove keywords below score thresholds
    
    Args:
        items: List of keyword items from pipeline output
        min_cluster_size: Minimum keywords per cluster
        max_word_count: Maximum words per keyword
        min_word_count: Minimum words per keyword
        min_opportunity_score: Minimum opportunity score (0-1)
        min_relative_interest: Minimum relative interest score (0-1)
        
    Returns:
        Tuple of (validated_items, qa_report)
    """
    original_count = len(items)
    
    # Step 1: Filter by keyword length
    length_filtered = []
    length_rejected = []
    
    for item in items:
        kw = item.get("keyword", "")
        is_valid, reason = validate_keyword_length(kw, min_word_count, max_word_count)
        if is_valid:
            length_filtered.append(item)
        else:
            length_rejected.append({"item": item, "reason": reason})
    
    # Step 2: Filter by score thresholds
    score_filtered = []
    score_rejected = []
    
    for item in length_filtered:
        opp_score = item.get("opportunity_score", 0.0)
        rel_interest = item.get("relative_interest", 0.0)
        
        if opp_score < min_opportunity_score:
            score_rejected.append({
                "item": item, 
                "reason": f"Low opportunity: {opp_score:.2f} < {min_opportunity_score}"
            })
        elif rel_interest < min_relative_interest:
            score_rejected.append({
                "item": item, 
                "reason": f"Low relative interest: {rel_interest:.2f} < {min_relative_interest}"
            })
        else:
            score_filtered.append(item)
    
    # Step 3: Group by cluster and filter small clusters
    clusters: Dict[str, List[Dict]] = {}
    for item in score_filtered:
        cluster = item.get("cluster", "unknown")
        clusters.setdefault(cluster, []).append(item)
    
    valid_items = []
    cluster_rejected = []
    
    for cluster_name, cluster_items in clusters.items():
        if len(cluster_items) >= min_cluster_size:
            valid_items.extend(cluster_items)
        else:
            cluster_rejected.append({
                "cluster": cluster_name,
                "items": cluster_items,
                "reason": f"Cluster too small: {len(cluster_items)} < {min_cluster_size}",
            })
    
    # Generate QA report
    qa_report = {
        "original_count": original_count,
        "final_count": len(valid_items),
        "removed_count": original_count - len(valid_items),
        "removal_rate": (original_count - len(valid_items)) / original_count if original_count > 0 else 0.0,
        "length_rejected": len(length_rejected),
        "score_rejected": len(score_rejected),
        "cluster_rejected": sum(len(cr["items"]) for cr in cluster_rejected),
        "clusters_removed": len(cluster_rejected),
        "validation_rules": {
            "min_cluster_size": min_cluster_size,
            "max_word_count": max_word_count,
            "min_word_count": min_word_count,
            "min_opportunity_score": min_opportunity_score,
            "min_relative_interest": min_relative_interest,
        },
        "details": {
            "length_rejected": length_rejected[:10],  # Limit detail output
            "score_rejected": score_rejected[:10],
            "cluster_rejected": cluster_rejected,
        },
    }
    
    logging.info(
        f"QA Validation: {original_count} → {len(valid_items)} items "
        f"({qa_report['removal_rate']:.1%} removed)"
    )
    
    return valid_items, qa_report


def run_qa_validation(
    items: List[Dict],
    config: Optional[Dict] = None,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Run QA validation with configuration.
    
    Convenience wrapper that reads QA settings from config dict.
    
    Args:
        items: List of keyword items from pipeline
        config: Optional config dict with 'qa' section
        
    Returns:
        Tuple of (validated_items, qa_report)
    """
    qa_cfg = (config or {}).get("qa", {})
    
    return validate_pipeline_output(
        items,
        min_cluster_size=int(qa_cfg.get("min_cluster_size", DEFAULT_MIN_CLUSTER_SIZE)),
        max_word_count=int(qa_cfg.get("max_word_count", DEFAULT_MAX_WORD_COUNT)),
        min_word_count=int(qa_cfg.get("min_word_count", DEFAULT_MIN_WORD_COUNT)),
        min_opportunity_score=float(qa_cfg.get("min_opportunity_score", DEFAULT_MIN_OPPORTUNITY_SCORE)),
        min_relative_interest=float(qa_cfg.get("min_relative_interest", DEFAULT_MIN_RELATIVE_INTEREST)),
    )


def print_qa_report(report: Dict[str, Any], verbose: bool = False) -> str:
    """
    Format QA report as human-readable string.
    
    Args:
        report: QA report dict from validate_pipeline_output
        verbose: If True, include detailed rejection lists
        
    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "QA VALIDATION REPORT",
        "=" * 60,
        "",
        f"Original keywords:  {report['original_count']}",
        f"Final keywords:     {report['final_count']}",
        f"Removed:            {report['removed_count']} ({report['removal_rate']:.1%})",
        "",
        "Rejection breakdown:",
        f"  - Length violations: {report['length_rejected']}",
        f"  - Score thresholds:  {report['score_rejected']}",
        f"  - Small clusters:    {report['cluster_rejected']} ({report['clusters_removed']} clusters)",
        "",
        "Validation rules applied:",
    ]
    
    rules = report.get("validation_rules", {})
    for rule, value in rules.items():
        lines.append(f"  - {rule}: {value}")
    
    if verbose and report.get("details"):
        details = report["details"]
        
        if details.get("cluster_rejected"):
            lines.append("")
            lines.append("Rejected clusters:")
            for cr in details["cluster_rejected"]:
                lines.append(f"  - {cr['cluster']}: {len(cr['items'])} keywords ({cr['reason']})")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)
