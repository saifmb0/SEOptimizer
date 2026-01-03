"""
Scoring Engine module for ORYX.

Encapsulates the logic for scoring keyword candidates:
- Metrics computation (relative interest, difficulty, CTR potential)
- Opportunity score calculation
- Naturalness and universal term penalties

This module is part of the Phase 3 architecture modularization.
"""

import logging
from typing import Dict, List, Optional, Set

from .metrics import (
    compute_metrics,
    opportunity_scores,
    generate_reference_phrases,
)
from .cluster import infer_intent


class ScoringEngine:
    """
    Scores keyword candidates based on multiple factors.
    
    Computes metrics and opportunity scores for keywords,
    applying naturalness scoring and universal term penalties.
    
    Usage:
        engine = ScoringEngine(
            business_goals="leads",
            geo="ae",
            niche="contracting",
        )
        scored = engine.score_candidates(
            candidates, clusters, freq, questions,
            documents=cleaned_texts,
            seed_topic="villa construction",
        )
    """
    
    def __init__(
        self,
        business_goals: str = "traffic",
        geo: str = "global",
        niche: Optional[str] = None,
        commercial_weight: float = 0.25,
        use_naturalness: bool = True,
        use_universal_penalty: bool = True,
        competitors: Optional[List[str]] = None,
        intent_rules: Optional[Dict] = None,
    ):
        """
        Initialize the scoring engine.
        
        Args:
            business_goals: Business goals ("traffic", "leads", etc.)
            geo: Geographic target
            niche: Optional niche for specialized scoring
            commercial_weight: Weight for commercial value boost
            use_naturalness: Whether to apply naturalness scoring
            use_universal_penalty: Whether to penalize universal terms
            competitors: List of competitor domains for intent detection
            intent_rules: Custom intent classification rules
        """
        self.business_goals = business_goals
        self.geo = geo
        self.niche = niche
        self.commercial_weight = commercial_weight
        self.use_naturalness = use_naturalness
        self.use_universal_penalty = use_universal_penalty
        self.competitors = competitors or []
        self.intent_rules = intent_rules
    
    def score_candidates(
        self,
        candidates: List[str],
        clusters: Dict[str, List[str]],
        freq: Dict[str, int],
        questions: Set[str],
        provider: str = "none",
        validated_keywords: Optional[Dict[str, bool]] = None,
        total_docs: int = 0,
        documents: Optional[List[str]] = None,
        seed_topic: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        Score all keyword candidates.
        
        Args:
            candidates: List of keyword candidates
            clusters: Cluster assignments
            freq: Document frequency counts
            questions: Set of question-style keywords
            provider: SERP provider used
            validated_keywords: Autocomplete validation results
            total_docs: Total documents in corpus
            documents: Document texts for universal term detection
            seed_topic: Seed topic for dynamic reference phrases
            
        Returns:
            Dict with 'metrics', 'intents', and 'opportunity_scores' keys
        """
        # Infer intent for all candidates
        intents = self._classify_intents(candidates)
        
        # Compute base metrics
        metrics = compute_metrics(
            candidates,
            clusters,
            intents,
            freq,
            questions,
            provider,
            serp_total_results=None,
            validated_keywords=validated_keywords or {},
            total_docs=total_docs,
        )
        
        # Generate dynamic reference phrases if seed_topic provided
        reference_phrases = None
        if seed_topic:
            reference_phrases = generate_reference_phrases(seed_topic)
        
        # Calculate opportunity scores
        opp_scores = opportunity_scores(
            metrics,
            intents,
            self.business_goals,
            geo=self.geo,
            niche=self.niche,
            commercial_weight=self.commercial_weight,
            documents=documents,
            use_naturalness=self.use_naturalness,
            use_universal_penalty=self.use_universal_penalty,
            reference_phrases=reference_phrases,
        )
        
        return {
            "metrics": metrics,
            "intents": intents,
            "opportunity_scores": opp_scores,
        }
    
    def _classify_intents(self, candidates: List[str]) -> Dict[str, str]:
        """Classify search intent for all candidates."""
        return {
            kw: infer_intent(kw, self.competitors, self.intent_rules)
            for kw in candidates
        }
    
    def get_top_candidates(
        self,
        scored_results: Dict[str, Dict],
        max_per_cluster: int = 12,
        clusters: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict]:
        """
        Get top candidates per cluster based on opportunity score.
        
        Args:
            scored_results: Results from score_candidates()
            max_per_cluster: Maximum keywords to return per cluster
            clusters: Cluster assignments
            
        Returns:
            List of top keyword dicts with scores
        """
        metrics = scored_results["metrics"]
        intents = scored_results["intents"]
        opp_scores = scored_results["opportunity_scores"]
        
        if not clusters:
            # Return all candidates sorted by opportunity score
            return sorted(
                [
                    {
                        "keyword": kw,
                        "intent": intents.get(kw, "informational"),
                        "opportunity_score": opp_scores.get(kw, 0.0),
                        **metrics.get(kw, {}),
                    }
                    for kw in metrics.keys()
                ],
                key=lambda x: (-x["opportunity_score"], x["keyword"]),
            )
        
        # Select top per cluster
        items = []
        for cluster_name, keywords in clusters.items():
            sorted_kws = sorted(
                keywords,
                key=lambda k: (
                    opp_scores.get(k, 0),
                    metrics.get(k, {}).get("relative_interest", 0),
                ),
                reverse=True,
            )
            
            for kw in sorted_kws[:max_per_cluster]:
                m = metrics.get(kw, {})
                items.append({
                    "keyword": kw.lower(),
                    "cluster": cluster_name.lower(),
                    "intent": intents.get(kw, "informational"),
                    "opportunity_score": float(opp_scores.get(kw, 0.0)),
                    "relative_interest": float(m.get("relative_interest", 0.0)),
                    "difficulty": float(m.get("difficulty", 0.0)),
                    "ctr_potential": float(m.get("ctr_potential", 1.0)),
                    "validated": bool(m.get("validated", False)),
                    "estimated": bool(m.get("estimated", True)),
                })
        
        # Sort globally by opportunity score
        return sorted(
            items,
            key=lambda x: (-x["opportunity_score"], -x["relative_interest"], x["keyword"]),
        )
