"""
Tests for UAE/Gulf market specialization features.

Tests commercial scoring, SERP CTR adjustment, and niche-specific logic.
"""
import pytest

from keyword_lab.metrics import (
    commercial_value,
    ctr_potential,
    estimate_serp_features,
    compute_metrics,
    opportunity_scores,
    COMMERCIAL_TRIGGERS,
    TRANSACTIONAL_TRIGGERS,
    UAE_COMMERCIAL_TRIGGERS,
    SERP_FEATURE_CTR_IMPACT,
)


class TestCommercialValue:
    """Tests for commercial_value() CPC proxy heuristic."""
    
    def test_transactional_intent_high_value(self):
        """Transactional intent should have high commercial value."""
        score = commercial_value("buy villa dubai", intent="transactional")
        assert score >= 0.6, "Transactional intent should score >= 0.6"
    
    def test_informational_intent_low_value(self):
        """Informational intent should have low commercial value."""
        score = commercial_value("what is villa", intent="informational")
        assert score <= 0.4, "Informational intent should score <= 0.4"
    
    def test_commercial_triggers_boost(self):
        """Commercial trigger words should boost score."""
        base = commercial_value("villa", intent="informational")
        with_trigger = commercial_value("villa contractors", intent="informational")
        assert with_trigger > base, "Commercial triggers should boost score"
    
    def test_uae_geo_boost(self):
        """UAE geo should enable UAE-specific triggers."""
        base = commercial_value("fit out dubai", geo="us")
        uae = commercial_value("fit out dubai", geo="ae")
        assert uae > base, "UAE geo should boost UAE-specific terms"
    
    def test_contracting_niche_boost(self):
        """Contracting niche should boost construction terms."""
        base = commercial_value("villa renovation", niche=None)
        contracting = commercial_value("villa renovation", niche="contracting")
        assert contracting > base, "Contracting niche should boost construction terms"
    
    def test_score_bounds(self):
        """Score should be bounded 0.0-1.0."""
        # Stack all boosts
        score = commercial_value(
            "buy villa fit out contractors quote price near me",
            intent="transactional",
            geo="ae",
            niche="contracting",
        )
        assert 0.0 <= score <= 1.0, "Score should be bounded"
    
    def test_transactional_triggers(self):
        """Transactional triggers should significantly boost score."""
        base = commercial_value("villa", intent="commercial")
        with_trigger = commercial_value("get villa quote", intent="commercial")
        # Transactional triggers like "get...quote" add at least 0.1 boost
        assert with_trigger > base, "Transactional triggers should boost score"
    
    def test_b2b_triggers(self):
        """B2B triggers should boost commercial value."""
        base = commercial_value("software", intent="commercial")
        b2b = commercial_value("enterprise software solutions", intent="commercial")
        assert b2b > base, "B2B triggers should boost score"


class TestCTRPotential:
    """Tests for ctr_potential() SERP feature adjustment."""
    
    def test_no_features_full_ctr(self):
        """Keywords without SERP features should have high CTR potential."""
        # Generic long-tail unlikely to trigger features
        score = ctr_potential("specific niche contracting services dubai", intent="commercial")
        assert score >= 0.6, "Few SERP features should maintain high CTR"
    
    def test_how_to_reduces_ctr(self):
        """How-to queries often get featured snippets."""
        score = ctr_potential("how to renovate villa", intent="informational")
        assert score < 1.0, "Featured snippet queries should reduce CTR"
    
    def test_local_queries_reduced_ctr(self):
        """Local queries get map pack."""
        score = ctr_potential("contractors near me", intent="local")
        assert score < 0.9, "Local pack should reduce CTR"
    
    def test_commercial_queries_reduced_ctr(self):
        """Commercial queries get ads."""
        score = ctr_potential("buy office furniture", intent="transactional")
        assert score < 0.95, "Commercial queries attract ads"
    
    def test_ctr_floor(self):
        """CTR should never go below 0.2."""
        # Stack many feature triggers
        score = ctr_potential(
            "how to buy cheap furniture near me tutorial",
            intent="transactional"
        )
        assert score >= 0.2, "CTR should have floor of 0.2"
    
    def test_ctr_ceiling(self):
        """CTR should never exceed 1.0."""
        score = ctr_potential("xyz123", intent="navigational")
        assert score <= 1.0, "CTR should be capped at 1.0"


class TestEstimateSerpFeatures:
    """Tests for estimate_serp_features()."""
    
    def test_how_to_triggers_snippet(self):
        """How-to queries should trigger featured snippet."""
        features = estimate_serp_features("how to paint villa", intent="informational")
        assert "featured_snippet" in features, "How-to should trigger featured_snippet"
    
    def test_what_is_triggers_knowledge_panel(self):
        """What-is queries should trigger knowledge panel."""
        features = estimate_serp_features("what is mep contractor", intent="informational")
        assert "knowledge_panel" in features, "What-is should trigger knowledge_panel"
    
    def test_near_me_triggers_local_pack(self):
        """Near-me queries should trigger local pack."""
        features = estimate_serp_features("contractors near me", intent="local")
        assert "local_pack" in features, "Near-me should trigger local_pack"
    
    def test_buy_triggers_shopping(self):
        """Buy queries should trigger shopping results."""
        features = estimate_serp_features("buy office furniture online", intent="transactional")
        assert "shopping_results" in features, "Buy should trigger shopping_results"
    
    def test_commercial_gets_ads(self):
        """Commercial intent should trigger ads."""
        features = estimate_serp_features("office fit out companies", intent="commercial")
        assert "top_ads" in features, "Commercial queries should get ads"


class TestComputeMetrics:
    """Tests for compute_metrics() with new CTR fields."""
    
    def test_metrics_include_ctr_potential(self):
        """Metrics should include ctr_potential field."""
        keywords = ["villa renovation dubai"]
        clusters = {"cluster-0": keywords}
        intents = {"villa renovation dubai": "commercial"}
        freq = {"villa renovation dubai": 5}
        
        metrics = compute_metrics(
            keywords, clusters, intents, freq, set(), "none"
        )
        
        assert "ctr_potential" in metrics["villa renovation dubai"]
        assert 0.0 <= metrics["villa renovation dubai"]["ctr_potential"] <= 1.0
    
    def test_metrics_include_serp_features(self):
        """Metrics should include serp_features field."""
        keywords = ["how to renovate villa"]
        clusters = {"cluster-0": keywords}
        intents = {"how to renovate villa": "informational"}
        freq = {"how to renovate villa": 3}
        
        metrics = compute_metrics(
            keywords, clusters, intents, freq, set(), "none"
        )
        
        assert "serp_features" in metrics["how to renovate villa"]
        assert isinstance(metrics["how to renovate villa"]["serp_features"], list)


class TestOpportunityScores:
    """Tests for opportunity_scores() with commercial and CTR adjustments."""
    
    def test_ctr_reduces_opportunity(self):
        """Lower CTR potential should reduce opportunity score."""
        # Create metrics with different CTR potentials
        metrics = {
            "high_ctr_kw": {"search_volume": 0.5, "difficulty": 0.3, "ctr_potential": 0.9},
            "low_ctr_kw": {"search_volume": 0.5, "difficulty": 0.3, "ctr_potential": 0.4},
        }
        intents = {"high_ctr_kw": "commercial", "low_ctr_kw": "commercial"}
        
        scores = opportunity_scores(metrics, intents, "traffic")
        
        assert scores["high_ctr_kw"] > scores["low_ctr_kw"], \
            "Higher CTR potential should mean higher opportunity"
    
    def test_lead_goals_boost_commercial(self):
        """Lead-focused goals should boost commercial keywords."""
        metrics = {
            "commercial_kw": {"search_volume": 0.5, "difficulty": 0.3, "ctr_potential": 0.8},
            "info_kw": {"search_volume": 0.5, "difficulty": 0.3, "ctr_potential": 0.8},
        }
        intents = {"commercial_kw": "transactional", "info_kw": "informational"}
        
        # With lead goals
        lead_scores = opportunity_scores(metrics, intents, "leads", niche="contracting")
        
        assert lead_scores["commercial_kw"] > lead_scores["info_kw"], \
            "Lead goals should prioritize commercial keywords"
    
    def test_niche_scoring(self):
        """Niche parameter should affect scoring."""
        metrics = {
            "fit out dubai": {"search_volume": 0.5, "difficulty": 0.3, "ctr_potential": 0.7},
        }
        intents = {"fit out dubai": "commercial"}
        
        # Compare with and without niche
        no_niche = opportunity_scores(metrics, intents, "leads", geo="ae", niche=None)
        with_niche = opportunity_scores(metrics, intents, "leads", geo="ae", niche="contracting")
        
        assert with_niche["fit out dubai"] >= no_niche["fit out dubai"], \
            "Contracting niche should boost fit-out keywords"
    
    def test_ctr_adjustment_can_be_disabled(self):
        """CTR adjustment should be disableable."""
        metrics = {
            "kw": {"search_volume": 0.5, "difficulty": 0.3, "ctr_potential": 0.5},
        }
        intents = {"kw": "commercial"}
        
        with_ctr = opportunity_scores(metrics, intents, "traffic", use_ctr_adjustment=True)
        without_ctr = opportunity_scores(metrics, intents, "traffic", use_ctr_adjustment=False)
        
        assert without_ctr["kw"] >= with_ctr["kw"], \
            "Disabling CTR adjustment should increase score"


class TestUAESpecificLogic:
    """Tests for UAE/Gulf market specific features."""
    
    def test_uae_triggers_exist(self):
        """UAE commercial triggers should exist."""
        assert "fit out" in UAE_COMMERCIAL_TRIGGERS
        assert "mep" in UAE_COMMERCIAL_TRIGGERS
        assert "turnkey" in UAE_COMMERCIAL_TRIGGERS
    
    def test_serp_triggers_include_dubai(self):
        """SERP feature triggers should include UAE locations."""
        from keyword_lab.metrics import SERP_FEATURE_TRIGGERS
        local_triggers = SERP_FEATURE_TRIGGERS.get("local_pack", [])
        assert "in dubai" in local_triggers, "Dubai should trigger local pack"
        assert "in abu dhabi" in local_triggers, "Abu Dhabi should trigger local pack"
    
    def test_uae_geo_codes(self):
        """UAE and Gulf geo codes should enable special handling."""
        gulf_geos = ["ae", "sa", "qa", "kw", "bh", "om"]
        for geo in gulf_geos:
            score = commercial_value("fit out project", geo=geo)
            us_score = commercial_value("fit out project", geo="us")
            assert score > us_score, f"Geo {geo} should boost UAE terms"
