"""Tests for SEO metadata auto-generation."""

import pytest

from keyword_lab.io import generate_seo_metadata, _suggest_schema_type


class TestMetadataGeneration:
    """Test SEO metadata generation."""
    
    @pytest.fixture
    def sample_keywords(self):
        return [
            {
                "keyword": "best contractor dubai",
                "cluster": "contractors",
                "parent_topic": "contracting services",
                "opportunity_score": 0.9,
                "intent": "transactional",
            },
            {
                "keyword": "villa contractor abu dhabi",
                "cluster": "contractors",
                "parent_topic": "contracting services",
                "opportunity_score": 0.7,
                "intent": "transactional",
            },
            {
                "keyword": "how to find contractor",
                "cluster": "guides",
                "parent_topic": "contractor guides",
                "opportunity_score": 0.6,
                "intent": "informational",
            },
        ]
    
    def test_generates_metadata_for_clusters(self, sample_keywords):
        """Should generate metadata for each unique cluster."""
        metadata = generate_seo_metadata(sample_keywords)
        
        assert len(metadata) == 2  # contractors and guides
        
        clusters = [m["cluster"] for m in metadata]
        assert "contractors" in clusters
        assert "guides" in clusters
    
    def test_title_length(self, sample_keywords):
        """Titles should be within SEO best practice (50-60 chars)."""
        metadata = generate_seo_metadata(sample_keywords)
        
        for m in metadata:
            title = m.get("title", "")
            # Allow some flexibility but should be under 70
            assert len(title) <= 70, f"Title too long: {title}"
    
    def test_description_length(self, sample_keywords):
        """Descriptions should be within SEO best practice (150-160 chars)."""
        metadata = generate_seo_metadata(sample_keywords)
        
        for m in metadata:
            desc = m.get("description", "")
            assert len(desc) <= 160, f"Description too long: {desc}"
    
    def test_primary_keyword_is_highest_opportunity(self, sample_keywords):
        """Primary keyword should be the one with highest opportunity."""
        metadata = generate_seo_metadata(sample_keywords)
        
        contractors_meta = next(m for m in metadata if m["cluster"] == "contractors")
        # best contractor dubai has 0.9, villa contractor has 0.7
        assert contractors_meta["primary_keyword"] == "best contractor dubai"
    
    def test_includes_schema_suggestion(self, sample_keywords):
        """Should suggest appropriate schema type."""
        metadata = generate_seo_metadata(sample_keywords)
        
        for m in metadata:
            assert "suggested_schema" in m
            assert m["suggested_schema"] in ["LocalBusiness", "Service", "FAQPage", "Product", "WebPage"]


class TestSchemaTypeSuggestion:
    """Test schema type suggestion logic."""
    
    def test_service_cluster_suggests_localbusiness(self):
        """Service-related clusters should suggest LocalBusiness."""
        schema = _suggest_schema_type("transactional", "contractor services")
        assert schema == "LocalBusiness"
    
    def test_informational_suggests_faqpage(self):
        """Informational intent should suggest FAQPage."""
        schema = _suggest_schema_type("informational", "general questions")
        assert schema == "FAQPage"
    
    def test_product_cluster_suggests_product(self):
        """Product-related clusters should suggest Product schema."""
        schema = _suggest_schema_type("transactional", "building materials product")
        assert schema == "Product"
    
    def test_local_intent_suggests_localbusiness(self):
        """Local intent should suggest LocalBusiness."""
        schema = _suggest_schema_type("local", "near me searches")
        assert schema == "LocalBusiness"
