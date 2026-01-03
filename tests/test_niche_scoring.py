"""Tests for niche-adaptive naturalness scoring."""

import pytest

from oryx.metrics import (
    generate_reference_phrases,
    calculate_naturalness_score,
    HAS_SENTENCE_TRANSFORMERS,
)


class TestNicheAdaptation:
    """Test suite for dynamic, niche-adaptive scoring."""

    def test_generate_reference_phrases_construction(self):
        """Reference phrases should be generated for construction niche."""
        phrases = generate_reference_phrases("villa construction")
        
        assert len(phrases) >= 5, "Should generate multiple reference phrases"
        assert any("villa construction" in p.lower() for p in phrases)
        assert any("cost" in p.lower() for p in phrases)
        assert any("pricing" in p.lower() for p in phrases)

    def test_generate_reference_phrases_saas(self):
        """Reference phrases should be generated for SaaS niche."""
        phrases = generate_reference_phrases("SaaS CRM")
        
        assert len(phrases) >= 5, "Should generate multiple reference phrases"
        assert any("saas crm" in p.lower() for p in phrases)
        assert any("reviews" in p.lower() or "comparison" in p.lower() for p in phrases)

    def test_generate_reference_phrases_healthcare(self):
        """Reference phrases should be generated for healthcare niche."""
        phrases = generate_reference_phrases("dental implants")
        
        assert len(phrases) >= 5, "Should generate multiple reference phrases"
        assert any("dental implants" in p.lower() for p in phrases)

    def test_reference_phrases_are_unique_per_niche(self):
        """Different niches should produce different reference phrases."""
        construction_phrases = generate_reference_phrases("home renovation")
        tech_phrases = generate_reference_phrases("cloud computing")
        
        # Convert to sets for comparison
        construction_set = set(construction_phrases)
        tech_set = set(tech_phrases)
        
        # Should have no overlap (different topics)
        assert construction_set != tech_set, "Different niches should produce different phrases"
        assert len(construction_set & tech_set) == 0, "No overlap expected"

    def test_empty_seed_topic(self):
        """Empty seed topic should still generate phrases."""
        phrases = generate_reference_phrases("")
        
        # Should still generate generic phrases
        assert len(phrases) >= 5

    @pytest.mark.skipif(
        not HAS_SENTENCE_TRANSFORMERS,
        reason="sentence-transformers not installed"
    )
    def test_naturalness_uses_custom_references(self):
        """Naturalness scoring should accept custom reference phrases."""
        custom_phrases = [
            "cloud computing services",
            "aws migration",
            "kubernetes deployment",
        ]
        
        # This should not raise an error
        score = calculate_naturalness_score(
            "kubernetes tutorial",
            reference_phrases=custom_phrases
        )
        
        assert 0.0 <= score <= 1.0, "Score should be between 0 and 1"

    @pytest.mark.skipif(
        not HAS_SENTENCE_TRANSFORMERS,
        reason="sentence-transformers not installed"
    )
    def test_garbage_scores_low(self):
        """Garbage/nonsense phrases should score low."""
        garbage_phrases = [
            "en ae ar login menu",
            "cookie privacy terms footer",
            "template owners developers facility",
        ]
        
        for phrase in garbage_phrases:
            score = calculate_naturalness_score(phrase)
            assert score < 0.5, f"Garbage phrase should score low: {phrase} = {score}"

    @pytest.mark.skipif(
        not HAS_SENTENCE_TRANSFORMERS,
        reason="sentence-transformers not installed"
    )
    def test_natural_scores_high(self):
        """Natural keyword phrases should score higher."""
        natural_phrases = [
            "best home renovation services",
            "how to choose a contractor",
            "kitchen remodeling cost",
        ]
        
        for phrase in natural_phrases:
            score = calculate_naturalness_score(phrase)
            assert score > 0.3, f"Natural phrase should score reasonably: {phrase} = {score}"
