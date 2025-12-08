"""Tests for NLP negative sentiment filtering."""

import pytest

from keyword_lab.nlp import (
    contains_negative_sentiment,
    NEGATIVE_SENTIMENT_TERMS,
)


class TestNegativeSentimentFilter:
    """Test the negative sentiment shield."""
    
    def test_detects_scam_keywords(self):
        """Keywords with 'scam' should be flagged."""
        assert contains_negative_sentiment("is hagcc a scam")
        assert contains_negative_sentiment("contractor scams dubai")
        assert contains_negative_sentiment("scammer companies uae")
    
    def test_detects_fraud_keywords(self):
        """Keywords with 'fraud' should be flagged."""
        assert contains_negative_sentiment("fraud contractors abu dhabi")
        assert contains_negative_sentiment("fraudulent construction companies")
    
    def test_detects_complaint_keywords(self):
        """Keywords with 'complaint' should be flagged."""
        assert contains_negative_sentiment("hagcc complaints")
        assert contains_negative_sentiment("complaint about contractor")
    
    def test_allows_normal_keywords(self):
        """Normal keywords should not be flagged."""
        assert not contains_negative_sentiment("best contractors dubai")
        assert not contains_negative_sentiment("villa renovation abu dhabi")
        assert not contains_negative_sentiment("commercial building contractor")
    
    def test_case_insensitive(self):
        """Detection should be case insensitive."""
        assert contains_negative_sentiment("SCAM contractors")
        assert contains_negative_sentiment("Fraud company")
    
    def test_terms_list_not_empty(self):
        """NEGATIVE_SENTIMENT_TERMS should have entries."""
        assert len(NEGATIVE_SENTIMENT_TERMS) >= 10
        assert "scam" in NEGATIVE_SENTIMENT_TERMS
        assert "fraud" in NEGATIVE_SENTIMENT_TERMS
