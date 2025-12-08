"""Tests for competitor gap analysis module."""

import pytest

from keyword_lab.analysis import (
    identify_competitor_gaps,
    calculate_gap_score,
    prioritize_quick_wins,
    generate_gap_report,
)


class TestGapIdentification:
    """Test competitor gap identification."""
    
    @pytest.fixture
    def sample_keywords(self):
        return [
            {"keyword": "best contractor dubai", "opportunity_score": 0.8, "difficulty": 0.3, "cluster": "contractors"},
            {"keyword": "villa renovation", "opportunity_score": 0.7, "difficulty": 0.4, "cluster": "renovation"},
            {"keyword": "cheap construction", "opportunity_score": 0.4, "difficulty": 0.8, "cluster": "budget"},
            {"keyword": "ac repair", "opportunity_score": 0.6, "difficulty": 0.2, "cluster": "services"},
        ]
    
    def test_finds_high_opportunity_gaps(self, sample_keywords):
        """Should find keywords with high opportunity."""
        gaps = identify_competitor_gaps(sample_keywords, opportunity_threshold=0.6)
        
        # Should include high-opp keywords
        gap_kws = [g["keyword"] for g in gaps]
        assert "best contractor dubai" in gap_kws
        assert "villa renovation" in gap_kws
        
        # Should exclude low-opp keywords
        assert "cheap construction" not in gap_kws
    
    def test_filters_high_difficulty(self, sample_keywords):
        """Should filter high difficulty keywords."""
        gaps = identify_competitor_gaps(
            sample_keywords, 
            opportunity_threshold=0.5, 
            difficulty_threshold=0.3
        )
        
        gap_kws = [g["keyword"] for g in gaps]
        # Only ac repair has low enough difficulty
        assert "ac repair" in gap_kws
    
    def test_with_competitor_data(self, sample_keywords):
        """Should boost keywords missing from competitors."""
        competitor_rankings = {
            "competitor1": {"villa renovation", "cheap construction"},
        }
        
        gaps = identify_competitor_gaps(
            sample_keywords,
            competitor_rankings=competitor_rankings,
            opportunity_threshold=0.6,
        )
        
        # Keywords not in competitor set should have competitor_missing in reason
        for gap in gaps:
            if gap["keyword"] == "best contractor dubai":
                assert "competitor_missing" in gap.get("gap_reason", "")


class TestGapScoring:
    """Test gap score calculation."""
    
    def test_basic_score_calculation(self):
        """Score should be opportunity * (1 - difficulty)."""
        score = calculate_gap_score(0.8, 0.2, competitor_missing=False)
        assert abs(score - 0.64) < 0.01  # 0.8 * 0.8 = 0.64
    
    def test_competitor_boost(self):
        """Competitor missing should boost score by 50%."""
        base_score = calculate_gap_score(0.8, 0.2, competitor_missing=False)
        boosted_score = calculate_gap_score(0.8, 0.2, competitor_missing=True)
        
        assert boosted_score == base_score * 1.5


class TestQuickWins:
    """Test quick win prioritization."""
    
    def test_finds_quick_wins(self):
        """Should find low-difficulty, decent-opportunity keywords."""
        keywords = [
            {"keyword": "kw1", "difficulty": 0.2, "opportunity_score": 0.6, "validated": True},
            {"keyword": "kw2", "difficulty": 0.8, "opportunity_score": 0.9, "validated": True},
            {"keyword": "kw3", "difficulty": 0.1, "opportunity_score": 0.5, "validated": False},
        ]
        
        quick_wins = prioritize_quick_wins(keywords, max_difficulty=0.3, min_opportunity=0.5)
        
        qw_kws = [q["keyword"] for q in quick_wins]
        assert "kw1" in qw_kws
        assert "kw3" in qw_kws
        assert "kw2" not in qw_kws  # Too difficult


class TestGapReport:
    """Test comprehensive gap report generation."""
    
    def test_generates_report_structure(self):
        """Report should have expected structure."""
        keywords = [
            {"keyword": "test kw", "opportunity_score": 0.7, "difficulty": 0.3, "cluster": "test"},
        ]
        
        report = generate_gap_report(keywords)
        
        assert "total_keywords" in report
        assert "gap_opportunities" in report
        assert "quick_wins" in report
        assert "recommendations" in report
        assert report["total_keywords"] == 1
