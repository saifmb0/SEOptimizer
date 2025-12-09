from oryx.schema import COMPACT_SCHEMA, validate_items

def test_schema_validation():
    sample = [
        {
            "keyword": "best coffee beans",
            "cluster": "cluster-0",
            "parent_topic": "coffee beans",
            "intent": "commercial",
            "funnel_stage": "MOFU",
            "relative_interest": 0.8,
            "difficulty": 0.4,
            "ctr_potential": 0.85,
            "serp_features": ["top_ads", "shopping_results"],
            "estimated": True,
            "validated": True,
            "opportunity_score": 0.5,
        }
    ]
    
    # Action: Validate items and capture the result
    result = validate_items(sample)
    
    # Assertion 1: It should return a list (not True)
    assert isinstance(result, list), "Expected a list of KeywordData objects"
    
    # Assertion 2: The list should not be empty
    assert len(result) == 1
    
    # Assertion 3: Verify the data was correctly parsed into the Pydantic model
    # Note: Pydantic models support dot access
    assert result[0].keyword == "best coffee beans"
    assert result[0].opportunity_score == 0.5
    assert result[0].validated is True