import json
from keyword_lab.schema import COMPACT_SCHEMA, validate_items
from jsonschema import validate


def test_schema_validation():
    sample = [
        {
            "keyword": "best coffee beans",
            "cluster": "cluster-0",
            "intent": "commercial",
            "funnel_stage": "MOFU",
            "search_volume": 200,
            "difficulty": 40,
            "estimated": True,
            "opportunity_score": 0.5,
        }
    ]
    validate(instance=sample, schema=COMPACT_SCHEMA)
    assert validate_items(sample) is True
