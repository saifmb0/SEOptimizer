from jsonschema import Draft7Validator

COMPACT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "keyword": {"type": "string"},
            "cluster": {"type": "string"},
            "parent_topic": {"type": "string"},
            "intent": {"type": "string", "enum": [
                "informational", "commercial", "transactional", "navigational",
                # GEO-centric intents (added in Week 2)
                "direct_answer", "complex_research", "comparative", "local"
            ]},
            "funnel_stage": {"type": "string", "enum": ["TOFU", "MOFU", "BOFU"]},
            "search_volume": {"type": "number", "minimum": 0},
            "difficulty": {"type": "number", "minimum": 0, "maximum": 1},
            "ctr_potential": {"type": "number", "minimum": 0, "maximum": 1},
            "serp_features": {"type": "array", "items": {"type": "string"}},
            "estimated": {"type": "boolean"},
            "validated": {"type": "boolean"},
            "opportunity_score": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": [
            "keyword", "cluster", "parent_topic", "intent", "funnel_stage",
            "search_volume", "difficulty", "ctr_potential", "serp_features",
            "estimated", "validated", "opportunity_score"
        ]
    }
}

validator = Draft7Validator(COMPACT_SCHEMA)

def validate_items(items):
    errors = list(validator.iter_errors(items))
    if errors:
        messages = []
        for e in errors:
            messages.append(f"Validation error at {list(e.path)}: {e.message}")
        raise ValueError("; ".join(messages))
    return True
