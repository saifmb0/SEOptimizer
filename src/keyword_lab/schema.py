from jsonschema import Draft7Validator

COMPACT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "keyword": {"type": "string"},
            "cluster": {"type": "string"},
            "intent": {"type": "string", "enum": [
                "informational", "commercial", "transactional", "navigational"
            ]},
            "funnel_stage": {"type": "string", "enum": ["TOFU", "MOFU", "BOFU"]},
            "search_volume": {"type": "number", "minimum": 0, "maximum": 1},
            "difficulty": {"type": "number", "minimum": 0, "maximum": 1},
            "estimated": {"type": "boolean"},
            "opportunity_score": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": [
            "keyword", "cluster", "intent", "funnel_stage",
            "search_volume", "difficulty", "estimated", "opportunity_score"
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
