"""
Keyword data schema validation for ORYX (Keyword Lab).

Uses Pydantic V2 for strict validation with type coercion and custom validators.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


# ============================================================================
# Enum Definitions
# ============================================================================

class IntentType(str, Enum):
    """Search intent types including GEO-centric intents."""
    INFORMATIONAL = "informational"
    COMMERCIAL = "commercial"
    TRANSACTIONAL = "transactional"
    NAVIGATIONAL = "navigational"
    # GEO-centric intents (for AI search optimization)
    DIRECT_ANSWER = "direct_answer"
    COMPLEX_RESEARCH = "complex_research"
    COMPARATIVE = "comparative"
    LOCAL = "local"


class FunnelStage(str, Enum):
    """Marketing funnel stages."""
    TOFU = "TOFU"  # Top of Funnel (awareness)
    MOFU = "MOFU"  # Middle of Funnel (consideration)
    BOFU = "BOFU"  # Bottom of Funnel (decision)


# ============================================================================
# Keyword Data Model
# ============================================================================

class KeywordData(BaseModel):
    """
    Individual keyword data with strict validation.
    
    This schema ensures data integrity throughout the ORYX pipeline.
    """
    
    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for extensibility
        str_strip_whitespace=True,  # Auto-strip whitespace
        validate_assignment=True,  # Validate on attribute assignment
    )
    
    # Core keyword fields
    keyword: str = Field(..., min_length=1, max_length=500, description="The keyword phrase")
    cluster: str = Field(default="unclustered", description="Cluster/topic group")
    parent_topic: str = Field(default="", description="Parent topic for content hierarchy")
    
    # Intent classification
    intent: IntentType = Field(default=IntentType.INFORMATIONAL, description="Search intent type")
    funnel_stage: FunnelStage = Field(default=FunnelStage.TOFU, description="Marketing funnel stage")
    
    # Metrics (with validation bounds)
    search_volume: float = Field(default=0, ge=0, description="Monthly search volume")
    difficulty: float = Field(default=0.5, ge=0, le=1, description="Keyword difficulty (0-1)")
    ctr_potential: float = Field(default=0.5, ge=0, le=1, description="CTR potential (0-1)")
    opportunity_score: float = Field(default=0.5, ge=0, le=1, description="Opportunity score (0-1)")
    
    # SERP features
    serp_features: List[str] = Field(default_factory=list, description="SERP features present")
    
    # Data quality flags
    estimated: bool = Field(default=True, description="Whether metrics are estimated")
    validated: bool = Field(default=False, description="Whether data has been validated")
    
    # GEO-specific fields (for AI search optimization)
    info_gain_score: Optional[float] = Field(default=None, ge=0, le=1, description="Information gain score")
    geo_suitability: Optional[float] = Field(default=None, ge=0, le=1, description="GEO query suitability")
    
    # UAE localization fields
    uae_entities: List[str] = Field(default_factory=list, description="Detected UAE entities")
    emirate: Optional[str] = Field(default=None, description="Target emirate")
    
    @field_validator("keyword")
    @classmethod
    def clean_keyword(cls, v: str) -> str:
        """Clean and normalize keyword."""
        # Remove excessive whitespace
        v = " ".join(v.split())
        # Convert to lowercase for consistency
        return v.lower()
    
    @field_validator("cluster", "parent_topic")
    @classmethod
    def clean_topic(cls, v: str) -> str:
        """Clean topic/cluster names."""
        return " ".join(v.split()) if v else v
    
    @field_validator("serp_features", mode="before")
    @classmethod
    def ensure_serp_list(cls, v: Any) -> List[str]:
        """Ensure SERP features is a list."""
        if v is None:
            return []
        if isinstance(v, str):
            # Handle comma-separated strings
            return [s.strip() for s in v.split(",") if s.strip()]
        return list(v)
    
    @field_validator("uae_entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v: Any) -> List[str]:
        """Ensure UAE entities is a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return list(v)
    
    # NOTE: opportunity_score auto-calculation removed - pipeline explicitly sets this value
    # via metrics.opportunity_scores(). Auto-calculation caused issues when validating
    # data with explicitly-set opportunity_score values (e.g., in tests or external data).
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(exclude_none=True)


# ============================================================================
# Validation Functions
# ============================================================================

def validate_keyword(data: Dict[str, Any]) -> KeywordData:
    """
    Validate a single keyword dictionary.
    
    Args:
        data: Keyword data dictionary
        
    Returns:
        Validated KeywordData instance
        
    Raises:
        pydantic.ValidationError: If validation fails
    """
    return KeywordData(**data)


def validate_items(items: List[Dict[str, Any]]) -> List[KeywordData]:
    """
    Validate a list of keyword items.
    
    Args:
        items: List of keyword data dictionaries
        
    Returns:
        List of validated KeywordData instances
        
    Raises:
        ValueError: If validation fails with detailed error messages
        
    Example:
        >>> items = [{"keyword": "construction abu dhabi"}]
        >>> validated = validate_items(items)
        >>> print(validated[0].keyword)
        "construction abu dhabi"
    """
    from pydantic import ValidationError
    
    validated = []
    errors = []
    
    for i, item in enumerate(items):
        try:
            validated.append(KeywordData(**item))
        except ValidationError as e:
            for err in e.errors():
                loc = ".".join(str(l) for l in err["loc"])
                errors.append(f"Item {i}, {loc}: {err['msg']}")
    
    if errors:
        raise ValueError("; ".join(errors))
    
    return validated


def items_to_dicts(items: List[KeywordData]) -> List[Dict[str, Any]]:
    """
    Convert KeywordData instances back to dictionaries.
    
    Args:
        items: List of KeywordData instances
        
    Returns:
        List of dictionaries suitable for JSON serialization
    """
    return [item.to_dict() for item in items]


# ============================================================================
# Legacy Support (for backward compatibility with jsonschema usage)
# ============================================================================

# Legacy schema definition (kept for reference/migration)
COMPACT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "keyword": {"type": "string"},
            "cluster": {"type": "string"},
            "parent_topic": {"type": "string"},
            "intent": {"type": "string"},
            "funnel_stage": {"type": "string", "enum": ["TOFU", "MOFU", "BOFU"]},
            "search_volume": {"type": "number", "minimum": 0},
            "difficulty": {"type": "number", "minimum": 0, "maximum": 1},
            "ctr_potential": {"type": "number", "minimum": 0, "maximum": 1},
            "serp_features": {"type": "array", "items": {"type": "string"}},
            "estimated": {"type": "boolean"},
            "validated": {"type": "boolean"},
            "opportunity_score": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["keyword"]
    }
}
