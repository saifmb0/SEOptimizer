"""
Configuration loading, merging, and validation for ORYX.

Uses Pydantic V2 for strict validation with enterprise-grade error messages.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


# ============================================================================
# Pydantic Configuration Models
# ============================================================================

class IntentRulesConfig(BaseModel):
    """Intent classification rules mapping intent types to keyword patterns."""
    
    model_config = ConfigDict(extra="forbid", frozen=True)
    
    informational: List[str] = Field(
        default=["who", "what", "why", "how", "guide", "tutorial", "tips", "checklist", "template"],
        description="Keywords triggering informational intent"
    )
    commercial: List[str] = Field(
        default=["best", "top", "review", "compare", "vs", "alternatives", "pricing"],
        description="Keywords triggering commercial intent"
    )
    transactional: List[str] = Field(
        default=["buy", "discount", "coupon", "deal", "near me", "order", "book", "hire"],
        description="Keywords triggering transactional intent"
    )
    navigational: List[str] = Field(
        default=[],
        description="Keywords triggering navigational intent"
    )
    # GEO-centric intents (for AI search optimization)
    direct_answer: List[str] = Field(
        default=["what is", "define", "meaning of", "cost of"],
        description="Keywords suitable for AI direct answers"
    )
    local: List[str] = Field(
        default=["near me", "in dubai", "in abu dhabi", "in uae", "dubai", "abu dhabi"],
        description="Keywords triggering local intent"
    )


class NlpConfig(BaseModel):
    """NLP processing configuration."""
    
    model_config = ConfigDict(extra="forbid", frozen=True)
    
    ngram_min_df: int = Field(default=2, ge=1, description="Minimum document frequency for n-grams")
    ngram_range: Tuple[int, int] = Field(default=(1, 3), description="Range of n-gram sizes")
    top_terms_per_doc: int = Field(default=10, ge=1, description="Top terms to extract per document")
    
    @field_validator("ngram_range")
    @classmethod
    def validate_ngram_range(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        if v[0] < 1 or v[1] < v[0]:
            raise ValueError(f"ngram_range must be (min, max) where min >= 1 and max >= min, got {v}")
        return v


class ScrapeConfig(BaseModel):
    """Web scraping configuration with UAE-specific defaults."""
    
    model_config = ConfigDict(extra="forbid", frozen=True)
    
    timeout: int = Field(default=30, ge=1, le=120, description="Request timeout in seconds")
    retries: int = Field(default=3, ge=0, le=10, description="Number of retry attempts")
    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        description="User agent string for requests"
    )
    max_serp_results: int = Field(default=10, ge=1, le=100, description="Max SERP results to fetch")
    cache_enabled: bool = Field(default=True, description="Enable request caching")
    cache_dir: str = Field(default=".cache/scrape", description="Cache directory path")
    
    # UAE/Gulf proxy configuration
    proxy_enabled: bool = Field(default=False, description="Enable proxy rotation")
    proxy_urls: List[str] = Field(default=[], description="List of proxy URLs")
    
    # Rate limiting for respectful scraping
    min_delay_ms: int = Field(default=500, ge=0, description="Minimum delay between requests (ms)")
    max_delay_ms: int = Field(default=2000, ge=0, description="Maximum delay between requests (ms)")
    
    @model_validator(mode="after")
    def validate_delays(self) -> "ScrapeConfig":
        if self.max_delay_ms < self.min_delay_ms:
            raise ValueError(f"max_delay_ms ({self.max_delay_ms}) must be >= min_delay_ms ({self.min_delay_ms})")
        return self


class ClusterConfig(BaseModel):
    """Clustering algorithm configuration."""
    
    model_config = ConfigDict(extra="forbid", frozen=True)
    
    max_clusters: int = Field(default=15, ge=1, le=100, description="Maximum number of clusters")
    max_keywords_per_cluster: int = Field(default=20, ge=1, description="Max keywords per cluster")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    use_silhouette: bool = Field(default=True, description="Use silhouette scoring for optimal k")
    silhouette_k_range: Tuple[int, int] = Field(default=(3, 15), description="Range for silhouette k search")
    
    @field_validator("silhouette_k_range")
    @classmethod
    def validate_k_range(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        if v[0] < 2 or v[1] <= v[0]:
            raise ValueError(f"silhouette_k_range must be (min, max) where min >= 2 and max > min, got {v}")
        return v


class LlmConfig(BaseModel):
    """LLM provider configuration."""
    
    model_config = ConfigDict(extra="forbid", frozen=True)
    
    provider: Literal["auto", "gemini", "openai", "anthropic", "none"] = Field(
        default="auto", description="LLM provider to use"
    )
    max_expansion_results: int = Field(default=10, ge=1, le=50, description="Max keyword expansion results")
    model: Optional[str] = Field(default=None, description="Specific model name (e.g., 'gemini-1.5-pro')")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=2048, ge=100, le=16384, description="Max tokens per request")


class OutputConfig(BaseModel):
    """Output format configuration."""
    
    model_config = ConfigDict(extra="forbid", frozen=True)
    
    format: Literal["json", "csv", "xlsx"] = Field(default="json", description="Output file format")
    pretty_print: bool = Field(default=True, description="Pretty-print JSON output")
    include_metadata: bool = Field(default=True, description="Include pipeline metadata in output")


class GeoConfig(BaseModel):
    """Geographic/locale configuration for UAE market."""
    
    model_config = ConfigDict(extra="forbid", frozen=True)
    
    region: str = Field(default="ae", description="Target region code")
    locale: str = Field(default="en-AE", description="Locale for search queries")
    emirates: List[str] = Field(
        default=["Abu Dhabi", "Dubai", "Sharjah", "Ajman", "RAK", "Fujairah", "UAQ"],
        description="Target emirates list"
    )
    primary_emirate: str = Field(default="Abu Dhabi", description="Primary target emirate")
    currency: str = Field(default="AED", description="Currency code")
    bilingual: bool = Field(default=True, description="Enable Arabic-English bilingual processing")


class KeywordLabConfig(BaseModel):
    """
    Complete ORYX configuration schema.
    
    All fields have sensible UAE-focused defaults.
    Use load_config() to get a fully merged configuration.
    
    Example:
        >>> config = load_config("config.yaml")
        >>> clusters = config.cluster.max_clusters
    """
    
    model_config = ConfigDict(extra="allow", frozen=False)  # Allow extra for flexibility
    
    intent_rules: IntentRulesConfig = Field(default_factory=IntentRulesConfig)
    question_prefixes: List[str] = Field(
        default=["how", "what", "best", "vs", "for", "near me", "beginner", "advanced", 
                 "guide", "checklist", "template", "why", "cost", "price", "cheap"],
        description="Prefixes for question generation"
    )
    nlp: NlpConfig = Field(default_factory=NlpConfig)
    scrape: ScrapeConfig = Field(default_factory=ScrapeConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    geo: GeoConfig = Field(default_factory=GeoConfig)
    
    # Niche-specific metadata (from presets)
    niche: Optional[str] = Field(default=None, description="Target niche (e.g., 'contracting')")
    preset_name: Optional[str] = Field(default=None, description="Loaded preset name")


# ============================================================================
# Configuration Loading Functions
# ============================================================================

class ConfigValidationError(Exception):
    """Raised when config validation fails with detailed Pydantic errors."""
    
    def __init__(self, errors: List[str]):
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_default_config() -> Dict[str, Any]:
    """Load the bundled default configuration."""
    default_path = Path(__file__).parent / "default_config.yaml"
    if default_path.exists():
        return yaml.safe_load(default_path.read_text()) or {}
    return {}


def load_preset(preset_path: str) -> Dict[str, Any]:
    """
    Load a niche-specific preset file.
    
    Presets are YAML files containing market/locale-specific configurations
    such as geo, language, niche terminology, and scoring weights.
    
    Args:
        preset_path: Path to preset YAML file (e.g., "presets/contracting_ae.yaml")
        
    Returns:
        Preset configuration dictionary
        
    Raises:
        FileNotFoundError: If preset file doesn't exist
        ConfigValidationError: If preset is malformed
        
    Example:
        >>> preset = load_preset("presets/contracting_ae.yaml")
        >>> print(preset["geo"]["region"])  # "ae"
        >>> print(preset["niche"])  # "contracting"
    """
    preset_file = Path(preset_path)
    if not preset_file.exists():
        raise FileNotFoundError(f"Preset file not found: {preset_path}")
    
    try:
        preset = yaml.safe_load(preset_file.read_text()) or {}
        preset["preset_name"] = preset_file.stem
        logging.info(f"Loaded preset from {preset_path}")
        return preset
    except yaml.YAMLError as e:
        raise ConfigValidationError([f"Failed to parse preset {preset_path}: {e}"])


def _normalize_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize flat preset config values to nested Pydantic structure.
    
    Presets use flat values like `geo: "ae"` for user convenience,
    but Pydantic expects nested structures like `geo: {region: "ae"}`.
    
    This function converts:
    - geo: "ae" -> geo: {region: "ae"}
    - max_clusters: 12 -> cluster: {max_clusters: 12}
    - etc.
    """
    result = config_dict.copy()
    
    # Normalize geo: string -> geo: {region: string}
    if "geo" in result and isinstance(result["geo"], str):
        geo_str = result["geo"]
        result["geo"] = {"region": geo_str}
    
    # Normalize flat cluster settings
    cluster_keys = ["max_clusters", "max_keywords_per_cluster", "random_state", 
                    "use_silhouette", "silhouette_k_range"]
    for key in cluster_keys:
        if key in result and "cluster" not in result:
            result["cluster"] = {}
        if key in result:
            if "cluster" not in result:
                result["cluster"] = {}
            result["cluster"][key] = result.pop(key)
    
    # Normalize flat llm settings
    llm_keys = ["provider", "model", "temperature", "max_tokens", "max_expansion_results"]
    for key in llm_keys:
        if key in result and key not in ["niche", "preset_name"]:
            # Only move provider if it looks like an LLM provider value
            if key == "provider" and result[key] in ["auto", "gemini", "openai", "anthropic", "none"]:
                if "llm" not in result:
                    result["llm"] = {}
                result["llm"][key] = result.pop(key)
    
    # Normalize flat scrape settings
    scrape_keys = ["timeout", "retries", "user_agent", "max_serp_results", 
                   "proxy_enabled", "proxy_urls", "min_delay_ms", "max_delay_ms"]
    for key in scrape_keys:
        if key in result:
            if "scrape" not in result:
                result["scrape"] = {}
            result["scrape"][key] = result.pop(key)
    
    return result


def load_config(
    config_path: Optional[str] = None, 
    preset_path: Optional[str] = None,
    validate: bool = True
) -> KeywordLabConfig:
    """
    Load configuration, merging preset and user config with defaults.
    
    Priority (highest to lowest):
    1. User config file (explicit --config or ./config.yaml)
    2. Preset file (--preset for niche-specific settings)
    3. Default config bundled with package
    
    Args:
        config_path: Optional path to user config file
        preset_path: Optional path to niche preset file
        validate: Whether to validate config (always True with Pydantic)
        
    Returns:
        Validated KeywordLabConfig instance
        
    Raises:
        ConfigValidationError: If config is invalid
        
    Example:
        >>> config = load_config(preset_path="presets/contracting_ae.yaml")
        >>> print(config.geo.region)  # "ae"
        >>> print(config.niche)  # "contracting"
    """
    # Start with defaults
    config_dict = load_default_config()
    
    # Merge preset if provided
    if preset_path:
        preset = load_preset(preset_path)
        config_dict = _deep_merge(config_dict, preset)
        logging.debug(f"Merged preset from {preset_path}")
    
    # Determine user config path
    user_config_path = Path(config_path) if config_path else Path("config.yaml")
    
    # Merge user config if it exists
    if user_config_path.exists():
        try:
            user_config = yaml.safe_load(user_config_path.read_text()) or {}
            config_dict = _deep_merge(config_dict, user_config)
            logging.debug(f"Loaded user config from {user_config_path}")
        except Exception as e:
            logging.warning(f"Failed to read config {user_config_path}: {e}")
    
    # Normalize flat preset values to nested Pydantic structure
    config_dict = _normalize_config(config_dict)
    
    # Validate and create Pydantic model
    try:
        return KeywordLabConfig(**config_dict)
    except Exception as e:
        # Convert Pydantic validation errors to our format
        from pydantic import ValidationError
        if isinstance(e, ValidationError):
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            raise ConfigValidationError(errors)
        raise


def get_intent_rules(config) -> Dict[str, List[str]]:
    """
    Get intent rules from config as a dictionary.
    
    Handles both Pydantic KeywordLabConfig and legacy dict configs.
    """
    if config is None:
        return IntentRulesConfig().model_dump()
    if isinstance(config, KeywordLabConfig):
        return config.intent_rules.model_dump()
    if isinstance(config, dict):
        rules = config.get("intent_rules", {})
        if isinstance(rules, IntentRulesConfig):
            return rules.model_dump()
        return rules if rules else IntentRulesConfig().model_dump()
    return IntentRulesConfig().model_dump()


def get_question_prefixes(config) -> List[str]:
    """
    Get question prefixes from config.
    
    Handles both Pydantic KeywordLabConfig and legacy dict configs.
    """
    default_prefixes = ["how", "what", "best", "vs", "for", "near me", "beginner", "advanced", 
                        "guide", "checklist", "template", "why", "cost", "price", "cheap"]
    if config is None:
        return default_prefixes
    if isinstance(config, KeywordLabConfig):
        return config.question_prefixes
    if isinstance(config, dict):
        return config.get("question_prefixes", default_prefixes)
    return default_prefixes


def config_to_dict(config) -> Dict[str, Any]:
    """
    Convert config (Pydantic or dict) to a nested dictionary.
    
    This allows the pipeline to use dict-style access for backwards compatibility.
    """
    if config is None:
        return {}
    if isinstance(config, KeywordLabConfig):
        return config.model_dump()
    if isinstance(config, dict):
        return config
    return {}
