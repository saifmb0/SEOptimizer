"""Configuration loading, merging, and validation for Keyword Lab."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict, NotRequired

import yaml
from jsonschema import Draft7Validator, ValidationError


# ============================================================================
# Typed Configuration Dictionaries
# ============================================================================

class IntentRulesConfig(TypedDict, total=False):
    """Intent classification rules mapping intent types to keyword patterns."""
    informational: List[str]
    commercial: List[str]
    transactional: List[str]
    navigational: List[str]


class NlpConfig(TypedDict, total=False):
    """NLP processing configuration."""
    ngram_min_df: int
    ngram_range: Tuple[int, int]
    top_terms_per_doc: int


class ScrapeConfig(TypedDict, total=False):
    """Web scraping configuration."""
    timeout: int
    retries: int
    user_agent: str
    max_serp_results: int
    cache_enabled: bool
    cache_dir: str


class ClusterConfig(TypedDict, total=False):
    """Clustering algorithm configuration."""
    max_clusters: int
    max_keywords_per_cluster: int
    random_state: int
    use_silhouette: bool
    silhouette_k_range: Tuple[int, int]


class LlmConfig(TypedDict, total=False):
    """LLM provider configuration."""
    provider: str  # "auto", "gemini", "openai", "anthropic", "none"
    max_expansion_results: int
    model: Optional[str]


class OutputConfig(TypedDict, total=False):
    """Output format configuration."""
    format: str  # "json", "csv", "xlsx"
    pretty_print: bool


class KeywordLabConfig(TypedDict, total=False):
    """
    Complete Keyword Lab configuration schema.
    
    All fields are optional as they fall back to defaults.
    Use load_config() to get a fully merged configuration.
    
    Example:
        >>> config = load_config("config.yaml")
        >>> clusters = config.get("cluster", {}).get("max_clusters", 10)
    """
    intent_rules: IntentRulesConfig
    question_prefixes: List[str]
    nlp: NlpConfig
    scrape: ScrapeConfig
    cluster: ClusterConfig
    llm: LlmConfig
    output: OutputConfig


# JSON Schema for config validation
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "intent_rules": {
            "type": "object",
            "properties": {
                "informational": {"type": "array", "items": {"type": "string"}},
                "commercial": {"type": "array", "items": {"type": "string"}},
                "transactional": {"type": "array", "items": {"type": "string"}},
                "navigational": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
        "question_prefixes": {"type": "array", "items": {"type": "string"}},
        "nlp": {
            "type": "object",
            "properties": {
                "ngram_min_df": {"type": "integer", "minimum": 1},
                "ngram_range": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                "top_terms_per_doc": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
        "scrape": {
            "type": "object",
            "properties": {
                "timeout": {"type": "integer", "minimum": 1},
                "retries": {"type": "integer", "minimum": 0},
                "user_agent": {"type": "string"},
                "max_serp_results": {"type": "integer", "minimum": 1},
                "cache_enabled": {"type": "boolean"},
                "cache_dir": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "cluster": {
            "type": "object",
            "properties": {
                "max_clusters": {"type": "integer", "minimum": 1},
                "max_keywords_per_cluster": {"type": "integer", "minimum": 1},
                "random_state": {"type": "integer"},
                "use_silhouette": {"type": "boolean"},
                "silhouette_k_range": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
            },
            "additionalProperties": False,
        },
        "llm": {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "enum": ["auto", "gemini", "openai", "anthropic", "none"]},
                "max_expansion_results": {"type": "integer", "minimum": 1},
                "model": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        },
        "output": {
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["json", "csv", "xlsx"]},
                "pretty_print": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": True,  # Allow extra top-level keys for flexibility
}

_config_validator = Draft7Validator(CONFIG_SCHEMA)


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    
    def __init__(self, errors: List[str]):
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration against schema.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    for error in _config_validator.iter_errors(config):
        path = ".".join(str(p) for p in error.path) if error.path else "(root)"
        errors.append(f"{path}: {error.message}")
    return errors


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


def load_config(config_path: Optional[str] = None, validate: bool = True) -> KeywordLabConfig:
    """
    Load configuration, merging user config with defaults.
    
    Priority (highest to lowest):
    1. User config file (explicit --config or ./config.yaml)
    2. Default config bundled with package
    
    Args:
        config_path: Optional path to user config file
        validate: Whether to validate config against schema
        
    Returns:
        Merged configuration dictionary with type hints
        
    Raises:
        ConfigValidationError: If validation is enabled and config is invalid
    """
    # Start with defaults
    config = load_default_config()
    
    # Determine user config path
    user_config_path = Path(config_path) if config_path else Path("config.yaml")
    
    # Merge user config if it exists
    if user_config_path.exists():
        try:
            user_config = yaml.safe_load(user_config_path.read_text()) or {}
            config = _deep_merge(config, user_config)
            logging.debug(f"Loaded user config from {user_config_path}")
        except Exception as e:
            logging.warning(f"Failed to read config {user_config_path}: {e}")
    
    # Validate merged config
    if validate:
        errors = validate_config(config)
        if errors:
            raise ConfigValidationError(errors)
    
    return config


def get_intent_rules(config: KeywordLabConfig) -> IntentRulesConfig:
    """Get intent rules from config, with fallback defaults."""
    return config.get("intent_rules", {
        "informational": ["who", "what", "why", "how", "guide", "tutorial", "tips", "checklist", "template"],
        "commercial": ["best", "top", "review", "compare", "vs", "alternatives", "pricing"],
        "transactional": ["buy", "discount", "coupon", "deal", "near me"],
        "navigational": [],
    })


def get_question_prefixes(config: KeywordLabConfig) -> List[str]:
    """Get question prefixes from config, with fallback defaults."""
    return config.get("question_prefixes", [
        "how", "what", "best", "vs", "for", "near me", 
        "beginner", "advanced", "guide", "checklist", "template", "why"
    ])
