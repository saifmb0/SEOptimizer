"""ORYX - SEO keyword discovery and clustering CLI."""

# Import typed config classes for convenience
from oryx.config import (
    KeywordLabConfig,
    IntentRulesConfig,
    NlpConfig,
    ScrapeConfig,
    ClusterConfig,
    LlmConfig,
    OutputConfig,
    load_config,
    ConfigValidationError,
)

__all__ = [
    # Modules
    "io",
    "scrape",
    "nlp",
    "cluster",
    "metrics",
    "schema",
    "pipeline",
    "config",
    # Config types
    "KeywordLabConfig",
    "IntentRulesConfig",
    "NlpConfig",
    "ScrapeConfig",
    "ClusterConfig",
    "LlmConfig",
    "OutputConfig",
    "load_config",
    "ConfigValidationError",
]

__version__ = "3.4.0"
