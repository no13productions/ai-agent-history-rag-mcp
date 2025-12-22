"""Configuration management for the decision engine.

C4 fix: Provides environment variable and runtime configuration support
so tuning doesn't require code changes.

Environment variables:
    DECISION_ENGINE_CACHE_ENABLED: Enable/disable caching (default: true)
    DECISION_ENGINE_CACHE_TTL: Cache TTL in seconds (default: 300)
    DECISION_ENGINE_CACHE_MAXSIZE: Maximum cache entries (default: 100)
    DECISION_ENGINE_REFINEMENT_ENABLED: Enable query refinement (default: true)
    DECISION_ENGINE_SYNTHESIS_ENABLED: Enable result synthesis (default: true)
    DECISION_ENGINE_ADEQUACY_THRESHOLD: Evaluator adequacy threshold (default: 0.5)
    DECISION_ENGINE_COMPLETENESS_THRESHOLD: Evaluator completeness threshold (default: 0.4)
    DECISION_ENGINE_MAX_REFINEMENT_ATTEMPTS: Max refinement retries (default: 1)
"""

import logging
import os
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _get_bool_env(name: str, default: bool) -> bool:
    """Get boolean from environment variable.

    Args:
        name: Environment variable name
        default: Default value if not set

    Returns:
        Boolean value (recognizes 'true', '1', 'yes' as True)
    """
    value = os.environ.get(name, "").lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    return default


def _get_int_env(
    name: str, default: int, min_val: int | None = None, max_val: int | None = None
) -> int:
    """Get integer from environment variable.

    Args:
        name: Environment variable name
        default: Default value if not set
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Integer value, clamped to bounds
    """
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        result = int(value)
        # Raise ValueError for out-of-range values instead of silent clamping
        if min_val is not None and result < min_val:
            raise ValueError(f"{name}={result} below minimum {min_val}")
        if max_val is not None and result > max_val:
            raise ValueError(f"{name}={result} exceeds maximum {max_val}")
        return result
    except ValueError as e:
        logger.error(f"Invalid integer for {name}: {value} ({e}), using default {default}")
        return default


def _get_float_env(
    name: str, default: float, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Get float from environment variable.

    Args:
        name: Environment variable name
        default: Default value if not set
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Float value, clamped to bounds
    """
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        result = float(value)
        # Raise ValueError for out-of-range values instead of silent clamping
        if min_val is not None and result < min_val:
            raise ValueError(f"{name}={result} below minimum {min_val}")
        if max_val is not None and result > max_val:
            raise ValueError(f"{name}={result} exceeds maximum {max_val}")
        return result
    except ValueError as e:
        logger.error(f"Invalid float for {name}: {value} ({e}), using default {default}")
        return default


@dataclass
class DecisionEngineConfig:
    """Configuration for the decision engine.

    Can be initialized from environment variables or programmatically.
    All settings have sensible defaults.
    """

    # Cache settings
    cache_enabled: bool = field(
        default_factory=lambda: _get_bool_env("DECISION_ENGINE_CACHE_ENABLED", True)
    )
    cache_ttl_seconds: int = field(
        default_factory=lambda: _get_int_env(
            "DECISION_ENGINE_CACHE_TTL", 300, min_val=1, max_val=86400
        )
    )
    cache_maxsize: int = field(
        default_factory=lambda: _get_int_env(
            "DECISION_ENGINE_CACHE_MAXSIZE", 100, min_val=1, max_val=10000
        )
    )

    # Feature toggles
    refinement_enabled: bool = field(
        default_factory=lambda: _get_bool_env("DECISION_ENGINE_REFINEMENT_ENABLED", True)
    )
    synthesis_enabled: bool = field(
        default_factory=lambda: _get_bool_env("DECISION_ENGINE_SYNTHESIS_ENABLED", True)
    )

    # Evaluator thresholds
    adequacy_threshold: float = field(
        default_factory=lambda: _get_float_env(
            "DECISION_ENGINE_ADEQUACY_THRESHOLD", 0.5, min_val=0.0, max_val=1.0
        )
    )
    completeness_threshold: float = field(
        default_factory=lambda: _get_float_env(
            "DECISION_ENGINE_COMPLETENESS_THRESHOLD", 0.4, min_val=0.0, max_val=1.0
        )
    )

    # Refinement settings
    max_refinement_attempts: int = field(
        default_factory=lambda: _get_int_env(
            "DECISION_ENGINE_MAX_REFINEMENT_ATTEMPTS", 1, min_val=0, max_val=5
        )
    )

    def __post_init__(self):
        """Log configuration after initialization."""
        logger.debug(
            f"DecisionEngineConfig loaded: cache={self.cache_enabled}, "
            f"ttl={self.cache_ttl_seconds}s, refinement={self.refinement_enabled}, "
            f"synthesis={self.synthesis_enabled}"
        )


# Global config instance (lazy initialization with thread safety)
_global_config: DecisionEngineConfig | None = None
_global_config_lock = threading.Lock()


def get_config() -> DecisionEngineConfig:
    """Get or create the global configuration instance.

    Uses double-check locking pattern for thread-safe lazy initialization.
    Configuration is read from environment variables on first call.

    Returns:
        Global DecisionEngineConfig instance
    """
    global _global_config
    if _global_config is None:
        with _global_config_lock:
            # Double-check after acquiring lock
            if _global_config is None:
                _global_config = DecisionEngineConfig()
    return _global_config


def reset_config() -> None:
    """Reset global config to force re-reading environment variables.

    Useful for testing or when environment changes at runtime.
    """
    global _global_config
    with _global_config_lock:
        _global_config = None
