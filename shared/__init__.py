"""
Shared Components for EXPLORER PRIME Unified System

This module provides common infrastructure used across all pipeline stages:
- Explorer Agent v3.0 (Generation)
- LSM (Language Strategy Model)
- HIFA v2.0 (Historical Validation)
- Forward Testing (Hinance)

Key Components:
- UnifiedStrategy: Universal strategy format
- StrategyGenome: Strategy logic representation
- StrategyAdapter: Format conversion utilities
- Feature schema: 60-dimension feature vector
"""

from .unified_strategy import (
    StrategyGenome,
    UnifiedStrategy,
    HIFAResult,
    ForwardTestResult,
    ProductionStrategy,
    GateResult,
    StatisticalScores,
    BacktestMetrics,
    ShadowMetrics,
)

from .adapters import (
    StrategyAdapter,
    ExplorerAdapter,
    LSMAdapter,
    HinanceAdapter,
)

from .features import (
    FEATURE_SCHEMA,
    FEATURE_DIMENSIONS,
    extract_features,
    validate_features,
    FeatureVector,
)

from .constants import (
    # Thresholds
    DEFAULT_DSR_THRESHOLD,
    DEFAULT_PBO_THRESHOLD,
    DEFAULT_TRANSFER_RATIO_MIN,
    DEFAULT_MAX_DD_RATIO,
    # Limits
    MAX_CONCURRENT_SHADOW_STRATEGIES,
    MIN_SHADOW_DURATION_DAYS,
    MIN_SHADOW_TRADES,
    # Regimes
    REGIME_NORMAL,
    REGIME_ELEVATED,
    REGIME_CRISIS,
    VIX_NORMAL_THRESHOLD,
    VIX_ELEVATED_THRESHOLD,
)

__all__ = [
    # Strategy formats
    'StrategyGenome',
    'UnifiedStrategy',
    'HIFAResult',
    'ForwardTestResult',
    'ProductionStrategy',
    'GateResult',
    'StatisticalScores',
    'BacktestMetrics',
    'ShadowMetrics',

    # Adapters
    'StrategyAdapter',
    'ExplorerAdapter',
    'LSMAdapter',
    'HinanceAdapter',

    # Features
    'FEATURE_SCHEMA',
    'FEATURE_DIMENSIONS',
    'extract_features',
    'validate_features',
    'FeatureVector',

    # Constants
    'DEFAULT_DSR_THRESHOLD',
    'DEFAULT_PBO_THRESHOLD',
    'DEFAULT_TRANSFER_RATIO_MIN',
    'DEFAULT_MAX_DD_RATIO',
    'MAX_CONCURRENT_SHADOW_STRATEGIES',
    'MIN_SHADOW_DURATION_DAYS',
    'MIN_SHADOW_TRADES',
    'REGIME_NORMAL',
    'REGIME_ELEVATED',
    'REGIME_CRISIS',
    'VIX_NORMAL_THRESHOLD',
    'VIX_ELEVATED_THRESHOLD',
]

__version__ = '1.0.0'
