"""
Shared Constants for EXPLORER PRIME Unified System

Centralized constants used across all pipeline components.
"""

# ==============================================================================
# HIFA v2.0 Validation Thresholds
# ==============================================================================

# Gate 4: Statistical Tests
DEFAULT_DSR_THRESHOLD = 0.95          # Deflated Sharpe Ratio threshold
DEFAULT_PBO_THRESHOLD = 0.50          # Probability of Backtest Overfitting max
DEFAULT_T_STAT_THRESHOLD = 3.0        # Minimum t-statistic for significance
DEFAULT_FDR_LEVEL = 0.05              # False Discovery Rate level

# Gate 5: CPCV Configuration
DEFAULT_CPCV_N_GROUPS = 6             # Number of groups for CPCV
DEFAULT_CPCV_K_TEST = 2               # Test paths per split
DEFAULT_CPCV_PURGE_DAYS = 20          # Purge buffer days
DEFAULT_CPCV_EMBARGO_DAYS = 60        # Embargo period days

# Gate 6: Clustering
DEFAULT_SIMILARITY_THRESHOLD = 0.70   # Max correlation for redundancy

# Gate 7: Regime Validation
REGIME_NORMAL = "normal"
REGIME_ELEVATED = "elevated"
REGIME_CRISIS = "crisis"

VIX_NORMAL_THRESHOLD = 20             # Below = normal
VIX_ELEVATED_THRESHOLD = 30           # Above = crisis

DEFAULT_MIN_REGIME_SHARPE = 0.0       # Minimum Sharpe per regime

# ==============================================================================
# Forward Testing (Hinance) Configuration
# ==============================================================================

# Shadow Trading Parameters
DEFAULT_TRANSFER_RATIO_MIN = 0.5      # Minimum shadow_sharpe / backtest_sharpe
DEFAULT_MAX_DD_RATIO = 1.5            # Maximum shadow_dd / backtest_dd

MAX_CONCURRENT_SHADOW_STRATEGIES = 50  # Capacity limit
MIN_SHADOW_DURATION_DAYS = 14         # Minimum shadow period
MAX_SHADOW_DURATION_DAYS = 28         # Maximum shadow period
MIN_SHADOW_TRADES = 20                # Minimum trades for validity

# Execution Simulation
DEFAULT_SLIPPAGE_MIN_PCT = 0.01       # 1 bps minimum slippage
DEFAULT_SLIPPAGE_MAX_PCT = 0.10       # 10 bps maximum slippage
DEFAULT_FEE_TAKER = 0.001             # 10 bps taker fee
DEFAULT_FEE_MAKER = 0.0               # 0 bps maker fee
DEFAULT_LATENCY_MIN_MS = 50           # Minimum latency simulation
DEFAULT_LATENCY_MAX_MS = 200          # Maximum latency simulation

# Forward Test Configuration Dictionary (for easy import)
FORWARD_TEST_CONFIG = {
    'TRANSFER_RATIO_MIN': DEFAULT_TRANSFER_RATIO_MIN,
    'MAX_DD_RATIO': DEFAULT_MAX_DD_RATIO,
    'MAX_CONCURRENT': MAX_CONCURRENT_SHADOW_STRATEGIES,
    'MIN_DURATION_DAYS': MIN_SHADOW_DURATION_DAYS,
    'MAX_DURATION_DAYS': MAX_SHADOW_DURATION_DAYS,
    'SHADOW_DURATION_DAYS': MIN_SHADOW_DURATION_DAYS,
    'MIN_TRADES': MIN_SHADOW_TRADES,
}

# HIFA Thresholds Dictionary (for easy import)
HIFA_THRESHOLDS = {
    'DSR_THRESHOLD': DEFAULT_DSR_THRESHOLD,
    'PBO_THRESHOLD': DEFAULT_PBO_THRESHOLD,
    'T_STAT_THRESHOLD': DEFAULT_T_STAT_THRESHOLD,
    'FDR_LEVEL': DEFAULT_FDR_LEVEL,
    'SIMILARITY_THRESHOLD': DEFAULT_SIMILARITY_THRESHOLD,
}

# ==============================================================================
# Generation Budget Allocation
# ==============================================================================

ENGINE_BUDGET_EVOLUTIONARY = 0.40     # 40% - Evolutionary Search
ENGINE_BUDGET_GENAI = 0.25            # 25% - GenAI Generation
ENGINE_BUDGET_PATTERN = 0.15          # 15% - Pattern Discovery
ENGINE_BUDGET_RECOMBINE = 0.10        # 10% - Recombine Crossover
ENGINE_BUDGET_LSM = 0.10              # 10% - Language Strategy Model

# ==============================================================================
# Feature Vector Configuration
# ==============================================================================

FEATURE_DIMENSIONS = 60               # Total feature vector size

# Feature Groups
PRICE_FEATURE_COUNT = 15              # Price-based features
VOLUME_FEATURE_COUNT = 10             # Volume-based features
VOLATILITY_FEATURE_COUNT = 10         # Volatility features
MOMENTUM_FEATURE_COUNT = 10           # Momentum indicators
MICROSTRUCTURE_FEATURE_COUNT = 8      # Market microstructure
REGIME_FEATURE_COUNT = 4              # Regime state features
TIME_FEATURE_COUNT = 3                # Time-based features

# ==============================================================================
# Strategy Complexity Limits
# ==============================================================================

MAX_STRATEGY_PARAMETERS = 20          # Maximum tunable parameters
MAX_STRATEGY_RULES = 15               # Maximum entry/exit rules
MAX_STRATEGY_TREE_DEPTH = 5           # Maximum condition nesting
MAX_STRATEGY_INDICATORS = 10          # Maximum indicators used

# Minimum requirements
MIN_STRATEGY_TRADES = 100             # Minimum trades for backtest validity
MIN_STRATEGY_DAYS = 365               # Minimum backtest period (1 year)

# ==============================================================================
# Pipeline Pass Rates (Expected)
# ==============================================================================

EXPECTED_GATE_1_PASS_RATE = 0.95      # Syntax check
EXPECTED_GATE_2_PASS_RATE = 0.80      # Complexity + BIC
EXPECTED_GATE_3_PASS_RATE = 0.70      # Quick test + MBL
EXPECTED_GATE_4_PASS_RATE = 0.40      # DSR + PBO + FDR
EXPECTED_GATE_5_PASS_RATE = 0.60      # CPCV
EXPECTED_GATE_6_PASS_RATE = 0.70      # HRP Clustering
EXPECTED_GATE_7_PASS_RATE = 0.60      # Regime Validation

# Cumulative expected pass rate
EXPECTED_HIFA_TOTAL_PASS_RATE = 0.03  # ~3% of candidates pass all gates

# Forward testing pass rate
EXPECTED_FORWARD_PASS_RATE = 0.50     # 50% of HIFA survivors
EXPECTED_PRODUCTION_RATE = 0.015      # ~1.5% of initial candidates

# ==============================================================================
# Logging & Monitoring
# ==============================================================================

LOG_LEVEL_DEFAULT = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Monitoring intervals
MONITOR_INTERVAL_SECONDS = 300        # 5 minutes
DRIFT_CHECK_INTERVAL_HOURS = 24       # Daily drift check

# Alert thresholds
ALERT_TRANSFER_RATIO_DROP = 0.3       # Alert if TR drops below 0.3
ALERT_DD_RATIO_SPIKE = 2.0            # Alert if DD ratio exceeds 2.0
ALERT_CONSECUTIVE_LOSSES = 5          # Alert on 5 consecutive losing trades

# ==============================================================================
# Storage Configuration
# ==============================================================================

EMT_VERSION = "2.0"
STRATEGY_FILE_EXTENSION = ".strategy"
METADATA_FILE_EXTENSION = ".meta"
AUDIT_LOG_RETENTION_DAYS = 365

# Production limits
MAX_PRODUCTION_STRATEGIES = 200       # Maximum strategies in production

# Batch processing
DEFAULT_BATCH_SIZE = 100              # Strategies per batch
MAX_BATCH_SIZE = 1000                 # Maximum batch size
