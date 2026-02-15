"""
Feature Registry & Projection Layer

Single source of truth for all features with versioned schemas, status
tracking, and projection layer for transparent cross-version evaluation.

Explorer Prime v2.0 - Phase 2
"""

import math
import hashlib
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from datetime import datetime, timedelta
import numpy as np


# ==============================================================================
# Enums
# ==============================================================================

class FeatureStatus(str, Enum):
    """Feature maturity status.

    NOTE: Do NOT confuse with C1/C2 framework terms.
    C1 = fixed perceptual primitives, C2 = recombinations (strategies).
    These labels describe feature maturity only.
    """
    EXPERIMENTAL = "experimental"  # Under evaluation, not in CORE schema
    VALIDATED = "validated"        # Passed initial tests, in active schema
    CORE = "core"                  # Battle-tested, in both active and core schemas
    DEPRECATED = "deprecated"      # Scheduled for removal


class DataClass(str, Enum):
    """Feature data class categories."""
    PRICE = "price"                 # OHLCV-derived features
    VOLUME = "volume"               # Volume profile, VWAP, cumulative delta
    ORDER_FLOW = "order_flow"       # Bid/ask imbalance, trade aggression, OFI
    VOLATILITY = "volatility"       # Realized vol, implied vol proxies, ATR
    MICROSTRUCTURE = "microstructure"  # Spread, depth, impact estimates
    REGIME = "regime"               # HMM states, VIX-derived, correlation regimes
    CROSS_ASSET = "cross_asset"     # Lead-lag, correlation, relative strength
    DERIVED = "derived"             # Features computed from other features


class MaturityDecision(str, Enum):
    """Trial evaluation outcome."""
    PROMOTE = "promote"
    DEPRECATE = "deprecate"
    EXTEND = "extend"


# ==============================================================================
# Feature Definition
# ==============================================================================

@dataclass
class FeatureDefinition:
    """A single registered feature."""
    feature_id: str                             # Unique, immutable once registered
    name: str                                   # Human-readable name
    data_class: DataClass                       # Category
    compute_fn: Optional[Callable] = None       # Function to compute from raw data
    status: FeatureStatus = FeatureStatus.EXPERIMENTAL
    added_version: str = "1.0.0"
    deprecated_version: Optional[str] = None
    correlation_with_core: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "name": self.name,
            "data_class": self.data_class.value,
            "status": self.status.value,
            "added_version": self.added_version,
            "deprecated_version": self.deprecated_version,
            "correlation_with_core": self.correlation_with_core,
            "metadata": self.metadata,
        }


# ==============================================================================
# Trial Record (for maturity pipeline)
# ==============================================================================

@dataclass
class TrialRecord:
    """Tracks a feature's maturity trial."""
    feature_id: str
    start_date: datetime = field(default_factory=datetime.now)
    strategies_generated: int = 0
    hifa_passed: int = 0
    shadow_entries: int = 0
    shadow_passes: int = 0
    functional_correlation: Optional[float] = None
    extended: bool = False
    extension_date: Optional[datetime] = None
    ended: bool = False
    decision: Optional[MaturityDecision] = None


# ==============================================================================
# Feature Registry
# ==============================================================================

class FeatureRegistry:
    """
    Single source of truth for all features.

    Pre-populated with 60 CORE features matching Explorer v3.0's feature space.
    Supports versioned schemas, promotion/deprecation, and class-based queries.
    """

    def __init__(self, pre_populate: bool = True):
        self._features: Dict[str, FeatureDefinition] = {}
        self._schema_version: str = "1.0.0"
        self._version_history: List[Tuple[str, datetime, str]] = [
            ("1.0.0", datetime.now(), "Initial 60-feature CORE schema"),
        ]
        self._schema_snapshots: Dict[str, List[str]] = {}

        if pre_populate:
            self._populate_core_features()
            self._schema_snapshots["1.0.0"] = self.get_core_schema()

    # --------------------------------------------------------------------------
    # Registration
    # --------------------------------------------------------------------------

    def register(
        self,
        feature_id: str,
        name: str,
        data_class: DataClass,
        compute_fn: Optional[Callable] = None,
        status: FeatureStatus = FeatureStatus.EXPERIMENTAL,
        metadata: Optional[Dict] = None,
    ) -> FeatureDefinition:
        """Register a new feature. ID must be unique."""
        if feature_id in self._features:
            raise ValueError(f"Feature '{feature_id}' already registered")

        feat = FeatureDefinition(
            feature_id=feature_id,
            name=name,
            data_class=data_class,
            compute_fn=compute_fn,
            status=status,
            added_version=self._schema_version,
            metadata=metadata or {},
        )
        self._features[feature_id] = feat
        return feat

    # --------------------------------------------------------------------------
    # Schema queries
    # --------------------------------------------------------------------------

    def get_active_schema(self) -> List[str]:
        """Return feature IDs with status CORE or VALIDATED, sorted."""
        return sorted([
            fid for fid, f in self._features.items()
            if f.status in (FeatureStatus.CORE, FeatureStatus.VALIDATED)
        ])

    def get_core_schema(self) -> List[str]:
        """Return feature IDs with status CORE only, sorted."""
        return sorted([
            fid for fid, f in self._features.items()
            if f.status == FeatureStatus.CORE
        ])

    def get_schema_at_version(self, version: str) -> List[str]:
        """Return historical schema snapshot at a given version."""
        if version in self._schema_snapshots:
            return self._schema_snapshots[version]
        # Fallback: current core schema
        return self.get_core_schema()

    def get_features_by_class(self, data_class: DataClass) -> List[FeatureDefinition]:
        """Return all features (any status) in a given data class."""
        return [
            f for f in self._features.values()
            if f.data_class == data_class
        ]

    def get_feature(self, feature_id: str) -> Optional[FeatureDefinition]:
        """Lookup a single feature by ID."""
        return self._features.get(feature_id)

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def version_history(self) -> List[Tuple[str, datetime, str]]:
        return list(self._version_history)

    @property
    def feature_count(self) -> int:
        return len(self._features)

    @property
    def core_count(self) -> int:
        return len(self.get_core_schema())

    # --------------------------------------------------------------------------
    # Promotion / Deprecation
    # --------------------------------------------------------------------------

    def promote(self, feature_id: str, new_status: FeatureStatus) -> None:
        """Change a feature's status. Increments version if CORE status changes."""
        if feature_id not in self._features:
            raise KeyError(f"Feature '{feature_id}' not found")

        feat = self._features[feature_id]
        old_status = feat.status

        # Validate transition
        valid_transitions = {
            FeatureStatus.EXPERIMENTAL: [FeatureStatus.VALIDATED, FeatureStatus.DEPRECATED],
            FeatureStatus.VALIDATED: [FeatureStatus.CORE, FeatureStatus.DEPRECATED],
            FeatureStatus.CORE: [FeatureStatus.DEPRECATED],
            FeatureStatus.DEPRECATED: [],  # Terminal state
        }
        if new_status not in valid_transitions.get(old_status, []):
            raise ValueError(
                f"Invalid transition: {old_status.value} → {new_status.value}"
            )

        feat.status = new_status

        # Increment version if CORE boundary changes
        if new_status == FeatureStatus.CORE or old_status == FeatureStatus.CORE:
            self._increment_version(
                f"Feature '{feature_id}' status: {old_status.value} → {new_status.value}"
            )

    def deprecate(self, feature_id: str) -> None:
        """Deprecate a feature. Records in version history."""
        if feature_id not in self._features:
            raise KeyError(f"Feature '{feature_id}' not found")

        feat = self._features[feature_id]
        if feat.status == FeatureStatus.DEPRECATED:
            return  # Already deprecated

        old_status = feat.status
        feat.status = FeatureStatus.DEPRECATED
        feat.deprecated_version = self._schema_version

        # Increment version if was CORE
        if old_status == FeatureStatus.CORE:
            self._increment_version(
                f"Feature '{feature_id}' deprecated from CORE"
            )
        else:
            self._version_history.append(
                (self._schema_version, datetime.now(),
                 f"Feature '{feature_id}' deprecated from {old_status.value}")
            )

    def _increment_version(self, description: str):
        """Bump the schema version (minor increment)."""
        parts = self._schema_version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        self._schema_version = ".".join(parts)
        self._version_history.append(
            (self._schema_version, datetime.now(), description)
        )
        self._schema_snapshots[self._schema_version] = self.get_core_schema()

    # --------------------------------------------------------------------------
    # Pre-populated 60 CORE features
    # --------------------------------------------------------------------------

    def _populate_core_features(self):
        """Register 60 CORE features matching Explorer v3.0's feature space."""

        # 20 PRICE features
        price_features = [
            ("price_return_1m", "1-Minute Return"),
            ("price_return_5m", "5-Minute Return"),
            ("price_return_15m", "15-Minute Return"),
            ("price_return_1h", "1-Hour Return"),
            ("price_return_4h", "4-Hour Return"),
            ("price_return_24h", "24-Hour Return"),
            ("price_macd_signal", "MACD Signal Line"),
            ("price_macd_histogram", "MACD Histogram"),
            ("price_rsi_14", "RSI (14-period)"),
            ("price_rsi_7", "RSI (7-period)"),
            ("price_bollinger_upper", "Bollinger Upper Band"),
            ("price_bollinger_lower", "Bollinger Lower Band"),
            ("price_bollinger_width", "Bollinger Band Width"),
            ("price_ema_12", "EMA 12"),
            ("price_ema_26", "EMA 26"),
            ("price_sma_50", "SMA 50"),
            ("price_sma_200", "SMA 200"),
            ("price_high_low_range", "High-Low Range"),
            ("price_close_to_vwap", "Close-to-VWAP Ratio"),
            ("price_momentum_12", "12-Period Momentum"),
        ]

        # 10 VOLUME features
        volume_features = [
            ("vol_vwap", "VWAP"),
            ("vol_obv", "On-Balance Volume"),
            ("vol_profile_poc", "Volume Profile POC"),
            ("vol_cumulative_delta", "Cumulative Delta"),
            ("vol_relative_1h", "Relative Volume 1H"),
            ("vol_relative_24h", "Relative Volume 24H"),
            ("vol_buy_sell_ratio", "Buy/Sell Volume Ratio"),
            ("vol_tick_volume", "Tick Volume"),
            ("vol_rolling_mean_1h", "Rolling Mean Volume 1H"),
            ("vol_spike_indicator", "Volume Spike Indicator"),
        ]

        # 10 ORDER_FLOW features
        order_flow_features = [
            ("ofi_order_flow_imbalance", "Order Flow Imbalance"),
            ("ofi_bid_ask_imbalance", "Bid-Ask Imbalance"),
            ("ofi_trade_aggression", "Trade Aggression"),
            ("ofi_taker_buy_ratio", "Taker Buy Ratio"),
            ("ofi_large_order_flow", "Large Order Flow"),
            ("ofi_cumulative_ofi", "Cumulative OFI"),
            ("ofi_depth_imbalance_l1", "L1 Depth Imbalance"),
            ("ofi_depth_imbalance_l5", "L5 Depth Imbalance"),
            ("ofi_trade_intensity", "Trade Intensity"),
            ("ofi_net_aggressive", "Net Aggressive Flow"),
        ]

        # 8 VOLATILITY features
        volatility_features = [
            ("vlt_realized_1h", "Realized Volatility 1H"),
            ("vlt_realized_4h", "Realized Volatility 4H"),
            ("vlt_realized_24h", "Realized Volatility 24H"),
            ("vlt_atr_14", "ATR 14-period"),
            ("vlt_garman_klass", "Garman-Klass Estimator"),
            ("vlt_parkinson", "Parkinson Estimator"),
            ("vlt_vol_of_vol", "Volatility of Volatility"),
            ("vlt_implied_proxy", "Implied Vol Proxy"),
        ]

        # 6 MICROSTRUCTURE features
        microstructure_features = [
            ("mcs_spread_bps", "Bid-Ask Spread (bps)"),
            ("mcs_depth_ratio", "Depth Ratio Top 5"),
            ("mcs_impact_estimate", "Market Impact Estimate"),
            ("mcs_effective_spread", "Effective Spread"),
            ("mcs_kyle_lambda", "Kyle's Lambda"),
            ("mcs_amihud_illiquidity", "Amihud Illiquidity"),
        ]

        # 6 REGIME features
        regime_features = [
            ("reg_hmm_state", "HMM Regime State"),
            ("reg_vix_level", "VIX Level"),
            ("reg_correlation_regime", "Cross-Asset Correlation Regime"),
            ("reg_trend_strength", "Trend Strength ADX"),
            ("reg_market_phase", "Market Phase Indicator"),
            ("reg_volatility_regime", "Volatility Regime"),
        ]

        # Register all
        all_features = [
            (price_features, DataClass.PRICE),
            (volume_features, DataClass.VOLUME),
            (order_flow_features, DataClass.ORDER_FLOW),
            (volatility_features, DataClass.VOLATILITY),
            (microstructure_features, DataClass.MICROSTRUCTURE),
            (regime_features, DataClass.REGIME),
        ]

        for feature_list, data_class in all_features:
            for fid, name in feature_list:
                self._features[fid] = FeatureDefinition(
                    feature_id=fid,
                    name=name,
                    data_class=data_class,
                    status=FeatureStatus.CORE,
                    added_version="1.0.0",
                )


# ==============================================================================
# Feature Projector
# ==============================================================================

class FeatureProjector:
    """
    Handles cross-schema feature vector projections.

    Provides schema-aware distance for Gate 6 HRP clustering and
    projection utilities for evaluation across different schema versions.
    """

    def __init__(self, registry: FeatureRegistry):
        self.registry = registry

    def project(self, full_vector: np.ndarray, target_schema: List[str]) -> np.ndarray:
        """
        Extract features from full vector for a target schema.

        Args:
            full_vector: Full feature vector aligned to source schema
            target_schema: List of feature IDs to extract

        Returns:
            Projected vector with only target features
        """
        source_schema = self.registry.get_active_schema()
        result = np.zeros(len(target_schema), dtype=np.float64)

        for i, fid in enumerate(target_schema):
            if fid in source_schema:
                src_idx = source_schema.index(fid)
                if src_idx < len(full_vector):
                    result[i] = full_vector[src_idx]

        return result

    def pad_to_full(
        self,
        partial_vector: np.ndarray,
        source_schema: List[str],
    ) -> np.ndarray:
        """
        Expand partial vector to full active schema with zero padding.

        WARNING: Do NOT use for Gate 6 HRP clustering.
        Use schema_aware_distance instead for cross-schema comparisons.

        Args:
            partial_vector: Vector aligned to source_schema
            source_schema: Feature IDs in partial_vector

        Returns:
            Full-length vector aligned to active schema
        """
        active = self.registry.get_active_schema()
        result = np.zeros(len(active), dtype=np.float64)

        for i, fid in enumerate(source_schema):
            if fid in active and i < len(partial_vector):
                idx = active.index(fid)
                result[idx] = partial_vector[i]

        return result

    def schema_aware_distance(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
        schema_a: List[str],
        schema_b: List[str],
    ) -> float:
        """
        Compute schema-aware cosine distance between two feature vectors.

        Only compares over shared feature dimensions. Jaccard-weighted to
        normalize by intersection/union coverage. Returns 1.0 (max distance)
        if fewer than 5 shared features.

        USE THIS for all Gate 6 HRP clustering comparisons.

        Args:
            vec_a: Feature vector A
            vec_b: Feature vector B
            schema_a: Feature IDs for vec_a
            schema_b: Feature IDs for vec_b

        Returns:
            Distance in [0.0, 1.0+]. Lower = more similar.
        """
        shared = set(schema_a) & set(schema_b)

        if len(shared) < 5:
            return 1.0

        # Extract shared dimensions
        indices_a = [schema_a.index(f) for f in sorted(shared)]
        indices_b = [schema_b.index(f) for f in sorted(shared)]

        sub_a = vec_a[indices_a]
        sub_b = vec_b[indices_b]

        # Cosine distance
        norm_a = np.linalg.norm(sub_a)
        norm_b = np.linalg.norm(sub_b)

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 1.0

        cos_sim = np.dot(sub_a, sub_b) / (norm_a * norm_b)
        cos_dist = 1.0 - cos_sim

        # Jaccard coverage
        union = set(schema_a) | set(schema_b)
        coverage = len(shared) / len(union) if union else 0.0

        if coverage < 1e-10:
            return 1.0

        return cos_dist / coverage


# ==============================================================================
# Feature Maturity Pipeline
# ==============================================================================

class FeatureMaturityPipeline:
    """
    Manages feature promotion trials.

    Features start as EXPERIMENTAL, undergo a trial period, and are either
    promoted to CORE (via VALIDATED) or deprecated based on strategy-level
    evidence.
    """

    # Thresholds
    MIN_STRATEGIES: int = 500
    MIN_MARGINAL_PASS_RATE: float = 1.0  # Relative to CORE-only base rate
    MAX_CORE_CORRELATION: float = 0.7    # Functional, not raw
    MIN_SHADOW_ENTRIES: int = 3
    MIN_SHADOW_PASSES: int = 1
    TRIAL_DURATION_DAYS: int = 90
    EXTENSION_DURATION_DAYS: int = 45

    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
        self._trials: Dict[str, TrialRecord] = {}

    def start_trial(self, feature_id: str) -> TrialRecord:
        """Start a maturity trial for a feature."""
        if feature_id not in self.registry._features:
            raise KeyError(f"Feature '{feature_id}' not in registry")

        trial = TrialRecord(feature_id=feature_id)
        self._trials[feature_id] = trial
        return trial

    def update_trial(
        self,
        feature_id: str,
        strategies_generated: int = 0,
        hifa_passed: int = 0,
        shadow_entries: int = 0,
        shadow_passes: int = 0,
    ) -> None:
        """Update trial metrics incrementally."""
        if feature_id not in self._trials:
            raise KeyError(f"No active trial for '{feature_id}'")

        trial = self._trials[feature_id]
        trial.strategies_generated += strategies_generated
        trial.hifa_passed += hifa_passed
        trial.shadow_entries += shadow_entries
        trial.shadow_passes += shadow_passes

    def evaluate_trial(
        self,
        feature_id: str,
        functional_correlation: Optional[float] = None,
        base_pass_rate: float = 0.03,
    ) -> MaturityDecision:
        """
        Evaluate a feature's maturity trial.

        Args:
            feature_id: Feature under evaluation
            functional_correlation: Pre-computed functional correlation
            base_pass_rate: CORE-only HIFA pass rate for comparison

        Returns:
            MaturityDecision: PROMOTE, DEPRECATE, or EXTEND
        """
        if feature_id not in self._trials:
            raise KeyError(f"No active trial for '{feature_id}'")

        trial = self._trials[feature_id]

        if functional_correlation is not None:
            trial.functional_correlation = functional_correlation

        # Check 1: Sufficient strategies generated?
        if trial.strategies_generated < self.MIN_STRATEGIES:
            if not trial.extended:
                trial.extended = True
                trial.extension_date = datetime.now()
                trial.decision = MaturityDecision.EXTEND
                return MaturityDecision.EXTEND
            else:
                trial.decision = MaturityDecision.DEPRECATE
                trial.ended = True
                return MaturityDecision.DEPRECATE

        # Check 2: Marginal pass rate
        if trial.strategies_generated > 0:
            trial_pass_rate = trial.hifa_passed / trial.strategies_generated
            marginal_rate = trial_pass_rate / base_pass_rate if base_pass_rate > 0 else 0
            if marginal_rate < self.MIN_MARGINAL_PASS_RATE:
                trial.decision = MaturityDecision.DEPRECATE
                trial.ended = True
                return MaturityDecision.DEPRECATE

        # Check 3: Functional correlation (redundancy)
        if trial.functional_correlation is not None:
            if trial.functional_correlation > self.MAX_CORE_CORRELATION:
                trial.decision = MaturityDecision.DEPRECATE
                trial.ended = True
                return MaturityDecision.DEPRECATE

        # Check 4: Shadow trading entries
        if trial.shadow_entries < self.MIN_SHADOW_ENTRIES:
            trial.decision = MaturityDecision.DEPRECATE
            trial.ended = True
            return MaturityDecision.DEPRECATE

        # Check 5: Shadow trading passes
        if trial.shadow_passes < self.MIN_SHADOW_PASSES:
            trial.decision = MaturityDecision.DEPRECATE
            trial.ended = True
            return MaturityDecision.DEPRECATE

        # All checks passed → PROMOTE
        trial.decision = MaturityDecision.PROMOTE
        trial.ended = True
        return MaturityDecision.PROMOTE

    def compute_functional_correlation(
        self,
        feature_performance_vector: np.ndarray,
        core_performance_vectors: Dict[str, np.ndarray],
    ) -> float:
        """
        Compute functional correlation using strategy-level performance.

        Not raw feature correlation. Measures: do strategies using this feature
        BEHAVE differently from those using each core feature?

        Args:
            feature_performance_vector: Sharpe contributions of strategies using this feature
            core_performance_vectors: Sharpe contributions of strategies using each core feature

        Returns:
            Max correlation with any core feature's performance vector
        """
        if len(feature_performance_vector) == 0 or not core_performance_vectors:
            return 0.0

        max_corr = 0.0
        for core_fid, core_vec in core_performance_vectors.items():
            if len(core_vec) == 0:
                continue

            # Align lengths
            min_len = min(len(feature_performance_vector), len(core_vec))
            if min_len < 3:
                continue

            a = feature_performance_vector[:min_len]
            b = core_vec[:min_len]

            # Compute correlation
            if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                continue

            corr = abs(np.corrcoef(a, b)[0, 1])
            max_corr = max(max_corr, corr)

        return max_corr

    def get_trial(self, feature_id: str) -> Optional[TrialRecord]:
        """Get trial record for a feature."""
        return self._trials.get(feature_id)


# ==============================================================================
# Shadow Re-Evolution Manager
# ==============================================================================

class ShadowReEvolution:
    """
    Manages re-evolution of legacy strategies when schema version changes.

    When CORE features change (promotion or deprecation), existing production
    strategies may become stale. This manager tracks legacy strategies and
    their re-evolution attempts.
    """

    REEVOLUTION_WINDOW_DAYS: int = 60
    LEGACY_RISK_CAP: float = 0.05  # Max 5% of portfolio risk

    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
        self._legacy_strategies: Dict[str, Dict[str, Any]] = {}
        self._reevolution_tasks: List[Dict[str, Any]] = []

    def check_for_reevolution(
        self,
        production_strategies: List[Dict[str, Any]],
        current_schema: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Identify strategies needing re-evolution due to schema change.

        Args:
            production_strategies: List of production strategy dicts
                Each must have 'strategy_id', 'schema_version', 'performance_sharpe'
            current_schema: Current active schema feature IDs

        Returns:
            List of re-evolution task dicts
        """
        tasks = []
        current_version = self.registry.schema_version

        for strat in production_strategies:
            strat_version = strat.get("schema_version", "1.0.0")
            if strat_version != current_version:
                task = {
                    "strategy_id": strat["strategy_id"],
                    "old_version": strat_version,
                    "new_version": current_version,
                    "target_sharpe": strat.get("performance_sharpe", 0.0),
                    "deadline": datetime.now() + timedelta(days=self.REEVOLUTION_WINDOW_DAYS),
                    "status": "pending",
                }
                tasks.append(task)
                self._reevolution_tasks.append(task)

        return tasks

    def tag_as_legacy(self, strategy_id: str, metadata: Dict[str, Any] = None):
        """Tag a strategy as LEGACY (re-evolution failed or not attempted)."""
        self._legacy_strategies[strategy_id] = {
            "tagged_date": datetime.now(),
            "metadata": metadata or {},
        }

    def get_legacy_risk_allocation(
        self,
        portfolio_strategies: List[Dict[str, Any]],
    ) -> float:
        """
        Compute the total risk allocation to LEGACY strategies.

        Args:
            portfolio_strategies: All portfolio strategies with 'strategy_id' and 'risk_weight'

        Returns:
            Total risk weight of LEGACY strategies
        """
        total_legacy_risk = 0.0
        for strat in portfolio_strategies:
            if strat["strategy_id"] in self._legacy_strategies:
                total_legacy_risk += strat.get("risk_weight", 0.0)
        return total_legacy_risk

    def is_legacy_risk_within_cap(
        self,
        portfolio_strategies: List[Dict[str, Any]],
    ) -> bool:
        """Check if LEGACY strategies are within the 5% risk cap."""
        return self.get_legacy_risk_allocation(portfolio_strategies) <= self.LEGACY_RISK_CAP

    @property
    def legacy_count(self) -> int:
        return len(self._legacy_strategies)

    @property
    def pending_reevolution_count(self) -> int:
        return sum(1 for t in self._reevolution_tasks if t["status"] == "pending")
