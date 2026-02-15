"""
Anomaly Signature — Structured L0 Gap Detection Output

The AnomalySignature IS the L0 output format. It transforms a binary
gap-detected signal into a multi-dimensional gap profile that L1-L3
and the intervention protocol can act on.

Explorer Prime v2.0 - Phase 3
"""

import uuid
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np


# ==============================================================================
# Enums
# ==============================================================================

class GapType(str, Enum):
    """Classification of the gap detected by anomaly diagnostic."""
    STRUCTURAL = "structural"   # Existing features sufficient, tree can't express
    FEATURE = "feature"         # Signal lives in unmeasured data
    AMBIGUOUS = "ambiguous"     # Mid-range core_overlap, needs sequential protocol
    PATTERN = "pattern"         # Pattern-based gap (used by engine allocator)
    UNKNOWN = "unknown"         # Insufficient data to classify


# ==============================================================================
# Trade Opportunity
# ==============================================================================

@dataclass
class TradeOpportunity:
    """A missed profitable trade identified by the diagnostic.

    Represents a time period where the strategy was flat (no position)
    but a profitable trade existed (forward return > 2x transaction cost).
    """
    timestamp: datetime
    asset: str
    feature_vector: np.ndarray      # Feature values at missed entry
    forward_return: float           # Realized return if entered
    transaction_cost: float         # Estimated transaction cost
    horizon_bars: int               # Number of bars to realize return
    regime: str = "RANGE"           # BULL/BEAR/RANGE at time of miss
    vol_regime: str = "MEDIUM"      # LOW/MEDIUM/HIGH/EXTREME
    strategy_id: Optional[str] = None

    def profit_ratio(self) -> float:
        """Return multiple of transaction cost."""
        if self.transaction_cost <= 0:
            return float('inf') if self.forward_return > 0 else 0.0
        return self.forward_return / self.transaction_cost


# ==============================================================================
# Profile Data Classes
# ==============================================================================

@dataclass
class TemporalProfile:
    """Temporal clustering of missed trades."""
    hour_distribution: Dict[int, float] = field(default_factory=dict)
    day_of_week_distribution: Dict[int, float] = field(default_factory=dict)
    has_strong_pattern: bool = False   # entropy < 0.7 of uniform
    peak_hours: List[int] = field(default_factory=list)

    @staticmethod
    def compute_entropy(distribution: Dict[int, float]) -> float:
        """Compute Shannon entropy of a probability distribution."""
        values = [v for v in distribution.values() if v > 0]
        if not values:
            return 0.0
        total = sum(values)
        if total == 0:
            return 0.0
        probs = [v / total for v in values]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    @staticmethod
    def uniform_entropy(n_bins: int) -> float:
        """Max entropy for n bins (uniform distribution)."""
        if n_bins <= 0:
            return 0.0
        return math.log2(n_bins)

    def detect_pattern(self, n_top_hours: int = 3) -> None:
        """Detect temporal patterns and set flags.

        Strong pattern means entropy < 0.7 of uniform (concentrated).
        If only 1-2 unique hours, that IS a strong pattern by definition.
        """
        if not self.hour_distribution:
            self.has_strong_pattern = False
            return

        n_unique = len(self.hour_distribution)
        actual_entropy = self.compute_entropy(self.hour_distribution)

        # With very few unique values, it's inherently concentrated
        if n_unique <= 2:
            self.has_strong_pattern = True
        else:
            # Compare to maximum possible entropy (24 hours uniform)
            max_entropy = self.uniform_entropy(24)
            if max_entropy > 0:
                self.has_strong_pattern = actual_entropy < 0.7 * max_entropy
            else:
                self.has_strong_pattern = False

        # Find peak hours
        sorted_hours = sorted(
            self.hour_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        self.peak_hours = [h for h, _ in sorted_hours[:n_top_hours]]


@dataclass
class RegimeProfile:
    """Regime distribution of missed trades."""
    regime_counts: Dict[str, int] = field(default_factory=dict)
    concentrated_at_transitions: bool = False  # >40% within 2h of regime change
    transition_types: List[Tuple[str, str]] = field(default_factory=list)
    transition_fraction: float = 0.0

    def detect_concentration(self, transition_timestamps: List[datetime],
                              miss_timestamps: List[datetime],
                              window_hours: float = 2.0) -> None:
        """Detect if misses cluster around regime transitions."""
        if not transition_timestamps or not miss_timestamps:
            self.concentrated_at_transitions = False
            return

        near_transition = 0
        window_seconds = window_hours * 3600

        for miss_ts in miss_timestamps:
            for trans_ts in transition_timestamps:
                if abs((miss_ts - trans_ts).total_seconds()) <= window_seconds:
                    near_transition += 1
                    break

        self.transition_fraction = near_transition / len(miss_timestamps) if miss_timestamps else 0.0
        self.concentrated_at_transitions = self.transition_fraction > 0.40


@dataclass
class AssetProfile:
    """Asset concentration of missed trades."""
    asset_counts: Dict[str, int] = field(default_factory=dict)
    dominant_asset: Optional[str] = None  # If any asset > 50%
    skew_coefficient: float = 0.0

    def analyze(self) -> None:
        """Compute asset concentration metrics."""
        if not self.asset_counts:
            return
        total = sum(self.asset_counts.values())
        if total == 0:
            return

        # Check for dominant asset
        for asset, count in self.asset_counts.items():
            if count / total > 0.50:
                self.dominant_asset = asset
                break

        # Compute skew: ratio of max to mean
        counts = list(self.asset_counts.values())
        mean_count = total / len(counts)
        if mean_count > 0:
            self.skew_coefficient = max(counts) / mean_count
        else:
            self.skew_coefficient = 0.0


@dataclass
class VolatilityProfile:
    """Volatility context of missed trades."""
    vol_at_miss: List[float] = field(default_factory=list)
    vol_regime_at_miss: List[str] = field(default_factory=list)
    concentrated_in_regime: Optional[str] = None  # If >60% in one vol regime

    def analyze(self) -> None:
        """Detect vol regime concentration."""
        if not self.vol_regime_at_miss:
            return

        counts: Dict[str, int] = {}
        for vr in self.vol_regime_at_miss:
            counts[vr] = counts.get(vr, 0) + 1

        total = len(self.vol_regime_at_miss)
        for regime, count in counts.items():
            if count / total > 0.60:
                self.concentrated_in_regime = regime
                break


@dataclass
class PrecedingPattern:
    """Feature vectors and cluster structure of conditions preceding misses."""
    feature_vectors_30min_before: Optional[np.ndarray] = None  # (n_misses, n_features)
    umap_embedding: Optional[np.ndarray] = None                # (n_misses, 2)
    n_subclusters: int = 0
    subcluster_labels: Optional[np.ndarray] = None

    def has_clusters(self) -> bool:
        """Whether meaningful subclusters were found."""
        return self.n_subclusters > 1


@dataclass
class LeadLagProfile:
    """Cross-asset lead-lag structure around missed trades."""
    cross_asset_correlations: Dict[str, float] = field(default_factory=dict)
    leading_instruments: List[str] = field(default_factory=list)
    has_significant_leads: bool = False  # Any lead corr > 0.3 with p < 0.05

    def identify_leads(self, correlations: Dict[str, float],
                       p_values: Dict[str, float],
                       corr_threshold: float = 0.3,
                       p_threshold: float = 0.05) -> None:
        """Identify significant leading instruments."""
        self.cross_asset_correlations = correlations
        self.leading_instruments = []

        for asset, corr in correlations.items():
            p_val = p_values.get(asset, 1.0)
            if abs(corr) > corr_threshold and p_val < p_threshold:
                self.leading_instruments.append(asset)

        self.has_significant_leads = len(self.leading_instruments) > 0


# ==============================================================================
# Anomaly Signature — The L0 Output
# ==============================================================================

@dataclass
class AnomalySignature:
    """Structured output of L0 gap detection.

    This IS the L0 output — a multi-dimensional gap profile that transforms
    a binary gap-detected signal into actionable characterization for
    L1-L3 and the intervention protocol.
    """
    anomaly_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    discovery_date: datetime = field(default_factory=datetime.utcnow)
    missed_trades: List[TradeOpportunity] = field(default_factory=list)

    # Observable characteristics
    temporal_clustering: TemporalProfile = field(default_factory=TemporalProfile)
    regime_distribution: RegimeProfile = field(default_factory=RegimeProfile)
    asset_concentration: AssetProfile = field(default_factory=AssetProfile)
    volatility_context: VolatilityProfile = field(default_factory=VolatilityProfile)
    preceding_market_pattern: PrecedingPattern = field(default_factory=PrecedingPattern)
    lead_lag_structure: LeadLagProfile = field(default_factory=LeadLagProfile)

    @property
    def n_misses(self) -> int:
        return len(self.missed_trades)

    def summary(self) -> Dict[str, Any]:
        """Compact summary for logging."""
        return {
            "anomaly_id": self.anomaly_id,
            "n_misses": self.n_misses,
            "has_temporal_pattern": self.temporal_clustering.has_strong_pattern,
            "regime_concentrated_at_transitions": self.regime_distribution.concentrated_at_transitions,
            "dominant_asset": self.asset_concentration.dominant_asset,
            "vol_concentrated": self.volatility_context.concentrated_in_regime,
            "n_subclusters": self.preceding_market_pattern.n_subclusters,
            "has_lead_lag": self.lead_lag_structure.has_significant_leads,
        }


# ==============================================================================
# Gap Classification Output
# ==============================================================================

@dataclass
class GapClassification:
    """Result of Random Forest gap classification.

    core_overlap determines gap type:
    - >= 0.7 → STRUCTURAL (existing features sufficient)
    - <= 0.4 → FEATURE (signal in unmeasured data)
    - 0.4 < overlap < 0.7 → AMBIGUOUS (needs sequential protocol)
    """
    core_overlap: float                              # Fraction of top-5 in CORE
    gap_type: GapType
    top_features: List[Tuple[str, float]]            # (feature_id, importance)
    rf_classifier: Any = None                        # Trained RF classifier
    baseline_auc: float = 0.0                        # AUC of classifier
    missing_features: List[str] = field(default_factory=list)  # Top features NOT in CORE

    # L1 extension: data class distribution of top features
    data_class_distribution: Dict[str, float] = field(default_factory=dict)


# ==============================================================================
# Diagnosis Result — Full Diagnostic Output
# ==============================================================================

@dataclass
class DiagnosisResult:
    """Complete diagnostic result combining signature + classification."""
    signature: AnomalySignature
    classification: GapClassification
    recommended_action: str  # "structural_seeds" / "feature_scout" / "sequential_intervention"
    tree_topology_seeds: Optional[List] = None          # RF splits as tree templates
    missing_feature_candidates: Optional[List[str]] = None
    confidence: float = 0.0                             # RF AUC = distinguishability

    def __post_init__(self):
        # Derive recommended action from gap type
        if not self.recommended_action:
            if self.classification.gap_type == GapType.STRUCTURAL:
                self.recommended_action = "structural_seeds"
            elif self.classification.gap_type == GapType.FEATURE:
                self.recommended_action = "feature_scout"
            else:
                self.recommended_action = "sequential_intervention"
