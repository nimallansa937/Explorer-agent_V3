"""
Failure Archive — Behavioral Similarity Negative Seeding

Ring buffer of failed strategy records. Penalizes new candidates whose
BEHAVIORAL signature (trade signal history) correlates with past failures.

Key design: Uses trade signal correlation, NOT feature activation vectors.
Two strategies with identical features but different tree topologies trade
differently — behavioral correlation captures this distinction.

Explorer Prime v2.0 - Phase 5
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import Counter
import numpy as np


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class FailureRecord:
    """Record of a failed production strategy."""
    strategy_id: str
    genome: Any = None                          # HierarchicalGenome
    failure_date: datetime = field(default_factory=datetime.utcnow)
    failure_regime: str = "RANGE"               # BULL/BEAR/RANGE
    decay_type: str = "unknown"                 # structural/feature/ambiguous
    anomaly_signature: Any = None               # AnomalySignature
    feature_activation_vector: Optional[np.ndarray] = None
    trade_signal_history: Optional[np.ndarray] = None  # Behavioral signature
    time_to_failure_days: int = 0               # Days from production to retirement


@dataclass
class FailureDistribution:
    """Aggregate statistics from the failure archive."""
    time_to_failure_histogram: Dict[str, int] = field(default_factory=dict)
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    gap_type_distribution: Dict[str, int] = field(default_factory=dict)
    median_time_to_failure: float = 0.0
    n_records: int = 0
    ttf_values: List[int] = field(default_factory=list)


# ==============================================================================
# Failure Archive
# ==============================================================================

class FailureArchive:
    """Ring buffer of failed strategies with behavioral similarity penalty.

    Capacity: 10000 records (oldest evicted when full).
    Penalty formula:
        For each archived failure:
          1. Behavioral similarity (correlation of trade signal histories)
          2. Time decay: exp(-days_since_failure / 120)  (120-day half-life)
          3. Regime factor: 0.3 + 0.7 * regime_match
          4. penalty = similarity * time_factor * regime_factor
        Return max(penalties) across all records.
        Applied as: fitness *= (1.0 - 0.5 * penalty)  (50% cap)
    """

    TIME_DECAY_HALF_LIFE: float = 120.0     # Days
    BEHAVIORAL_SIMILARITY_THRESHOLD: float = 0.7
    REGIME_FLOOR: float = 0.3
    MAX_PENALTY_FACTOR: float = 0.5
    DEFAULT_MAX_RECORDS: int = 10000

    def __init__(self, max_records: int = 10000):
        self._records: List[FailureRecord] = []
        self._max_records = max_records

    @property
    def records(self) -> List[FailureRecord]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def add(self, record: FailureRecord) -> None:
        """Add a failure record, evicting oldest if at capacity."""
        if len(self._records) >= self._max_records:
            self._records.pop(0)  # Evict oldest
        self._records.append(record)

    def penalty(
        self,
        candidate_signal_history: Optional[np.ndarray],
        current_regime: str,
        reference_date: Optional[datetime] = None,
    ) -> float:
        """Compute behavioral penalty for a candidate strategy.

        Args:
            candidate_signal_history: Trade signal history of the candidate
            current_regime: Current market regime
            reference_date: Date for time decay (default: now)

        Returns:
            Penalty in [0.0, ~1.0], to be applied as fitness *= (1 - 0.5 * penalty)
        """
        if candidate_signal_history is None or len(self._records) == 0:
            return 0.0

        if reference_date is None:
            reference_date = datetime.utcnow()

        max_penalty = 0.0

        for record in self._records:
            if record.trade_signal_history is None:
                continue

            # 1. Behavioral similarity (correlation)
            similarity = self._behavioral_similarity(
                candidate_signal_history, record.trade_signal_history
            )
            if similarity < self.BEHAVIORAL_SIMILARITY_THRESHOLD:
                continue

            # 2. Time decay
            days_since = max(0, (reference_date - record.failure_date).days)
            time_factor = math.exp(-days_since * math.log(2) / self.TIME_DECAY_HALF_LIFE)

            # 3. Regime match
            regime_match = 1.0 if current_regime == record.failure_regime else 0.0
            regime_factor = self.REGIME_FLOOR + (1.0 - self.REGIME_FLOOR) * regime_match

            # 4. Combined penalty
            p = similarity * time_factor * regime_factor
            max_penalty = max(max_penalty, p)

        return max_penalty

    def apply_penalty(self, fitness: float, penalty: float) -> float:
        """Apply penalty to fitness score.

        fitness *= (1.0 - MAX_PENALTY_FACTOR * penalty)
        Caps at 50% reduction.
        """
        return fitness * (1.0 - self.MAX_PENALTY_FACTOR * penalty)

    def get_failure_distribution(self) -> FailureDistribution:
        """Compute aggregate failure statistics."""
        if not self._records:
            return FailureDistribution()

        ttf_values = [r.time_to_failure_days for r in self._records]
        regime_counts = Counter(r.failure_regime for r in self._records)
        gap_type_counts = Counter(r.decay_type for r in self._records)

        # Time-to-failure histogram (10-day bins)
        ttf_hist: Dict[str, int] = {}
        for ttf in ttf_values:
            bin_label = f"{(ttf // 10) * 10}-{(ttf // 10 + 1) * 10}"
            ttf_hist[bin_label] = ttf_hist.get(bin_label, 0) + 1

        sorted_ttf = sorted(ttf_values)
        median_ttf = sorted_ttf[len(sorted_ttf) // 2] if sorted_ttf else 0.0

        return FailureDistribution(
            time_to_failure_histogram=ttf_hist,
            regime_distribution=dict(regime_counts),
            gap_type_distribution=dict(gap_type_counts),
            median_time_to_failure=median_ttf,
            n_records=len(self._records),
            ttf_values=ttf_values,
        )

    @staticmethod
    def _behavioral_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute behavioral similarity via Pearson correlation.

        Handles different-length signals by truncating to the shorter.
        """
        min_len = min(len(a), len(b))
        if min_len < 2:
            return 0.0

        a_t = a[:min_len]
        b_t = b[:min_len]

        a_mean = a_t.mean()
        b_mean = b_t.mean()
        a_centered = a_t - a_mean
        b_centered = b_t - b_mean

        denom = np.sqrt((a_centered ** 2).sum() * (b_centered ** 2).sum())
        if denom == 0:
            return 0.0

        return float((a_centered * b_centered).sum() / denom)
