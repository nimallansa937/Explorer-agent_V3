"""
Dynamic Engine Allocation — Thompson Sampling with Gap-Driven Priors

The fixed 40/25/15/10/10 allocation is replaced by Thompson sampling where
gap diagnostics shift the Beta distribution priors. This provides the correct
balance between exploitation and exploration at the meta-level.

Key Design:
- Gap diagnostic as PRIOR SHIFT, not deterministic override
- Per-engine decay rates reflect regime sensitivity
- 8% exploration floor prevents engine starvation

Explorer Prime v2.0 - Phase 6
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import numpy as np

# Import GapType from diagnostics
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from diagnostics.anomaly_signature import GapType
except ImportError:
    # Fallback if diagnostics not available
    class GapType(Enum):
        STRUCTURAL = "structural"
        FEATURE = "feature"
        PATTERN = "pattern"
        AMBIGUOUS = "ambiguous"
        UNKNOWN = "unknown"


# ==============================================================================
# Constants
# ==============================================================================

ENGINE_NAMES = ["evolutionary", "genai", "pattern", "recombine", "lsm"]

GAP_AFFINITY: Dict[GapType, Dict[str, float]] = {
    GapType.STRUCTURAL: {
        "evolutionary": 0.55, "genai": 0.15, "pattern": 0.10,
        "recombine": 0.15, "lsm": 0.05,
    },
    GapType.FEATURE: {
        "evolutionary": 0.20, "genai": 0.35, "pattern": 0.15,
        "recombine": 0.05, "lsm": 0.25,
    },
    GapType.PATTERN: {
        "evolutionary": 0.25, "genai": 0.10, "pattern": 0.40,
        "recombine": 0.15, "lsm": 0.10,
    },
    GapType.AMBIGUOUS: {
        "evolutionary": 0.35, "genai": 0.25, "pattern": 0.15,
        "recombine": 0.15, "lsm": 0.10,
    },
    GapType.UNKNOWN: {
        "evolutionary": 0.30, "genai": 0.25, "pattern": 0.15,
        "recombine": 0.10, "lsm": 0.20,
    },
}

# Per-engine decay rates (reflects regime sensitivity)
ENGINE_DECAY_RATES: Dict[str, float] = {
    "evolutionary": 0.990,   # 70-cycle half-life (highly regime-sensitive)
    "genai": 0.995,          # 138-cycle half-life (moderate sensitivity)
    "pattern": 0.993,        # 99-cycle half-life
    "recombine": 0.997,      # 231-cycle half-life (less regime-dependent)
    "lsm": 0.998,            # 346-cycle half-life (least regime-sensitive)
}


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class FeatureExplorationStats:
    """Per-feature exploration statistics."""
    feature_id: str
    strategies_generated: int = 0
    hifa_passed: int = 0
    shadow_results: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.strategies_generated == 0:
            return 0.0
        return self.hifa_passed / self.strategies_generated


@dataclass
class AllocationRecord:
    """Record of a single allocation decision."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    gap_type: Optional[GapType] = None
    allocations: Dict[str, int] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    n_strategies: int = 0


# ==============================================================================
# Engine Allocator (Thompson Sampling)
# ==============================================================================

class EngineAllocator:
    """Thompson sampling engine allocation with gap-driven prior shifts.

    For each engine, maintains a Beta(alpha, beta) distribution.
    On allocate():
      1. Get affinity vector for current gap type
      2. Add gap affinity * prior_bonus to alpha (prior shift)
      3. Sample from shifted Beta distribution
      4. Apply floor (no engine below 8%)
      5. Normalize and allocate integer counts

    On update():
      - alpha += n_hifa_passed
      - beta += (n_generated - n_hifa_passed)
      - Apply per-engine decay to prevent lock-in
    """

    DEFAULT_ALPHA: float = 2.0
    DEFAULT_BETA: float = 2.0

    def __init__(
        self,
        exploration_floor: float = 0.08,
        prior_bonus: float = 1.5,
        seed: Optional[int] = None,
    ):
        self.exploration_floor = exploration_floor
        self.prior_bonus = prior_bonus

        # Beta(alpha, beta) per engine — weak uniform prior
        self.betas: Dict[str, List[float]] = {
            engine: [self.DEFAULT_ALPHA, self.DEFAULT_BETA]
            for engine in ENGINE_NAMES
        }

        # Per-engine decay rates
        self.decay_rates: Dict[str, float] = dict(ENGINE_DECAY_RATES)

        # History
        self._history: List[AllocationRecord] = []

        # RNG
        self._rng = np.random.RandomState(seed)

    def allocate(
        self,
        gap_type: GapType,
        n_strategies: int = 1000,
    ) -> Dict[str, int]:
        """Allocate strategies across engines using Thompson sampling.

        Args:
            gap_type: Current gap classification
            n_strategies: Total strategies to allocate

        Returns:
            Dict mapping engine name → number of strategies
        """
        # Get affinity vector
        affinity = GAP_AFFINITY.get(gap_type, GAP_AFFINITY[GapType.UNKNOWN])

        # Sample from shifted Beta distributions
        raw_weights: Dict[str, float] = {}
        for engine in ENGINE_NAMES:
            alpha, beta = self.betas[engine]

            # Prior shift: add affinity * bonus to alpha
            alpha_shifted = alpha + affinity.get(engine, 0.2) * self.prior_bonus

            # Sample from Beta distribution
            sample = self._rng.beta(alpha_shifted, beta)

            # Apply floor
            sample = max(sample, self.exploration_floor)
            raw_weights[engine] = sample

        # Normalize to sum to 1.0
        total_weight = sum(raw_weights.values())
        weights: Dict[str, float] = {
            engine: w / total_weight for engine, w in raw_weights.items()
        }

        # Allocate integer counts
        allocations: Dict[str, int] = {}
        remaining = n_strategies
        engine_list = sorted(weights.keys())  # Deterministic order

        for engine in engine_list[:-1]:
            count = int(weights[engine] * n_strategies)
            allocations[engine] = count
            remaining -= count

        # Last engine gets remainder
        allocations[engine_list[-1]] = remaining

        # Record history
        self._history.append(AllocationRecord(
            gap_type=gap_type,
            allocations=dict(allocations),
            weights=dict(weights),
            n_strategies=n_strategies,
        ))

        return allocations

    def update(self, engine: str, n_generated: int, n_hifa_passed: int) -> None:
        """Update Beta distribution from HIFA results.

        Args:
            engine: Engine name
            n_generated: Total strategies generated
            n_hifa_passed: Strategies that passed HIFA
        """
        if engine not in self.betas:
            return

        alpha, beta = self.betas[engine]

        # Update with results
        alpha += n_hifa_passed
        beta += (n_generated - n_hifa_passed)

        # Apply per-engine decay
        decay_rate = self.decay_rates.get(engine, 0.995)
        alpha *= decay_rate
        beta *= decay_rate

        # Prevent parameters from collapsing below minimum
        alpha = max(1.0, alpha)
        beta = max(1.0, beta)

        self.betas[engine] = [alpha, beta]

    def get_current_weights(self) -> Dict[str, float]:
        """Return expected value of each engine's Beta distribution.

        Expected value of Beta(alpha, beta) = alpha / (alpha + beta).
        """
        weights = {}
        for engine in ENGINE_NAMES:
            alpha, beta = self.betas[engine]
            weights[engine] = alpha / (alpha + beta)
        return weights

    def get_allocation_history(self) -> List[AllocationRecord]:
        """Return full allocation history."""
        return list(self._history)


# ==============================================================================
# Exploration Budget Manager
# ==============================================================================

class ExplorationBudgetManager:
    """Manages exploration budget for experimental features.

    Reserves 20% of the daily strategy budget for strategies that MUST
    include at least one experimental feature. These go through normal
    HIFA pipeline — the exploration budget just ensures they get generated.
    """

    DEFAULT_EXPLORATION_FRACTION: float = 0.20

    def __init__(self, exploration_fraction: float = 0.20):
        self.exploration_fraction = exploration_fraction
        self._feature_stats: Dict[str, FeatureExplorationStats] = {}

    def allocate_exploration(
        self,
        total_strategies: int,
        experimental_features: List[str],
    ) -> int:
        """Return number of strategies that must include experimental features.

        Args:
            total_strategies: Total strategies being generated
            experimental_features: Feature IDs currently in experimental phase

        Returns:
            Number of strategies to reserve for experimental feature exploration
        """
        if not experimental_features:
            return 0

        n_exploration = int(self.exploration_fraction * total_strategies)

        # Initialize stats for any new features
        for feat_id in experimental_features:
            if feat_id not in self._feature_stats:
                self._feature_stats[feat_id] = FeatureExplorationStats(
                    feature_id=feat_id,
                )

        return n_exploration

    def record_generation(
        self,
        feature_id: str,
        n_generated: int,
        n_hifa_passed: int,
    ) -> None:
        """Record generation results for a feature.

        Args:
            feature_id: The experimental feature
            n_generated: Strategies generated with this feature
            n_hifa_passed: Strategies that passed HIFA
        """
        if feature_id not in self._feature_stats:
            self._feature_stats[feature_id] = FeatureExplorationStats(
                feature_id=feature_id,
            )

        stats = self._feature_stats[feature_id]
        stats.strategies_generated += n_generated
        stats.hifa_passed += n_hifa_passed

    def record_shadow_result(
        self,
        feature_id: str,
        result: Dict[str, Any],
    ) -> None:
        """Record shadow trading result for a feature.

        Args:
            feature_id: The experimental feature
            result: Shadow result (e.g., {'sharpe': 1.2, 'days': 14})
        """
        if feature_id not in self._feature_stats:
            self._feature_stats[feature_id] = FeatureExplorationStats(
                feature_id=feature_id,
            )

        self._feature_stats[feature_id].shadow_results.append(result)

    def get_feature_exploration_stats(self) -> Dict[str, FeatureExplorationStats]:
        """Return per-feature exploration statistics."""
        return dict(self._feature_stats)
