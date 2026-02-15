"""
Sequential Intervention Protocol

Enforces structure-first intervention with 45-day attribution window,
then conditionally routes to feature investigation. Prevents the lethal
attribution problem of simultaneous C1 + C2 interventions.

Explorer Prime v2.0 - Phase 4
"""

import uuid
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np

from .anomaly_signature import (
    AnomalySignature,
    GapType,
    GapClassification,
    DiagnosisResult,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Enums
# ==============================================================================

class InterventionType(str, Enum):
    """Types of interventions available."""
    STRUCTURAL_SEEDS = "structural_seeds"
    FEATURE_SCOUT = "feature_scout"
    CONDITIONAL_FEATURE_SCOUT = "conditional_feature_scout"


class PlanStatus(str, Enum):
    """Status of an intervention plan."""
    PENDING = "pending"
    PHASE_1_ACTIVE = "phase_1_active"
    AWAITING_ATTRIBUTION = "awaiting_attribution"
    PHASE_2_ACTIVE = "phase_2_active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Phase1Decision(str, Enum):
    """Outcome of Phase 1 evaluation."""
    CLOSE = "close"                       # >70% resolution, structural fix sufficient
    PROCEED_TO_FEATURE = "proceed_to_feature"  # <30% resolution, primarily feature gap
    PROCEED_CONFOUNDED = "proceed_confounded"  # 30-70% resolution, mixed


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class InterventionPlan:
    """A structured intervention plan linking diagnosis to action.

    Tracks the full lifecycle from routing decision through attribution
    window to completion.
    """
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    anomaly_id: str = ""
    phase_1: InterventionType = InterventionType.STRUCTURAL_SEEDS
    phase_2: Optional[InterventionType] = None
    attribution_window_days: int = 0      # 0 for non-ambiguous, 45 for ambiguous
    confounded_tag: bool = False          # If True, feature promotion threshold 1.5x
    created_at: datetime = field(default_factory=datetime.utcnow)
    phase_1_started: Optional[datetime] = None
    phase_1_completed: Optional[datetime] = None
    phase_2_started: Optional[datetime] = None
    status: PlanStatus = PlanStatus.PENDING

    # Regime stability tracking
    regime_at_start: Optional[str] = None
    paused_days: int = 0
    regime_transitions: List[Tuple[datetime, str, str]] = field(default_factory=list)

    def start_phase_1(self, regime: Optional[str] = None) -> None:
        """Activate phase 1."""
        self.phase_1_started = datetime.utcnow()
        self.status = PlanStatus.PHASE_1_ACTIVE
        if regime:
            self.regime_at_start = regime

    def complete_phase_1(self) -> None:
        """Move to awaiting attribution (if applicable) or completed."""
        self.phase_1_completed = datetime.utcnow()
        if self.phase_2 is not None:
            self.status = PlanStatus.AWAITING_ATTRIBUTION
        else:
            self.status = PlanStatus.COMPLETED

    def start_phase_2(self) -> None:
        """Activate phase 2."""
        self.phase_2_started = datetime.utcnow()
        self.status = PlanStatus.PHASE_2_ACTIVE

    def complete(self) -> None:
        """Mark plan as completed."""
        self.status = PlanStatus.COMPLETED

    def cancel(self) -> None:
        """Cancel the plan."""
        self.status = PlanStatus.CANCELLED

    def effective_attribution_days(self) -> int:
        """Get effective attribution window (excluding paused days)."""
        return max(0, self.attribution_window_days - self.paused_days)

    def elapsed_active_days(self) -> int:
        """Days elapsed since phase 1 completed (minus paused days)."""
        if self.phase_1_completed is None:
            return 0
        elapsed = (datetime.utcnow() - self.phase_1_completed).days
        return max(0, elapsed - self.paused_days)


@dataclass
class Phase1Result:
    """Result of evaluating Phase 1 intervention effectiveness."""
    resolution_rate: float          # 1.0 - (current_rate / original_rate)
    decision: Phase1Decision
    remaining_anomaly_rate: float   # current_anomaly_rate
    confounded: bool = False        # If True, feature promotion threshold 1.5x


@dataclass
class FeatureProposal:
    """A proposed new feature from the directed feature scout."""
    feature_id: str
    source_channel: str             # "event_driven" / "regime_transition" / "cross_asset"
    relevance_score: float          # 0.0 to 1.0
    compute_specification: str      # Description of how to compute
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeSeed:
    """A tree topology seed extracted from RF decision boundaries."""
    seed_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_anomaly_id: str = ""
    topology: Dict[str, Any] = field(default_factory=dict)
    feature_splits: List[Tuple[int, float]] = field(default_factory=list)
    depth: int = 0


# ==============================================================================
# Intervention Router
# ==============================================================================

class InterventionRouter:
    """Routes diagnostic results to appropriate interventions.

    Enforces the sequential protocol for AMBIGUOUS gaps:
    1. Structure-first intervention (45-day window)
    2. Conditional feature investigation if structural fix insufficient
    """

    AMBIGUOUS_LOW: float = 0.4
    AMBIGUOUS_HIGH: float = 0.7
    STRUCTURAL_WINDOW_DAYS: int = 45
    REGIME_STABILITY_REQUIRED: bool = True

    # Phase 1 evaluation thresholds
    CLOSE_THRESHOLD: float = 0.70       # >70% → structural fix sufficient
    PROCEED_THRESHOLD: float = 0.30     # <30% → proceed to feature
    # Between 30-70% → proceed confounded (stricter threshold)

    def route(self, diagnosis: DiagnosisResult) -> InterventionPlan:
        """Route diagnosis to an intervention plan.

        Args:
            diagnosis: Full diagnostic result with classification

        Returns:
            InterventionPlan with appropriate phases and windows
        """
        core_overlap = diagnosis.classification.core_overlap
        anomaly_id = diagnosis.signature.anomaly_id

        if core_overlap >= self.AMBIGUOUS_HIGH:
            # STRUCTURAL: seeds only, no attribution window
            return InterventionPlan(
                anomaly_id=anomaly_id,
                phase_1=InterventionType.STRUCTURAL_SEEDS,
                phase_2=None,
                attribution_window_days=0,
                confounded_tag=False,
            )

        elif core_overlap <= self.AMBIGUOUS_LOW:
            # FEATURE: direct feature scout, no window
            return InterventionPlan(
                anomaly_id=anomaly_id,
                phase_1=InterventionType.FEATURE_SCOUT,
                phase_2=None,
                attribution_window_days=0,
                confounded_tag=False,
            )

        else:
            # AMBIGUOUS: sequential protocol
            return InterventionPlan(
                anomaly_id=anomaly_id,
                phase_1=InterventionType.STRUCTURAL_SEEDS,
                phase_2=InterventionType.CONDITIONAL_FEATURE_SCOUT,
                attribution_window_days=self.STRUCTURAL_WINDOW_DAYS,
                confounded_tag=False,  # Set later if phase 1 partially resolves
            )

    def evaluate_phase_1(
        self,
        plan: InterventionPlan,
        original_anomaly_rate: float,
        current_anomaly_rate: float,
    ) -> Phase1Result:
        """Evaluate Phase 1 effectiveness after attribution window.

        Args:
            plan: The intervention plan
            original_anomaly_rate: Anomaly rate before intervention
            current_anomaly_rate: Current anomaly rate

        Returns:
            Phase1Result with resolution_rate and decision
        """
        if original_anomaly_rate <= 0:
            return Phase1Result(
                resolution_rate=1.0,
                decision=Phase1Decision.CLOSE,
                remaining_anomaly_rate=current_anomaly_rate,
            )

        resolution_rate = 1.0 - (current_anomaly_rate / original_anomaly_rate)
        resolution_rate = max(0.0, min(1.0, resolution_rate))

        if resolution_rate > self.CLOSE_THRESHOLD:
            decision = Phase1Decision.CLOSE
            confounded = False
        elif resolution_rate < self.PROCEED_THRESHOLD:
            decision = Phase1Decision.PROCEED_TO_FEATURE
            confounded = False
        else:
            decision = Phase1Decision.PROCEED_CONFOUNDED
            confounded = True
            plan.confounded_tag = True

        return Phase1Result(
            resolution_rate=resolution_rate,
            decision=decision,
            remaining_anomaly_rate=current_anomaly_rate,
            confounded=confounded,
        )

    def check_regime_stability(
        self,
        plan: InterventionPlan,
        regime_at_start: str,
        current_regime: str,
    ) -> bool:
        """Check if attribution window should be active or paused.

        Pauses the window during regime transitions to avoid contamination.

        Args:
            plan: The intervention plan
            regime_at_start: Regime when phase 1 started
            current_regime: Current market regime

        Returns:
            True if window is active, False if paused
        """
        if not self.REGIME_STABILITY_REQUIRED:
            return True

        if current_regime != regime_at_start:
            # Regime changed — pause the window
            plan.regime_transitions.append(
                (datetime.utcnow(), regime_at_start, current_regime)
            )
            plan.paused_days += 1
            logger.info(
                f"Regime transition {regime_at_start} → {current_regime}; "
                f"attribution window paused (plan {plan.plan_id})"
            )
            return False

        return True

    def get_confounded_promotion_multiplier(self, plan: InterventionPlan) -> float:
        """Get feature promotion threshold multiplier.

        When confounded_tag is True, the feature maturity pipeline requires
        1.5x the normal evidence threshold for promotion.
        """
        return 1.5 if plan.confounded_tag else 1.0


# ==============================================================================
# Directed Feature Scout
# ==============================================================================

class DirectedFeatureScout:
    """Generates TARGETED feature proposals from anomaly signatures.

    Three search channels:
    1. Event-driven (temporal clustering → event indicators)
    2. Regime-transition (transition concentration → regime predictors)
    3. Cross-asset (lead-lag → untapped data streams)
    """

    AUC_IMPROVEMENT_THRESHOLD: float = 0.08

    def __init__(self, feature_registry=None):
        self.feature_registry = feature_registry

    def search_from_signature(
        self, signature: AnomalySignature
    ) -> List[FeatureProposal]:
        """Generate targeted feature proposals from anomaly signature.

        Inspects signature profiles and activates relevant search channels.
        """
        proposals = []

        # Channel 1: Event-driven search
        if signature.temporal_clustering.has_strong_pattern:
            proposals.extend(self._event_driven_search(signature))

        # Channel 2: Regime-transition search
        if signature.regime_distribution.concentrated_at_transitions:
            proposals.extend(self._regime_transition_search(signature))

        # Channel 3: Cross-asset search
        if signature.lead_lag_structure.has_significant_leads:
            proposals.extend(self._cross_asset_search(signature))

        return proposals

    def _event_driven_search(
        self, signature: AnomalySignature
    ) -> List[FeatureProposal]:
        """Search for event-driven features from temporal clustering."""
        proposals = []
        peak_hours = signature.temporal_clustering.peak_hours

        for hour in peak_hours:
            # Economic release windows
            if 8 <= hour <= 10:
                proposals.append(FeatureProposal(
                    feature_id=f"event_economic_release_h{hour}",
                    source_channel="event_driven",
                    relevance_score=0.8,
                    compute_specification=(
                        f"Binary indicator for economic data release at hour {hour}. "
                        f"Source: economic calendar. Value: 1 during release window, 0 otherwise."
                    ),
                ))

            # Session boundary features
            if hour in [9, 16]:  # Market open/close
                proposals.append(FeatureProposal(
                    feature_id=f"event_session_boundary_h{hour}",
                    source_channel="event_driven",
                    relevance_score=0.7,
                    compute_specification=(
                        f"Session boundary indicator at hour {hour}. "
                        f"Measures order flow imbalance during session transitions."
                    ),
                ))

            # General time-of-day effect
            proposals.append(FeatureProposal(
                feature_id=f"event_tod_effect_h{hour}",
                source_channel="event_driven",
                relevance_score=0.5,
                compute_specification=(
                    f"Time-of-day effect feature for hour {hour}. "
                    f"Historical average return and volatility at this hour."
                ),
            ))

        return proposals

    def _regime_transition_search(
        self, signature: AnomalySignature
    ) -> List[FeatureProposal]:
        """Search for regime transition prediction features."""
        proposals = []
        transition_types = signature.regime_distribution.transition_types

        seen = set()
        for from_regime, to_regime in transition_types:
            key = f"{from_regime}_{to_regime}"
            if key in seen:
                continue
            seen.add(key)

            proposals.append(FeatureProposal(
                feature_id=f"regime_transition_{from_regime}_to_{to_regime}",
                source_channel="regime_transition",
                relevance_score=0.85,
                compute_specification=(
                    f"Regime transition predictor: {from_regime} → {to_regime}. "
                    f"Combines VIX term structure slope, credit spread change rate, "
                    f"and cross-asset correlation breakdown indicator."
                ),
            ))

        # General transition detection feature
        if signature.regime_distribution.transition_fraction > 0.5:
            proposals.append(FeatureProposal(
                feature_id="regime_transition_proximity",
                source_channel="regime_transition",
                relevance_score=0.9,
                compute_specification=(
                    "Regime transition proximity score. Rolling window measure of "
                    "how close current market conditions are to historical transition points. "
                    "Uses Mahalanobis distance to transition cluster centroids."
                ),
            ))

        return proposals

    def _cross_asset_search(
        self, signature: AnomalySignature
    ) -> List[FeatureProposal]:
        """Search for cross-asset features from lead-lag structure."""
        proposals = []

        for instrument in signature.lead_lag_structure.leading_instruments:
            corr = signature.lead_lag_structure.cross_asset_correlations.get(
                instrument, 0.0
            )

            proposals.append(FeatureProposal(
                feature_id=f"cross_asset_lead_{instrument}",
                source_channel="cross_asset",
                relevance_score=min(1.0, abs(corr) * 2.0),
                compute_specification=(
                    f"Cross-asset leading indicator from {instrument}. "
                    f"Lagged returns, volume, and order flow from {instrument} "
                    f"as predictors (lag corr={corr:.3f}). "
                    f"Features: lag_return_1m, lag_volume_change, lag_spread."
                ),
            ))

        return proposals

    def validate_fill(
        self,
        proposed_feature_values: np.ndarray,
        missed_features: np.ndarray,
        control_features: np.ndarray,
        baseline_auc: float,
    ) -> bool:
        """Validate if a proposed feature fills the gap.

        Augments the classification task with the proposed feature and
        checks if AUC improves by >= 0.08.

        Args:
            proposed_feature_values: New feature values (n_samples,)
            missed_features: Original missed-trade features (n_missed, n_features)
            control_features: Control set features (n_control, n_features)
            baseline_auc: Original AUC without the new feature

        Returns:
            True if AUC improvement >= 0.08
        """
        from .anomaly_diagnostic import SimpleRandomForest

        n_missed = len(missed_features)
        n_control = len(control_features)
        n_min = min(n_missed, n_control)

        # Augment feature vectors with proposed feature
        if len(proposed_feature_values) < n_missed + n_control:
            return False

        X_missed_aug = np.column_stack([
            missed_features[:n_min],
            proposed_feature_values[:n_min].reshape(-1, 1)
        ])
        X_control_aug = np.column_stack([
            control_features[:n_min],
            proposed_feature_values[n_min:2*n_min].reshape(-1, 1)
        ])

        X = np.vstack([X_missed_aug, X_control_aug])
        y = np.concatenate([np.ones(n_min), np.zeros(n_min)])

        rf = SimpleRandomForest(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(X, y)
        new_auc = rf._compute_auc(X, y)

        improvement = new_auc - baseline_auc
        return improvement >= self.AUC_IMPROVEMENT_THRESHOLD


# ==============================================================================
# Structural Seed Injector
# ==============================================================================

class StructuralSeedInjector:
    """Extracts RF decision boundaries as tree topology seeds.

    Seeds become initialization points for evolutionary search,
    replacing random initialization for 20% of the population.
    """

    SEED_POPULATION_FRACTION: float = 0.20

    def extract_tree_seeds(
        self,
        rf_classifier: Any,
        source_anomaly_id: str = "",
    ) -> List[TreeSeed]:
        """Convert RF trees into HierarchicalGenome tree templates.

        Takes the most informative RF trees and extracts their
        decision boundaries as topology seeds.
        """
        from .anomaly_diagnostic import SimpleRandomForest

        if not isinstance(rf_classifier, SimpleRandomForest):
            return []

        seeds = []
        # Extract from top 5 trees
        for i, tree in enumerate(rf_classifier.trees[:5]):
            splits = self._extract_splits(tree.tree)
            if splits:
                seeds.append(TreeSeed(
                    source_anomaly_id=source_anomaly_id,
                    topology=tree.tree or {},
                    feature_splits=splits,
                    depth=self._tree_depth(tree.tree),
                ))

        return seeds

    def _extract_splits(self, node: Optional[Dict]) -> List[Tuple[int, float]]:
        """Extract all feature splits from a tree node."""
        if node is None or node.get("leaf", True):
            return []

        splits = [(node["feature"], node["threshold"])]
        splits.extend(self._extract_splits(node.get("left")))
        splits.extend(self._extract_splits(node.get("right")))
        return splits

    def _tree_depth(self, node: Optional[Dict]) -> int:
        """Compute depth of tree."""
        if node is None or node.get("leaf", True):
            return 0
        left_depth = self._tree_depth(node.get("left"))
        right_depth = self._tree_depth(node.get("right"))
        return 1 + max(left_depth, right_depth)

    def inject_seeds(
        self,
        seeds: List[TreeSeed],
        population_size: int,
    ) -> Dict[str, Any]:
        """Compute seed injection parameters.

        Returns injection specification for the evolutionary engine:
        how many seeds to inject and which slots to replace.

        Args:
            seeds: Tree seeds to inject
            population_size: Total population size

        Returns:
            Dict with n_seeds, seed_slots, and seeds data
        """
        n_seed_slots = max(1, int(population_size * self.SEED_POPULATION_FRACTION))
        n_seeds = min(len(seeds), n_seed_slots)

        return {
            "n_seed_slots": n_seed_slots,
            "n_seeds_available": n_seeds,
            "seeds": seeds[:n_seeds],
            "replacement_fraction": self.SEED_POPULATION_FRACTION,
        }
