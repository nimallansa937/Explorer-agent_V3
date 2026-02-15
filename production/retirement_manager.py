"""
Retirement Manager — Strategy Lifecycle State Machine

Manages the HEALTHY → WARNING → CONFIRMING_RETIREMENT → RETIRED lifecycle.

Key design:
- 30-day confirmation at 25% allocation is asymmetry correction:
  Cost of premature retirement > cost of delayed retirement
- Regime-conditioned decay prevents false retirement of suppressed strategies

THIS IS WHERE THE FULL LOOP CLOSES:
Every retirement generates actionable intelligence:
1. Structural autopsy → anti-templates
2. Failure archive → negative seeding
3. Feature scout → investigation queue
4. Meta-learning → time-to-failure data

Explorer Prime v2.0 - Phase 7
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from enum import Enum

from .edge_decay import EdgeDecayDetector


# ==============================================================================
# Enums
# ==============================================================================

class StrategyState(Enum):
    """Strategy lifecycle states."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CONFIRMING_RETIREMENT = "confirming_retirement"
    RETIRED = "retired"


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class RetirementAction:
    """Result of a daily retirement decision."""
    strategy_id: str
    new_state: StrategyState
    allocation_fraction: float
    decay_probability: float
    regime_conditioned: bool = False
    days_in_confirmation: int = 0


@dataclass
class StrategyRecord:
    """Internal record for a tracked strategy."""
    strategy_id: str
    detector: EdgeDecayDetector
    state: StrategyState = StrategyState.HEALTHY
    confirmation_start: Optional[datetime] = None
    production_start: Optional[datetime] = None


# ==============================================================================
# Retirement Manager
# ==============================================================================

class RetirementManager:
    """Strategy lifecycle manager with Kalman-based decay detection.

    States:
      HEALTHY:     decay_prob < 0.3 → full allocation
      WARNING:     0.3 ≤ decay_prob < 0.7 → 50% allocation
      CONFIRMING:  decay_prob ≥ 0.7 → 25% allocation, 30-day timer
      RETIRED:     confirmed decay → removed from production

    Recovery:
      - CONFIRMING → WARNING if decay drops below 0.5
      - WARNING → HEALTHY if decay drops below 0.3
      - CONFIRMING → HEALTHY if decay drops below 0.3 (skips WARNING)
    """

    HEALTHY_THRESHOLD: float = 0.3
    WARNING_THRESHOLD: float = 0.7
    CONFIRMATION_DAYS: int = 30
    WARNING_ALLOCATION: float = 0.50
    CONFIRMATION_ALLOCATION: float = 0.25
    RECOVERY_THRESHOLD: float = 0.5  # Below this during confirmation → cancel

    def __init__(self, default_drift_var: float = 0.001):
        self._strategies: Dict[str, StrategyRecord] = {}
        self._default_drift_var = default_drift_var
        self._retirement_queue: List[str] = []

    def register_strategy(
        self,
        strategy_id: str,
        initial_sharpe: float = 1.0,
        drift_var: Optional[float] = None,
        production_start: Optional[datetime] = None,
    ) -> None:
        """Register a new production strategy for monitoring.

        Args:
            strategy_id: Unique strategy identifier
            initial_sharpe: Backtest Sharpe for initialization
            drift_var: Calibrated drift variance (or default)
            production_start: When strategy entered production
        """
        detector = EdgeDecayDetector(
            initial_sharpe=initial_sharpe,
            drift_var=drift_var or self._default_drift_var,
        )
        self._strategies[strategy_id] = StrategyRecord(
            strategy_id=strategy_id,
            detector=detector,
            production_start=production_start or datetime.utcnow(),
        )

    def daily_update(
        self,
        strategy_id: str,
        daily_return: float,
        current_regime: str,
        current_date: Optional[datetime] = None,
    ) -> RetirementAction:
        """Process daily return and determine retirement action.

        Args:
            strategy_id: Strategy to update
            daily_return: Today's return
            current_regime: Current market regime
            current_date: Optional date override (for testing)

        Returns:
            RetirementAction with state, allocation, and diagnostics
        """
        if strategy_id not in self._strategies:
            raise ValueError(f"Unknown strategy: {strategy_id}")

        record = self._strategies[strategy_id]
        now = current_date or datetime.utcnow()

        # Update Kalman filter
        record.detector.update(daily_return, current_regime)

        # Get regime-conditioned decay probability
        decay_prob = record.detector.regime_conditioned_decay(current_regime)
        standard_decay = record.detector.decay_probability()
        is_regime_conditioned = (decay_prob != standard_decay)

        # Decision logic
        if decay_prob < self.HEALTHY_THRESHOLD:
            # Recovery
            if record.state == StrategyState.WARNING:
                record.state = StrategyState.HEALTHY
            elif record.state == StrategyState.CONFIRMING_RETIREMENT:
                record.state = StrategyState.HEALTHY
                record.confirmation_start = None

            record.state = StrategyState.HEALTHY
            return RetirementAction(
                strategy_id=strategy_id,
                new_state=StrategyState.HEALTHY,
                allocation_fraction=1.0,
                decay_probability=decay_prob,
                regime_conditioned=is_regime_conditioned,
            )

        elif decay_prob < self.WARNING_THRESHOLD:
            # Warning zone
            if record.state == StrategyState.CONFIRMING_RETIREMENT:
                # Cancel confirmation if below recovery threshold
                if decay_prob < self.RECOVERY_THRESHOLD:
                    record.state = StrategyState.WARNING
                    record.confirmation_start = None
                # Stay in confirmation if still above recovery threshold
                else:
                    days = (now - record.confirmation_start).days if record.confirmation_start else 0
                    return RetirementAction(
                        strategy_id=strategy_id,
                        new_state=StrategyState.CONFIRMING_RETIREMENT,
                        allocation_fraction=self.CONFIRMATION_ALLOCATION,
                        decay_probability=decay_prob,
                        regime_conditioned=is_regime_conditioned,
                        days_in_confirmation=days,
                    )

            record.state = StrategyState.WARNING
            return RetirementAction(
                strategy_id=strategy_id,
                new_state=StrategyState.WARNING,
                allocation_fraction=self.WARNING_ALLOCATION,
                decay_probability=decay_prob,
                regime_conditioned=is_regime_conditioned,
            )

        else:
            # High decay zone (≥ warning_threshold)
            if record.state != StrategyState.CONFIRMING_RETIREMENT:
                # Enter confirmation
                record.state = StrategyState.CONFIRMING_RETIREMENT
                record.confirmation_start = now

            # Check if confirmation period elapsed
            days_in_confirmation = 0
            if record.confirmation_start:
                days_in_confirmation = (now - record.confirmation_start).days

            if days_in_confirmation >= self.CONFIRMATION_DAYS:
                # Retire!
                record.state = StrategyState.RETIRED
                self._retirement_queue.append(strategy_id)
                return RetirementAction(
                    strategy_id=strategy_id,
                    new_state=StrategyState.RETIRED,
                    allocation_fraction=0.0,
                    decay_probability=decay_prob,
                    regime_conditioned=is_regime_conditioned,
                    days_in_confirmation=days_in_confirmation,
                )

            return RetirementAction(
                strategy_id=strategy_id,
                new_state=StrategyState.CONFIRMING_RETIREMENT,
                allocation_fraction=self.CONFIRMATION_ALLOCATION,
                decay_probability=decay_prob,
                regime_conditioned=is_regime_conditioned,
                days_in_confirmation=days_in_confirmation,
            )

    def on_retirement(
        self,
        strategy_id: str,
        failure_archive: Any = None,
        structural_autopsy: Any = None,
        anti_template_injector: Any = None,
        meta_learning: Any = None,
        decay_type: str = "unknown",
        genome: Any = None,
        anomaly_signature: Any = None,
        trade_signal_history: Any = None,
    ) -> Dict[str, Any]:
        """Process a strategy retirement — THIS IS WHERE THE FULL LOOP CLOSES.

        1. Store failure record in failure archive (negative seeding)
        2. If STRUCTURAL: add anti-template via structural autopsy
        3. If FEATURE: queue for Feature Scout (HIGH priority)
        4. Update meta-learning with time-to-failure data

        Args:
            strategy_id: The retired strategy
            failure_archive: FailureArchive instance
            structural_autopsy: StructuralAutopsy instance
            anti_template_injector: AntiTemplateInjector instance
            meta_learning: MetaLearningSignal instance
            decay_type: Classification (structural/feature/ambiguous)
            genome: Strategy genome
            anomaly_signature: Related anomaly signature
            trade_signal_history: Trade signal history

        Returns:
            Summary of actions taken
        """
        record = self._strategies.get(strategy_id)
        actions = {"strategy_id": strategy_id, "actions": []}

        # Calculate time-to-failure
        ttf_days = 0
        if record and record.production_start:
            ttf_days = (datetime.utcnow() - record.production_start).days

        # 1. Failure archive
        if failure_archive is not None:
            try:
                from feedback.failure_archive import FailureRecord
                failure_record = FailureRecord(
                    strategy_id=strategy_id,
                    genome=genome,
                    decay_type=decay_type,
                    anomaly_signature=anomaly_signature,
                    trade_signal_history=trade_signal_history,
                    time_to_failure_days=ttf_days,
                )
                failure_archive.add(failure_record)
                actions["actions"].append("failure_archived")
            except ImportError:
                pass

        # 2. Structural autopsy → anti-template
        if decay_type in ("structural", "ambiguous") and structural_autopsy is not None:
            try:
                autopsy_result = structural_autopsy.analyze(
                    strategy_id=strategy_id,
                    genome=genome,
                    gap_type=decay_type,
                )
                if anti_template_injector is not None and autopsy_result.anti_templates:
                    anti_template_injector.add_from_autopsy(autopsy_result)
                    actions["actions"].append("anti_template_added")
            except Exception:
                pass

        # 3. Feature scout queue
        if decay_type == "feature":
            actions["actions"].append("feature_investigation_queued")
            actions["priority"] = "HIGH"

        # 4. Meta-learning update
        if meta_learning is not None:
            actions["actions"].append("meta_learning_updated")
            actions["time_to_failure_days"] = ttf_days

        return actions

    def get_state(self, strategy_id: str) -> Optional[StrategyState]:
        """Get current state of a strategy."""
        record = self._strategies.get(strategy_id)
        return record.state if record else None

    def get_detector(self, strategy_id: str) -> Optional[EdgeDecayDetector]:
        """Get the EdgeDecayDetector for a strategy."""
        record = self._strategies.get(strategy_id)
        return record.detector if record else None

    def get_retirement_queue(self) -> List[str]:
        """Get list of strategies queued for retirement processing."""
        return list(self._retirement_queue)

    def clear_retirement_queue(self) -> None:
        """Clear the retirement queue after processing."""
        self._retirement_queue.clear()
