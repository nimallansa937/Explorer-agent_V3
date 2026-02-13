"""
Transfer Gate (Gate 8) - Forward Testing Validation

The final validation gate that determines if a strategy's live performance
matches its historical backtest. This is the critical bridge between
paper trading and production deployment.

Key Metrics:
- Transfer Ratio: shadow_sharpe / backtest_sharpe (min 0.5)
- Drawdown Ratio: shadow_max_dd / backtest_max_dd (max 1.5)
- Trade Count: Minimum trades required for statistical validity
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
import statistics
import math

from shared.unified_strategy import (
    UnifiedStrategy,
    ForwardTestResult,
    ShadowMetrics,
    StrategyStatus,
)
from shared.constants import FORWARD_TEST_CONFIG

from .analytics.performance import (
    PerformanceMetrics,
    TransferMetrics,
    DrawdownMetrics,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class TransferGateConfig:
    """Configuration for Transfer Gate validation."""

    # Core thresholds
    min_transfer_ratio: float = 0.5  # shadow_sharpe / backtest_sharpe
    max_drawdown_ratio: float = 1.5  # shadow_max_dd / backtest_max_dd
    min_trades: int = 30  # Minimum trades for statistical validity

    # Secondary thresholds
    max_return_degradation: float = 0.5  # Max 50% return loss
    max_win_rate_degradation: float = 0.15  # Max 15% win rate drop
    max_profit_factor_degradation: float = 0.4  # Max 40% profit factor drop

    # Execution quality thresholds
    max_avg_slippage_bps: float = 20.0  # Max average slippage
    max_execution_drag_pct: float = 2.0  # Max total execution cost

    # Duration requirements
    min_duration_days: int = 7
    recommended_duration_days: int = 14
    max_duration_days: int = 28

    # Confidence adjustments
    require_positive_pnl: bool = False  # Can pass with negative PnL if ratios ok
    require_positive_sharpe: bool = True  # Shadow Sharpe must be positive

    @classmethod
    def from_constants(cls) -> 'TransferGateConfig':
        """Create config from shared constants."""
        return cls(
            min_transfer_ratio=FORWARD_TEST_CONFIG.get('TRANSFER_RATIO_MIN', 0.5),
            max_drawdown_ratio=FORWARD_TEST_CONFIG.get('MAX_DD_RATIO', 1.5),
            min_trades=FORWARD_TEST_CONFIG.get('MIN_TRADES', 30),
            min_duration_days=FORWARD_TEST_CONFIG.get('MIN_DURATION_DAYS', 7),
            recommended_duration_days=FORWARD_TEST_CONFIG.get('SHADOW_DURATION_DAYS', 14),
            max_duration_days=FORWARD_TEST_CONFIG.get('MAX_DURATION_DAYS', 28),
        )


# ==============================================================================
# Gate Result
# ==============================================================================

class GateCheckResult(Enum):
    """Individual gate check result."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


@dataclass
class GateCheck:
    """Result of a single gate check."""
    name: str
    result: GateCheckResult
    actual_value: float
    threshold: float
    message: str = ""
    weight: float = 1.0  # Importance weight for scoring

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'result': self.result.value,
            'actual_value': self.actual_value,
            'threshold': self.threshold,
            'message': self.message,
            'weight': self.weight,
        }


@dataclass
class TransferGateResult:
    """Complete result of Transfer Gate evaluation."""

    # Overall result
    passed: bool = False
    confidence_score: float = 0.0  # 0-1 confidence in result
    gate_score: float = 0.0  # Weighted score across all checks

    # Individual checks
    checks: List[GateCheck] = field(default_factory=list)

    # Summary metrics
    transfer_ratio: float = 0.0
    drawdown_ratio: float = 0.0
    return_degradation: float = 0.0

    # Backtest reference
    backtest_sharpe: float = 0.0
    backtest_max_dd: float = 0.0
    backtest_trades: int = 0

    # Shadow results
    shadow_sharpe: float = 0.0
    shadow_max_dd: float = 0.0
    shadow_trades: int = 0
    shadow_duration_days: float = 0.0

    # Execution impact
    total_slippage_drag: float = 0.0
    total_fee_drag: float = 0.0

    # Failure details
    failure_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Timestamps
    evaluated_at: datetime = field(default_factory=datetime.now)

    @property
    def passed_checks(self) -> int:
        return sum(1 for c in self.checks if c.result == GateCheckResult.PASSED)

    @property
    def failed_checks(self) -> int:
        return sum(1 for c in self.checks if c.result == GateCheckResult.FAILED)

    @property
    def warning_checks(self) -> int:
        return sum(1 for c in self.checks if c.result == GateCheckResult.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'confidence_score': self.confidence_score,
            'gate_score': self.gate_score,
            'transfer_ratio': self.transfer_ratio,
            'drawdown_ratio': self.drawdown_ratio,
            'backtest_sharpe': self.backtest_sharpe,
            'shadow_sharpe': self.shadow_sharpe,
            'shadow_trades': self.shadow_trades,
            'shadow_duration_days': self.shadow_duration_days,
            'passed_checks': self.passed_checks,
            'failed_checks': self.failed_checks,
            'failure_reasons': self.failure_reasons,
            'warnings': self.warnings,
            'checks': [c.to_dict() for c in self.checks],
            'evaluated_at': self.evaluated_at.isoformat(),
        }

    def to_forward_test_result(self) -> ForwardTestResult:
        """Convert to ForwardTestResult for UnifiedStrategy."""
        return ForwardTestResult(
            transfer_ratio=self.transfer_ratio,
            shadow_sharpe=self.shadow_sharpe,
            shadow_max_dd=self.shadow_max_dd,
            drawdown_ratio=self.drawdown_ratio,
            shadow_trades=self.shadow_trades,
            passed=self.passed,
            shadow_metrics=ShadowMetrics(
                duration_days=self.shadow_duration_days,
                total_trades=self.shadow_trades,
                sharpe_ratio=self.shadow_sharpe,
                max_drawdown=self.shadow_max_dd,
            ),
        )


# ==============================================================================
# Transfer Gate
# ==============================================================================

class TransferGate:
    """
    Gate 8 - Transfer Ratio Validation

    Evaluates whether a strategy's shadow trading performance is consistent
    with its historical backtest results. This is the final gate before
    production deployment.

    The key insight is that strategies often perform worse in live trading
    due to:
    - Execution costs (slippage, fees)
    - Market impact
    - Timing differences
    - Regime changes

    The transfer ratio captures this degradation:
        transfer_ratio = shadow_sharpe / backtest_sharpe

    A ratio of 0.5 means the strategy retains 50% of its backtest
    risk-adjusted performance, which is acceptable for deployment.

    Usage:
        gate = TransferGate()

        # Evaluate strategy
        result = gate.evaluate(
            strategy=strategy,
            shadow_metrics=performance_metrics,
            transfer_metrics=transfer_metrics,
        )

        if result.passed:
            # Strategy ready for production
            strategy.status = StrategyStatus.PRODUCTION_READY
    """

    def __init__(self, config: Optional[TransferGateConfig] = None):
        """
        Initialize Transfer Gate.

        Args:
            config: Gate configuration (uses defaults if None)
        """
        self.config = config or TransferGateConfig.from_constants()
        self.evaluation_count = 0
        self.pass_count = 0
        self.fail_count = 0

        logger.info(f"TransferGate initialized: min_ratio={self.config.min_transfer_ratio}, max_dd_ratio={self.config.max_drawdown_ratio}")

    def evaluate(
        self,
        strategy: UnifiedStrategy,
        shadow_metrics: PerformanceMetrics,
        transfer_metrics: Optional[TransferMetrics] = None,
    ) -> TransferGateResult:
        """
        Evaluate a strategy against Transfer Gate criteria.

        Args:
            strategy: UnifiedStrategy with HIFA results
            shadow_metrics: Performance metrics from shadow trading
            transfer_metrics: Pre-calculated transfer metrics (optional)

        Returns:
            TransferGateResult with detailed evaluation
        """
        result = TransferGateResult()
        self.evaluation_count += 1

        # Validate inputs
        if not strategy.hifa_result:
            result.failure_reasons.append("Strategy missing HIFA results")
            return result

        # Extract backtest reference
        backtest_sharpe = strategy.hifa_result.backtest_sharpe
        backtest_max_dd = strategy.hifa_result.backtest_max_dd
        backtest_trades = 0
        backtest_win_rate = 0.0
        backtest_profit_factor = 0.0

        if strategy.hifa_result.backtest_metrics:
            backtest_trades = strategy.hifa_result.backtest_metrics.total_trades
            backtest_win_rate = strategy.hifa_result.backtest_metrics.win_rate
            backtest_profit_factor = strategy.hifa_result.backtest_metrics.profit_factor

        result.backtest_sharpe = backtest_sharpe
        result.backtest_max_dd = backtest_max_dd
        result.backtest_trades = backtest_trades

        # Extract shadow results
        result.shadow_sharpe = shadow_metrics.sharpe_ratio
        result.shadow_max_dd = shadow_metrics.drawdown.max_drawdown_pct
        result.shadow_trades = shadow_metrics.total_trades
        result.shadow_duration_days = shadow_metrics.duration_days

        # Calculate transfer metrics if not provided
        if transfer_metrics:
            result.transfer_ratio = transfer_metrics.transfer_ratio
            result.drawdown_ratio = transfer_metrics.drawdown_expansion
            result.return_degradation = transfer_metrics.return_degradation
            result.total_slippage_drag = transfer_metrics.slippage_drag_pct
            result.total_fee_drag = transfer_metrics.fee_drag_pct
        else:
            # Calculate from raw metrics
            if backtest_sharpe > 0:
                result.transfer_ratio = shadow_metrics.sharpe_ratio / backtest_sharpe
            if backtest_max_dd > 0:
                result.drawdown_ratio = shadow_metrics.drawdown.max_drawdown_pct / backtest_max_dd

        # Run all checks
        checks = []

        # 1. Transfer Ratio Check (Critical)
        checks.append(self._check_transfer_ratio(result.transfer_ratio))

        # 2. Drawdown Ratio Check (Critical)
        checks.append(self._check_drawdown_ratio(result.drawdown_ratio))

        # 3. Minimum Trades Check (Critical)
        checks.append(self._check_minimum_trades(shadow_metrics.total_trades))

        # 4. Duration Check
        checks.append(self._check_duration(shadow_metrics.duration_days))

        # 5. Positive Sharpe Check (if required)
        if self.config.require_positive_sharpe:
            checks.append(self._check_positive_sharpe(shadow_metrics.sharpe_ratio))

        # 6. Win Rate Degradation Check
        if backtest_win_rate > 0:
            win_rate_degradation = (backtest_win_rate - shadow_metrics.win_rate) / backtest_win_rate
            checks.append(self._check_win_rate_degradation(win_rate_degradation))

        # 7. Profit Factor Degradation Check
        if backtest_profit_factor > 0 and shadow_metrics.profit_factor > 0:
            pf_degradation = 1 - (shadow_metrics.profit_factor / backtest_profit_factor)
            checks.append(self._check_profit_factor_degradation(pf_degradation))

        # 8. Execution Quality Check
        checks.append(self._check_execution_quality(shadow_metrics.avg_slippage_bps))

        result.checks = checks

        # Calculate gate score
        result.gate_score = self._calculate_gate_score(checks)

        # Calculate confidence score
        result.confidence_score = self._calculate_confidence(
            shadow_metrics.total_trades,
            shadow_metrics.duration_days,
        )

        # Determine pass/fail
        critical_checks_passed = all(
            c.result == GateCheckResult.PASSED
            for c in checks
            if c.name in ['transfer_ratio', 'drawdown_ratio', 'minimum_trades']
        )

        result.passed = critical_checks_passed

        # Collect failure reasons
        for check in checks:
            if check.result == GateCheckResult.FAILED:
                result.failure_reasons.append(check.message)
            elif check.result == GateCheckResult.WARNING:
                result.warnings.append(check.message)

        # Update statistics
        if result.passed:
            self.pass_count += 1
        else:
            self.fail_count += 1

        logger.info(
            f"TransferGate evaluation: passed={result.passed}, "
            f"transfer_ratio={result.transfer_ratio:.3f}, "
            f"score={result.gate_score:.3f}"
        )

        return result

    def _check_transfer_ratio(self, ratio: float) -> GateCheck:
        """Check transfer ratio threshold."""
        passed = ratio >= self.config.min_transfer_ratio

        # Determine result level
        if passed:
            if ratio >= 0.7:
                result = GateCheckResult.PASSED
                msg = f"Excellent transfer ratio: {ratio:.3f}"
            else:
                result = GateCheckResult.PASSED
                msg = f"Transfer ratio acceptable: {ratio:.3f}"
        else:
            result = GateCheckResult.FAILED
            msg = f"Transfer ratio {ratio:.3f} below minimum {self.config.min_transfer_ratio}"

        return GateCheck(
            name='transfer_ratio',
            result=result,
            actual_value=ratio,
            threshold=self.config.min_transfer_ratio,
            message=msg,
            weight=3.0,  # Critical check
        )

    def _check_drawdown_ratio(self, ratio: float) -> GateCheck:
        """Check drawdown expansion ratio."""
        passed = ratio <= self.config.max_drawdown_ratio

        if passed:
            if ratio <= 1.0:
                result = GateCheckResult.PASSED
                msg = f"Drawdown contained: {ratio:.3f}x backtest"
            else:
                result = GateCheckResult.PASSED
                msg = f"Drawdown expansion acceptable: {ratio:.3f}x"
        else:
            result = GateCheckResult.FAILED
            msg = f"Drawdown expansion {ratio:.3f}x exceeds max {self.config.max_drawdown_ratio}x"

        return GateCheck(
            name='drawdown_ratio',
            result=result,
            actual_value=ratio,
            threshold=self.config.max_drawdown_ratio,
            message=msg,
            weight=2.5,
        )

    def _check_minimum_trades(self, trades: int) -> GateCheck:
        """Check minimum trade count."""
        passed = trades >= self.config.min_trades

        if passed:
            result = GateCheckResult.PASSED
            msg = f"Sufficient trades: {trades}"
        else:
            result = GateCheckResult.FAILED
            msg = f"Insufficient trades: {trades} < {self.config.min_trades} required"

        return GateCheck(
            name='minimum_trades',
            result=result,
            actual_value=float(trades),
            threshold=float(self.config.min_trades),
            message=msg,
            weight=2.0,
        )

    def _check_duration(self, days: float) -> GateCheck:
        """Check shadow trading duration."""
        if days >= self.config.recommended_duration_days:
            result = GateCheckResult.PASSED
            msg = f"Full evaluation period: {days:.1f} days"
        elif days >= self.config.min_duration_days:
            result = GateCheckResult.WARNING
            msg = f"Short evaluation: {days:.1f} days (recommended: {self.config.recommended_duration_days})"
        else:
            result = GateCheckResult.FAILED
            msg = f"Insufficient duration: {days:.1f} days < {self.config.min_duration_days} minimum"

        return GateCheck(
            name='duration',
            result=result,
            actual_value=days,
            threshold=float(self.config.min_duration_days),
            message=msg,
            weight=1.0,
        )

    def _check_positive_sharpe(self, sharpe: float) -> GateCheck:
        """Check for positive Sharpe ratio."""
        if sharpe > 0.5:
            result = GateCheckResult.PASSED
            msg = f"Strong Sharpe ratio: {sharpe:.3f}"
        elif sharpe > 0:
            result = GateCheckResult.PASSED
            msg = f"Positive Sharpe: {sharpe:.3f}"
        else:
            result = GateCheckResult.FAILED
            msg = f"Negative Sharpe ratio: {sharpe:.3f}"

        return GateCheck(
            name='positive_sharpe',
            result=result,
            actual_value=sharpe,
            threshold=0.0,
            message=msg,
            weight=1.5,
        )

    def _check_win_rate_degradation(self, degradation: float) -> GateCheck:
        """Check win rate degradation."""
        if degradation <= 0:
            result = GateCheckResult.PASSED
            msg = "Win rate improved in shadow trading"
        elif degradation <= self.config.max_win_rate_degradation:
            result = GateCheckResult.PASSED
            msg = f"Win rate degradation acceptable: {degradation:.1%}"
        else:
            result = GateCheckResult.WARNING
            msg = f"Win rate degradation high: {degradation:.1%}"

        return GateCheck(
            name='win_rate_degradation',
            result=result,
            actual_value=degradation,
            threshold=self.config.max_win_rate_degradation,
            message=msg,
            weight=0.8,
        )

    def _check_profit_factor_degradation(self, degradation: float) -> GateCheck:
        """Check profit factor degradation."""
        if degradation <= 0:
            result = GateCheckResult.PASSED
            msg = "Profit factor improved"
        elif degradation <= self.config.max_profit_factor_degradation:
            result = GateCheckResult.PASSED
            msg = f"Profit factor degradation acceptable: {degradation:.1%}"
        else:
            result = GateCheckResult.WARNING
            msg = f"Profit factor degradation high: {degradation:.1%}"

        return GateCheck(
            name='profit_factor_degradation',
            result=result,
            actual_value=degradation,
            threshold=self.config.max_profit_factor_degradation,
            message=msg,
            weight=0.8,
        )

    def _check_execution_quality(self, avg_slippage_bps: float) -> GateCheck:
        """Check execution quality via slippage."""
        if avg_slippage_bps <= self.config.max_avg_slippage_bps * 0.5:
            result = GateCheckResult.PASSED
            msg = f"Excellent execution: {avg_slippage_bps:.1f} bps avg slippage"
        elif avg_slippage_bps <= self.config.max_avg_slippage_bps:
            result = GateCheckResult.PASSED
            msg = f"Acceptable execution: {avg_slippage_bps:.1f} bps"
        else:
            result = GateCheckResult.WARNING
            msg = f"High slippage: {avg_slippage_bps:.1f} bps"

        return GateCheck(
            name='execution_quality',
            result=result,
            actual_value=avg_slippage_bps,
            threshold=self.config.max_avg_slippage_bps,
            message=msg,
            weight=0.5,
        )

    def _calculate_gate_score(self, checks: List[GateCheck]) -> float:
        """Calculate weighted gate score from all checks."""
        total_weight = sum(c.weight for c in checks)
        if total_weight == 0:
            return 0.0

        score = 0.0
        for check in checks:
            if check.result == GateCheckResult.PASSED:
                score += check.weight * 1.0
            elif check.result == GateCheckResult.WARNING:
                score += check.weight * 0.5
            # FAILED and SKIPPED contribute 0

        return score / total_weight

    def _calculate_confidence(self, trades: int, days: float) -> float:
        """
        Calculate confidence in the evaluation result.

        More trades and longer duration = higher confidence.
        """
        # Trade confidence (logistic curve)
        trade_conf = 1 / (1 + math.exp(-0.1 * (trades - 50)))

        # Duration confidence
        duration_conf = min(1.0, days / self.config.recommended_duration_days)

        # Combined (geometric mean)
        return math.sqrt(trade_conf * duration_conf)

    def apply_to_strategy(
        self,
        strategy: UnifiedStrategy,
        result: TransferGateResult,
    ) -> UnifiedStrategy:
        """
        Apply gate result to strategy.

        Args:
            strategy: Strategy to update
            result: Gate evaluation result

        Returns:
            Updated strategy
        """
        # Set forward test result
        strategy.forward_result = result.to_forward_test_result()

        # Update status
        if result.passed:
            strategy.status = StrategyStatus.PRODUCTION_READY
        else:
            strategy.status = StrategyStatus.FORWARD_FAILED

        return strategy

    def get_statistics(self) -> Dict[str, Any]:
        """Get gate statistics."""
        return {
            'total_evaluations': self.evaluation_count,
            'passed': self.pass_count,
            'failed': self.fail_count,
            'pass_rate': self.pass_count / self.evaluation_count if self.evaluation_count > 0 else 0,
            'config': {
                'min_transfer_ratio': self.config.min_transfer_ratio,
                'max_drawdown_ratio': self.config.max_drawdown_ratio,
                'min_trades': self.config.min_trades,
            }
        }
