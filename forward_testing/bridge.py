"""
Forward Testing Shadow Bridge

Connects HIFA-validated strategies to the shadow trading infrastructure.
Manages strategy deployment, execution monitoring, and performance tracking.

Migrated from Hinance V2 shadow_bridge.py with integration to shared module.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import asyncio
import logging
from collections import defaultdict

from shared.unified_strategy import (
    UnifiedStrategy,
    StrategyStatus,
    ForwardTestResult,
)
from shared.adapters import HinanceAdapter
from shared.constants import (
    FORWARD_TEST_CONFIG,
    HIFA_THRESHOLDS,
)

from .models import (
    Order,
    Trade,
    Position,
    Account,
    ExecutionResult,
    MarketState,
    OrderSide,
    OrderType,
    VolatilityRegime,
    PositionSide,
)
from .execution.engine import ExecutionEngine
from .analytics.performance import (
    PerformanceAnalyzer,
    TradeRecord,
    PerformanceMetrics,
    TransferMetrics,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Structures
# ==============================================================================

class DeploymentStatus(Enum):
    """Strategy deployment status."""
    PENDING = "PENDING"
    DEPLOYING = "DEPLOYING"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class ShadowPerformance:
    """Real-time shadow trading performance snapshot."""
    strategy_id: str = ""

    # Time tracking
    start_time: datetime = field(default_factory=datetime.now)
    current_time: datetime = field(default_factory=datetime.now)
    duration_days: float = 0.0

    # Performance metrics
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Transfer metrics (vs backtest)
    transfer_ratio: float = 0.0
    drawdown_expansion: float = 0.0

    # Execution quality
    avg_slippage_bps: float = 0.0
    total_fees: float = 0.0

    # Status
    current_position: Optional[Position] = None
    pending_orders: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_id': self.strategy_id,
            'duration_days': self.duration_days,
            'total_pnl': self.total_pnl,
            'total_return_pct': self.total_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'current_drawdown_pct': self.current_drawdown_pct,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'transfer_ratio': self.transfer_ratio,
            'avg_slippage_bps': self.avg_slippage_bps,
        }


@dataclass
class DeploymentResult:
    """Result of strategy deployment to shadow trading."""
    strategy_id: str = ""
    status: DeploymentStatus = DeploymentStatus.PENDING

    # Deployment details
    deployed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Performance summary
    performance: Optional[ShadowPerformance] = None
    transfer_metrics: Optional[TransferMetrics] = None

    # Validation outcome
    passed_forward_test: bool = False
    failure_reason: Optional[str] = None

    # Errors
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_id': self.strategy_id,
            'status': self.status.value,
            'deployed_at': self.deployed_at.isoformat() if self.deployed_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'passed_forward_test': self.passed_forward_test,
            'failure_reason': self.failure_reason,
            'errors': self.errors,
            'performance': self.performance.to_dict() if self.performance else None,
            'transfer_metrics': self.transfer_metrics.to_dict() if self.transfer_metrics else None,
        }


@dataclass
class ShadowSession:
    """Active shadow trading session for a strategy."""
    strategy_id: str
    strategy: UnifiedStrategy

    # Components
    execution_engine: ExecutionEngine = field(default_factory=ExecutionEngine)
    analyzer: PerformanceAnalyzer = field(default_factory=PerformanceAnalyzer)

    # State
    status: DeploymentStatus = DeploymentStatus.PENDING
    account: Account = field(default_factory=Account)
    positions: Dict[str, Position] = field(default_factory=dict)
    pending_orders: Dict[str, Order] = field(default_factory=dict)

    # Configuration
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% of capital
    target_leverage: float = 1.0

    # Timing
    started_at: Optional[datetime] = None
    last_signal_at: Optional[datetime] = None
    last_trade_at: Optional[datetime] = None

    # Backtest reference for transfer ratio
    backtest_sharpe: float = 0.0
    backtest_return: float = 0.0
    backtest_max_dd: float = 0.0
    backtest_trades: int = 0


# ==============================================================================
# Shadow Bridge
# ==============================================================================

class ForwardTestingBridge:
    """
    Bridge between HIFA validation and shadow trading infrastructure.

    Responsibilities:
    - Deploy HIFA-validated strategies to shadow trading
    - Manage concurrent shadow sessions (up to 50)
    - Monitor performance and calculate transfer metrics
    - Determine forward test pass/fail

    Usage:
        bridge = ForwardTestingBridge()

        # Deploy strategy
        result = await bridge.deploy_strategy(strategy)

        # Get performance
        perf = bridge.get_performance(strategy_id)

        # Complete forward test
        final_result = await bridge.complete_forward_test(strategy_id)
    """

    # Configuration
    MAX_CONCURRENT_STRATEGIES = 50
    DEFAULT_SHADOW_DURATION_DAYS = 14
    MIN_SHADOW_DURATION_DAYS = 7
    MAX_SHADOW_DURATION_DAYS = 28

    # Transfer validation thresholds
    TRANSFER_RATIO_MIN = 0.5
    DRAWDOWN_RATIO_MAX = 1.5
    MIN_TRADES_REQUIRED = 30

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_strategies: int = 50,
        shadow_duration_days: int = 14,
    ):
        """
        Initialize the forward testing bridge.

        Args:
            initial_capital: Starting capital per strategy
            max_strategies: Maximum concurrent shadow strategies
            shadow_duration_days: Default shadow trading duration
        """
        self.initial_capital = initial_capital
        self.max_strategies = min(max_strategies, self.MAX_CONCURRENT_STRATEGIES)
        self.shadow_duration_days = shadow_duration_days

        # Active sessions
        self.sessions: Dict[str, ShadowSession] = {}

        # Completed results
        self.completed_results: Dict[str, DeploymentResult] = {}

        # Callbacks
        self.on_trade_callback: Optional[Callable] = None
        self.on_signal_callback: Optional[Callable] = None
        self.on_completion_callback: Optional[Callable] = None

        # Statistics
        self.total_deployed = 0
        self.total_passed = 0
        self.total_failed = 0

        logger.info(f"ForwardTestingBridge initialized: max_strategies={max_strategies}, duration={shadow_duration_days}d")

    async def deploy_strategy(
        self,
        strategy: UnifiedStrategy,
        shadow_duration_days: Optional[int] = None,
        initial_capital: Optional[float] = None,
    ) -> DeploymentResult:
        """
        Deploy a HIFA-validated strategy to shadow trading.

        Args:
            strategy: UnifiedStrategy that passed HIFA validation
            shadow_duration_days: Override default shadow period
            initial_capital: Override default starting capital

        Returns:
            DeploymentResult with deployment status
        """
        result = DeploymentResult(strategy_id=strategy.strategy_id)

        # Validate strategy
        if not strategy.is_hifa_validated:
            result.status = DeploymentStatus.FAILED
            result.failure_reason = "Strategy must pass HIFA validation before forward testing"
            result.errors.append("HIFA validation required")
            return result

        # Check capacity
        if len(self.sessions) >= self.max_strategies:
            result.status = DeploymentStatus.FAILED
            result.failure_reason = f"Maximum concurrent strategies ({self.max_strategies}) reached"
            result.errors.append("Capacity limit reached")
            return result

        # Check for duplicate
        if strategy.strategy_id in self.sessions:
            result.status = DeploymentStatus.FAILED
            result.failure_reason = "Strategy already deployed"
            result.errors.append("Duplicate deployment")
            return result

        try:
            # Create session
            capital = initial_capital or self.initial_capital
            duration = shadow_duration_days or self.shadow_duration_days

            session = ShadowSession(
                strategy_id=strategy.strategy_id,
                strategy=strategy,
                execution_engine=ExecutionEngine(),
                analyzer=PerformanceAnalyzer(initial_capital=capital),
                initial_capital=capital,
            )

            # Set backtest reference from HIFA result
            if strategy.hifa_result:
                session.backtest_sharpe = strategy.hifa_result.backtest_sharpe
                session.backtest_return = getattr(strategy.hifa_result, 'backtest_return', 0)
                session.backtest_max_dd = strategy.hifa_result.backtest_max_dd
                session.backtest_trades = getattr(strategy.hifa_result.backtest_metrics, 'total_trades', 0) if strategy.hifa_result.backtest_metrics else 0

            # Initialize account
            session.account.wallet_balance = capital
            session.account.available_balance = capital

            # Start session
            session.status = DeploymentStatus.ACTIVE
            session.started_at = datetime.now()

            self.sessions[strategy.strategy_id] = session
            self.total_deployed += 1

            result.status = DeploymentStatus.ACTIVE
            result.deployed_at = session.started_at

            logger.info(f"Deployed strategy {strategy.strategy_id} to shadow trading")

        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.failure_reason = str(e)
            result.errors.append(f"Deployment error: {e}")
            logger.error(f"Failed to deploy strategy {strategy.strategy_id}: {e}")

        return result

    async def process_signal(
        self,
        strategy_id: str,
        signal: Dict[str, Any],
        market_state: MarketState,
    ) -> Optional[ExecutionResult]:
        """
        Process a trading signal from a shadow strategy.

        Args:
            strategy_id: Strategy that generated the signal
            signal: Signal dictionary with action, size, etc.
            market_state: Current market state

        Returns:
            ExecutionResult if trade executed, None otherwise
        """
        session = self.sessions.get(strategy_id)
        if not session or session.status != DeploymentStatus.ACTIVE:
            return None

        session.last_signal_at = datetime.now()

        # Parse signal
        action = signal.get('action', 'HOLD')
        if action == 'HOLD':
            return None

        # Determine order parameters
        side = OrderSide.BUY if action in ['BUY', 'LONG'] else OrderSide.SELL
        size = signal.get('size', session.max_position_size)

        # Calculate position size
        position_value = session.account.available_balance * size
        quantity = position_value / market_state.last_price

        # Check if we have an existing position
        current_position = session.positions.get(market_state.symbol)

        if current_position and current_position.size > 0:
            # Check if signal is to close position
            if self._should_close_position(current_position, side):
                return await self._close_position(session, current_position, market_state)

        # Execute new trade
        result = await session.execution_engine.execute_market_order(
            symbol=market_state.symbol,
            side=side,
            quantity=quantity,
            market_state=market_state,
        )

        if result.success:
            await self._process_fill(session, result, market_state)
            session.last_trade_at = datetime.now()

            if self.on_trade_callback:
                self.on_trade_callback(strategy_id, result)

        return result

    async def _close_position(
        self,
        session: ShadowSession,
        position: Position,
        market_state: MarketState,
    ) -> ExecutionResult:
        """Close an existing position."""
        # Opposite side to close
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY

        result = await session.execution_engine.execute_market_order(
            symbol=position.symbol,
            side=close_side,
            quantity=position.size,
            market_state=market_state,
            reduce_only=True,
        )

        if result.success:
            # Calculate realized P&L
            if position.side == PositionSide.LONG:
                pnl = (result.avg_fill_price - position.entry_price) * position.size
            else:
                pnl = (position.entry_price - result.avg_fill_price) * position.size

            pnl -= result.fee

            # Create trade record
            trade_record = TradeRecord(
                trade_id=result.order_id,
                strategy_id=session.strategy_id,
                symbol=position.symbol,
                side=position.side.value,
                entry_time=position.opened_at,
                entry_price=position.entry_price,
                entry_size=position.size,
                exit_time=datetime.now(),
                exit_price=result.avg_fill_price,
                exit_fee=result.fee,
                exit_slippage_bps=result.slippage_total_bps,
                exit_reason='signal',
                gross_pnl=pnl + result.fee,
                net_pnl=pnl,
                return_pct=(pnl / (position.entry_price * position.size)) * 100,
                regime_at_exit=market_state.volatility_regime,
            )

            session.analyzer.add_trade(trade_record)

            # Update account
            session.account.wallet_balance += pnl
            session.account.update_available_balance()

            # Remove position
            del session.positions[position.symbol]

        return result

    async def _process_fill(
        self,
        session: ShadowSession,
        result: ExecutionResult,
        market_state: MarketState,
    ) -> None:
        """Process a filled order."""
        # Create/update position
        position_side = PositionSide.LONG if result.side == OrderSide.BUY else PositionSide.SHORT

        position = Position(
            strategy_id=session.strategy_id,
            symbol=result.symbol,
            side=position_side,
            size=result.filled_qty,
            entry_price=result.avg_fill_price,
            mark_price=market_state.last_price,
            margin=result.filled_qty * result.avg_fill_price / session.target_leverage,
            leverage=session.target_leverage,
        )

        session.positions[result.symbol] = position

        # Update account margin
        session.account.position_margin += position.margin
        session.account.update_available_balance()

    def _should_close_position(self, position: Position, signal_side: OrderSide) -> bool:
        """Check if signal indicates closing the position."""
        if position.side == PositionSide.LONG and signal_side == OrderSide.SELL:
            return True
        if position.side == PositionSide.SHORT and signal_side == OrderSide.BUY:
            return True
        return False

    def get_performance(self, strategy_id: str) -> Optional[ShadowPerformance]:
        """
        Get current performance snapshot for a strategy.

        Args:
            strategy_id: Strategy to get performance for

        Returns:
            ShadowPerformance snapshot or None if not found
        """
        session = self.sessions.get(strategy_id)
        if not session:
            return None

        # Calculate current metrics
        metrics = session.analyzer.calculate_metrics()

        # Calculate transfer metrics if we have backtest reference
        transfer_ratio = 0.0
        drawdown_expansion = 0.0

        if session.backtest_sharpe > 0:
            transfer_ratio = metrics.sharpe_ratio / session.backtest_sharpe

        if session.backtest_max_dd > 0 and metrics.drawdown.max_drawdown_pct > 0:
            drawdown_expansion = metrics.drawdown.max_drawdown_pct / session.backtest_max_dd

        # Get current position
        current_position = None
        if session.positions:
            current_position = list(session.positions.values())[0]

        return ShadowPerformance(
            strategy_id=strategy_id,
            start_time=session.started_at or datetime.now(),
            current_time=datetime.now(),
            duration_days=metrics.duration_days,
            total_pnl=metrics.total_pnl,
            total_return_pct=metrics.total_return_pct,
            sharpe_ratio=metrics.sharpe_ratio,
            max_drawdown_pct=metrics.drawdown.max_drawdown_pct,
            current_drawdown_pct=metrics.drawdown.current_drawdown_pct,
            total_trades=metrics.total_trades,
            winning_trades=metrics.winning_trades,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            transfer_ratio=transfer_ratio,
            drawdown_expansion=drawdown_expansion,
            avg_slippage_bps=metrics.avg_slippage_bps,
            total_fees=metrics.total_fees,
            current_position=current_position,
            pending_orders=len(session.pending_orders),
        )

    async def complete_forward_test(
        self,
        strategy_id: str,
        force: bool = False,
    ) -> DeploymentResult:
        """
        Complete forward testing and evaluate results.

        Args:
            strategy_id: Strategy to complete testing for
            force: Force completion even if minimum duration not reached

        Returns:
            DeploymentResult with final evaluation
        """
        session = self.sessions.get(strategy_id)
        if not session:
            return DeploymentResult(
                strategy_id=strategy_id,
                status=DeploymentStatus.FAILED,
                failure_reason="Session not found",
            )

        # Check minimum duration
        if session.started_at:
            duration = (datetime.now() - session.started_at).total_seconds() / 86400
            if duration < self.MIN_SHADOW_DURATION_DAYS and not force:
                return DeploymentResult(
                    strategy_id=strategy_id,
                    status=DeploymentStatus.ACTIVE,
                    failure_reason=f"Minimum duration ({self.MIN_SHADOW_DURATION_DAYS}d) not reached",
                )

        # Close any open positions
        for symbol, position in list(session.positions.items()):
            # Create mock market state for closing
            market_state = MarketState(
                symbol=symbol,
                last_price=position.mark_price,
                bid=position.mark_price * 0.9999,
                ask=position.mark_price * 1.0001,
            )
            await self._close_position(session, position, market_state)

        # Calculate final metrics
        metrics = session.analyzer.calculate_metrics()

        # Calculate transfer metrics
        transfer_metrics = session.analyzer.calculate_transfer_metrics(
            backtest_sharpe=session.backtest_sharpe,
            backtest_return=session.backtest_return,
            backtest_max_dd=session.backtest_max_dd,
            backtest_trades=session.backtest_trades,
        )

        # Determine pass/fail
        passed = (
            transfer_metrics.transfer_ratio >= self.TRANSFER_RATIO_MIN and
            transfer_metrics.drawdown_expansion <= self.DRAWDOWN_RATIO_MAX and
            metrics.total_trades >= self.MIN_TRADES_REQUIRED
        )

        # Create result
        result = DeploymentResult(
            strategy_id=strategy_id,
            status=DeploymentStatus.COMPLETED,
            deployed_at=session.started_at,
            completed_at=datetime.now(),
            performance=self.get_performance(strategy_id),
            transfer_metrics=transfer_metrics,
            passed_forward_test=passed,
        )

        if not passed:
            reasons = []
            if transfer_metrics.transfer_ratio < self.TRANSFER_RATIO_MIN:
                reasons.append(f"Transfer ratio {transfer_metrics.transfer_ratio:.2f} < {self.TRANSFER_RATIO_MIN}")
            if transfer_metrics.drawdown_expansion > self.DRAWDOWN_RATIO_MAX:
                reasons.append(f"Drawdown expansion {transfer_metrics.drawdown_expansion:.2f} > {self.DRAWDOWN_RATIO_MAX}")
            if metrics.total_trades < self.MIN_TRADES_REQUIRED:
                reasons.append(f"Trades {metrics.total_trades} < {self.MIN_TRADES_REQUIRED}")
            result.failure_reason = "; ".join(reasons)

        # Update strategy with forward test result
        session.strategy.forward_result = ForwardTestResult(
            transfer_ratio=transfer_metrics.transfer_ratio,
            shadow_sharpe=metrics.sharpe_ratio,
            shadow_max_dd=metrics.drawdown.max_drawdown_pct,
            drawdown_ratio=transfer_metrics.drawdown_expansion,
            shadow_trades=metrics.total_trades,
            passed=passed,
        )

        # Update statistics
        if passed:
            self.total_passed += 1
            session.strategy.status = StrategyStatus.PRODUCTION_READY
        else:
            self.total_failed += 1
            session.strategy.status = StrategyStatus.FORWARD_FAILED

        # Move to completed
        self.completed_results[strategy_id] = result
        del self.sessions[strategy_id]

        # Callback
        if self.on_completion_callback:
            self.on_completion_callback(strategy_id, result)

        logger.info(f"Forward test completed for {strategy_id}: passed={passed}")

        return result

    def pause_strategy(self, strategy_id: str) -> bool:
        """Pause shadow trading for a strategy."""
        session = self.sessions.get(strategy_id)
        if session and session.status == DeploymentStatus.ACTIVE:
            session.status = DeploymentStatus.PAUSED
            return True
        return False

    def resume_strategy(self, strategy_id: str) -> bool:
        """Resume shadow trading for a strategy."""
        session = self.sessions.get(strategy_id)
        if session and session.status == DeploymentStatus.PAUSED:
            session.status = DeploymentStatus.ACTIVE
            return True
        return False

    def stop_strategy(self, strategy_id: str) -> bool:
        """Stop and remove a strategy from shadow trading."""
        session = self.sessions.get(strategy_id)
        if session:
            session.status = DeploymentStatus.STOPPED
            del self.sessions[strategy_id]
            return True
        return False

    def get_active_strategies(self) -> List[str]:
        """Get list of active strategy IDs."""
        return [
            sid for sid, session in self.sessions.items()
            if session.status == DeploymentStatus.ACTIVE
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            'total_deployed': self.total_deployed,
            'total_passed': self.total_passed,
            'total_failed': self.total_failed,
            'active_sessions': len(self.sessions),
            'pass_rate': self.total_passed / self.total_deployed if self.total_deployed > 0 else 0,
            'capacity_used': len(self.sessions) / self.max_strategies,
        }
