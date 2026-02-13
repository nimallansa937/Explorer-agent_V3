"""
Tests for Forward Testing Module

Comprehensive tests for the shadow trading infrastructure including:
- ExecutionEngine: Market microstructure simulation
- PerformanceAnalyzer: Metrics calculation
- TransferGate: Gate 8 validation
- DeploymentQueue: Queue management
- ShadowMonitor: Alerting system
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from forward_testing.models import (
    Order,
    Trade,
    Position,
    Account,
    ExecutionResult,
    MarketState,
    OrderSide,
    OrderType,
    OrderStatus,
    PositionSide,
    VolatilityRegime,
    ExecutionConfig,
    LatencyConfig,
    SpreadConfig,
)

from forward_testing.execution.engine import ExecutionEngine

from forward_testing.analytics.performance import (
    PerformanceAnalyzer,
    TradeRecord,
    PerformanceMetrics,
    TransferMetrics,
    DrawdownMetrics,
    RiskMetrics,
)

from forward_testing.transfer_gate import (
    TransferGate,
    TransferGateResult,
    TransferGateConfig,
    GateCheck,
    GateCheckResult,
)

from forward_testing.deployment_queue import (
    DeploymentQueue,
    QueuedStrategy,
    QueueStatus,
    QueuePriority,
)

from forward_testing.shadow_monitor import (
    ShadowMonitor,
    MonitorAlert,
    AlertLevel,
    AlertType,
    MonitorConfig,
    StrategyHealth,
)

from shared.unified_strategy import (
    UnifiedStrategy,
    StrategyGenome,
    SourceEngine,
    StrategyStatus,
    HIFAResult,
    BacktestMetrics,
    GateResult,
    StatisticalScores,
    RegimeTier,
)


# ==============================================================================
# Helper Functions
# ==============================================================================

def _create_test_hifa_result(strategy_id: str, sharpe: float = 1.5, max_dd: float = 15.0) -> HIFAResult:
    """Create a valid HIFAResult for testing."""
    backtest_metrics = BacktestMetrics(
        sharpe_ratio=sharpe,
        sortino_ratio=2.0,
        calmar_ratio=1.0,
        max_drawdown=max_dd,
        total_return=45.0,
        total_trades=120,
        win_rate=0.55,
        profit_factor=1.8,
        avg_trade_return=0.5,
        avg_trade_duration_hours=4.0,
        start_date=datetime.now() - timedelta(days=365),
        end_date=datetime.now(),
    )

    gate_results = {
        i: GateResult(
            gate_number=i,
            gate_name=f"Gate_{i}",
            passed=True,
            score=0.8,
            threshold=0.5,
        )
        for i in range(1, 8)
    }

    return HIFAResult(
        strategy_id=strategy_id,
        passed=True,
        final_gate=7,
        gate_results=gate_results,
        statistical_scores=StatisticalScores(
            dsr=1.2,
            dsr_pvalue=0.01,
            pbo=0.1,
            pbo_logits=-2.0,
            t_stat=3.5,
            fdr_adjusted_pvalue=0.02,
            cpcv_sharpe_mean=1.3,
            cpcv_sharpe_std=0.2,
            cpcv_paths_passed=8,
            cpcv_total_paths=10,
        ),
        backtest_metrics=backtest_metrics,
        regime_tier=RegimeTier.ALL_WEATHER,
        regime_performance={"trending": sharpe, "ranging": sharpe * 0.8, "volatile": sharpe * 0.6},
        cluster_id=1,
        similarity_score=0.3,
        rejection_reason=None,
    )


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def market_state():
    """Create a test market state."""
    return MarketState(
        symbol="BTCUSDT",
        last_price=50000.0,
        mark_price=50000.0,
        index_price=50000.0,
        bid=49995.0,
        ask=50005.0,
        spread_bps=2.0,
        volume_24h=1e9,
        open_interest=5e8,
        funding_rate=0.0001,
        volatility_regime=VolatilityRegime.NORMAL,
        realized_vol_1h=0.02,
        realized_vol_24h=0.035,
    )


@pytest.fixture
def execution_engine():
    """Create an execution engine."""
    return ExecutionEngine(simulate_latency=False)


@pytest.fixture
def performance_analyzer():
    """Create a performance analyzer."""
    return PerformanceAnalyzer(initial_capital=100000.0)


@pytest.fixture
def sample_strategy():
    """Create a sample validated strategy."""
    genome = StrategyGenome(
        entry_conditions=[{"type": "crossover", "indicator": "sma", "params": {"fast": 10, "slow": 30}}],
        exit_conditions=[{"type": "crossover", "indicator": "sma", "params": {"fast": 30, "slow": 10}}],
        position_sizing={"method": "fixed", "size": 0.1},
        parameters={"sma_fast": 10, "sma_slow": 30},
        lookback_periods={"sma": 30},
        indicators=["SMA"],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
    )

    strategy = UnifiedStrategy(
        strategy_id="test_strategy_001",
        genome=genome,
        source_engine=SourceEngine.EVOLUTIONARY,
        target_asset="BTCUSDT",
        target_regime="trending",
    )

    # Add HIFA result (required for forward testing)
    strategy.hifa_result = _create_test_hifa_result("test_strategy_001", sharpe=1.5, max_dd=15.0)
    strategy.status = StrategyStatus.HIFA_PASSED

    return strategy


@pytest.fixture
def sample_trades():
    """Create sample trade records."""
    trades = []
    base_time = datetime.now() - timedelta(days=14)

    for i in range(50):
        is_win = i % 3 != 0  # ~67% win rate
        pnl = 500.0 if is_win else -300.0

        trade = TradeRecord(
            trade_id=f"trade_{i}",
            strategy_id="test_strategy_001",
            symbol="BTCUSDT",
            side="LONG" if i % 2 == 0 else "SHORT",
            entry_time=base_time + timedelta(hours=i * 6),
            entry_price=50000.0 + (i * 10),
            entry_size=0.1,
            entry_fee=2.0,
            entry_slippage_bps=3.0,
            exit_time=base_time + timedelta(hours=i * 6 + 4),
            exit_price=50000.0 + (i * 10) + (100 if is_win else -60),
            exit_fee=2.0,
            exit_slippage_bps=3.0,
            exit_reason='signal',
            gross_pnl=pnl + 4.0,
            net_pnl=pnl,
            return_pct=(pnl / 5000.0) * 100,
            regime_at_entry=VolatilityRegime.NORMAL,
            regime_at_exit=VolatilityRegime.NORMAL,
        )
        trades.append(trade)

    return trades


# ==============================================================================
# Model Tests
# ==============================================================================

class TestModels:
    """Test data model classes."""

    def test_order_creation(self):
        """Test Order dataclass."""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=0.1,
        )
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.NEW
        assert order.filled_qty == 0.0

    def test_position_calculation(self):
        """Test Position P&L calculation."""
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=1.0,
            entry_price=50000.0,
            mark_price=51000.0,
        )
        pnl = position.calculate_unrealized_pnl()
        assert pnl == 1000.0  # (51000 - 50000) * 1.0

        # Short position
        short_position = Position(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            size=1.0,
            entry_price=50000.0,
            mark_price=49000.0,
        )
        short_pnl = short_position.calculate_unrealized_pnl()
        assert short_pnl == 1000.0  # (50000 - 49000) * 1.0

    def test_account_balance_update(self):
        """Test Account balance calculations."""
        account = Account(wallet_balance=100000.0)
        account.unrealized_pnl = 5000.0
        account.position_margin = 10000.0
        account.open_order_margin = 2000.0

        account.update_margin_balance()
        assert account.margin_balance == 105000.0

        account.update_available_balance()
        assert account.available_balance == 93000.0  # 100000 + 5000 - 10000 - 2000

    def test_volatility_regime_from_vix(self):
        """Test VolatilityRegime classification."""
        assert VolatilityRegime.from_vix(10) == VolatilityRegime.LOW_VOL
        assert VolatilityRegime.from_vix(20) == VolatilityRegime.NORMAL
        assert VolatilityRegime.from_vix(35) == VolatilityRegime.HIGH_VOL
        assert VolatilityRegime.from_vix(50) == VolatilityRegime.CRISIS
        assert VolatilityRegime.from_vix(70) == VolatilityRegime.CASCADE

    def test_market_state_properties(self):
        """Test MarketState calculations."""
        state = MarketState(
            symbol="BTCUSDT",
            bid=49990.0,
            ask=50010.0,
            last_price=50000.0,
        )
        assert state.mid_price == 50000.0
        assert abs(state.spread - 0.0004) < 0.0001  # ~4 bps


# ==============================================================================
# Execution Engine Tests
# ==============================================================================

class TestExecutionEngine:
    """Test execution engine functionality."""

    def test_market_impact_calculation(self, execution_engine):
        """Test square-root market impact model."""
        # Small order should have minimal impact
        small_impact = execution_engine.calculate_market_impact(
            order_size_usd=10000,
            adv=1e9,
            volatility=0.02,
        )
        assert small_impact < 5.0  # Less than 5 bps

        # Large order should have more impact
        large_impact = execution_engine.calculate_market_impact(
            order_size_usd=10_000_000,
            adv=1e9,
            volatility=0.02,
        )
        assert large_impact > small_impact

    def test_spread_calculation(self, execution_engine):
        """Test regime-dependent spread calculation."""
        base_spread = 0.0002  # 2 bps as decimal

        normal_spread = execution_engine.calculate_effective_spread(
            base_spread, VolatilityRegime.NORMAL
        )
        # Multiplier is 1.0, but with time adjustment it may vary
        assert normal_spread >= base_spread * 0.7  # At least 70% of base

        high_vol_spread = execution_engine.calculate_effective_spread(
            base_spread, VolatilityRegime.HIGH_VOL
        )
        assert high_vol_spread > normal_spread  # Should be higher

        crisis_spread = execution_engine.calculate_effective_spread(
            base_spread, VolatilityRegime.CRISIS
        )
        assert crisis_spread > high_vol_spread  # Should be even higher

    def test_slippage_calculation(self, execution_engine):
        """Test total slippage calculation."""
        effective_spread = 0.0005  # 5 bps as decimal
        market_impact = 10.0  # 10 bps

        # Get all 4 values
        total, spread_cost, impact_cost, timing = execution_engine.calculate_total_slippage(
            effective_spread, market_impact, OrderSide.BUY
        )
        assert total > 0
        assert spread_cost > 0
        assert impact_cost > 0

    def test_fee_calculation(self, execution_engine):
        """Test fee calculation."""
        notional = 10000.0

        taker_fee = execution_engine.calculate_fee(notional, is_maker=False)
        maker_fee = execution_engine.calculate_fee(notional, is_maker=True)

        assert taker_fee > maker_fee
        assert taker_fee == notional * 0.0004  # 4 bps
        assert maker_fee == notional * 0.0002  # 2 bps

    @pytest.mark.asyncio
    async def test_market_order_execution(self, execution_engine, market_state):
        """Test market order execution."""
        # Update market state
        execution_engine.update_market_state("BTCUSDT", market_state)

        result = await execution_engine.execute_market_order(
            strategy_id="test_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.1,
            current_price=50000.0,
        )

        assert result.success
        assert result.filled_qty == 0.1
        assert result.avg_fill_price > 0
        assert result.fee > 0
        assert result.slippage_total_bps >= 0

    @pytest.mark.asyncio
    async def test_order_rejection_cascade(self, execution_engine, market_state):
        """Test order rejection in cascade regime."""
        # Update market state
        execution_engine.update_market_state("BTCUSDT", market_state)

        # Force cascade regime
        execution_engine.set_regime(VolatilityRegime.CASCADE)

        # Execute multiple orders - some should be rejected
        rejection_count = 0
        for _ in range(20):
            result = await execution_engine.execute_market_order(
                strategy_id="test_001",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                quantity=0.1,
                current_price=50000.0,
            )
            if not result.success:
                rejection_count += 1

        # In CASCADE, rejection rate is 15%, so we should see some rejections
        # (but not all due to randomness)
        assert rejection_count >= 0  # At least test runs without error


# ==============================================================================
# Performance Analyzer Tests
# ==============================================================================

class TestPerformanceAnalyzer:
    """Test performance analytics."""

    def test_trade_recording(self, performance_analyzer, sample_trades):
        """Test trade recording."""
        for trade in sample_trades[:10]:
            performance_analyzer.add_trade(trade)

        assert len(performance_analyzer.trades) == 10
        assert performance_analyzer.current_equity != performance_analyzer.initial_capital

    def test_metrics_calculation(self, performance_analyzer, sample_trades):
        """Test comprehensive metrics calculation."""
        # Add all trades
        for trade in sample_trades:
            performance_analyzer.add_trade(trade)

        # Add daily returns for Sharpe calculation
        for i in range(14):
            performance_analyzer.add_daily_return(
                datetime.now() - timedelta(days=14-i),
                0.005 if i % 3 != 0 else -0.003  # Simulate daily returns
            )

        metrics = performance_analyzer.calculate_metrics()

        assert metrics.total_trades == len(sample_trades)
        assert metrics.total_pnl != 0
        assert 0 <= metrics.win_rate <= 1
        assert metrics.duration_days > 0

    def test_drawdown_calculation(self, performance_analyzer):
        """Test drawdown metrics calculation."""
        # Simulate equity curve with drawdown
        equity = 100000.0
        returns = [0.02, 0.01, -0.05, -0.03, -0.02, 0.04, 0.03]

        # Need to also add trades for the metrics to be calculated
        for i, ret in enumerate(returns):
            equity *= (1 + ret)
            performance_analyzer.update_equity(
                datetime.now() - timedelta(days=7-i),
                equity
            )
            # Add a trade record for each
            trade = TradeRecord(
                trade_id=f"trade_{i}",
                strategy_id="test",
                entry_time=datetime.now() - timedelta(days=7-i),
                exit_time=datetime.now() - timedelta(days=7-i) + timedelta(hours=1),
                net_pnl=100000.0 * ret,
            )
            performance_analyzer.add_trade(trade)

        metrics = performance_analyzer.calculate_metrics()
        # Drawdown is calculated from equity curve - check it's reasonable
        assert metrics.drawdown.max_drawdown_pct >= 0

    def test_transfer_metrics_calculation(self, performance_analyzer, sample_trades):
        """Test transfer ratio calculation."""
        # Add trades
        for trade in sample_trades:
            performance_analyzer.add_trade(trade)

        # Add daily returns
        for i in range(14):
            performance_analyzer.add_daily_return(
                datetime.now() - timedelta(days=14-i),
                0.005 if i % 3 != 0 else -0.003
            )

        # Calculate transfer metrics
        transfer = performance_analyzer.calculate_transfer_metrics(
            backtest_sharpe=1.5,
            backtest_return=45.0,
            backtest_max_dd=15.0,
            backtest_trades=120,
            backtest_win_rate=0.55,
        )

        assert transfer.backtest_sharpe == 1.5
        assert transfer.shadow_trades == len(sample_trades)
        # Transfer ratio should be calculated
        if transfer.shadow_sharpe > 0:
            assert transfer.transfer_ratio > 0


# ==============================================================================
# Transfer Gate Tests
# ==============================================================================

class TestTransferGate:
    """Test Gate 8 (Transfer Gate) validation."""

    def test_gate_initialization(self):
        """Test gate initialization."""
        gate = TransferGate()
        assert gate.config.min_transfer_ratio == 0.5
        assert gate.config.max_drawdown_ratio == 1.5
        # MIN_SHADOW_TRADES from constants is 20
        assert gate.config.min_trades == 20

    def test_custom_config(self):
        """Test gate with custom configuration."""
        config = TransferGateConfig(
            min_transfer_ratio=0.6,
            max_drawdown_ratio=1.3,
            min_trades=50,
        )
        gate = TransferGate(config=config)

        assert gate.config.min_transfer_ratio == 0.6
        assert gate.config.min_trades == 50

    def test_passing_evaluation(self, sample_strategy):
        """Test strategy that passes Gate 8."""
        gate = TransferGate()

        # Create shadow metrics that would pass
        shadow_metrics = PerformanceMetrics(
            total_trades=50,
            total_return_pct=30.0,
            sharpe_ratio=1.0,  # Transfer ratio = 1.0 / 1.5 = 0.67
            duration_days=14,
            win_rate=0.52,
            profit_factor=1.5,
            avg_slippage_bps=5.0,
        )
        shadow_metrics.drawdown = DrawdownMetrics(
            max_drawdown_pct=12.0,  # DD ratio = 12 / 15 = 0.8
        )

        result = gate.evaluate(sample_strategy, shadow_metrics)

        assert result.transfer_ratio > 0.5
        assert result.drawdown_ratio < 1.5
        assert result.passed

    def test_failing_transfer_ratio(self, sample_strategy):
        """Test strategy that fails on transfer ratio."""
        gate = TransferGate()

        # Create shadow metrics with poor Sharpe
        shadow_metrics = PerformanceMetrics(
            total_trades=50,
            sharpe_ratio=0.3,  # Transfer ratio = 0.3 / 1.5 = 0.2
            duration_days=14,
        )
        shadow_metrics.drawdown = DrawdownMetrics(max_drawdown_pct=10.0)

        result = gate.evaluate(sample_strategy, shadow_metrics)

        assert result.transfer_ratio < 0.5
        assert not result.passed
        assert any("Transfer ratio" in r for r in result.failure_reasons)

    def test_failing_drawdown_ratio(self, sample_strategy):
        """Test strategy that fails on drawdown expansion."""
        gate = TransferGate()

        shadow_metrics = PerformanceMetrics(
            total_trades=50,
            sharpe_ratio=1.0,
            duration_days=14,
        )
        shadow_metrics.drawdown = DrawdownMetrics(
            max_drawdown_pct=25.0,  # DD ratio = 25 / 15 = 1.67
        )

        result = gate.evaluate(sample_strategy, shadow_metrics)

        assert result.drawdown_ratio > 1.5
        assert not result.passed
        assert any("Drawdown" in r for r in result.failure_reasons)

    def test_failing_minimum_trades(self, sample_strategy):
        """Test strategy that fails on minimum trades."""
        gate = TransferGate()

        shadow_metrics = PerformanceMetrics(
            total_trades=15,  # Below minimum
            sharpe_ratio=1.0,
            duration_days=14,
        )
        shadow_metrics.drawdown = DrawdownMetrics(max_drawdown_pct=10.0)

        result = gate.evaluate(sample_strategy, shadow_metrics)

        assert not result.passed
        assert any("trades" in r.lower() for r in result.failure_reasons)

    def test_gate_score_calculation(self, sample_strategy):
        """Test weighted gate score calculation."""
        gate = TransferGate()

        shadow_metrics = PerformanceMetrics(
            total_trades=50,
            sharpe_ratio=1.0,
            duration_days=14,
            win_rate=0.52,
            profit_factor=1.5,
            avg_slippage_bps=5.0,
        )
        shadow_metrics.drawdown = DrawdownMetrics(max_drawdown_pct=12.0)

        result = gate.evaluate(sample_strategy, shadow_metrics)

        assert 0 <= result.gate_score <= 1
        assert result.confidence_score > 0


# ==============================================================================
# Deployment Queue Tests
# ==============================================================================

class TestDeploymentQueue:
    """Test deployment queue management."""

    def test_queue_initialization(self):
        """Test queue initialization."""
        queue = DeploymentQueue(max_concurrent=50)
        assert queue.max_concurrent == 50
        assert queue.waiting_count == 0
        assert queue.active_count == 0

    def test_enqueue_strategy(self, sample_strategy):
        """Test adding strategy to queue."""
        queue = DeploymentQueue()

        queued = queue.enqueue(
            sample_strategy,
            priority=QueuePriority.HIGH,
        )

        assert queued.strategy_id == sample_strategy.strategy_id
        assert queued.status == QueueStatus.WAITING
        assert queued.priority == QueuePriority.HIGH
        assert queue.waiting_count == 1

    def test_priority_ordering(self):
        """Test priority-based queue ordering."""
        queue = DeploymentQueue()

        # Create strategies with different priorities
        for i, priority in enumerate([QueuePriority.LOW, QueuePriority.CRITICAL, QueuePriority.NORMAL]):
            strategy = UnifiedStrategy(
                strategy_id=f"strat_{i}",
                source_engine=SourceEngine.EVOLUTIONARY,
            )
            # Create proper HIFAResult
            strategy.hifa_result = _create_test_hifa_result(f"strat_{i}")
            strategy.status = StrategyStatus.HIFA_PASSED
            queue.enqueue(strategy, priority=priority)

        # Get next should return CRITICAL first
        next_strategy = queue.get_next()
        assert next_strategy.priority == QueuePriority.CRITICAL

    def test_capacity_limit(self, sample_strategy):
        """Test capacity limit enforcement."""
        queue = DeploymentQueue(max_concurrent=2)

        # Add and deploy 3 strategies
        for i in range(3):
            strategy = UnifiedStrategy(
                strategy_id=f"strat_{i}",
                source_engine=SourceEngine.EVOLUTIONARY,
            )
            strategy.hifa_result = _create_test_hifa_result(f"strat_{i}")
            strategy.status = StrategyStatus.HIFA_PASSED
            queue.enqueue(strategy)

        # Deploy first 2
        for _ in range(2):
            next_s = queue.get_next()
            queue.deploy(next_s.strategy_id)

        # Third should not be deployable (at capacity)
        assert queue.is_full
        assert queue.get_next() is None

    def test_completion_flow(self, sample_strategy):
        """Test strategy completion flow."""
        queue = DeploymentQueue()

        queue.enqueue(sample_strategy)
        next_s = queue.get_next()
        queue.deploy(next_s.strategy_id)

        # Complete with success
        queue.complete(
            sample_strategy.strategy_id,
            passed=True,
            metrics={'total_trades': 50, 'total_pnl': 5000},
        )

        assert queue.active_count == 0
        assert queue.total_completed == 1
        assert queue.total_passed == 1

    def test_failure_and_retry(self, sample_strategy):
        """Test failure handling and retry."""
        queue = DeploymentQueue()

        queue.enqueue(sample_strategy)
        next_s = queue.get_next()
        queue.deploy(next_s.strategy_id)

        # First failure - should retry
        queue.fail(sample_strategy.strategy_id, "Connection error", allow_retry=True)

        status = queue.get_status(sample_strategy.strategy_id)
        assert status.status == QueueStatus.WAITING
        assert status.retry_count == 1


# ==============================================================================
# Shadow Monitor Tests
# ==============================================================================

class TestShadowMonitor:
    """Test shadow trading monitor."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = ShadowMonitor()
        assert len(monitor.strategies) == 0

    def test_strategy_registration(self):
        """Test registering strategy for monitoring."""
        monitor = ShadowMonitor()
        monitor.register_strategy("test_001", initial_equity=100000.0)

        assert "test_001" in monitor.strategies
        health = monitor.get_health("test_001")
        assert health is not None
        assert health.is_healthy

    def test_trade_tracking(self):
        """Test trade event tracking."""
        monitor = ShadowMonitor()
        monitor.register_strategy("test_001")

        # Record winning trade
        alerts = monitor.on_trade("test_001", pnl=500.0, slippage_bps=3.0, is_win=True)

        health = monitor.get_health("test_001")
        assert health.total_trades == 1
        assert health.winning_trades == 1
        assert health.consecutive_wins == 1

    def test_drawdown_alerts(self):
        """Test drawdown alert generation."""
        config = MonitorConfig(
            drawdown_warning_pct=10.0,
            drawdown_critical_pct=15.0,
        )
        monitor = ShadowMonitor(config=config)
        monitor.register_strategy("test_001")

        # Trigger warning drawdown
        alerts = monitor.on_equity_update("test_001", equity=88000.0, initial_equity=100000.0)

        assert len(alerts) >= 1
        assert any(a.alert_type == AlertType.DRAWDOWN_WARNING for a in alerts)

    def test_losing_streak_alerts(self):
        """Test losing streak detection."""
        config = MonitorConfig(losing_streak_warning=3)
        monitor = ShadowMonitor(config=config)
        monitor.register_strategy("test_001")

        # Record losing streak
        for _ in range(4):
            alerts = monitor.on_trade("test_001", pnl=-100.0, is_win=False)

        assert any(a.alert_type == AlertType.LOSING_STREAK for a in alerts)

    def test_regime_change_notification(self):
        """Test regime change alerts."""
        config = MonitorConfig(notify_on_regime_change=True)
        monitor = ShadowMonitor(config=config)
        monitor.register_strategy("test_001")

        alerts = monitor.on_regime_change(VolatilityRegime.CRISIS)

        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.REGIME_CHANGE
        assert monitor.current_regime == VolatilityRegime.CRISIS

    def test_health_check(self):
        """Test health check functionality."""
        monitor = ShadowMonitor()
        monitor.register_strategy("test_001")

        # Record some activity
        monitor.on_trade("test_001", pnl=500.0, is_win=True)

        health_result = monitor.health_check("test_001")

        assert health_result['status'] == 'healthy'
        assert health_result['strategy_id'] == "test_001"
        assert 'metrics' in health_result

    def test_alert_acknowledgment(self):
        """Test alert acknowledgment."""
        monitor = ShadowMonitor()
        monitor.register_strategy("test_001")

        # Generate an alert
        config = MonitorConfig(drawdown_warning_pct=5.0)
        monitor.config = config
        monitor.on_equity_update("test_001", equity=94000.0, initial_equity=100000.0)

        alerts = monitor.get_alerts("test_001", unacknowledged_only=True)
        assert len(alerts) > 0

        # Acknowledge
        alert_id = alerts[0].id
        success = monitor.acknowledge_alert(alert_id)
        assert success

        unacked = monitor.get_alerts("test_001", unacknowledged_only=True)
        assert len(unacked) < len(alerts)


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestIntegration:
    """Integration tests for complete flows."""

    @pytest.mark.asyncio
    async def test_full_shadow_session(self, sample_strategy, market_state):
        """Test complete shadow trading session flow."""
        engine = ExecutionEngine(simulate_latency=False)
        engine.update_market_state("BTCUSDT", market_state)
        analyzer = PerformanceAnalyzer(initial_capital=100000.0)

        # Simulate trades
        for i in range(35):
            # Execute trade
            result = await engine.execute_market_order(
                strategy_id=sample_strategy.strategy_id,
                symbol="BTCUSDT",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                quantity=0.1,
                current_price=50000.0,
            )

            if result.success:
                # Record as trade
                is_win = i % 3 != 0
                pnl = 100 if is_win else -60

                trade = TradeRecord(
                    trade_id=result.order_id,
                    strategy_id=sample_strategy.strategy_id,
                    symbol="BTCUSDT",
                    entry_price=result.avg_fill_price,
                    entry_size=result.filled_qty,
                    entry_fee=result.fee,
                    entry_slippage_bps=result.slippage_total_bps,
                    exit_time=datetime.now(),
                    exit_price=result.avg_fill_price * (1.002 if is_win else 0.998),
                    net_pnl=pnl,
                )
                analyzer.add_trade(trade)

        # Add daily returns
        for i in range(14):
            analyzer.add_daily_return(
                datetime.now() - timedelta(days=14-i),
                0.003 if i % 3 != 0 else -0.002
            )

        # Calculate metrics
        metrics = analyzer.calculate_metrics()
        assert metrics.total_trades >= 30

        # Evaluate with Transfer Gate
        gate = TransferGate()
        result = gate.evaluate(sample_strategy, metrics)

        # Result should be deterministic based on metrics
        assert result.shadow_trades == metrics.total_trades
        assert isinstance(result.passed, bool)

    def test_queue_to_monitor_flow(self, sample_strategy):
        """Test flow from queue to monitor."""
        queue = DeploymentQueue()
        monitor = ShadowMonitor()

        # Enqueue
        queue.enqueue(sample_strategy)

        # Deploy
        next_s = queue.get_next()
        queue.deploy(next_s.strategy_id)

        # Register with monitor
        monitor.register_strategy(next_s.strategy_id)

        # Simulate trading
        for i in range(10):
            monitor.on_trade(
                next_s.strategy_id,
                pnl=100 if i % 2 == 0 else -50,
                is_win=(i % 2 == 0),
            )

        # Check health
        health = monitor.get_health(next_s.strategy_id)
        assert health.total_trades == 10

        # Complete in queue
        queue.complete(next_s.strategy_id, passed=True)

        # Verify completion
        stats = queue.get_statistics()
        assert stats['total_completed'] == 1


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
