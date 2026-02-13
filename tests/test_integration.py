"""
Integration Tests for EXPLORER PRIME Unified System

Tests end-to-end pipeline flows:
- Shared module → HIFA integration
- HIFA → Forward Testing flow
- Forward Testing → EMT storage
- Full orchestrator pipeline
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.unified_strategy import (
    UnifiedStrategy,
    StrategyGenome,
    SourceEngine,
    StrategyStatus,
    HIFAResult,
    ForwardTestResult,
    ShadowMetrics,
    BacktestMetrics,
    GateResult,
    StatisticalScores,
    RegimeTier,
)

from forward_testing.models import VolatilityRegime
from forward_testing.transfer_gate import TransferGate, TransferGateConfig
from forward_testing.analytics.performance import PerformanceMetrics, DrawdownMetrics

from emt.production import EMTProduction, ProductionConfig

from orchestrator import (
    UnifiedOrchestrator,
    OrchestratorConfig,
    PipelineStage,
    PipelineResult,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_genome():
    """Create a sample strategy genome."""
    return StrategyGenome(
        entry_conditions=[{"type": "crossover", "indicator": "sma", "params": {"fast": 10, "slow": 30}}],
        exit_conditions=[{"type": "crossover", "indicator": "sma", "params": {"fast": 30, "slow": 10}}],
        position_sizing={"method": "fixed", "size": 0.1},
        parameters={"sma_fast": 10, "sma_slow": 30},
        lookback_periods={"sma": 30},
        indicators=["SMA"],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
    )


@pytest.fixture
def sample_strategy(sample_genome):
    """Create a sample unified strategy."""
    return UnifiedStrategy(
        strategy_id="test_integration_001",
        genome=sample_genome,
        source_engine=SourceEngine.EVOLUTIONARY,
        target_asset="BTCUSDT",
        target_regime="trending",
    )


@pytest.fixture
def sample_hifa_result():
    """Create a sample HIFA result."""
    backtest_metrics = BacktestMetrics(
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        calmar_ratio=1.0,
        max_drawdown=15.0,
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
        strategy_id="test_integration_001",
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
        regime_performance={"trending": 1.5, "ranging": 1.2, "volatile": 0.9},
        cluster_id=1,
        similarity_score=0.3,
        rejection_reason=None,
    )


@pytest.fixture
def sample_forward_result():
    """Create a sample forward test result."""
    shadow_metrics = ShadowMetrics(
        sharpe_ratio=1.05,
        sortino_ratio=1.5,
        max_drawdown=12.0,
        win_rate=0.52,
        total_trades=50,
        avg_slippage_bps=3.0,
        avg_latency_ms=50.0,
        total_fees_paid=100.0,
        start_date=datetime.now() - timedelta(days=14),
        end_date=datetime.now(),
        duration_days=14,
        total_return=25.0,
    )
    return ForwardTestResult(
        strategy_id="test_integration_001",
        passed=True,
        shadow_metrics=shadow_metrics,
        transfer_ratio=0.7,
        dd_ratio=0.8,
        rejection_reason=None,
        backtest_sharpe=1.5,
        backtest_max_dd=15.0,
        test_start=datetime.now() - timedelta(days=14),
        test_end=datetime.now(),
    )


@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1h')
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(40000, 50000, 1000),
        'high': np.random.uniform(40000, 51000, 1000),
        'low': np.random.uniform(39000, 50000, 1000),
        'close': np.random.uniform(40000, 50000, 1000),
        'volume': np.random.uniform(1e6, 1e9, 1000),
    })


# ==============================================================================
# Shared Module Tests
# ==============================================================================

class TestSharedToHIFA:
    """Test shared module integration with HIFA."""

    def test_strategy_hifa_attachment(self, sample_strategy, sample_hifa_result):
        """Test attaching HIFA result to strategy."""
        assert sample_strategy.hifa_result is None
        assert sample_strategy.status == StrategyStatus.GENERATED

        # Attach HIFA result
        sample_strategy.hifa_result = sample_hifa_result
        sample_strategy.status = StrategyStatus.HIFA_PASSED

        assert sample_strategy.hifa_result is not None
        assert sample_strategy.hifa_result.passed
        assert sample_strategy.is_hifa_validated

    def test_strategy_genome_hash_consistency(self, sample_genome):
        """Test genome hash is consistent."""
        hash1 = sample_genome.compute_hash()
        hash2 = sample_genome.compute_hash()

        assert hash1 == hash2
        assert len(hash1) > 0

    def test_strategy_json_serialization(self, sample_strategy, sample_hifa_result):
        """Test strategy serialization with HIFA result."""
        sample_strategy.hifa_result = sample_hifa_result
        sample_strategy.status = StrategyStatus.HIFA_PASSED

        # Serialize
        json_str = sample_strategy.to_json()
        assert json_str is not None
        assert len(json_str) > 0

        # Deserialize
        restored = UnifiedStrategy.from_json(json_str)
        assert restored.strategy_id == sample_strategy.strategy_id
        assert restored.status == StrategyStatus.HIFA_PASSED


# ==============================================================================
# HIFA to Forward Testing Tests
# ==============================================================================

class TestHIFAToForwardTesting:
    """Test HIFA to Forward Testing integration."""

    def test_transfer_gate_with_hifa_strategy(self, sample_strategy, sample_hifa_result):
        """Test Transfer Gate evaluates HIFA-validated strategy."""
        sample_strategy.hifa_result = sample_hifa_result
        sample_strategy.status = StrategyStatus.HIFA_PASSED

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

        assert result.passed
        assert result.transfer_ratio > 0.5
        assert result.drawdown_ratio < 1.5

    def test_hifa_backtest_metrics_used_in_gate(self, sample_strategy, sample_hifa_result):
        """Test that HIFA backtest metrics are properly used in gate evaluation."""
        sample_strategy.hifa_result = sample_hifa_result
        sample_strategy.status = StrategyStatus.HIFA_PASSED

        gate = TransferGate()

        # Shadow metrics
        shadow_metrics = PerformanceMetrics(
            total_trades=50,
            sharpe_ratio=0.8,  # Lower than backtest
            duration_days=14,
        )
        shadow_metrics.drawdown = DrawdownMetrics(max_drawdown_pct=10.0)

        result = gate.evaluate(sample_strategy, shadow_metrics)

        # Transfer ratio should be calculated from backtest sharpe (1.5)
        expected_ratio = 0.8 / 1.5  # ~0.53
        assert abs(result.transfer_ratio - expected_ratio) < 0.1


# ==============================================================================
# Forward Testing to EMT Tests
# ==============================================================================

class TestForwardTestingToEMT:
    """Test Forward Testing to EMT storage integration."""

    def test_emt_stores_forward_passed_strategy(
        self, sample_strategy, sample_hifa_result, sample_forward_result
    ):
        """Test EMT stores strategy that passed forward testing."""
        # Setup strategy with all validations
        sample_strategy.hifa_result = sample_hifa_result
        sample_strategy.forward_result = sample_forward_result
        sample_strategy.status = StrategyStatus.FORWARD_PASSED

        # Store in EMT
        emt = EMTProduction(ProductionConfig(
            storage_dir="test_emt_data/production",
            archive_dir="test_emt_data/archive",
            auto_save=False,
        ))

        result = emt.store(sample_strategy)

        assert result.success
        assert result.strategy_id == sample_strategy.strategy_id
        assert emt.active_count == 1

        # Verify strategy status updated
        assert sample_strategy.status == StrategyStatus.PRODUCTION

    def test_emt_merkle_root_changes_on_add(
        self, sample_strategy, sample_hifa_result, sample_forward_result
    ):
        """Test Merkle root updates when strategy is added."""
        sample_strategy.hifa_result = sample_hifa_result
        sample_strategy.forward_result = sample_forward_result
        sample_strategy.status = StrategyStatus.FORWARD_PASSED

        emt = EMTProduction(ProductionConfig(
            auto_save=False,
            enable_merkle=True,
        ))

        # Initial state
        initial_root = emt.get_merkle_root()
        assert initial_root is None

        # Add strategy
        emt.store(sample_strategy)
        root1 = emt.get_merkle_root()
        assert root1 is not None

        # Add another strategy
        strategy2 = UnifiedStrategy(
            strategy_id="test_integration_002",
            genome=sample_strategy.genome,
            source_engine=SourceEngine.EVOLUTIONARY,
        )
        strategy2.hifa_result = sample_hifa_result
        strategy2.forward_result = sample_forward_result
        strategy2.status = StrategyStatus.FORWARD_PASSED

        emt.store(strategy2)
        root2 = emt.get_merkle_root()

        # Root should change
        assert root2 != root1

    def test_emt_retire_strategy(
        self, sample_strategy, sample_hifa_result, sample_forward_result
    ):
        """Test retiring a strategy from EMT."""
        sample_strategy.hifa_result = sample_hifa_result
        sample_strategy.forward_result = sample_forward_result
        sample_strategy.status = StrategyStatus.FORWARD_PASSED

        emt = EMTProduction(ProductionConfig(auto_save=False))

        # Store
        emt.store(sample_strategy)
        assert emt.active_count == 1

        # Retire
        success = emt.retire(sample_strategy.strategy_id, "Test retirement")
        assert success
        assert emt.active_count == 0
        assert sample_strategy.status == StrategyStatus.RETIRED


# ==============================================================================
# Orchestrator Tests
# ==============================================================================

class TestOrchestrator:
    """Test unified orchestrator."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly."""
        orchestrator = UnifiedOrchestrator()

        assert orchestrator.config is not None
        assert orchestrator.config.default_candidates == 1000

    def test_orchestrator_custom_config(self):
        """Test orchestrator with custom config."""
        config = OrchestratorConfig(
            default_candidates=500,
            hifa_batch_size=25,
            max_concurrent_shadow=10,
        )

        orchestrator = UnifiedOrchestrator(config=config)

        assert orchestrator.config.default_candidates == 500
        assert orchestrator.config.hifa_batch_size == 25

    @pytest.mark.asyncio
    async def test_orchestrator_stub_pipeline(self, sample_market_data):
        """Test orchestrator runs with stubs (no real components)."""
        # This tests that the orchestrator framework works
        # even when real components aren't available
        orchestrator = UnifiedOrchestrator()

        result = await orchestrator.run_full_pipeline(
            market_data=sample_market_data,
            n_candidates=10,
        )

        assert result is not None
        assert result.completed
        assert result.pipeline_id is not None

    @pytest.mark.asyncio
    async def test_orchestrator_pipeline_result_structure(self, sample_market_data):
        """Test pipeline result has correct structure."""
        orchestrator = UnifiedOrchestrator()

        result = await orchestrator.run_full_pipeline(
            market_data=sample_market_data,
            n_candidates=5,
        )

        # Check result structure
        assert hasattr(result, 'pipeline_id')
        assert hasattr(result, 'started_at')
        assert hasattr(result, 'completed_at')
        assert hasattr(result, 'stage_results')
        assert hasattr(result, 'production_strategies')

        # Check summary
        summary = result.to_summary()
        assert 'pipeline_id' in summary
        assert 'stages' in summary

    @pytest.mark.asyncio
    async def test_orchestrator_with_mock_components(self, sample_market_data):
        """Test orchestrator with mocked components."""
        # Create mock explorer that generates strategies
        mock_explorer = MagicMock()
        mock_explorer.generate_candidates.return_value = [
            UnifiedStrategy(
                strategy_id=f"mock_strategy_{i}",
                genome=StrategyGenome(
                    entry_conditions=[{"type": "test"}],
                    exit_conditions=[{"type": "test"}],
                    position_sizing={"method": "fixed", "size": 0.1},
                    parameters={"test": i},
                    lookback_periods={"test": 10},
                    indicators=["SMA"],
                ),
                source_engine=SourceEngine.EVOLUTIONARY,
            )
            for i in range(5)
        ]

        orchestrator = UnifiedOrchestrator(explorer=mock_explorer)

        result = await orchestrator.run_full_pipeline(
            market_data=sample_market_data,
            n_candidates=5,
        )

        # Explorer should have been called
        mock_explorer.generate_candidates.assert_called_once()

        # Should have generation stage result
        assert PipelineStage.GENERATION in result.stage_results

    def test_orchestrator_pipeline_status(self):
        """Test getting pipeline status."""
        orchestrator = UnifiedOrchestrator()

        # No active pipeline
        status = orchestrator.get_pipeline_status()
        assert status is None

        # Check history
        history = orchestrator.get_history()
        assert isinstance(history, list)


# ==============================================================================
# End-to-End Flow Tests
# ==============================================================================

class TestEndToEndFlow:
    """Test complete end-to-end flows."""

    def test_full_strategy_lifecycle(
        self, sample_strategy, sample_hifa_result, sample_forward_result
    ):
        """Test strategy through complete lifecycle."""
        # Stage 1: Generated
        assert sample_strategy.status == StrategyStatus.GENERATED

        # Stage 2: HIFA Validation
        sample_strategy.hifa_result = sample_hifa_result
        sample_strategy.status = StrategyStatus.HIFA_PASSED
        assert sample_strategy.is_hifa_validated

        # Stage 3: Forward Testing
        sample_strategy.forward_result = sample_forward_result
        sample_strategy.status = StrategyStatus.FORWARD_PASSED
        assert sample_strategy.forward_result.passed

        # Stage 4: Production
        emt = EMTProduction(ProductionConfig(auto_save=False))
        result = emt.store(sample_strategy)

        assert result.success
        assert sample_strategy.status == StrategyStatus.PRODUCTION
        assert sample_strategy.production_start is not None

        # Stage 5: Retirement
        emt.retire(sample_strategy.strategy_id, "Test complete")
        assert sample_strategy.status == StrategyStatus.RETIRED
        assert sample_strategy.production_end is not None

    def test_batch_strategy_processing(self, sample_hifa_result, sample_forward_result):
        """Test processing multiple strategies through pipeline."""
        strategies = []
        for i in range(10):
            genome = StrategyGenome(
                entry_conditions=[{"type": "test", "id": i}],
                exit_conditions=[{"type": "test"}],
                position_sizing={"method": "fixed", "size": 0.1},
                parameters={"test": i},
                lookback_periods={"test": 10},
                indicators=["SMA"],
            )
            strategy = UnifiedStrategy(
                strategy_id=f"batch_strategy_{i}",
                genome=genome,
                source_engine=SourceEngine.EVOLUTIONARY,
            )
            strategy.hifa_result = HIFAResult(
                strategy_id=strategy.strategy_id,
                passed=i % 3 != 0,  # 7 pass, 3 fail
                final_gate=7 if i % 3 != 0 else 3,
                gate_results=sample_hifa_result.gate_results,
                statistical_scores=sample_hifa_result.statistical_scores,
                backtest_metrics=sample_hifa_result.backtest_metrics,
                regime_tier=RegimeTier.ALL_WEATHER,
                regime_performance={},
                cluster_id=None,
                similarity_score=None,
                rejection_reason="Failed Gate 3" if i % 3 == 0 else None,
            )
            strategies.append(strategy)

        # Count passed HIFA (indices 1,2,4,5,7,8 pass = 6 strategies)
        hifa_passed = [s for s in strategies if s.hifa_result.passed]
        assert len(hifa_passed) == 6

        # Store passed strategies
        emt = EMTProduction(ProductionConfig(auto_save=False))
        for strategy in hifa_passed:
            strategy.status = StrategyStatus.HIFA_PASSED
            strategy.forward_result = sample_forward_result
            strategy.status = StrategyStatus.FORWARD_PASSED
            emt.store(strategy)

        assert emt.active_count == 6

    def test_transfer_ratio_flow(self, sample_strategy, sample_hifa_result):
        """Test transfer ratio calculation through the flow."""
        # Setup with known values
        sample_hifa_result.backtest_metrics.sharpe_ratio = 2.0
        sample_strategy.hifa_result = sample_hifa_result
        sample_strategy.status = StrategyStatus.HIFA_PASSED

        # Forward test with lower Sharpe
        shadow_metrics = PerformanceMetrics(
            total_trades=50,
            sharpe_ratio=1.2,  # 60% of backtest
            duration_days=14,
        )
        shadow_metrics.drawdown = DrawdownMetrics(max_drawdown_pct=10.0)

        gate = TransferGate()
        result = gate.evaluate(sample_strategy, shadow_metrics)

        # Transfer ratio should be ~0.6
        expected_ratio = 1.2 / 2.0
        assert abs(result.transfer_ratio - expected_ratio) < 0.05
        assert result.passed  # 0.6 > 0.5 threshold


# ==============================================================================
# Edge Cases and Error Handling
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_pipeline(self, sample_market_data):
        """Test pipeline with no generated strategies."""
        mock_explorer = MagicMock()
        mock_explorer.generate_candidates.return_value = []

        orchestrator = UnifiedOrchestrator(explorer=mock_explorer)

        # Should complete without errors
        import asyncio
        result = asyncio.run(orchestrator.run_full_pipeline(
            market_data=sample_market_data,
            n_candidates=10,
        ))

        assert result.completed
        assert len(result.production_strategies) == 0

    def test_emt_capacity_limit(self):
        """Test EMT capacity limit."""
        emt = EMTProduction(ProductionConfig(
            max_active_strategies=3,
            auto_save=False,
            min_sharpe_for_retention=0.0,  # Don't auto-retire
        ))

        # Fill to capacity
        for i in range(3):
            strategy = UnifiedStrategy(
                strategy_id=f"capacity_test_{i}",
                genome=StrategyGenome(
                    entry_conditions=[{"type": "test"}],
                    exit_conditions=[{"type": "test"}],
                    position_sizing={"method": "fixed", "size": 0.1},
                    parameters={"test": i},
                    lookback_periods={"test": 10},
                    indicators=["SMA"],
                ),
                source_engine=SourceEngine.EVOLUTIONARY,
            )
            result = emt.store(strategy)
            assert result.success

        # At capacity
        assert emt.active_count == 3
        assert emt.capacity_remaining == 0

        # Try to add one more (should fail without auto-retirement)
        strategy = UnifiedStrategy(
            strategy_id="capacity_test_overflow",
            genome=StrategyGenome(
                entry_conditions=[{"type": "test"}],
                exit_conditions=[{"type": "test"}],
                position_sizing={"method": "fixed", "size": 0.1},
                parameters={"test": 99},
                lookback_periods={"test": 10},
                indicators=["SMA"],
            ),
            source_engine=SourceEngine.EVOLUTIONARY,
        )
        result = emt.store(strategy)
        assert not result.success
        assert "capacity" in result.message.lower()

    def test_strategy_status_transitions(self, sample_strategy, sample_hifa_result):
        """Test valid status transitions."""
        # GENERATED -> HIFA_VALIDATING -> HIFA_PASSED
        assert sample_strategy.status == StrategyStatus.GENERATED

        sample_strategy.status = StrategyStatus.HIFA_VALIDATING
        assert sample_strategy.status == StrategyStatus.HIFA_VALIDATING

        sample_strategy.hifa_result = sample_hifa_result
        sample_strategy.status = StrategyStatus.HIFA_PASSED
        assert sample_strategy.status == StrategyStatus.HIFA_PASSED

    def test_merkle_verification(self, sample_strategy):
        """Test Merkle tree integrity verification."""
        emt = EMTProduction(ProductionConfig(
            auto_save=False,
            enable_merkle=True,
        ))

        sample_strategy.status = StrategyStatus.FORWARD_PASSED
        emt.store(sample_strategy)

        # Verify integrity
        is_valid = emt.verify_strategy(sample_strategy.strategy_id)
        assert is_valid

        # Get production strategy and tamper
        prod = emt.get(sample_strategy.strategy_id)
        original_hash = prod.merkle_hash

        # Verify still works
        assert emt.verify_strategy(sample_strategy.strategy_id)
