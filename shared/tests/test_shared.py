"""
Tests for EXPLORER PRIME shared module.

Tests cover:
- StrategyGenome creation and validation
- UnifiedStrategy lifecycle
- Format adapters (Explorer, LSM, Hinance)
- Feature vector operations
"""

import pytest
import numpy as np
import json
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.unified_strategy import (
    StrategyGenome,
    UnifiedStrategy,
    HIFAResult,
    ForwardTestResult,
    ProductionStrategy,
    GateResult,
    StatisticalScores,
    BacktestMetrics,
    ShadowMetrics,
    StrategyStatus,
    SourceEngine,
    RegimeTier,
)
from shared.adapters import (
    StrategyAdapter,
    ExplorerAdapter,
    LSMAdapter,
    HinanceAdapter,
    BatchAdapter,
)
from shared.features import (
    FEATURE_SCHEMA,
    FEATURE_DIMENSIONS,
    FeatureVector,
    FeatureGroup,
    get_feature_names,
    get_feature_indices,
)
from shared.constants import (
    DEFAULT_DSR_THRESHOLD,
    DEFAULT_PBO_THRESHOLD,
    MAX_STRATEGY_PARAMETERS,
    FEATURE_DIMENSIONS as CONST_FEATURE_DIMS,
)


# ==============================================================================
# StrategyGenome Tests
# ==============================================================================

class TestStrategyGenome:
    """Tests for StrategyGenome class."""

    @pytest.fixture
    def sample_genome(self):
        """Create sample strategy genome."""
        return StrategyGenome(
            entry_conditions=[
                {"operator": "and", "left": {"indicator": "RSI", "op": "lt", "value": 30}, "right": {"indicator": "MACD", "op": "cross_above", "value": 0}},
            ],
            exit_conditions=[
                {"operator": "or", "left": {"indicator": "RSI", "op": "gt", "value": 70}, "right": {"indicator": "stop_loss", "op": "triggered"}},
            ],
            position_sizing={"method": "kelly", "max_size": 0.5},
            parameters={"rsi_period": 14, "macd_fast": 12, "macd_slow": 26},
            lookback_periods={"rsi": 14, "macd": 26},
            indicators=["RSI", "MACD", "BB"],
            stop_loss_pct=0.02,
            take_profit_pct=0.06,
        )

    def test_genome_creation(self, sample_genome):
        """Test genome creation with valid data."""
        assert sample_genome.n_parameters == 3
        assert sample_genome.n_rules == 2
        assert len(sample_genome.indicators) == 3
        assert sample_genome.stop_loss_pct == 0.02

    def test_genome_hash(self, sample_genome):
        """Test genome hash computation."""
        hash1 = sample_genome.compute_hash()
        assert len(hash1) == 16  # 16 hex characters

        # Same genome should have same hash
        genome2 = StrategyGenome(
            entry_conditions=sample_genome.entry_conditions.copy(),
            exit_conditions=sample_genome.exit_conditions.copy(),
            position_sizing=sample_genome.position_sizing.copy(),
            parameters=sample_genome.parameters.copy(),
            lookback_periods=sample_genome.lookback_periods.copy(),
            indicators=sample_genome.indicators.copy(),
        )
        assert genome2.compute_hash() == hash1

    def test_genome_json_serialization(self, sample_genome):
        """Test JSON serialization/deserialization."""
        json_str = sample_genome.to_json()
        assert isinstance(json_str, str)

        # Parse and verify
        data = json.loads(json_str)
        assert "entry_conditions" in data
        assert "parameters" in data
        assert len(data["indicators"]) == 3

        # Deserialize
        restored = StrategyGenome.from_json(json_str)
        assert restored.n_parameters == sample_genome.n_parameters
        assert restored.indicators == sample_genome.indicators

    def test_genome_validation_pass(self, sample_genome):
        """Test genome validation with valid genome."""
        is_valid, errors = sample_genome.validate()
        assert is_valid
        assert len(errors) == 0

    def test_genome_validation_fail_no_entry(self):
        """Test genome validation fails without entry conditions."""
        genome = StrategyGenome(
            entry_conditions=[],  # Empty!
            exit_conditions=[{"operator": "gt", "left": "RSI", "right": 70}],
            position_sizing={"method": "fixed"},
            parameters={},
            lookback_periods={},
            indicators=["RSI"],
        )
        is_valid, errors = genome.validate()
        assert not is_valid
        assert "No entry conditions defined" in errors

    def test_genome_to_executable(self, sample_genome):
        """Test code generation."""
        code = sample_genome.to_executable("python")
        assert "class GeneratedStrategy" in code
        assert "INDICATORS" in code
        assert "should_enter" in code

    def test_genome_tree_depth(self):
        """Test tree depth calculation."""
        # Simple flat genome
        genome = StrategyGenome(
            entry_conditions=[{"operator": "gt", "left": "RSI", "right": 30}],
            exit_conditions=[{"operator": "lt", "left": "RSI", "right": 70}],
            position_sizing={"method": "fixed"},
            parameters={},
            lookback_periods={},
            indicators=["RSI"],
        )
        assert genome.tree_depth >= 1


# ==============================================================================
# UnifiedStrategy Tests
# ==============================================================================

class TestUnifiedStrategy:
    """Tests for UnifiedStrategy class."""

    @pytest.fixture
    def sample_strategy(self):
        """Create sample unified strategy."""
        genome = StrategyGenome(
            entry_conditions=[{"operator": "gt", "left": "RSI", "right": 30}],
            exit_conditions=[{"operator": "lt", "left": "RSI", "right": 70}],
            position_sizing={"method": "fixed", "size": 0.5},
            parameters={"rsi_period": 14},
            lookback_periods={"rsi": 14},
            indicators=["RSI"],
        )
        return UnifiedStrategy(
            genome=genome,
            source_engine=SourceEngine.EVOLUTIONARY,
            target_asset="BTCUSDT",
            target_regime="normal",
        )

    def test_strategy_creation(self, sample_strategy):
        """Test strategy creation with valid data."""
        assert sample_strategy.strategy_id.startswith("strat_")
        assert sample_strategy.source_engine == SourceEngine.EVOLUTIONARY
        assert sample_strategy.status == StrategyStatus.GENERATED
        assert not sample_strategy.is_hifa_validated
        assert not sample_strategy.is_production_ready

    def test_strategy_genome_hash(self, sample_strategy):
        """Test genome hash accessor."""
        hash_val = sample_strategy.genome_hash
        assert hash_val is not None
        assert len(hash_val) == 16

    def test_strategy_status_transitions(self, sample_strategy):
        """Test valid status transitions."""
        # GENERATED -> HIFA_VALIDATING
        sample_strategy.update_status(StrategyStatus.HIFA_VALIDATING)
        assert sample_strategy.status == StrategyStatus.HIFA_VALIDATING

        # HIFA_VALIDATING -> HIFA_PASSED
        sample_strategy.update_status(StrategyStatus.HIFA_PASSED)
        assert sample_strategy.status == StrategyStatus.HIFA_PASSED

    def test_strategy_invalid_status_transition(self, sample_strategy):
        """Test invalid status transition raises error."""
        with pytest.raises(ValueError):
            # Can't go directly from GENERATED to PRODUCTION
            sample_strategy.update_status(StrategyStatus.PRODUCTION)

    def test_attach_hifa_result(self, sample_strategy):
        """Test attaching HIFA result."""
        hifa_result = HIFAResult(
            strategy_id=sample_strategy.strategy_id,
            passed=True,
            final_gate=7,
            gate_results={},
            statistical_scores=None,
            backtest_metrics=BacktestMetrics(
                sharpe_ratio=1.5,
                sortino_ratio=2.0,
                max_drawdown=0.15,
                calmar_ratio=10.0,
                win_rate=0.55,
                profit_factor=1.8,
                total_trades=150,
                avg_trade_return=0.002,
                avg_trade_duration_hours=24,
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2024, 1, 1),
                total_return=0.45,
            ),
            regime_tier=RegimeTier.ALL_WEATHER,
            regime_performance={"normal": 1.2, "elevated": 0.8, "crisis": 0.3},
            cluster_id=1,
            similarity_score=0.3,
            rejection_reason=None,
        )

        sample_strategy.update_status(StrategyStatus.HIFA_VALIDATING)
        sample_strategy.attach_hifa_result(hifa_result)

        assert sample_strategy.is_hifa_validated
        assert sample_strategy.status == StrategyStatus.HIFA_PASSED
        assert sample_strategy.hifa_result.backtest_sharpe == 1.5

    def test_strategy_json_serialization(self, sample_strategy):
        """Test JSON serialization."""
        json_str = sample_strategy.to_json()
        assert isinstance(json_str, str)

        data = json.loads(json_str)
        assert "strategy_id" in data
        assert "genome" in data
        assert data["source_engine"] == "evolutionary"

    def test_strategy_from_json(self, sample_strategy):
        """Test JSON deserialization."""
        json_str = sample_strategy.to_json()
        restored = UnifiedStrategy.from_json(json_str)

        assert restored.strategy_id == sample_strategy.strategy_id
        assert restored.source_engine == sample_strategy.source_engine
        assert restored.genome is not None


# ==============================================================================
# Adapter Tests
# ==============================================================================

class TestAdapters:
    """Tests for strategy format adapters."""

    def test_hinance_adapter_from_json(self):
        """Test Hinance JSON to UnifiedStrategy conversion."""
        genome_json = json.dumps({
            "entry_conditions": [{"operator": "gt", "left": "RSI", "right": 30}],
            "exit_conditions": [{"operator": "lt", "left": "RSI", "right": 70}],
            "position_sizing": {"method": "fixed"},
            "parameters": {"rsi_period": 14},
            "lookback_periods": {"rsi": 14},
            "indicators": ["RSI"],
        })

        strategy = HinanceAdapter.from_json(genome_json, {"strategy_id": "test_123"})

        assert strategy.strategy_id == "test_123"
        assert strategy.genome is not None
        assert "RSI" in strategy.genome.indicators

    def test_hinance_adapter_to_deployment(self):
        """Test UnifiedStrategy to Hinance deployment format."""
        genome = StrategyGenome(
            entry_conditions=[{"operator": "gt", "left": "RSI", "right": 30}],
            exit_conditions=[{"operator": "lt", "left": "RSI", "right": 70}],
            position_sizing={"method": "fixed"},
            parameters={},
            lookback_periods={},
            indicators=["RSI"],
        )
        strategy = UnifiedStrategy(
            strategy_id="test_deploy",
            genome=genome,
            source_engine=SourceEngine.EVOLUTIONARY,
        )

        # Need to attach HIFA result first
        hifa_result = HIFAResult(
            strategy_id="test_deploy",
            passed=True,
            final_gate=7,
            gate_results={},
            statistical_scores=None,
            backtest_metrics=BacktestMetrics(
                sharpe_ratio=1.5, sortino_ratio=2.0, max_drawdown=0.15,
                calmar_ratio=10.0, win_rate=0.55, profit_factor=1.8,
                total_trades=100, avg_trade_return=0.002,
                avg_trade_duration_hours=24,
                start_date=datetime.now(), end_date=datetime.now(),
                total_return=0.45,
            ),
            regime_tier=RegimeTier.ALL_WEATHER,
            regime_performance={},
            cluster_id=1,
            similarity_score=0.3,
            rejection_reason=None,
        )
        strategy.hifa_result = hifa_result
        strategy.status = StrategyStatus.HIFA_PASSED

        deployment = HinanceAdapter.to_deployment_format(strategy)

        assert deployment["strategy_id"] == "test_deploy"
        assert "genome_json" in deployment
        assert deployment["backtest_sharpe"] == 1.5
        assert deployment["hifa_gates_passed"] == 7

    def test_strategy_adapter_factory(self):
        """Test StrategyAdapter factory methods."""
        genome_json = json.dumps({
            "entry_conditions": [],
            "exit_conditions": [],
            "position_sizing": {"method": "fixed"},
            "parameters": {},
            "lookback_periods": {},
            "indicators": [],
        })

        # Test from_hinance_json
        strategy = StrategyAdapter.from_hinance_json(genome_json)
        assert strategy is not None
        assert isinstance(strategy, UnifiedStrategy)


# ==============================================================================
# Feature Vector Tests
# ==============================================================================

class TestFeatureVector:
    """Tests for FeatureVector class."""

    def test_feature_vector_creation(self):
        """Test feature vector creation."""
        values = np.random.randn(FEATURE_DIMENSIONS)
        fv = FeatureVector(values=values)

        assert len(fv.values) == FEATURE_DIMENSIONS
        assert fv.asset == "BTCUSDT"

    def test_feature_vector_wrong_dimension(self):
        """Test feature vector rejects wrong dimensions."""
        with pytest.raises(ValueError):
            FeatureVector(values=np.zeros(50))  # Wrong size

    def test_feature_vector_get_set(self):
        """Test feature access by name."""
        fv = FeatureVector()

        fv["rsi_14"] = 65.0
        assert fv["rsi_14"] == 65.0

        with pytest.raises(KeyError):
            _ = fv["nonexistent_feature"]

    def test_feature_vector_groups(self):
        """Test feature group accessors."""
        fv = FeatureVector()
        fv["rsi_14"] = 50.0
        fv["macd_signal"] = 1.0

        momentum = fv.get_momentum_features()
        assert "rsi_14" in momentum
        assert momentum["rsi_14"] == 50.0

    def test_feature_vector_validation(self):
        """Test feature vector validation."""
        fv = FeatureVector()
        is_valid, errors = fv.validate()
        assert is_valid  # All zeros should be valid

        # Set out of range value
        fv.values[35] = 150.0  # RSI > 100
        is_valid, errors = fv.validate()
        assert not is_valid
        assert any("rsi_14" in e for e in errors)

    def test_feature_vector_clip(self):
        """Test feature clipping to bounds."""
        fv = FeatureVector()
        fv.values[35] = 150.0  # RSI > 100

        clipped = fv.clip_to_bounds()
        assert clipped.values[35] == 100.0

    def test_feature_vector_to_dict(self):
        """Test conversion to dictionary."""
        fv = FeatureVector()
        fv["rsi_14"] = 55.0

        d = fv.to_dict()
        assert "rsi_14" in d
        assert d["rsi_14"] == 55.0

    def test_feature_vector_from_dict(self):
        """Test creation from dictionary."""
        data = {"rsi_14": 45.0, "macd_signal": -1.0}
        fv = FeatureVector.from_dict(data)

        assert fv["rsi_14"] == 45.0
        assert fv["macd_signal"] == -1.0


# ==============================================================================
# Feature Schema Tests
# ==============================================================================

class TestFeatureSchema:
    """Tests for feature schema definition."""

    def test_feature_count(self):
        """Test correct number of features."""
        assert len(FEATURE_SCHEMA) == FEATURE_DIMENSIONS
        assert CONST_FEATURE_DIMS == 60

    def test_feature_indices_unique(self):
        """Test all feature indices are unique."""
        indices = [f.index for f in FEATURE_SCHEMA.values()]
        assert len(indices) == len(set(indices))

    def test_feature_indices_continuous(self):
        """Test feature indices are 0 to N-1."""
        indices = sorted([f.index for f in FEATURE_SCHEMA.values()])
        assert indices == list(range(FEATURE_DIMENSIONS))

    def test_feature_groups_complete(self):
        """Test all feature groups are represented."""
        groups = set(f.group for f in FEATURE_SCHEMA.values())
        expected = {
            FeatureGroup.PRICE,
            FeatureGroup.VOLUME,
            FeatureGroup.VOLATILITY,
            FeatureGroup.MOMENTUM,
            FeatureGroup.MICROSTRUCTURE,
            FeatureGroup.REGIME,
            FeatureGroup.TIME,
        }
        assert groups == expected

    def test_get_feature_names(self):
        """Test get_feature_names helper."""
        all_names = get_feature_names()
        assert len(all_names) == FEATURE_DIMENSIONS

        momentum_names = get_feature_names(FeatureGroup.MOMENTUM)
        assert "rsi_14" in momentum_names
        assert len(momentum_names) == 10

    def test_get_feature_indices(self):
        """Test get_feature_indices helper."""
        price_indices = get_feature_indices(FeatureGroup.PRICE)
        assert len(price_indices) == 15
        assert 0 in price_indices  # First price feature


# ==============================================================================
# Constants Tests
# ==============================================================================

class TestConstants:
    """Tests for shared constants."""

    def test_threshold_values(self):
        """Test threshold constants have reasonable values."""
        assert 0 < DEFAULT_DSR_THRESHOLD <= 1
        assert 0 < DEFAULT_PBO_THRESHOLD <= 1

    def test_limit_values(self):
        """Test limit constants are positive."""
        assert MAX_STRATEGY_PARAMETERS > 0
        assert FEATURE_DIMENSIONS == 60


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestIntegration:
    """Integration tests for shared module."""

    def test_full_strategy_lifecycle(self):
        """Test strategy from creation to production-ready."""
        # Create genome
        genome = StrategyGenome(
            entry_conditions=[{"operator": "gt", "left": "RSI", "right": 30}],
            exit_conditions=[{"operator": "lt", "left": "RSI", "right": 70}],
            position_sizing={"method": "fixed", "size": 0.5},
            parameters={"rsi_period": 14},
            lookback_periods={"rsi": 14},
            indicators=["RSI"],
        )

        # Create strategy
        strategy = UnifiedStrategy(
            genome=genome,
            source_engine=SourceEngine.EVOLUTIONARY,
        )
        assert strategy.status == StrategyStatus.GENERATED

        # Simulate HIFA validation
        strategy.update_status(StrategyStatus.HIFA_VALIDATING)

        hifa_result = HIFAResult(
            strategy_id=strategy.strategy_id,
            passed=True,
            final_gate=7,
            gate_results={},
            statistical_scores=None,
            backtest_metrics=BacktestMetrics(
                sharpe_ratio=1.8, sortino_ratio=2.5, max_drawdown=0.12,
                calmar_ratio=15.0, win_rate=0.58, profit_factor=2.0,
                total_trades=200, avg_trade_return=0.003,
                avg_trade_duration_hours=18,
                start_date=datetime(2023, 1, 1), end_date=datetime(2024, 1, 1),
                total_return=0.65,
            ),
            regime_tier=RegimeTier.ALL_WEATHER,
            regime_performance={"normal": 1.5, "elevated": 1.0, "crisis": 0.5},
            cluster_id=2,
            similarity_score=0.25,
            rejection_reason=None,
        )
        strategy.attach_hifa_result(hifa_result)
        assert strategy.is_hifa_validated

        # Simulate forward testing
        strategy.update_status(StrategyStatus.SHADOW_TRADING)

        forward_result = ForwardTestResult(
            strategy_id=strategy.strategy_id,
            passed=True,
            shadow_metrics=ShadowMetrics(
                sharpe_ratio=1.2,
                sortino_ratio=1.8,
                max_drawdown=0.08,
                win_rate=0.52,
                total_trades=45,
                avg_slippage_bps=3.5,
                avg_latency_ms=85,
                total_fees_paid=120.0,
                start_date=datetime(2024, 1, 15),
                end_date=datetime(2024, 2, 1),
                duration_days=17,
                total_return=0.08,
            ),
            transfer_ratio=0.67,  # 1.2 / 1.8
            dd_ratio=0.67,        # 0.08 / 0.12
            rejection_reason=None,
            backtest_sharpe=1.8,
            backtest_max_dd=0.12,
            test_start=datetime(2024, 1, 15),
            test_end=datetime(2024, 2, 1),
        )
        strategy.attach_forward_result(forward_result)
        assert strategy.is_forward_validated
        assert strategy.is_production_ready

        # Note: Deployment format is used BEFORE forward testing starts
        # Transfer ratio comes from forward_result, not deployment format
        assert strategy.forward_result.transfer_ratio == 0.67
        assert strategy.hifa_result.backtest_sharpe == 1.8

    def test_batch_conversion(self):
        """Test batch conversion of strategies."""
        # Create batch of Hinance JSON strings
        jsons = []
        for i in range(5):
            jsons.append(json.dumps({
                "entry_conditions": [{"indicator": "RSI", "op": "lt", "value": 30 + i}],
                "exit_conditions": [{"indicator": "RSI", "op": "gt", "value": 70 - i}],
                "position_sizing": {"method": "fixed"},
                "parameters": {"period": 14 + i},
                "lookback_periods": {},
                "indicators": ["RSI"],
            }))

        strategies = BatchAdapter.convert_batch(jsons, "hinance")
        assert len(strategies) == 5
        assert all(isinstance(s, UnifiedStrategy) for s in strategies)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
