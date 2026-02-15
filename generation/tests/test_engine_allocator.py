"""
Tests for Phase 6: Dynamic Engine Allocation (Thompson Sampling)

Covers all 10 required tests from the v2.0 guide plus edge cases.

Explorer Prime v2.0 - Phase 6
"""

import pytest
import numpy as np
from collections import Counter

from generation.engine_allocator import (
    EngineAllocator,
    ExplorationBudgetManager,
    FeatureExplorationStats,
    GAP_AFFINITY,
    ENGINE_NAMES,
    ENGINE_DECAY_RATES,
)
from diagnostics.anomaly_signature import GapType


# ==============================================================================
# Test 1: Allocate Respects Floor
# ==============================================================================

class TestAllocateRespectsFloor:
    """test_allocate_respects_floor: No engine below 8%."""

    def test_all_engines_above_floor(self):
        """Each engine should get at least exploration_floor fraction."""
        allocator = EngineAllocator(exploration_floor=0.08, seed=42)
        alloc = allocator.allocate(GapType.STRUCTURAL, n_strategies=1000)

        # Check each engine has at least floor
        floor_count = int(0.08 * 1000) - 5  # Small tolerance for rounding
        for engine in ENGINE_NAMES:
            assert alloc[engine] >= floor_count, \
                f"{engine} got {alloc[engine]}, expected >= {floor_count}"

    def test_floor_with_extreme_bias(self):
        """Even with strong structural bias, no engine is starved."""
        allocator = EngineAllocator(exploration_floor=0.08, seed=123)

        # Run multiple times to check stochastic floor
        for _ in range(10):
            alloc = allocator.allocate(GapType.STRUCTURAL, n_strategies=1000)
            for engine in ENGINE_NAMES:
                # With 5 engines and floor=0.08, worst case after normalization
                # is 0.08/sum(weights), so count should be > 0
                assert alloc[engine] > 0, f"{engine} was starved (count=0)"


# ==============================================================================
# Test 2: Allocate Sums to Total
# ==============================================================================

class TestAllocateSumsToTotal:
    """test_allocate_sums_to_total: Allocations sum to n_strategies."""

    def test_sums_to_1000(self):
        allocator = EngineAllocator(seed=42)
        alloc = allocator.allocate(GapType.STRUCTURAL, n_strategies=1000)
        assert sum(alloc.values()) == 1000

    def test_sums_to_500(self):
        allocator = EngineAllocator(seed=42)
        alloc = allocator.allocate(GapType.FEATURE, n_strategies=500)
        assert sum(alloc.values()) == 500

    def test_sums_to_1(self):
        """Edge case: single strategy."""
        allocator = EngineAllocator(seed=42)
        alloc = allocator.allocate(GapType.UNKNOWN, n_strategies=1)
        assert sum(alloc.values()) == 1

    def test_all_gap_types_sum_correctly(self):
        """Test all gap types produce correct sums."""
        allocator = EngineAllocator(seed=42)
        for gap_type in GapType:
            alloc = allocator.allocate(gap_type, n_strategies=1000)
            assert sum(alloc.values()) == 1000, f"Failed for {gap_type}"


# ==============================================================================
# Test 3: Gap Affinity Shifts Prior
# ==============================================================================

class TestGapAffinityShiftsPrior:
    """test_gap_affinity_shifts_prior: Structural gap increases evolutionary sampling."""

    def test_structural_favors_evolutionary(self):
        """With structural gap, evolutionary should get higher average allocation."""
        evo_counts = []
        for seed in range(50):
            allocator = EngineAllocator(seed=seed)
            alloc = allocator.allocate(GapType.STRUCTURAL, n_strategies=1000)
            evo_counts.append(alloc["evolutionary"])

        avg_evo = np.mean(evo_counts)
        # Structural gap has 0.55 affinity for evolutionary — should be highest on average
        assert avg_evo > 150, f"Evolutionary avg {avg_evo} too low for structural gap"

    def test_feature_favors_genai(self):
        """Feature gap should boost GenAI allocation."""
        genai_counts = []
        for seed in range(50):
            allocator = EngineAllocator(seed=seed)
            alloc = allocator.allocate(GapType.FEATURE, n_strategies=1000)
            genai_counts.append(alloc["genai"])

        avg_genai = np.mean(genai_counts)
        assert avg_genai > 150, f"GenAI avg {avg_genai} too low for feature gap"

    def test_pattern_favors_pattern(self):
        """Pattern gap should boost pattern engine."""
        pattern_counts = []
        for seed in range(50):
            allocator = EngineAllocator(seed=seed)
            alloc = allocator.allocate(GapType.PATTERN, n_strategies=1000)
            pattern_counts.append(alloc["pattern"])

        avg_pattern = np.mean(pattern_counts)
        assert avg_pattern > 150, f"Pattern avg {avg_pattern} too low for pattern gap"


# ==============================================================================
# Test 4: Update Adjusts Beta
# ==============================================================================

class TestUpdateAdjustsBeta:
    """test_update_adjusts_beta: HIFA results update distributions correctly."""

    def test_success_increases_alpha(self):
        """Successful generation should increase alpha (success parameter)."""
        allocator = EngineAllocator(seed=42)
        initial_alpha = allocator.betas["evolutionary"][0]

        # 100 generated, 80 passed HIFA
        allocator.update("evolutionary", n_generated=100, n_hifa_passed=80)

        new_alpha = allocator.betas["evolutionary"][0]
        # After update and decay: (initial + 80) * decay_rate
        expected = (initial_alpha + 80) * ENGINE_DECAY_RATES["evolutionary"]
        assert abs(new_alpha - expected) < 0.01

    def test_failure_increases_beta(self):
        """Failed generation should increase beta (failure parameter)."""
        allocator = EngineAllocator(seed=42)
        initial_beta = allocator.betas["evolutionary"][1]

        # 100 generated, 20 passed (80 failed)
        allocator.update("evolutionary", n_generated=100, n_hifa_passed=20)

        new_beta = allocator.betas["evolutionary"][1]
        expected = (initial_beta + 80) * ENGINE_DECAY_RATES["evolutionary"]
        assert abs(new_beta - expected) < 0.01

    def test_weight_shifts_after_update(self):
        """Expected weight should reflect accumulated success/failure."""
        allocator = EngineAllocator(seed=42)

        # Evolutionary does very well
        for _ in range(5):
            allocator.update("evolutionary", n_generated=100, n_hifa_passed=90)

        # LSM does poorly
        for _ in range(5):
            allocator.update("lsm", n_generated=100, n_hifa_passed=5)

        weights = allocator.get_current_weights()
        assert weights["evolutionary"] > weights["lsm"]


# ==============================================================================
# Test 5: Decay Prevents Lock-in
# ==============================================================================

class TestDecayPreventsLockin:
    """test_decay_prevents_lockin: Old success decays over time."""

    def test_weight_decays_without_updates(self):
        """After initial success, weight should drift toward prior without new data."""
        allocator = EngineAllocator(seed=42)

        # Initial strong success
        allocator.update("evolutionary", n_generated=100, n_hifa_passed=95)
        weight_after_success = allocator.get_current_weights()["evolutionary"]

        # Many cycles of no new data (just decay)
        for _ in range(200):
            # Simulate decay by applying a neutral update
            allocator.update("evolutionary", n_generated=0, n_hifa_passed=0)

        weight_after_decay = allocator.get_current_weights()["evolutionary"]

        # Weight should drift toward 0.5 (uniform prior Beta(1,1))
        # After enough decay, alpha and beta both approach 1.0
        assert weight_after_decay < weight_after_success or \
            abs(weight_after_decay - 0.5) < abs(weight_after_success - 0.5)


# ==============================================================================
# Test 6: Per-Engine Decay Rates
# ==============================================================================

class TestPerEngineDecayRates:
    """test_per_engine_decay_rates: Evolutionary decays faster than LSM."""

    def test_evolutionary_decays_faster(self):
        """Evolutionary's success memory fades faster than LSM's."""
        allocator = EngineAllocator(seed=42)

        # Both start with same strong success
        allocator.update("evolutionary", n_generated=100, n_hifa_passed=90)
        allocator.update("lsm", n_generated=100, n_hifa_passed=90)

        # Apply 100 cycles of zero updates (pure decay)
        for _ in range(100):
            allocator.update("evolutionary", n_generated=0, n_hifa_passed=0)
            allocator.update("lsm", n_generated=0, n_hifa_passed=0)

        weights = allocator.get_current_weights()

        # LSM should retain more of its historical success
        # After same initial success, slower decay → higher alpha relative to beta
        # Both started at same point, but evolutionary decayed faster
        assert allocator.betas["lsm"][0] > allocator.betas["evolutionary"][0]

    def test_decay_rates_are_different(self):
        """Verify the decay rates are actually per-engine, not uniform."""
        allocator = EngineAllocator()
        assert allocator.decay_rates["evolutionary"] < allocator.decay_rates["lsm"]
        assert allocator.decay_rates["evolutionary"] == 0.990
        assert allocator.decay_rates["lsm"] == 0.998


# ==============================================================================
# Test 7: Thompson Stochastic
# ==============================================================================

class TestThompsonStochastic:
    """test_thompson_stochastic: Multiple allocate() calls produce variance."""

    def test_different_seeds_different_allocations(self):
        """Same gap type but different seeds should produce different allocations."""
        allocations = []
        for seed in range(10):
            allocator = EngineAllocator(seed=seed)
            alloc = allocator.allocate(GapType.STRUCTURAL, n_strategies=1000)
            allocations.append(alloc["evolutionary"])

        # Should have variance in evolutionary allocations
        assert len(set(allocations)) > 1, "Thompson sampling produced identical results"

    def test_repeated_calls_vary(self):
        """Same allocator, repeated calls should produce different allocations."""
        allocator = EngineAllocator(seed=42)
        allocs = [
            allocator.allocate(GapType.AMBIGUOUS, n_strategies=1000)["evolutionary"]
            for _ in range(20)
        ]
        assert len(set(allocs)) > 1, "Repeated Thompson sampling is deterministic"


# ==============================================================================
# Test 8: Exploration Budget 20%
# ==============================================================================

class TestExplorationBudget:
    """test_exploration_budget_20_percent: 200 of 1000 go to experimental features."""

    def test_default_20_percent(self):
        manager = ExplorationBudgetManager()
        n = manager.allocate_exploration(1000, ["feat_A", "feat_B"])
        assert n == 200  # 20% of 1000

    def test_custom_fraction(self):
        manager = ExplorationBudgetManager(exploration_fraction=0.30)
        n = manager.allocate_exploration(1000, ["feat_A"])
        assert n == 300

    def test_no_features_no_exploration(self):
        manager = ExplorationBudgetManager()
        n = manager.allocate_exploration(1000, [])
        assert n == 0

    def test_small_total(self):
        manager = ExplorationBudgetManager()
        n = manager.allocate_exploration(10, ["feat_A"])
        assert n == 2  # 20% of 10


# ==============================================================================
# Test 9: Exploration Stats Tracked
# ==============================================================================

class TestExplorationStatsTracked:
    """test_exploration_stats_tracked: Per-feature metrics maintained."""

    def test_records_generation(self):
        manager = ExplorationBudgetManager()
        manager.allocate_exploration(1000, ["feat_A", "feat_B"])

        manager.record_generation("feat_A", n_generated=50, n_hifa_passed=10)
        manager.record_generation("feat_A", n_generated=50, n_hifa_passed=15)

        stats = manager.get_feature_exploration_stats()
        assert "feat_A" in stats
        assert stats["feat_A"].strategies_generated == 100
        assert stats["feat_A"].hifa_passed == 25
        assert stats["feat_A"].pass_rate == 0.25

    def test_records_shadow_results(self):
        manager = ExplorationBudgetManager()
        manager.record_shadow_result("feat_A", {"sharpe": 1.2, "days": 14})
        manager.record_shadow_result("feat_A", {"sharpe": 0.8, "days": 14})

        stats = manager.get_feature_exploration_stats()
        assert len(stats["feat_A"].shadow_results) == 2

    def test_multiple_features(self):
        manager = ExplorationBudgetManager()
        manager.record_generation("feat_A", 50, 10)
        manager.record_generation("feat_B", 50, 20)

        stats = manager.get_feature_exploration_stats()
        assert stats["feat_A"].pass_rate == 0.2
        assert stats["feat_B"].pass_rate == 0.4


# ==============================================================================
# Test 10: History Recorded
# ==============================================================================

class TestHistoryRecorded:
    """test_history_recorded: Allocation history retrievable."""

    def test_history_tracks_allocations(self):
        allocator = EngineAllocator(seed=42)
        allocator.allocate(GapType.STRUCTURAL, n_strategies=1000)
        allocator.allocate(GapType.FEATURE, n_strategies=500)

        history = allocator.get_allocation_history()
        assert len(history) == 2
        assert history[0].gap_type == GapType.STRUCTURAL
        assert history[0].n_strategies == 1000
        assert history[1].gap_type == GapType.FEATURE
        assert history[1].n_strategies == 500

    def test_history_has_weights(self):
        allocator = EngineAllocator(seed=42)
        allocator.allocate(GapType.AMBIGUOUS, n_strategies=1000)

        history = allocator.get_allocation_history()
        assert len(history[0].weights) == 5
        assert abs(sum(history[0].weights.values()) - 1.0) < 0.01

    def test_empty_history(self):
        allocator = EngineAllocator(seed=42)
        assert len(allocator.get_allocation_history()) == 0


# ==============================================================================
# Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Additional edge case tests."""

    def test_gap_affinity_all_types_present(self):
        """All gap types should have affinity vectors."""
        for gap_type in GapType:
            assert gap_type in GAP_AFFINITY
            assert len(GAP_AFFINITY[gap_type]) == 5

    def test_affinity_sums_to_one(self):
        """Each affinity vector should sum to ~1.0."""
        for gap_type, affinity in GAP_AFFINITY.items():
            total = sum(affinity.values())
            assert abs(total - 1.0) < 0.01, f"{gap_type} affinity sums to {total}"

    def test_unknown_engine_update(self):
        """Updating non-existent engine should be no-op."""
        allocator = EngineAllocator(seed=42)
        allocator.update("nonexistent", 100, 50)  # Should not crash

    def test_zero_strategies(self):
        """Allocating 0 strategies."""
        allocator = EngineAllocator(seed=42)
        alloc = allocator.allocate(GapType.UNKNOWN, n_strategies=0)
        assert sum(alloc.values()) == 0

    def test_feature_stats_empty(self):
        """Empty stats for new feature."""
        stats = FeatureExplorationStats(feature_id="test")
        assert stats.pass_rate == 0.0
        assert stats.strategies_generated == 0
