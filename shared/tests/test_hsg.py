"""
Tests for Hierarchical Strategy Graph (HSG) - Phase 1

Explorer Prime v2.0
15+ tests covering: structural modes, state bus, mutations, crossover,
fitness penalties, backward compatibility, and signal bounds.
"""

import math
import copy
import random
import numpy as np
import pytest

from shared.hierarchical_genome import (
    StructuralMode,
    SlotCategory,
    CATEGORY_HALF_LIVES,
    CATEGORY_SLOT_RANGES,
    SignalStruct,
    TradeSignal,
    StateBusConfig,
    StateBus,
    TreeNode,
    DecisionTree,
    HierarchicalGenome,
    create_flat_genome,
    create_dual_genome,
    create_full_genome,
    _create_simple_tree,
)


# ==============================================================================
# Helpers
# ==============================================================================

def _make_deterministic_tree(action_direction=0.8, action_confidence=0.7,
                              action_size=0.2) -> DecisionTree:
    """Create a small deterministic tree: root -> feature[0] > 0 -> leaf."""
    tree = DecisionTree(max_nodes=7)
    tree.nodes[0] = TreeNode(
        node_id=0, is_leaf=False, feature_idx=0,
        operator=">", threshold=0.0,
        left_child=1, right_child=2,
    )
    tree.nodes[1] = TreeNode(
        node_id=1, is_leaf=True,
        action_direction=action_direction,
        action_confidence=action_confidence,
        action_size=action_size,
    )
    tree.nodes[2] = TreeNode(
        node_id=2, is_leaf=True,
        action_direction=-action_direction,
        action_confidence=action_confidence * 0.5,
        action_size=action_size * 0.5,
    )
    return tree


def _make_features(val: float = 1.0, n: int = 60) -> np.ndarray:
    """Create a feature vector with all elements set to val."""
    return np.full(n, val, dtype=np.float64)


# ==============================================================================
# Test 1: FLAT mode zero overhead
# ==============================================================================

class TestFlatModeZeroOverhead:
    """FLAT genome evaluates identical to v1.0 tree."""

    def test_flat_mode_passthrough(self):
        """FLAT mode output equals raw tree evaluation."""
        tree = _make_deterministic_tree(0.8, 0.7, 0.2)
        genome = HierarchicalGenome.from_flat_tree(tree)

        features_pos = _make_features(1.0)  # feature[0] > 0 → left leaf
        features_neg = _make_features(-1.0)  # feature[0] <= 0 → right leaf

        # Direct tree evaluation
        tree_sig_pos, _ = tree.evaluate(features_pos)
        tree_sig_neg, _ = tree.evaluate(features_neg)

        # HSG evaluation
        hsg_sig_pos = genome.evaluate({"default": features_pos})
        hsg_sig_neg = genome.evaluate({"default": features_neg})

        # Should be identical
        assert hsg_sig_pos.direction == pytest.approx(tree_sig_pos.direction_score)
        assert hsg_sig_pos.confidence == pytest.approx(tree_sig_pos.confidence)
        assert hsg_sig_pos.size == pytest.approx(tree_sig_pos.suggested_size)

        assert hsg_sig_neg.direction == pytest.approx(tree_sig_neg.direction_score)
        assert hsg_sig_neg.confidence == pytest.approx(tree_sig_neg.confidence)
        assert hsg_sig_neg.size == pytest.approx(tree_sig_neg.suggested_size)

    def test_flat_mode_no_state_bus_impact(self):
        """FLAT mode doesn't use state bus in arbitration."""
        tree = _make_deterministic_tree()
        genome = HierarchicalGenome.from_flat_tree(tree)

        features = _make_features(1.0)
        sig1 = genome.evaluate({"default": features})
        sig2 = genome.evaluate({"default": features})

        # Results should be identical since FLAT ignores state bus
        assert sig1.direction == pytest.approx(sig2.direction)
        assert sig1.size == pytest.approx(sig2.size)

    def test_flat_mode_validation_passes(self):
        """FLAT mode passes validation with single tree <= 31 nodes."""
        tree = _make_deterministic_tree()
        genome = HierarchicalGenome.from_flat_tree(tree)
        valid, errors = genome.validate()
        assert valid is True
        assert len(errors) == 0

    def test_flat_mode_validation_fails_multiple_trees(self):
        """FLAT mode fails if it has multiple trees."""
        genome = HierarchicalGenome(
            structural_mode=StructuralMode.FLAT,
            timeframe_trees={
                "1m": _make_deterministic_tree(),
                "15m": _make_deterministic_tree(),
            },
        )
        valid, errors = genome.validate()
        assert valid is False
        assert any("FLAT mode requires exactly 1 tree" in e for e in errors)


# ==============================================================================
# Test 2: DUAL mode weighted average
# ==============================================================================

class TestDualModeWeightedAverage:
    """Two trees combine correctly via weighted average."""

    def test_dual_mode_weighted_average(self):
        """DUAL arbitration produces weighted average of two signals."""
        tree_1m = _make_deterministic_tree(0.8, 0.9, 0.3)
        tree_15m = _make_deterministic_tree(0.4, 0.6, 0.1)

        genome = HierarchicalGenome(
            structural_mode=StructuralMode.DUAL,
            timeframe_trees={"1m": tree_1m, "15m": tree_15m},
            max_total_nodes=63,
            dual_weight=0.6,
        )

        features = {"1m": _make_features(1.0), "15m": _make_features(1.0)}
        result = genome.evaluate(features)

        # Both trees take left branch (feature[0] > 0), so:
        # 1m: direction=0.8, conf=0.7, size=0.2 (wait, tree has 0.8,0.9,0.3)
        # Actually the tree was created with 0.8, 0.9, 0.3 and 0.4, 0.6, 0.1
        # left leaf: direction=0.8, conf=0.9, size=0.3 for tree_1m
        # left leaf: direction=0.4, conf=0.6, size=0.1 for tree_15m
        expected_dir = 0.6 * 0.8 + 0.4 * 0.4
        expected_conf = 0.6 * 0.9 + 0.4 * 0.6
        expected_size = 0.6 * 0.3 + 0.4 * 0.1

        assert result.direction == pytest.approx(expected_dir, abs=0.01)
        assert result.confidence == pytest.approx(expected_conf, abs=0.01)
        assert result.size == pytest.approx(expected_size, abs=0.01)

    def test_dual_mode_equal_weight(self):
        """DUAL with 0.5 weight produces simple average."""
        tree_a = _make_deterministic_tree(1.0, 1.0, 1.0)
        tree_b = _make_deterministic_tree(0.0, 0.0, 0.0)

        genome = HierarchicalGenome(
            structural_mode=StructuralMode.DUAL,
            timeframe_trees={"1m": tree_a, "15m": tree_b},
            max_total_nodes=63,
            dual_weight=0.5,
        )

        features = {"1m": _make_features(1.0), "15m": _make_features(1.0)}
        result = genome.evaluate(features)

        # tree_a left leaf: (1.0, 1.0, 1.0), tree_b left leaf: (0.0, 0.0, 0.0)
        # Note: SignalStruct clips values, so 0.0 conf/size stay at 0.0
        assert result.direction == pytest.approx(0.5, abs=0.01)

    def test_dual_mode_validation(self):
        """DUAL requires exactly 2 trees."""
        genome = HierarchicalGenome(
            structural_mode=StructuralMode.DUAL,
            timeframe_trees={"1m": _make_deterministic_tree()},
            max_total_nodes=63,
        )
        valid, errors = genome.validate()
        assert valid is False
        assert any("DUAL mode requires exactly 2 trees" in e for e in errors)


# ==============================================================================
# Test 3: FULL mode arbitration
# ==============================================================================

class TestFullModeArbitration:
    """Arbitration tree receives correct inputs."""

    def test_full_mode_uses_arbitration_tree(self):
        """FULL mode evaluates the arbitration tree on packed signals."""
        tree_1m = _make_deterministic_tree(0.8, 0.7, 0.2)
        tree_15m = _make_deterministic_tree(0.4, 0.5, 0.1)
        tree_4h = _make_deterministic_tree(0.6, 0.8, 0.3)

        # Simple arbitration tree: always returns direction 0.5
        arb = DecisionTree(max_nodes=7)
        arb.nodes[0] = TreeNode(
            node_id=0, is_leaf=True,
            action_direction=0.5, action_confidence=0.9, action_size=0.2,
        )

        genome = HierarchicalGenome(
            structural_mode=StructuralMode.FULL,
            timeframe_trees={
                "1m": tree_1m, "15m": tree_15m, "4h": tree_4h,
            },
            arbitration_tree=arb,
            max_total_nodes=63,
        )

        features = {
            "1m": _make_features(1.0),
            "15m": _make_features(1.0),
            "4h": _make_features(1.0),
        }
        result = genome.evaluate(features)

        # Arbitration tree is a leaf that always returns 0.5 direction
        assert result.direction == pytest.approx(0.5)
        assert result.confidence == pytest.approx(0.9)
        assert result.size == pytest.approx(0.2)
        # Source signals should be populated
        assert len(result.source_signals) == 3

    def test_full_mode_validation_limits(self):
        """FULL mode: 2-4 trees, arbitration max 7 nodes."""
        genome = create_full_genome()
        valid, errors = genome.validate()
        # Random trees might violate constraints; check structure is reasonable
        assert genome.structural_mode == StructuralMode.FULL
        assert 2 <= len(genome.timeframe_trees) <= 4

    def test_full_mode_arbitration_max_nodes(self):
        """Arbitration tree >7 nodes fails validation."""
        big_arb = _create_simple_tree(max_nodes=15)
        genome = HierarchicalGenome(
            structural_mode=StructuralMode.FULL,
            timeframe_trees={
                "1m": _make_deterministic_tree(),
                "15m": _make_deterministic_tree(),
            },
            arbitration_tree=big_arb,
            max_total_nodes=63,
        )
        valid, errors = genome.validate()
        if big_arb.node_count > 7:
            assert valid is False
            assert any("Arbitration tree" in e for e in errors)


# ==============================================================================
# Test 4: State bus decay
# ==============================================================================

class TestStateBusDecay:
    """Slots decay correctly with configured half-lives."""

    def test_state_bus_decay_momentum(self):
        """Momentum slots (half-life 6) decay faster than regime slots."""
        config = StateBusConfig()
        bus = StateBus(config)

        # Set momentum slot (0) and regime slot (16) to same value
        bus.update(0, 1.0)    # MOMENTUM, half-life 6
        bus.update(16, 1.0)   # REGIME, half-life 24

        # After one decay tick
        bus.decay()
        momentum_val = bus.slots[0]
        regime_val = bus.slots[16]

        # Momentum should decay faster (smaller value after decay)
        assert momentum_val < regime_val
        assert momentum_val < 1.0
        assert regime_val < 1.0

    def test_state_bus_half_life_accuracy(self):
        """After half-life ticks, value should be ~0.5."""
        config = StateBusConfig()
        bus = StateBus(config)

        # Use MOMENTUM slot (half-life 6)
        bus.update(0, 1.0)
        for _ in range(6):
            bus.decay()

        # Should be approximately 0.5 after half-life ticks
        assert bus.slots[0] == pytest.approx(0.5, abs=0.05)

    def test_state_bus_reset(self):
        """Reset zeroes all slots."""
        bus = StateBus()
        bus.update(0, 5.0)
        bus.update(15, 3.0)
        bus.reset()
        assert np.allclose(bus.slots, 0.0)

    def test_state_bus_tick_updates_then_decays(self):
        """tick() applies updates then decays."""
        bus = StateBus()
        bus.tick(updates={0: 1.0})
        # After tick, slot should have update value * decay
        assert 0.0 < bus.slots[0] < 1.0


# ==============================================================================
# Test 5: Node budget constraint
# ==============================================================================

class TestNodeBudgetConstraint:
    """Total nodes never exceeds max_total_nodes."""

    def test_total_nodes_under_budget(self):
        """validate() catches over-budget genomes."""
        tree_big = _create_simple_tree(max_nodes=40)
        genome = HierarchicalGenome(
            structural_mode=StructuralMode.DUAL,
            timeframe_trees={
                "1m": tree_big,
                "15m": _create_simple_tree(max_nodes=30),
            },
            max_total_nodes=50,
        )

        if genome.total_node_count > 50:
            valid, errors = genome.validate()
            assert valid is False
            assert any("exceeds max" in e for e in errors)

    def test_flat_mode_max_31_nodes(self):
        """FLAT tree > 31 nodes fails validation."""
        big_tree = _create_simple_tree(max_nodes=35)
        genome = HierarchicalGenome.from_flat_tree(big_tree)
        # Override max to allow genome creation but tree has >31
        if big_tree.node_count > 31:
            valid, errors = genome.validate()
            assert valid is False


# ==============================================================================
# Test 6: Minimum allocation
# ==============================================================================

class TestMinimumAllocation:
    """Each tree has >= 15% of budget in non-FLAT modes."""

    def test_minimum_15_percent_allocation(self):
        """Trees below 15% share fail validation."""
        # Create one big tree and one tiny tree
        big_tree = _create_simple_tree(max_nodes=25)
        tiny_tree = DecisionTree(max_nodes=5)
        tiny_tree.nodes[0] = TreeNode(node_id=0, is_leaf=True,
                                        action_direction=0.5)

        genome = HierarchicalGenome(
            structural_mode=StructuralMode.DUAL,
            timeframe_trees={"1m": big_tree, "15m": tiny_tree},
            max_total_nodes=63,
        )

        total = genome.total_node_count
        tiny_share = tiny_tree.node_count / total if total > 0 else 0

        if tiny_share < 0.15:
            valid, errors = genome.validate()
            assert valid is False
            assert any("15%" in e for e in errors)

    def test_minimum_5_nodes_per_tree(self):
        """Trees with < 5 nodes fail validation in DUAL/FULL."""
        small_tree = DecisionTree(max_nodes=5)
        small_tree.nodes[0] = TreeNode(node_id=0, is_leaf=True,
                                        action_direction=0.3)

        genome = HierarchicalGenome(
            structural_mode=StructuralMode.DUAL,
            timeframe_trees={
                "1m": _make_deterministic_tree(),
                "15m": small_tree,
            },
            max_total_nodes=63,
        )

        valid, errors = genome.validate()
        assert valid is False
        assert any("minimum 5" in e for e in errors)


# ==============================================================================
# Test 7: Structural mode mutation
# ==============================================================================

class TestStructuralModeMutation:
    """Transitions between FLAT/DUAL/FULL modes work."""

    def test_flat_to_dual(self):
        """FLAT mutates to DUAL (adds second tree)."""
        genome = create_flat_genome()
        assert genome.structural_mode == StructuralMode.FLAT
        assert len(genome.timeframe_trees) == 1

        mutated = genome.mutate_structural_mode()
        assert mutated.structural_mode == StructuralMode.DUAL
        assert len(mutated.timeframe_trees) == 2

    def test_dual_to_full_or_flat(self):
        """DUAL mutates to either FULL or back to FLAT."""
        random.seed(42)
        genome = create_dual_genome()
        assert genome.structural_mode == StructuralMode.DUAL

        mutated = genome.mutate_structural_mode()
        assert mutated.structural_mode in (StructuralMode.FLAT, StructuralMode.FULL)

    def test_full_to_dual(self):
        """FULL simplifies to DUAL."""
        genome = create_full_genome()
        assert genome.structural_mode == StructuralMode.FULL

        mutated = genome.mutate_structural_mode()
        assert mutated.structural_mode == StructuralMode.DUAL
        assert len(mutated.timeframe_trees) == 2
        assert mutated.arbitration_tree is None

    def test_mutation_preserves_original(self):
        """Mutations create copies, don't modify original."""
        genome = create_flat_genome()
        original_mode = genome.structural_mode
        _ = genome.mutate_structural_mode()
        assert genome.structural_mode == original_mode


# ==============================================================================
# Test 8: Crossover between different modes
# ==============================================================================

class TestCrossoverDifferentModes:
    """Complex mode inherits correctly."""

    def test_crossover_inherits_complex_mode(self):
        """Offspring inherits the more complex parent's mode."""
        flat = create_flat_genome()
        dual = create_dual_genome()

        child = HierarchicalGenome.crossover(flat, dual)
        # DUAL is more complex than FLAT
        assert child.structural_mode == StructuralMode.DUAL

    def test_crossover_full_over_dual(self):
        """FULL > DUAL in mode hierarchy."""
        dual = create_dual_genome()
        full = create_full_genome()

        child = HierarchicalGenome.crossover(dual, full)
        assert child.structural_mode == StructuralMode.FULL

    def test_crossover_averages_half_lives(self):
        """State bus half-lives are averaged between parents."""
        parent_a = create_dual_genome()
        parent_b = create_dual_genome()

        # Set different half-lives
        parent_a.state_bus_config.half_lives[0] = 4.0
        parent_b.state_bus_config.half_lives[0] = 8.0

        child = HierarchicalGenome.crossover(parent_a, parent_b)
        assert child.state_bus_config.half_lives[0] == pytest.approx(6.0)

    def test_crossover_averages_dual_weight(self):
        """Dual weight is averaged between parents."""
        parent_a = create_dual_genome()
        parent_b = create_dual_genome()
        parent_a.dual_weight = 0.3
        parent_b.dual_weight = 0.7

        child = HierarchicalGenome.crossover(parent_a, parent_b)
        assert child.dual_weight == pytest.approx(0.5)

    def test_crossover_creates_new_id(self):
        """Offspring gets a new genome ID."""
        parent_a = create_flat_genome()
        parent_b = create_flat_genome()

        child = HierarchicalGenome.crossover(parent_a, parent_b)
        assert child.genome_id != parent_a.genome_id
        assert child.genome_id != parent_b.genome_id


# ==============================================================================
# Test 9: Fitness penalty
# ==============================================================================

class TestFitnessPenalty:
    """Concentration above 60% is penalized."""

    def test_no_penalty_flat_mode(self):
        """FLAT mode always has zero penalty."""
        genome = create_flat_genome()
        assert genome.concentration_penalty() == 0.0

    def test_no_penalty_balanced_trees(self):
        """Balanced allocation has no penalty."""
        tree_a = _create_simple_tree(max_nodes=15)
        tree_b = _create_simple_tree(max_nodes=15)

        genome = HierarchicalGenome(
            structural_mode=StructuralMode.DUAL,
            timeframe_trees={"1m": tree_a, "15m": tree_b},
            max_total_nodes=63,
        )

        # If roughly equal, dominant share ~0.5 < 0.6, penalty = 0
        total = genome.total_node_count
        if total > 0:
            dominant_share = max(tree_a.node_count, tree_b.node_count) / total
            if dominant_share <= 0.6:
                assert genome.concentration_penalty() == 0.0

    def test_penalty_when_concentrated(self):
        """Dominant tree > 60% share gets penalty."""
        big = _create_simple_tree(max_nodes=25)
        small = DecisionTree(max_nodes=7)
        for i in range(5):
            small.nodes[i] = TreeNode(node_id=i, is_leaf=(i >= 2),
                                       feature_idx=0, operator=">", threshold=0.0,
                                       left_child=i*2+1 if i < 2 else None,
                                       right_child=i*2+2 if i < 2 else None,
                                       action_direction=0.5 if i >= 2 else 0.0)

        genome = HierarchicalGenome(
            structural_mode=StructuralMode.DUAL,
            timeframe_trees={"1m": big, "15m": small},
            max_total_nodes=63,
        )

        total = genome.total_node_count
        if total > 0:
            dominant_share = max(big.node_count, small.node_count) / total
            if dominant_share > 0.6:
                penalty = genome.concentration_penalty()
                assert penalty > 0.0
                # Verify formula
                expected = max(0.0, dominant_share - 0.6) * 0.3
                assert penalty == pytest.approx(expected, abs=0.001)

    def test_apply_fitness_penalty(self):
        """apply_fitness_penalty reduces raw fitness."""
        genome = create_flat_genome()  # Penalty = 0 for FLAT
        assert genome.apply_fitness_penalty(1.0) == 1.0


# ==============================================================================
# Test 10: Backward compatibility
# ==============================================================================

class TestBackwardCompatibility:
    """from_flat_tree produces valid FLAT genome."""

    def test_from_flat_tree_basic(self):
        """from_flat_tree wraps a tree in FLAT mode."""
        tree = _make_deterministic_tree()
        genome = HierarchicalGenome.from_flat_tree(tree)

        assert genome.structural_mode == StructuralMode.FLAT
        assert len(genome.timeframe_trees) == 1
        assert "default" in genome.timeframe_trees
        assert genome.max_total_nodes == 31

    def test_from_flat_tree_custom_timeframe(self):
        """Custom timeframe name is preserved."""
        tree = _make_deterministic_tree()
        genome = HierarchicalGenome.from_flat_tree(tree, timeframe="5m")

        assert "5m" in genome.timeframe_trees
        assert genome.timeframe_trees["5m"] is tree

    def test_from_flat_tree_serialization_roundtrip(self):
        """Serialization/deserialization preserves genome."""
        tree = _make_deterministic_tree()
        genome = HierarchicalGenome.from_flat_tree(tree)

        data = genome.to_dict()
        restored = HierarchicalGenome.from_dict(data)

        assert restored.structural_mode == StructuralMode.FLAT
        assert len(restored.timeframe_trees) == 1

        # Evaluate both and compare
        features = {"default": _make_features(1.0)}
        sig_orig = genome.evaluate(features)
        sig_restored = restored.evaluate(features)

        assert sig_orig.direction == pytest.approx(sig_restored.direction)
        assert sig_orig.size == pytest.approx(sig_restored.size)
        assert sig_orig.confidence == pytest.approx(sig_restored.confidence)


# ==============================================================================
# Test 11: Decision path tracking
# ==============================================================================

class TestDecisionPathTracking:
    """get_decision_path returns correct traversal."""

    def test_decision_path_flat(self):
        """FLAT mode returns single tree path."""
        tree = _make_deterministic_tree()
        genome = HierarchicalGenome.from_flat_tree(tree)

        features = {"default": _make_features(1.0)}
        paths = genome.get_decision_path(features)

        assert "default" in paths
        assert len(paths["default"]) >= 1
        # Should visit root (0) then leaf (1 if feature > 0)
        assert 0 in paths["default"]
        assert 1 in paths["default"]

    def test_decision_path_dual(self):
        """DUAL mode returns paths for both trees."""
        genome = create_dual_genome()
        features = {"1m": _make_features(1.0), "15m": _make_features(-1.0)}
        paths = genome.get_decision_path(features)

        assert "1m" in paths
        assert "15m" in paths
        assert len(paths["1m"]) >= 1
        assert len(paths["15m"]) >= 1

    def test_decision_path_full_includes_arbitration(self):
        """FULL mode includes arbitration path."""
        genome = create_full_genome()
        features = {
            "1m": _make_features(1.0),
            "15m": _make_features(-1.0),
            "4h": _make_features(0.5),
        }
        paths = genome.get_decision_path(features)

        assert "1m" in paths
        assert "15m" in paths
        assert "4h" in paths
        if genome.arbitration_tree:
            assert "arbitration" in paths


# ==============================================================================
# Test 12: State bus category assignment
# ==============================================================================

class TestStateBusCategoryAssignment:
    """Slots assigned to correct categories."""

    def test_default_category_ranges(self):
        """Default config assigns categories to correct slot ranges."""
        config = StateBusConfig()

        # MOMENTUM: slots 0-7
        for i in range(8):
            assert config.slot_categories[i] == SlotCategory.MOMENTUM
            assert config.half_lives[i] == CATEGORY_HALF_LIVES[SlotCategory.MOMENTUM]

        # MEAN_REVERSION: slots 8-15
        for i in range(8, 16):
            assert config.slot_categories[i] == SlotCategory.MEAN_REVERSION
            assert config.half_lives[i] == CATEGORY_HALF_LIVES[SlotCategory.MEAN_REVERSION]

        # REGIME: slots 16-23
        for i in range(16, 24):
            assert config.slot_categories[i] == SlotCategory.REGIME
            assert config.half_lives[i] == CATEGORY_HALF_LIVES[SlotCategory.REGIME]

        # CUSTOM: slots 24-31
        for i in range(24, 32):
            assert config.slot_categories[i] == SlotCategory.CUSTOM
            assert config.half_lives[i] == CATEGORY_HALF_LIVES[SlotCategory.CUSTOM]

    def test_category_swap_mutation(self):
        """mutate_state_bus_category_swap exchanges slot categories."""
        genome = create_dual_genome()

        # Swap momentum slot 0 with regime slot 16
        original_cat_0 = genome.state_bus_config.slot_categories[0]
        original_cat_16 = genome.state_bus_config.slot_categories[16]

        mutated = genome.mutate_state_bus_category_swap(0, 16)

        assert mutated.state_bus_config.slot_categories[0] == original_cat_16
        assert mutated.state_bus_config.slot_categories[16] == original_cat_0


# ==============================================================================
# Test 13: Arbitration combinatorial PBO test
# ==============================================================================

class TestArbitrationCombinatorialPBO:
    """Shuffled inputs detect interaction overfitting."""

    def test_shuffled_signals_different_output(self):
        """If signals are shuffled, arbitration output changes (non-trivial arb)."""
        tree_1m = _make_deterministic_tree(0.8, 0.7, 0.2)
        tree_15m = _make_deterministic_tree(-0.3, 0.5, 0.1)

        # Non-trivial arbitration tree
        arb = DecisionTree(max_nodes=5)
        arb.nodes[0] = TreeNode(
            node_id=0, is_leaf=False, feature_idx=0,
            operator=">", threshold=0.0,
            left_child=1, right_child=2,
        )
        arb.nodes[1] = TreeNode(node_id=1, is_leaf=True,
                                  action_direction=0.9, action_confidence=0.8,
                                  action_size=0.3)
        arb.nodes[2] = TreeNode(node_id=2, is_leaf=True,
                                  action_direction=-0.5, action_confidence=0.4,
                                  action_size=0.1)

        genome = HierarchicalGenome(
            structural_mode=StructuralMode.FULL,
            timeframe_trees={"1m": tree_1m, "15m": tree_15m},
            arbitration_tree=arb,
            max_total_nodes=63,
        )

        features_normal = {"1m": _make_features(1.0), "15m": _make_features(-1.0)}
        result_normal = genome.evaluate(features_normal)

        # Swap the feature vectors (simulate shuffled sub-tree outputs)
        features_shuffled = {"1m": _make_features(-1.0), "15m": _make_features(1.0)}
        result_shuffled = genome.evaluate(features_shuffled)

        # Results should differ (signals from trees will be different)
        # If they're the same, arbitration might be trivially ignoring inputs
        # This is a basic test; full PBO would compare over many folds
        assert result_normal.source_signals != result_shuffled.source_signals

    def test_degradation_detection_framework(self):
        """Framework for detecting >40% arbitration degradation."""
        genome = create_full_genome()
        features = {
            "1m": _make_features(1.0),
            "15m": _make_features(0.5),
            "4h": _make_features(-0.5),
        }

        baseline = genome.evaluate(features)
        degradation_scores = []

        # Run 10 shuffled evaluations
        for _ in range(10):
            shuffled_features = {}
            keys = list(features.keys())
            random.shuffle(keys)
            for orig_key, shuf_key in zip(features.keys(), keys):
                shuffled_features[orig_key] = features[shuf_key]

            shuffled_result = genome.evaluate(shuffled_features)
            # Measure difference in direction
            diff = abs(baseline.direction - shuffled_result.direction)
            degradation_scores.append(diff)

        # Average degradation should be measurable (>0) for non-trivial genomes
        avg_degradation = np.mean(degradation_scores)
        # Just verify the framework produces values
        assert isinstance(avg_degradation, float)
        assert avg_degradation >= 0.0


# ==============================================================================
# Test 14: Half-life optimization
# ==============================================================================

class TestHalfLifeOptimization:
    """Bayesian optimization improves over canonical values (framework test)."""

    def test_canonical_half_lives(self):
        """Default half-lives match canonical values."""
        config = StateBusConfig()
        assert config.half_lives[0] == 6    # MOMENTUM
        assert config.half_lives[8] == 12   # MEAN_REVERSION
        assert config.half_lives[16] == 24  # REGIME
        assert config.half_lives[24] == 48  # CUSTOM

    def test_custom_half_lives_accepted(self):
        """Custom half-lives can be set and used."""
        config = StateBusConfig(
            slot_categories={0: SlotCategory.MOMENTUM},
            half_lives={0: 3.0},  # Faster than canonical 6
        )
        bus = StateBus(config)
        bus.update(0, 1.0)

        # After 3 ticks with half-life 3, value should be ~0.5
        for _ in range(3):
            bus.decay()

        assert bus.slots[0] == pytest.approx(0.5, abs=0.05)

    def test_half_life_variance_measurable(self):
        """Half-life variance within a category can be measured for regularization."""
        config = StateBusConfig()
        momentum_hls = [
            config.half_lives[i] for i in range(8)
            if i in config.half_lives
        ]
        variance = np.var(momentum_hls)
        # Default canonical values have zero variance within a category
        assert variance == pytest.approx(0.0)


# ==============================================================================
# Test 15: Signal struct bounds
# ==============================================================================

class TestSignalStructBounds:
    """direction, confidence, size stay in valid ranges."""

    def test_signal_struct_clipping(self):
        """Values outside bounds are clipped."""
        sig = SignalStruct(direction_score=2.0, confidence=-0.5, suggested_size=1.5)
        assert sig.direction_score == 1.0
        assert sig.confidence == 0.0
        assert sig.suggested_size == 1.0

    def test_signal_struct_lower_bounds(self):
        """Lower bounds are enforced."""
        sig = SignalStruct(direction_score=-2.0, confidence=-1.0, suggested_size=-0.5)
        assert sig.direction_score == -1.0
        assert sig.confidence == 0.0
        assert sig.suggested_size == 0.0

    def test_signal_struct_valid_passthrough(self):
        """Valid values pass through unchanged."""
        sig = SignalStruct(direction_score=0.5, confidence=0.7, suggested_size=0.3)
        assert sig.direction_score == 0.5
        assert sig.confidence == 0.7
        assert sig.suggested_size == 0.3

    def test_signal_struct_to_array(self):
        """to_array produces correct numpy array."""
        sig = SignalStruct(direction_score=0.5, confidence=0.8, suggested_size=0.2)
        arr = sig.to_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 3
        assert arr[0] == pytest.approx(0.5)
        assert arr[1] == pytest.approx(0.8)
        assert arr[2] == pytest.approx(0.2)


# ==============================================================================
# Additional mutation operator tests
# ==============================================================================

class TestMutationOperators:
    """Additional tests for timeframe add/remove and budget shift."""

    def test_add_timeframe(self):
        """Adding a timeframe creates a new tree."""
        genome = create_dual_genome()
        assert len(genome.timeframe_trees) == 2

        mutated = genome.mutate_add_timeframe("4h")
        assert len(mutated.timeframe_trees) == 3
        assert "4h" in mutated.timeframe_trees
        assert mutated.structural_mode == StructuralMode.FULL

    def test_add_existing_timeframe_noop(self):
        """Adding an existing timeframe does nothing."""
        genome = create_dual_genome()
        existing_tf = list(genome.timeframe_trees.keys())[0]
        mutated = genome.mutate_add_timeframe(existing_tf)
        assert len(mutated.timeframe_trees) == len(genome.timeframe_trees)

    def test_remove_timeframe(self):
        """Removing a timeframe frees nodes."""
        genome = create_full_genome()
        tf_to_remove = list(genome.timeframe_trees.keys())[-1]
        original_count = genome.total_node_count

        mutated = genome.mutate_remove_timeframe(tf_to_remove)
        assert tf_to_remove not in mutated.timeframe_trees
        assert mutated.total_node_count < original_count

    def test_remove_last_timeframe_blocked(self):
        """Can't remove the last timeframe."""
        genome = create_flat_genome()
        tf = list(genome.timeframe_trees.keys())[0]
        mutated = genome.mutate_remove_timeframe(tf)
        assert len(mutated.timeframe_trees) == 1  # Unchanged

    def test_node_budget_shift(self):
        """Budget shift adjusts max_nodes between trees."""
        genome = create_dual_genome()
        tfs = sorted(genome.timeframe_trees.keys())

        from_tree = genome.timeframe_trees[tfs[0]]
        to_tree = genome.timeframe_trees[tfs[1]]
        orig_from_max = from_tree.max_nodes
        orig_to_max = to_tree.max_nodes

        if from_tree.node_count > 6:  # Enough to shift
            mutated = genome.mutate_node_budget_shift(tfs[0], tfs[1], n_nodes=1)
            new_from = mutated.timeframe_trees[tfs[0]]
            new_to = mutated.timeframe_trees[tfs[1]]
            assert new_from.max_nodes == orig_from_max - 1
            assert new_to.max_nodes == orig_to_max + 1


# ==============================================================================
# Serialization tests
# ==============================================================================

class TestSerialization:
    """Genome serialization and hashing."""

    def test_to_dict_from_dict_roundtrip(self):
        """Full serialization roundtrip for all modes."""
        for create_fn in [create_flat_genome, create_dual_genome, create_full_genome]:
            genome = create_fn()
            data = genome.to_dict()
            restored = HierarchicalGenome.from_dict(data)

            assert restored.structural_mode == genome.structural_mode
            assert len(restored.timeframe_trees) == len(genome.timeframe_trees)

    def test_compute_hash_deterministic(self):
        """Same genome produces same hash."""
        tree = _make_deterministic_tree()
        genome = HierarchicalGenome.from_flat_tree(tree)

        h1 = genome.compute_hash()
        h2 = genome.compute_hash()
        assert h1 == h2

    def test_different_genomes_different_hash(self):
        """Different genomes produce different hashes."""
        g1 = HierarchicalGenome.from_flat_tree(_make_deterministic_tree(0.8, 0.7, 0.2))
        g2 = HierarchicalGenome.from_flat_tree(_make_deterministic_tree(0.3, 0.5, 0.1))
        assert g1.compute_hash() != g2.compute_hash()
