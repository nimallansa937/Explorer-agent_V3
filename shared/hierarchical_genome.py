"""
Hierarchical Strategy Graph (HSG)

Replaces flat 31-node binary decision tree with multi-timeframe DAG
of sub-trees connected by a shared state bus. Supports three structural
modes (FLAT/DUAL/FULL) so simple strategies pay zero overhead.

Explorer Prime v2.0 - Phase 1
"""

import math
import random
import hashlib
import json
import uuid
import copy
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import numpy as np


# ==============================================================================
# Core Enums
# ==============================================================================

class StructuralMode(str, Enum):
    """Strategy structural complexity modes."""
    FLAT = "flat"   # Single tree, no state bus, zero overhead (v1.0 compat)
    DUAL = "dual"   # Two timeframe trees, state bus, weighted-average arbitration
    FULL = "full"   # 2+ timeframe trees, state bus, full arbitration tree


class SlotCategory(str, Enum):
    """State bus slot categories with associated decay half-lives."""
    MOMENTUM = "momentum"           # Slots 0-7,  half-life 6 ticks
    MEAN_REVERSION = "mean_reversion"  # Slots 8-15, half-life 12 ticks
    REGIME = "regime"               # Slots 16-23, half-life 24 ticks
    CUSTOM = "custom"               # Slots 24-31, half-life 48 ticks


CATEGORY_HALF_LIVES = {
    SlotCategory.MOMENTUM: 6,
    SlotCategory.MEAN_REVERSION: 12,
    SlotCategory.REGIME: 24,
    SlotCategory.CUSTOM: 48,
}

CATEGORY_SLOT_RANGES = {
    SlotCategory.MOMENTUM: (0, 8),
    SlotCategory.MEAN_REVERSION: (8, 16),
    SlotCategory.REGIME: (16, 24),
    SlotCategory.CUSTOM: (24, 32),
}


# ==============================================================================
# Signal Struct
# ==============================================================================

@dataclass
class SignalStruct:
    """Output from a timeframe sub-tree evaluation."""
    direction_score: float = 0.0    # -1.0 to +1.0
    confidence: float = 0.0         # 0.0 to 1.0
    suggested_size: float = 0.0     # 0.0 to 1.0
    source_timeframe: str = "1m"

    def __post_init__(self):
        self.direction_score = max(-1.0, min(1.0, float(self.direction_score)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.suggested_size = max(0.0, min(1.0, float(self.suggested_size)))

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for arbitration input."""
        return np.array([self.direction_score, self.confidence, self.suggested_size])


@dataclass
class TradeSignal:
    """Final output from HSG evaluation."""
    direction: float = 0.0       # -1.0 (short) to +1.0 (long)
    size: float = 0.0            # 0.0 to 1.0
    confidence: float = 0.0      # 0.0 to 1.0
    source_signals: Dict[str, SignalStruct] = field(default_factory=dict)


# ==============================================================================
# State Bus
# ==============================================================================

@dataclass
class StateBusConfig:
    """Configuration for the shared state bus."""
    n_slots: int = 32
    slot_categories: Dict[int, SlotCategory] = field(default_factory=dict)
    half_lives: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.slot_categories:
            # Default assignment
            for cat, (start, end) in CATEGORY_SLOT_RANGES.items():
                for i in range(start, min(end, self.n_slots)):
                    self.slot_categories[i] = cat
                    self.half_lives[i] = CATEGORY_HALF_LIVES[cat]


class StateBus:
    """Shared state bus with append-and-decay slots."""

    def __init__(self, config: Optional[StateBusConfig] = None):
        self.config = config or StateBusConfig()
        self.slots = np.zeros(self.config.n_slots, dtype=np.float64)
        self._decay_factors = np.array([
            math.exp(-math.log(2) / max(self.config.half_lives.get(i, 12), 1))
            for i in range(self.config.n_slots)
        ])

    def update(self, slot_idx: int, value: float):
        """Write a value to a slot (replaces current value)."""
        if 0 <= slot_idx < self.config.n_slots:
            self.slots[slot_idx] = value

    def decay(self):
        """Apply decay to all slots."""
        self.slots *= self._decay_factors

    def tick(self, updates: Optional[Dict[int, float]] = None):
        """One tick: apply updates then decay all slots."""
        if updates:
            for idx, val in updates.items():
                self.update(idx, val)
        self.decay()

    def get_contents(self) -> np.ndarray:
        """Return copy of current state."""
        return self.slots.copy()

    def reset(self):
        """Reset all slots to zero."""
        self.slots[:] = 0.0


# ==============================================================================
# Decision Tree Node
# ==============================================================================

@dataclass
class TreeNode:
    """A node in a decision tree."""
    node_id: int
    is_leaf: bool = False
    # Internal node fields
    feature_idx: int = 0           # Index into feature vector
    operator: str = ">"            # >, <, >=, <=
    threshold: float = 0.0
    left_child: Optional[int] = None   # Node ID for True branch
    right_child: Optional[int] = None  # Node ID for False branch
    # Leaf node fields
    action_direction: float = 0.0  # -1.0 to 1.0
    action_confidence: float = 0.5
    action_size: float = 0.1

    def evaluate_condition(self, feature_value: float) -> bool:
        """Evaluate this node's condition against a feature value."""
        if self.operator == ">":
            return feature_value > self.threshold
        elif self.operator == "<":
            return feature_value < self.threshold
        elif self.operator == ">=":
            return feature_value >= self.threshold
        elif self.operator == "<=":
            return feature_value <= self.threshold
        return False


class DecisionTree:
    """Binary decision tree for strategy evaluation."""

    def __init__(self, nodes: Optional[Dict[int, TreeNode]] = None,
                 root_id: int = 0, max_nodes: int = 31):
        self.nodes: Dict[int, TreeNode] = nodes or {}
        self.root_id = root_id
        self.max_nodes = max_nodes

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def evaluate(self, features: np.ndarray) -> Tuple[SignalStruct, List[int]]:
        """
        Evaluate the tree against a feature vector.
        Returns (signal, path) where path is list of visited node IDs.
        """
        path = []
        current_id = self.root_id

        if not self.nodes:
            return SignalStruct(), []

        max_depth = 50  # Safety
        for _ in range(max_depth):
            if current_id not in self.nodes:
                break

            node = self.nodes[current_id]
            path.append(current_id)

            if node.is_leaf:
                return SignalStruct(
                    direction_score=node.action_direction,
                    confidence=node.action_confidence,
                    suggested_size=node.action_size,
                ), path

            # Internal node: evaluate condition
            feat_val = 0.0
            if 0 <= node.feature_idx < len(features):
                feat_val = features[node.feature_idx]

            if node.evaluate_condition(feat_val):
                current_id = node.left_child if node.left_child is not None else current_id
            else:
                current_id = node.right_child if node.right_child is not None else current_id

        # Fallback: no signal
        return SignalStruct(), path

    def to_dict(self) -> Dict:
        """Serialize tree to dictionary."""
        return {
            "root_id": self.root_id,
            "max_nodes": self.max_nodes,
            "nodes": {
                str(nid): {
                    "node_id": n.node_id,
                    "is_leaf": n.is_leaf,
                    "feature_idx": n.feature_idx,
                    "operator": n.operator,
                    "threshold": n.threshold,
                    "left_child": n.left_child,
                    "right_child": n.right_child,
                    "action_direction": n.action_direction,
                    "action_confidence": n.action_confidence,
                    "action_size": n.action_size,
                }
                for nid, n in self.nodes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DecisionTree":
        """Deserialize from dictionary."""
        tree = cls(root_id=data.get("root_id", 0),
                   max_nodes=data.get("max_nodes", 31))
        for nid_str, node_data in data.get("nodes", {}).items():
            tree.nodes[int(nid_str)] = TreeNode(**node_data)
        return tree


# ==============================================================================
# Hierarchical Genome
# ==============================================================================

class HierarchicalGenome:
    """
    Multi-timeframe strategy representation using DAG of sub-trees
    connected by a shared state bus.
    """

    def __init__(
        self,
        structural_mode: StructuralMode = StructuralMode.FLAT,
        timeframe_trees: Optional[Dict[str, DecisionTree]] = None,
        state_bus_config: Optional[StateBusConfig] = None,
        arbitration_tree: Optional[DecisionTree] = None,
        max_total_nodes: int = 63,
        schema_version: str = "1.0",
        feature_ids: Optional[List[str]] = None,
        dual_weight: float = 0.5,
    ):
        self.structural_mode = structural_mode
        self.timeframe_trees = timeframe_trees or {}
        self.state_bus_config = state_bus_config or StateBusConfig()
        self.arbitration_tree = arbitration_tree
        self.max_total_nodes = max_total_nodes
        self.schema_version = schema_version
        self.feature_ids = feature_ids or []
        self.dual_weight = max(0.0, min(1.0, dual_weight))  # For DUAL mode
        self._state_bus = StateBus(self.state_bus_config)
        self._genome_id = str(uuid.uuid4())[:12]

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def genome_id(self) -> str:
        return self._genome_id

    @property
    def total_node_count(self) -> int:
        """Total nodes across all trees + arbitration."""
        count = sum(t.node_count for t in self.timeframe_trees.values())
        if self.arbitration_tree:
            count += self.arbitration_tree.node_count
        return count

    @property
    def timeframe_count(self) -> int:
        return len(self.timeframe_trees)

    # --------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate genome constraints. Returns (valid, error_messages)."""
        errors = []

        # Total node budget
        if self.total_node_count > self.max_total_nodes:
            errors.append(
                f"Total nodes {self.total_node_count} exceeds max {self.max_total_nodes}"
            )

        # Mode-specific constraints
        if self.structural_mode == StructuralMode.FLAT:
            if len(self.timeframe_trees) != 1:
                errors.append(f"FLAT mode requires exactly 1 tree, got {len(self.timeframe_trees)}")
            for tf, tree in self.timeframe_trees.items():
                if tree.node_count > 31:
                    errors.append(f"FLAT mode tree '{tf}' has {tree.node_count} nodes, max 31")

        elif self.structural_mode == StructuralMode.DUAL:
            if len(self.timeframe_trees) != 2:
                errors.append(f"DUAL mode requires exactly 2 trees, got {len(self.timeframe_trees)}")

        elif self.structural_mode == StructuralMode.FULL:
            if len(self.timeframe_trees) < 2 or len(self.timeframe_trees) > 4:
                errors.append(f"FULL mode requires 2-4 trees, got {len(self.timeframe_trees)}")
            if self.arbitration_tree and self.arbitration_tree.node_count > 7:
                errors.append(
                    f"Arbitration tree has {self.arbitration_tree.node_count} nodes, max 7"
                )

        # Per-tree minimum constraints (skip for FLAT)
        if self.structural_mode != StructuralMode.FLAT and self.total_node_count > 0:
            for tf, tree in self.timeframe_trees.items():
                if tree.node_count < 5:
                    errors.append(f"Tree '{tf}' has {tree.node_count} nodes, minimum 5")
                tree_share = tree.node_count / max(self.total_node_count, 1)
                if tree_share < 0.15:
                    errors.append(
                        f"Tree '{tf}' has {tree_share:.1%} of budget, minimum 15%"
                    )

        return len(errors) == 0, errors

    # --------------------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------------------

    def evaluate(self, feature_vectors: Dict[str, np.ndarray]) -> TradeSignal:
        """
        Evaluate the HSG against timeframe-keyed feature vectors.

        Args:
            feature_vectors: Dict mapping timeframe -> feature array
                e.g., {"1m": array(60,), "15m": array(60,), "4h": array(60,)}
                For FLAT mode, use {"default": array(60,)}

        Returns:
            TradeSignal with final direction, size, confidence
        """
        signals: Dict[str, SignalStruct] = {}

        # Evaluate each timeframe tree
        for tf, tree in self.timeframe_trees.items():
            features = feature_vectors.get(tf, feature_vectors.get("default", np.zeros(60)))
            signal, _ = tree.evaluate(features)
            signal.source_timeframe = tf
            signals[tf] = signal

            # Update state bus with signal info
            self._update_state_bus_from_signal(tf, signal)

        # Apply state bus decay
        self._state_bus.decay()

        # Arbitration
        if self.structural_mode == StructuralMode.FLAT:
            # Direct passthrough
            if signals:
                sig = list(signals.values())[0]
                return TradeSignal(
                    direction=sig.direction_score,
                    size=sig.suggested_size,
                    confidence=sig.confidence,
                    source_signals=signals,
                )
            return TradeSignal(source_signals=signals)

        elif self.structural_mode == StructuralMode.DUAL:
            # Weighted average
            return self._dual_arbitration(signals)

        elif self.structural_mode == StructuralMode.FULL:
            # Full arbitration tree
            return self._full_arbitration(signals)

        return TradeSignal(source_signals=signals)

    def get_decision_path(self, feature_vectors: Dict[str, np.ndarray]) -> Dict[str, List[int]]:
        """
        Get traversal path through all trees for dependency analysis.
        Returns dict mapping timeframe -> list of visited node IDs.
        """
        paths = {}
        for tf, tree in self.timeframe_trees.items():
            features = feature_vectors.get(tf, feature_vectors.get("default", np.zeros(60)))
            _, path = tree.evaluate(features)
            paths[tf] = path
        if self.arbitration_tree:
            arb_input = self._pack_arbitration_input({})
            _, arb_path = self.arbitration_tree.evaluate(arb_input)
            paths["arbitration"] = arb_path
        return paths

    def _update_state_bus_from_signal(self, timeframe: str, signal: SignalStruct):
        """Write signal components to appropriate state bus slots."""
        tf_idx = list(self.timeframe_trees.keys()).index(timeframe) if timeframe in self.timeframe_trees else 0
        base_slot = tf_idx * 3  # 3 slots per timeframe signal

        if base_slot < self.state_bus_config.n_slots:
            self._state_bus.update(base_slot, signal.direction_score)
        if base_slot + 1 < self.state_bus_config.n_slots:
            self._state_bus.update(base_slot + 1, signal.confidence)
        if base_slot + 2 < self.state_bus_config.n_slots:
            self._state_bus.update(base_slot + 2, signal.suggested_size)

    def _dual_arbitration(self, signals: Dict[str, SignalStruct]) -> TradeSignal:
        """Weighted average of two signals."""
        sigs = list(signals.values())
        if len(sigs) < 2:
            if sigs:
                s = sigs[0]
                return TradeSignal(direction=s.direction_score, size=s.suggested_size,
                                   confidence=s.confidence, source_signals=signals)
            return TradeSignal(source_signals=signals)

        w = self.dual_weight
        s1, s2 = sigs[0], sigs[1]

        direction = w * s1.direction_score + (1 - w) * s2.direction_score
        size = w * s1.suggested_size + (1 - w) * s2.suggested_size
        confidence = w * s1.confidence + (1 - w) * s2.confidence

        return TradeSignal(
            direction=max(-1.0, min(1.0, direction)),
            size=max(0.0, min(1.0, size)),
            confidence=max(0.0, min(1.0, confidence)),
            source_signals=signals,
        )

    def _full_arbitration(self, signals: Dict[str, SignalStruct]) -> TradeSignal:
        """Evaluate arbitration tree on packed input."""
        arb_input = self._pack_arbitration_input(signals)

        if self.arbitration_tree:
            arb_signal, _ = self.arbitration_tree.evaluate(arb_input)
            return TradeSignal(
                direction=arb_signal.direction_score,
                size=arb_signal.suggested_size,
                confidence=arb_signal.confidence,
                source_signals=signals,
            )
        # Fallback: average signals
        return self._dual_arbitration(signals)

    def _pack_arbitration_input(self, signals: Dict[str, SignalStruct]) -> np.ndarray:
        """Pack all signal structs + state bus contents into arbitration input."""
        parts = []
        # Signal arrays (3 values each)
        for tf in sorted(self.timeframe_trees.keys()):
            if tf in signals:
                parts.append(signals[tf].to_array())
            else:
                parts.append(np.zeros(3))

        # State bus contents
        parts.append(self._state_bus.get_contents())

        # Pad to at least 60 dimensions for tree compatibility
        packed = np.concatenate(parts) if parts else np.zeros(60)
        if len(packed) < 60:
            packed = np.pad(packed, (0, 60 - len(packed)))

        return packed

    # --------------------------------------------------------------------------
    # Fitness Penalty for Node Concentration
    # --------------------------------------------------------------------------

    def concentration_penalty(self) -> float:
        """
        Penalty for over-concentrating nodes in one tree.
        Penalty = max(0, dominant_share - 0.6) * 0.3
        Applied as: fitness *= (1.0 - penalty)
        """
        if self.structural_mode == StructuralMode.FLAT or not self.timeframe_trees:
            return 0.0

        total = self.total_node_count
        if total == 0:
            return 0.0

        max_tree_nodes = max(t.node_count for t in self.timeframe_trees.values())
        dominant_share = max_tree_nodes / total
        return max(0.0, dominant_share - 0.6) * 0.3

    def apply_fitness_penalty(self, raw_fitness: float) -> float:
        """Apply concentration penalty to raw fitness."""
        penalty = self.concentration_penalty()
        return raw_fitness * (1.0 - penalty)

    # --------------------------------------------------------------------------
    # Mutation Operators
    # --------------------------------------------------------------------------

    def mutate_structural_mode(self) -> "HierarchicalGenome":
        """Transition between FLAT/DUAL/FULL structural modes."""
        new = copy.deepcopy(self)

        if new.structural_mode == StructuralMode.FLAT:
            # Promote to DUAL: split existing tree, add second
            new.structural_mode = StructuralMode.DUAL
            existing_tf = list(new.timeframe_trees.keys())[0]
            existing_tree = new.timeframe_trees[existing_tf]

            # Create second tree as partial copy with mutations
            second_tree = copy.deepcopy(existing_tree)
            second_tf = "15m" if existing_tf != "15m" else "4h"
            new.timeframe_trees[second_tf] = second_tree
            new.dual_weight = 0.5

        elif new.structural_mode == StructuralMode.DUAL:
            # Can go to FLAT (simplify) or FULL (complexify)
            if random.random() < 0.5:
                new.structural_mode = StructuralMode.FULL
                # Add small arbitration tree
                new.arbitration_tree = _create_simple_tree(max_nodes=5)
            else:
                new.structural_mode = StructuralMode.FLAT
                # Keep only the first tree
                first_tf = sorted(new.timeframe_trees.keys())[0]
                new.timeframe_trees = {first_tf: new.timeframe_trees[first_tf]}
                new.arbitration_tree = None

        elif new.structural_mode == StructuralMode.FULL:
            # Simplify to DUAL
            new.structural_mode = StructuralMode.DUAL
            tfs = sorted(new.timeframe_trees.keys())[:2]
            new.timeframe_trees = {tf: new.timeframe_trees[tf] for tf in tfs}
            new.arbitration_tree = None
            new.dual_weight = 0.5

        return new

    def mutate_add_timeframe(self, timeframe: str) -> "HierarchicalGenome":
        """Add a new timeframe tree (splits node budget)."""
        new = copy.deepcopy(self)
        if timeframe in new.timeframe_trees:
            return new
        if new.structural_mode == StructuralMode.FLAT:
            return new  # Can't add timeframes in FLAT

        # Create small tree from budget
        available = new.max_total_nodes - new.total_node_count
        n_nodes = max(5, min(available, 9))
        new.timeframe_trees[timeframe] = _create_simple_tree(max_nodes=n_nodes)

        if len(new.timeframe_trees) > 2:
            new.structural_mode = StructuralMode.FULL
            if not new.arbitration_tree:
                new.arbitration_tree = _create_simple_tree(max_nodes=5)

        return new

    def mutate_remove_timeframe(self, timeframe: str) -> "HierarchicalGenome":
        """Remove a timeframe tree (frees node budget)."""
        new = copy.deepcopy(self)
        if timeframe not in new.timeframe_trees or len(new.timeframe_trees) <= 1:
            return new

        del new.timeframe_trees[timeframe]

        if len(new.timeframe_trees) == 1:
            new.structural_mode = StructuralMode.FLAT
            new.arbitration_tree = None
        elif len(new.timeframe_trees) == 2:
            new.structural_mode = StructuralMode.DUAL
            new.arbitration_tree = None
            new.dual_weight = 0.5

        return new

    def mutate_node_budget_shift(self, from_tf: str, to_tf: str, n_nodes: int = 1) -> "HierarchicalGenome":
        """Move nodes between trees (respecting minimums)."""
        new = copy.deepcopy(self)
        if from_tf not in new.timeframe_trees or to_tf not in new.timeframe_trees:
            return new

        from_tree = new.timeframe_trees[from_tf]
        to_tree = new.timeframe_trees[to_tf]

        # Check minimum constraint on source
        if from_tree.node_count - n_nodes < 5:
            return new

        # Just adjust max_nodes (actual node rebalancing happens during evolution)
        from_tree.max_nodes = max(5, from_tree.max_nodes - n_nodes)
        to_tree.max_nodes = to_tree.max_nodes + n_nodes

        return new

    def mutate_state_bus_category_swap(self, slot_a: int, slot_b: int) -> "HierarchicalGenome":
        """Reassign slot categories between two slots."""
        new = copy.deepcopy(self)
        cfg = new.state_bus_config

        if slot_a in cfg.slot_categories and slot_b in cfg.slot_categories:
            cfg.slot_categories[slot_a], cfg.slot_categories[slot_b] = \
                cfg.slot_categories[slot_b], cfg.slot_categories[slot_a]
            cfg.half_lives[slot_a], cfg.half_lives[slot_b] = \
                cfg.half_lives[slot_b], cfg.half_lives[slot_a]

        # Rebuild state bus with new config
        new._state_bus = StateBus(cfg)
        return new

    # --------------------------------------------------------------------------
    # Crossover
    # --------------------------------------------------------------------------

    @staticmethod
    def crossover(parent_a: "HierarchicalGenome",
                  parent_b: "HierarchicalGenome") -> "HierarchicalGenome":
        """
        Cross two HSG genomes.
        - Exchange sub-trees at same timeframe level
        - Offspring inherits the more complex structural mode
        - State bus configs merge by averaging decay rates
        """
        # Determine offspring mode (more complex parent wins)
        mode_order = {StructuralMode.FLAT: 0, StructuralMode.DUAL: 1, StructuralMode.FULL: 2}
        if mode_order[parent_a.structural_mode] >= mode_order[parent_b.structural_mode]:
            base, donor = parent_a, parent_b
        else:
            base, donor = parent_b, parent_a

        child = copy.deepcopy(base)

        # Exchange matching timeframe trees
        shared_tfs = set(base.timeframe_trees.keys()) & set(donor.timeframe_trees.keys())
        for tf in shared_tfs:
            if random.random() < 0.5:
                child.timeframe_trees[tf] = copy.deepcopy(donor.timeframe_trees[tf])

        # Merge state bus configs by averaging half-lives
        for slot in range(child.state_bus_config.n_slots):
            hl_a = parent_a.state_bus_config.half_lives.get(slot, 12)
            hl_b = parent_b.state_bus_config.half_lives.get(slot, 12)
            child.state_bus_config.half_lives[slot] = (hl_a + hl_b) / 2.0

        # Average dual_weight
        child.dual_weight = (parent_a.dual_weight + parent_b.dual_weight) / 2.0

        # Rebuild state bus
        child._state_bus = StateBus(child.state_bus_config)
        child._genome_id = str(uuid.uuid4())[:12]

        return child

    # --------------------------------------------------------------------------
    # Backward Compatibility
    # --------------------------------------------------------------------------

    @classmethod
    def from_flat_tree(cls, tree: DecisionTree,
                       timeframe: str = "default",
                       schema_version: str = "1.0") -> "HierarchicalGenome":
        """
        Convert a v1.0 flat decision tree to a FLAT-mode HierarchicalGenome.
        Produces a zero-overhead wrapper - identical evaluation to v1.0.
        """
        return cls(
            structural_mode=StructuralMode.FLAT,
            timeframe_trees={timeframe: tree},
            max_total_nodes=31,
            schema_version=schema_version,
        )

    # --------------------------------------------------------------------------
    # Serialization
    # --------------------------------------------------------------------------

    def compute_hash(self) -> str:
        """Compute deterministic hash of the genome."""
        data = {
            "mode": self.structural_mode.value,
            "trees": {tf: t.to_dict() for tf, t in sorted(self.timeframe_trees.items())},
            "schema_version": self.schema_version,
            "dual_weight": self.dual_weight,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "genome_id": self._genome_id,
            "structural_mode": self.structural_mode.value,
            "timeframe_trees": {
                tf: tree.to_dict() for tf, tree in self.timeframe_trees.items()
            },
            "state_bus_config": {
                "n_slots": self.state_bus_config.n_slots,
                "slot_categories": {
                    str(k): v.value for k, v in self.state_bus_config.slot_categories.items()
                },
                "half_lives": {str(k): v for k, v in self.state_bus_config.half_lives.items()},
            },
            "arbitration_tree": self.arbitration_tree.to_dict() if self.arbitration_tree else None,
            "max_total_nodes": self.max_total_nodes,
            "schema_version": self.schema_version,
            "feature_ids": self.feature_ids,
            "dual_weight": self.dual_weight,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "HierarchicalGenome":
        """Deserialize from dictionary."""
        # Parse state bus config
        sbc_data = data.get("state_bus_config", {})
        sbc = StateBusConfig(
            n_slots=sbc_data.get("n_slots", 32),
            slot_categories={
                int(k): SlotCategory(v)
                for k, v in sbc_data.get("slot_categories", {}).items()
            },
            half_lives={
                int(k): float(v)
                for k, v in sbc_data.get("half_lives", {}).items()
            },
        )

        # Parse trees
        trees = {
            tf: DecisionTree.from_dict(td)
            for tf, td in data.get("timeframe_trees", {}).items()
        }

        # Parse arbitration
        arb = None
        if data.get("arbitration_tree"):
            arb = DecisionTree.from_dict(data["arbitration_tree"])

        genome = cls(
            structural_mode=StructuralMode(data.get("structural_mode", "flat")),
            timeframe_trees=trees,
            state_bus_config=sbc,
            arbitration_tree=arb,
            max_total_nodes=data.get("max_total_nodes", 63),
            schema_version=data.get("schema_version", "1.0"),
            feature_ids=data.get("feature_ids", []),
            dual_weight=data.get("dual_weight", 0.5),
        )
        genome._genome_id = data.get("genome_id", genome._genome_id)
        return genome


# ==============================================================================
# Helper Functions
# ==============================================================================

def _create_simple_tree(max_nodes: int = 7, n_features: int = 60) -> DecisionTree:
    """Create a simple random decision tree for initialization."""
    tree = DecisionTree(max_nodes=max_nodes)
    n_internal = max(1, max_nodes // 2)
    n_leaves = n_internal + 1

    node_id = 0

    # Root internal node
    for i in range(n_internal):
        tree.nodes[node_id] = TreeNode(
            node_id=node_id,
            is_leaf=False,
            feature_idx=random.randint(0, n_features - 1),
            operator=random.choice([">", "<", ">=", "<="]),
            threshold=random.uniform(-1.0, 1.0),
            left_child=node_id * 2 + 1 if node_id * 2 + 1 < max_nodes else None,
            right_child=node_id * 2 + 2 if node_id * 2 + 2 < max_nodes else None,
        )
        node_id += 1

    # Leaf nodes
    for i in range(n_leaves):
        if node_id >= max_nodes:
            break
        tree.nodes[node_id] = TreeNode(
            node_id=node_id,
            is_leaf=True,
            action_direction=random.uniform(-1.0, 1.0),
            action_confidence=random.uniform(0.3, 0.9),
            action_size=random.uniform(0.05, 0.3),
        )
        node_id += 1

    return tree


def create_flat_genome(n_features: int = 60, max_nodes: int = 31) -> HierarchicalGenome:
    """Convenience: create a random FLAT mode genome."""
    tree = _create_simple_tree(max_nodes=max_nodes, n_features=n_features)
    return HierarchicalGenome.from_flat_tree(tree)


def create_dual_genome(n_features: int = 60, max_nodes: int = 63) -> HierarchicalGenome:
    """Convenience: create a random DUAL mode genome."""
    half = max(5, max_nodes // 3)
    return HierarchicalGenome(
        structural_mode=StructuralMode.DUAL,
        timeframe_trees={
            "1m": _create_simple_tree(max_nodes=half, n_features=n_features),
            "15m": _create_simple_tree(max_nodes=half, n_features=n_features),
        },
        max_total_nodes=max_nodes,
        dual_weight=0.5,
    )


def create_full_genome(n_features: int = 60, max_nodes: int = 63) -> HierarchicalGenome:
    """Convenience: create a random FULL mode genome."""
    third = max(5, max_nodes // 4)
    arb = _create_simple_tree(max_nodes=5, n_features=60)
    return HierarchicalGenome(
        structural_mode=StructuralMode.FULL,
        timeframe_trees={
            "1m": _create_simple_tree(max_nodes=third, n_features=n_features),
            "15m": _create_simple_tree(max_nodes=third, n_features=n_features),
            "4h": _create_simple_tree(max_nodes=third, n_features=n_features),
        },
        arbitration_tree=arb,
        max_total_nodes=max_nodes,
    )
