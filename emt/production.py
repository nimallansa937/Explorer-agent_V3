"""
EMT Production Storage

Manages production-ready strategies with:
- Merkle tree versioning for audit trails
- Strategy persistence and retrieval
- Production lifecycle management
- Retirement handling
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from pathlib import Path
import pickle

from shared.unified_strategy import (
    UnifiedStrategy,
    StrategyStatus,
)
from shared.constants import (
    MAX_PRODUCTION_STRATEGIES,
    STRATEGY_FILE_EXTENSION,
    METADATA_FILE_EXTENSION,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class ProductionConfig:
    """Configuration for EMT production storage."""

    # Storage paths
    storage_dir: str = "emt_data/production"
    archive_dir: str = "emt_data/archive"

    # Limits
    max_active_strategies: int = MAX_PRODUCTION_STRATEGIES

    # Merkle tree
    enable_merkle: bool = True
    merkle_hash_algo: str = "sha256"

    # Auto-retirement
    auto_retire_days: int = 90  # Retire after 90 days of poor performance
    min_sharpe_for_retention: float = 0.5

    # Persistence
    auto_save: bool = True
    save_interval_minutes: int = 60


# ==============================================================================
# Storage Result
# ==============================================================================

@dataclass
class StorageResult:
    """Result of a storage operation."""
    success: bool
    strategy_id: str
    merkle_root: Optional[str] = None
    version: int = 1
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# ==============================================================================
# Production Strategy Wrapper
# ==============================================================================

@dataclass
class ProductionStrategy:
    """
    Wrapper for strategies in production.

    Tracks production-specific metadata and performance.
    """
    strategy: UnifiedStrategy
    version: int = 1

    # Production timestamps
    deployed_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    retired_at: Optional[datetime] = None

    # Production metrics
    production_trades: int = 0
    production_pnl: float = 0.0
    production_sharpe: Optional[float] = None
    production_max_dd: float = 0.0

    # Merkle tracking
    merkle_hash: Optional[str] = None
    parent_hash: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if strategy is active in production."""
        return self.retired_at is None

    @property
    def production_days(self) -> int:
        """Days in production."""
        end = self.retired_at or datetime.now()
        return (end - self.deployed_at).days

    def compute_hash(self, algo: str = "sha256") -> str:
        """Compute hash for Merkle tree."""
        # Get genome hash - use compute_hash method or genome_hash property from strategy
        genome_hash = self.strategy.genome_hash or self.strategy.genome.compute_hash()

        data = {
            "strategy_id": self.strategy.strategy_id,
            "version": self.version,
            "genome_hash": genome_hash,
            "deployed_at": self.deployed_at.isoformat(),
            "parent_hash": self.parent_hash,
        }
        content = json.dumps(data, sort_keys=True).encode()

        if algo == "sha256":
            return hashlib.sha256(content).hexdigest()
        elif algo == "sha3_256":
            return hashlib.sha3_256(content).hexdigest()
        else:
            return hashlib.sha256(content).hexdigest()


# ==============================================================================
# Merkle Tree Node
# ==============================================================================

@dataclass
class MerkleNode:
    """Node in the Merkle tree for audit trail."""
    hash: str
    left: Optional["MerkleNode"] = None
    right: Optional["MerkleNode"] = None
    strategy_id: Optional[str] = None  # Only leaf nodes have strategy IDs

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


# ==============================================================================
# EMT Production Storage
# ==============================================================================

class EMTProduction:
    """
    EMT (Evolutionary Merkle Tree) Production Storage.

    Manages production-ready strategies with Merkle tree versioning
    for complete audit trails.

    Features:
    - Strategy storage and retrieval
    - Merkle tree versioning
    - Production lifecycle management
    - Performance tracking
    - Auto-retirement

    Usage:
        emt = EMTProduction()

        # Store strategy
        result = emt.store(strategy)

        # Retrieve
        prod_strategy = emt.get("strategy_001")

        # Retire
        emt.retire("strategy_001", reason="Poor performance")

        # Get Merkle root for audit
        root = emt.get_merkle_root()
    """

    def __init__(self, config: Optional[ProductionConfig] = None):
        """
        Initialize EMT production storage.

        Args:
            config: Production configuration
        """
        self.config = config or ProductionConfig()

        # Active strategies
        self._strategies: Dict[str, ProductionStrategy] = {}

        # Retired strategies (archive)
        self._archive: Dict[str, ProductionStrategy] = {}

        # Merkle tree
        self._merkle_root: Optional[MerkleNode] = None
        self._merkle_dirty: bool = False  # Needs rebuild

        # Version tracking
        self._global_version: int = 0
        self._version_history: List[Dict[str, Any]] = []

        # Setup storage directories
        self._setup_storage()

        logger.info(
            f"EMTProduction initialized: max_strategies={self.config.max_active_strategies}"
        )

    def _setup_storage(self):
        """Setup storage directories."""
        Path(self.config.storage_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.archive_dir).mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Core Operations
    # ==========================================================================

    def store(self, strategy: UnifiedStrategy) -> StorageResult:
        """
        Store a strategy in production.

        Args:
            strategy: UnifiedStrategy to store

        Returns:
            StorageResult with success status and merkle root
        """
        strategy_id = strategy.strategy_id

        # Check capacity
        if len(self._strategies) >= self.config.max_active_strategies:
            # Try to retire lowest performer
            retired = self._auto_retire_lowest()
            if not retired:
                return StorageResult(
                    success=False,
                    strategy_id=strategy_id,
                    message=f"At capacity ({self.config.max_active_strategies}), "
                            "no strategies eligible for auto-retirement",
                )

        # Check for duplicates
        if strategy_id in self._strategies:
            return self._update_existing(strategy)

        # Create production wrapper
        prod_strategy = ProductionStrategy(
            strategy=strategy,
            version=1,
            deployed_at=datetime.now(),
            last_updated=datetime.now(),
            parent_hash=self._get_latest_hash(),
        )

        # Compute hash
        prod_strategy.merkle_hash = prod_strategy.compute_hash(
            self.config.merkle_hash_algo
        )

        # Store
        self._strategies[strategy_id] = prod_strategy
        self._merkle_dirty = True
        self._global_version += 1

        # Update strategy status
        strategy.status = StrategyStatus.PRODUCTION
        strategy.production_start = datetime.now()

        # Persist if enabled
        if self.config.auto_save:
            self._persist_strategy(prod_strategy)

        # Rebuild Merkle tree
        if self.config.enable_merkle:
            self._rebuild_merkle_tree()

        logger.info(f"Stored strategy {strategy_id} in production (v{self._global_version})")

        return StorageResult(
            success=True,
            strategy_id=strategy_id,
            merkle_root=self.get_merkle_root(),
            version=prod_strategy.version,
            message="Strategy stored successfully",
        )

    def _update_existing(self, strategy: UnifiedStrategy) -> StorageResult:
        """Update an existing strategy (new version)."""
        strategy_id = strategy.strategy_id
        existing = self._strategies[strategy_id]

        # Create new version
        new_version = existing.version + 1
        parent_hash = existing.merkle_hash

        # Update wrapper
        existing.strategy = strategy
        existing.version = new_version
        existing.last_updated = datetime.now()
        existing.parent_hash = parent_hash
        existing.merkle_hash = existing.compute_hash(self.config.merkle_hash_algo)

        self._merkle_dirty = True
        self._global_version += 1

        # Persist
        if self.config.auto_save:
            self._persist_strategy(existing)

        # Rebuild Merkle tree
        if self.config.enable_merkle:
            self._rebuild_merkle_tree()

        logger.info(f"Updated strategy {strategy_id} to v{new_version}")

        return StorageResult(
            success=True,
            strategy_id=strategy_id,
            merkle_root=self.get_merkle_root(),
            version=new_version,
            message=f"Strategy updated to version {new_version}",
        )

    def get(self, strategy_id: str) -> Optional[ProductionStrategy]:
        """
        Get a production strategy by ID.

        Args:
            strategy_id: Strategy identifier

        Returns:
            ProductionStrategy if found, None otherwise
        """
        return self._strategies.get(strategy_id)

    def get_strategy(self, strategy_id: str) -> Optional[UnifiedStrategy]:
        """
        Get the underlying UnifiedStrategy by ID.

        Args:
            strategy_id: Strategy identifier

        Returns:
            UnifiedStrategy if found, None otherwise
        """
        prod = self._strategies.get(strategy_id)
        return prod.strategy if prod else None

    def list_active(self) -> List[ProductionStrategy]:
        """
        List all active production strategies.

        Returns:
            List of active ProductionStrategy objects
        """
        return list(self._strategies.values())

    def list_strategy_ids(self) -> List[str]:
        """
        List all active strategy IDs.

        Returns:
            List of strategy IDs
        """
        return list(self._strategies.keys())

    def contains(self, strategy_id: str) -> bool:
        """Check if strategy exists in production."""
        return strategy_id in self._strategies

    # ==========================================================================
    # Retirement
    # ==========================================================================

    def retire(
        self,
        strategy_id: str,
        reason: str = "Manual retirement",
    ) -> bool:
        """
        Retire a strategy from production.

        Args:
            strategy_id: Strategy to retire
            reason: Retirement reason

        Returns:
            True if retired successfully
        """
        if strategy_id not in self._strategies:
            logger.warning(f"Strategy {strategy_id} not found for retirement")
            return False

        prod_strategy = self._strategies.pop(strategy_id)
        prod_strategy.retired_at = datetime.now()
        prod_strategy.strategy.status = StrategyStatus.RETIRED
        prod_strategy.strategy.retirement_reason = reason
        prod_strategy.strategy.production_end = datetime.now()

        # Move to archive
        self._archive[strategy_id] = prod_strategy

        # Persist to archive
        self._archive_strategy(prod_strategy)

        self._merkle_dirty = True
        self._global_version += 1

        # Rebuild Merkle tree
        if self.config.enable_merkle:
            self._rebuild_merkle_tree()

        logger.info(f"Retired strategy {strategy_id}: {reason}")
        return True

    def _auto_retire_lowest(self) -> bool:
        """Auto-retire the lowest performing strategy."""
        if not self._strategies:
            return False

        # Find lowest Sharpe
        lowest = min(
            self._strategies.values(),
            key=lambda s: s.production_sharpe or float('inf'),
        )

        # Only retire if below threshold
        if (
            lowest.production_sharpe is not None
            and lowest.production_sharpe < self.config.min_sharpe_for_retention
        ):
            self.retire(
                lowest.strategy.strategy_id,
                reason=f"Auto-retired: Sharpe {lowest.production_sharpe:.2f} "
                       f"< {self.config.min_sharpe_for_retention}",
            )
            return True

        return False

    # ==========================================================================
    # Performance Tracking
    # ==========================================================================

    def record_trade(
        self,
        strategy_id: str,
        pnl: float,
        trade_time: Optional[datetime] = None,
    ) -> bool:
        """
        Record a trade for a production strategy.

        Args:
            strategy_id: Strategy that made the trade
            pnl: Profit/loss from the trade
            trade_time: When the trade occurred

        Returns:
            True if recorded successfully
        """
        prod_strategy = self._strategies.get(strategy_id)
        if prod_strategy is None:
            return False

        prod_strategy.production_trades += 1
        prod_strategy.production_pnl += pnl
        prod_strategy.last_updated = trade_time or datetime.now()

        # Track drawdown
        if pnl < 0:
            # Simplified max drawdown tracking
            # Real implementation would track equity curve
            pass

        return True

    def update_metrics(
        self,
        strategy_id: str,
        sharpe: Optional[float] = None,
        max_dd: Optional[float] = None,
        **metrics,
    ) -> bool:
        """
        Update production metrics for a strategy.

        Args:
            strategy_id: Strategy to update
            sharpe: Production Sharpe ratio
            max_dd: Maximum drawdown
            **metrics: Additional metrics for metadata

        Returns:
            True if updated successfully
        """
        prod_strategy = self._strategies.get(strategy_id)
        if prod_strategy is None:
            return False

        if sharpe is not None:
            prod_strategy.production_sharpe = sharpe

        if max_dd is not None:
            prod_strategy.production_max_dd = max_dd

        prod_strategy.metadata.update(metrics)
        prod_strategy.last_updated = datetime.now()

        return True

    # ==========================================================================
    # Merkle Tree
    # ==========================================================================

    def get_merkle_root(self) -> Optional[str]:
        """
        Get the current Merkle root hash.

        Returns:
            Root hash string or None if empty
        """
        if not self.config.enable_merkle:
            return None

        if self._merkle_dirty:
            self._rebuild_merkle_tree()

        return self._merkle_root.hash if self._merkle_root else None

    def _rebuild_merkle_tree(self):
        """Rebuild the Merkle tree from current strategies."""
        if not self._strategies:
            self._merkle_root = None
            self._merkle_dirty = False
            return

        # Create leaf nodes
        leaves = [
            MerkleNode(
                hash=prod.merkle_hash,
                strategy_id=prod.strategy.strategy_id,
            )
            for prod in sorted(
                self._strategies.values(),
                key=lambda s: s.strategy.strategy_id,
            )
        ]

        # Build tree bottom-up
        self._merkle_root = self._build_merkle_level(leaves)
        self._merkle_dirty = False

        # Record version
        self._version_history.append({
            "version": self._global_version,
            "merkle_root": self._merkle_root.hash,
            "timestamp": datetime.now().isoformat(),
            "strategy_count": len(self._strategies),
        })

    def _build_merkle_level(self, nodes: List[MerkleNode]) -> MerkleNode:
        """Build one level of the Merkle tree."""
        if len(nodes) == 1:
            return nodes[0]

        # Pair nodes and create parent level
        parents = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i + 1] if i + 1 < len(nodes) else left  # Duplicate if odd

            # Compute parent hash
            combined = (left.hash + right.hash).encode()
            parent_hash = hashlib.sha256(combined).hexdigest()

            parents.append(MerkleNode(
                hash=parent_hash,
                left=left,
                right=right,
            ))

        return self._build_merkle_level(parents)

    def verify_strategy(self, strategy_id: str) -> bool:
        """
        Verify a strategy's integrity using Merkle proof.

        Args:
            strategy_id: Strategy to verify

        Returns:
            True if integrity verified
        """
        prod_strategy = self._strategies.get(strategy_id)
        if prod_strategy is None:
            return False

        # Recompute hash
        computed = prod_strategy.compute_hash(self.config.merkle_hash_algo)

        # Compare with stored hash
        return computed == prod_strategy.merkle_hash

    def _get_latest_hash(self) -> Optional[str]:
        """Get hash of the most recently added strategy."""
        if not self._strategies:
            return None

        latest = max(
            self._strategies.values(),
            key=lambda s: s.deployed_at,
        )
        return latest.merkle_hash

    # ==========================================================================
    # Persistence
    # ==========================================================================

    def _persist_strategy(self, prod_strategy: ProductionStrategy):
        """Persist a strategy to disk."""
        strategy_id = prod_strategy.strategy.strategy_id
        filepath = Path(self.config.storage_dir) / f"{strategy_id}{STRATEGY_FILE_EXTENSION}"

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(prod_strategy, f)
        except Exception as e:
            logger.error(f"Failed to persist strategy {strategy_id}: {e}")

    def _archive_strategy(self, prod_strategy: ProductionStrategy):
        """Archive a retired strategy."""
        strategy_id = prod_strategy.strategy.strategy_id
        filepath = Path(self.config.archive_dir) / f"{strategy_id}{STRATEGY_FILE_EXTENSION}"

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(prod_strategy, f)

            # Remove from active storage
            active_path = Path(self.config.storage_dir) / f"{strategy_id}{STRATEGY_FILE_EXTENSION}"
            if active_path.exists():
                active_path.unlink()

        except Exception as e:
            logger.error(f"Failed to archive strategy {strategy_id}: {e}")

    def load_from_disk(self):
        """Load all strategies from disk."""
        storage_path = Path(self.config.storage_dir)

        if not storage_path.exists():
            return

        for filepath in storage_path.glob(f"*{STRATEGY_FILE_EXTENSION}"):
            try:
                with open(filepath, 'rb') as f:
                    prod_strategy = pickle.load(f)
                    self._strategies[prod_strategy.strategy.strategy_id] = prod_strategy
            except Exception as e:
                logger.error(f"Failed to load strategy from {filepath}: {e}")

        if self._strategies:
            self._merkle_dirty = True
            if self.config.enable_merkle:
                self._rebuild_merkle_tree()

            logger.info(f"Loaded {len(self._strategies)} strategies from disk")

    def save_all(self):
        """Save all strategies to disk."""
        for prod_strategy in self._strategies.values():
            self._persist_strategy(prod_strategy)

        logger.info(f"Saved {len(self._strategies)} strategies to disk")

    # ==========================================================================
    # Statistics
    # ==========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get production statistics."""
        active = list(self._strategies.values())

        if not active:
            return {
                "active_count": 0,
                "archived_count": len(self._archive),
                "global_version": self._global_version,
                "merkle_root": self.get_merkle_root(),
            }

        sharpes = [s.production_sharpe for s in active if s.production_sharpe is not None]
        pnls = [s.production_pnl for s in active]
        trades = [s.production_trades for s in active]

        return {
            "active_count": len(active),
            "archived_count": len(self._archive),
            "global_version": self._global_version,
            "merkle_root": self.get_merkle_root(),
            "total_trades": sum(trades),
            "total_pnl": sum(pnls),
            "avg_sharpe": sum(sharpes) / len(sharpes) if sharpes else None,
            "avg_production_days": sum(s.production_days for s in active) / len(active),
        }

    @property
    def active_count(self) -> int:
        """Number of active strategies."""
        return len(self._strategies)

    @property
    def capacity_remaining(self) -> int:
        """Remaining capacity for new strategies."""
        return self.config.max_active_strategies - len(self._strategies)
