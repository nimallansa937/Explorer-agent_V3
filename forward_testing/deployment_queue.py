"""
Deployment Queue for Forward Testing

Manages the queue of strategies waiting for and undergoing shadow trading.
Handles capacity limits, priority scheduling, and lifecycle management.

Features:
- Priority queue with configurable scheduling
- Capacity management (default 50 concurrent)
- Automatic promotion/demotion based on performance
- Queue persistence and recovery
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import heapq
import logging
from collections import defaultdict

from shared.unified_strategy import UnifiedStrategy, StrategyStatus
from shared.constants import FORWARD_TEST_CONFIG

logger = logging.getLogger(__name__)


# ==============================================================================
# Queue Data Structures
# ==============================================================================

class QueueStatus(Enum):
    """Status of a queued strategy."""
    WAITING = "WAITING"
    SCHEDULED = "SCHEDULED"
    DEPLOYING = "DEPLOYING"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class QueuePriority(Enum):
    """Priority levels for queue scheduling."""
    CRITICAL = 0  # Highest priority
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4  # Lowest priority


@dataclass(order=True)
class QueuedStrategy:
    """
    A strategy in the deployment queue.

    Ordering is by (priority, enqueued_time) for heap operations.
    """
    # Sorting key (priority, then FIFO)
    sort_key: tuple = field(compare=True, default=(2, 0.0))

    # Identity (not used for comparison)
    strategy_id: str = field(compare=False, default="")
    strategy: Optional[UnifiedStrategy] = field(compare=False, default=None)

    # Queue status
    status: QueueStatus = field(compare=False, default=QueueStatus.WAITING)
    priority: QueuePriority = field(compare=False, default=QueuePriority.NORMAL)

    # Timing
    enqueued_at: datetime = field(compare=False, default_factory=datetime.now)
    scheduled_at: Optional[datetime] = field(compare=False, default=None)
    deployed_at: Optional[datetime] = field(compare=False, default=None)
    completed_at: Optional[datetime] = field(compare=False, default=None)

    # Configuration
    target_duration_days: int = field(compare=False, default=14)
    max_wait_hours: int = field(compare=False, default=72)  # Expire if waiting too long

    # Progress tracking
    current_trades: int = field(compare=False, default=0)
    current_pnl: float = field(compare=False, default=0.0)
    current_sharpe: float = field(compare=False, default=0.0)

    # Metadata
    source_engine: str = field(compare=False, default="")
    target_asset: str = field(compare=False, default="BTCUSDT")
    tags: List[str] = field(compare=False, default_factory=list)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)

    # Retry tracking
    retry_count: int = field(compare=False, default=0)
    max_retries: int = field(compare=False, default=2)
    last_error: Optional[str] = field(compare=False, default=None)

    def __post_init__(self):
        """Set sort key after initialization."""
        self.sort_key = (self.priority.value, self.enqueued_at.timestamp())

    @property
    def wait_time(self) -> timedelta:
        """Time spent waiting in queue."""
        if self.deployed_at:
            return self.deployed_at - self.enqueued_at
        return datetime.now() - self.enqueued_at

    @property
    def wait_hours(self) -> float:
        """Wait time in hours."""
        return self.wait_time.total_seconds() / 3600

    @property
    def is_expired(self) -> bool:
        """Check if strategy has waited too long."""
        return self.wait_hours > self.max_wait_hours and self.status == QueueStatus.WAITING

    @property
    def active_duration(self) -> timedelta:
        """Time in active shadow trading."""
        if not self.deployed_at:
            return timedelta(0)
        end = self.completed_at or datetime.now()
        return end - self.deployed_at

    @property
    def active_days(self) -> float:
        """Active duration in days."""
        return self.active_duration.total_seconds() / 86400

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_id': self.strategy_id,
            'status': self.status.value,
            'priority': self.priority.value,
            'enqueued_at': self.enqueued_at.isoformat(),
            'deployed_at': self.deployed_at.isoformat() if self.deployed_at else None,
            'wait_hours': self.wait_hours,
            'active_days': self.active_days,
            'current_trades': self.current_trades,
            'current_pnl': self.current_pnl,
            'target_asset': self.target_asset,
            'retry_count': self.retry_count,
        }


# ==============================================================================
# Deployment Queue
# ==============================================================================

class DeploymentQueue:
    """
    Manages the deployment queue for forward testing.

    Features:
    - Priority-based scheduling with configurable policies
    - Capacity management with automatic slot allocation
    - Performance-based early termination
    - Queue persistence for crash recovery

    Usage:
        queue = DeploymentQueue(max_concurrent=50)

        # Add strategy to queue
        queue.enqueue(strategy, priority=QueuePriority.HIGH)

        # Get next strategy to deploy
        next_strategy = queue.get_next()

        # Update status
        queue.update_status(strategy_id, QueueStatus.ACTIVE)

        # Mark complete
        queue.complete(strategy_id, passed=True)
    """

    # Default configuration
    DEFAULT_MAX_CONCURRENT = 50
    DEFAULT_TARGET_DURATION = 14
    DEFAULT_MAX_WAIT_HOURS = 72

    def __init__(
        self,
        max_concurrent: int = 50,
        target_duration_days: int = 14,
        max_wait_hours: int = 72,
    ):
        """
        Initialize deployment queue.

        Args:
            max_concurrent: Maximum concurrent shadow sessions
            target_duration_days: Default shadow trading duration
            max_wait_hours: Maximum wait time before expiration
        """
        self.max_concurrent = max_concurrent
        self.target_duration_days = target_duration_days
        self.max_wait_hours = max_wait_hours

        # Queue storage
        self._waiting_heap: List[QueuedStrategy] = []  # Priority heap
        self._active: Dict[str, QueuedStrategy] = {}
        self._completed: Dict[str, QueuedStrategy] = {}
        self._failed: Dict[str, QueuedStrategy] = {}

        # Index by strategy ID
        self._by_id: Dict[str, QueuedStrategy] = {}

        # Statistics
        self.total_enqueued = 0
        self.total_deployed = 0
        self.total_completed = 0
        self.total_passed = 0
        self.total_failed = 0
        self.total_expired = 0

        # Callbacks
        self.on_deploy_callback: Optional[Callable] = None
        self.on_complete_callback: Optional[Callable] = None
        self.on_expire_callback: Optional[Callable] = None

        logger.info(f"DeploymentQueue initialized: max_concurrent={max_concurrent}")

    @property
    def waiting_count(self) -> int:
        """Number of strategies waiting."""
        return len(self._waiting_heap)

    @property
    def active_count(self) -> int:
        """Number of active shadow sessions."""
        return len(self._active)

    @property
    def available_slots(self) -> int:
        """Available deployment slots."""
        return self.max_concurrent - self.active_count

    @property
    def is_full(self) -> bool:
        """Check if at capacity."""
        return self.active_count >= self.max_concurrent

    def enqueue(
        self,
        strategy: UnifiedStrategy,
        priority: QueuePriority = QueuePriority.NORMAL,
        target_duration_days: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QueuedStrategy:
        """
        Add a strategy to the deployment queue.

        Args:
            strategy: UnifiedStrategy to queue
            priority: Queue priority level
            target_duration_days: Override default duration
            tags: Optional tags for filtering
            metadata: Optional metadata

        Returns:
            QueuedStrategy object
        """
        # Check if already in queue
        if strategy.strategy_id in self._by_id:
            existing = self._by_id[strategy.strategy_id]
            logger.warning(f"Strategy {strategy.strategy_id} already in queue: {existing.status}")
            return existing

        # Create queued entry
        queued = QueuedStrategy(
            strategy_id=strategy.strategy_id,
            strategy=strategy,
            status=QueueStatus.WAITING,
            priority=priority,
            enqueued_at=datetime.now(),
            target_duration_days=target_duration_days or self.target_duration_days,
            max_wait_hours=self.max_wait_hours,
            source_engine=strategy.source_engine.value,
            target_asset=strategy.target_asset,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Add to heap and index
        heapq.heappush(self._waiting_heap, queued)
        self._by_id[strategy.strategy_id] = queued

        self.total_enqueued += 1

        logger.info(f"Enqueued strategy {strategy.strategy_id} with priority {priority.name}")

        return queued

    def get_next(self) -> Optional[QueuedStrategy]:
        """
        Get the next strategy to deploy.

        Returns highest priority waiting strategy if slots available.

        Returns:
            QueuedStrategy or None if queue empty or at capacity
        """
        # Check capacity
        if self.is_full:
            return None

        # Process expired strategies
        self._process_expired()

        # Get highest priority waiting
        while self._waiting_heap:
            queued = heapq.heappop(self._waiting_heap)

            # Skip if status changed
            if queued.status != QueueStatus.WAITING:
                continue

            # Skip expired
            if queued.is_expired:
                self._handle_expiration(queued)
                continue

            # Found valid strategy
            queued.status = QueueStatus.SCHEDULED
            queued.scheduled_at = datetime.now()

            return queued

        return None

    def deploy(self, strategy_id: str) -> bool:
        """
        Mark a strategy as deployed (active shadow trading).

        Args:
            strategy_id: Strategy to deploy

        Returns:
            True if successful
        """
        queued = self._by_id.get(strategy_id)
        if not queued:
            return False

        if queued.status not in [QueueStatus.WAITING, QueueStatus.SCHEDULED]:
            logger.warning(f"Cannot deploy {strategy_id}: status is {queued.status}")
            return False

        queued.status = QueueStatus.ACTIVE
        queued.deployed_at = datetime.now()

        # Move to active
        self._active[strategy_id] = queued

        self.total_deployed += 1

        if self.on_deploy_callback:
            self.on_deploy_callback(queued)

        logger.info(f"Deployed strategy {strategy_id}")

        return True

    def complete(
        self,
        strategy_id: str,
        passed: bool,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Mark a strategy as completed.

        Args:
            strategy_id: Strategy that completed
            passed: Whether it passed forward testing
            metrics: Final performance metrics

        Returns:
            True if successful
        """
        queued = self._by_id.get(strategy_id)
        if not queued:
            return False

        if queued.status != QueueStatus.ACTIVE:
            logger.warning(f"Cannot complete {strategy_id}: not active")
            return False

        queued.status = QueueStatus.COMPLETED
        queued.completed_at = datetime.now()

        if metrics:
            queued.current_trades = metrics.get('total_trades', 0)
            queued.current_pnl = metrics.get('total_pnl', 0.0)
            queued.current_sharpe = metrics.get('sharpe_ratio', 0.0)
            queued.metadata['final_metrics'] = metrics

        # Move from active to completed
        if strategy_id in self._active:
            del self._active[strategy_id]
        self._completed[strategy_id] = queued

        self.total_completed += 1
        if passed:
            self.total_passed += 1
        else:
            self.total_failed += 1

        if self.on_complete_callback:
            self.on_complete_callback(queued, passed)

        logger.info(f"Completed strategy {strategy_id}: passed={passed}")

        return True

    def fail(
        self,
        strategy_id: str,
        error: str,
        allow_retry: bool = True,
    ) -> bool:
        """
        Mark a strategy as failed.

        Args:
            strategy_id: Strategy that failed
            error: Error message
            allow_retry: Whether to allow retry

        Returns:
            True if handled (either retried or marked failed)
        """
        queued = self._by_id.get(strategy_id)
        if not queued:
            return False

        queued.last_error = error
        queued.retry_count += 1

        # Check if can retry
        if allow_retry and queued.retry_count <= queued.max_retries:
            # Re-queue with lower priority
            queued.status = QueueStatus.WAITING
            queued.priority = QueuePriority(min(queued.priority.value + 1, 4))
            queued.sort_key = (queued.priority.value, datetime.now().timestamp())

            # Remove from active if there
            if strategy_id in self._active:
                del self._active[strategy_id]

            # Re-add to heap
            heapq.heappush(self._waiting_heap, queued)

            logger.info(f"Retrying strategy {strategy_id} (attempt {queued.retry_count})")
            return True

        # Mark as failed
        queued.status = QueueStatus.FAILED
        queued.completed_at = datetime.now()

        if strategy_id in self._active:
            del self._active[strategy_id]
        self._failed[strategy_id] = queued

        self.total_failed += 1

        logger.warning(f"Strategy {strategy_id} failed: {error}")

        return True

    def pause(self, strategy_id: str) -> bool:
        """Pause an active strategy."""
        queued = self._by_id.get(strategy_id)
        if queued and queued.status == QueueStatus.ACTIVE:
            queued.status = QueueStatus.PAUSED
            return True
        return False

    def resume(self, strategy_id: str) -> bool:
        """Resume a paused strategy."""
        queued = self._by_id.get(strategy_id)
        if queued and queued.status == QueueStatus.PAUSED:
            queued.status = QueueStatus.ACTIVE
            return True
        return False

    def cancel(self, strategy_id: str) -> bool:
        """Cancel a queued or active strategy."""
        queued = self._by_id.get(strategy_id)
        if not queued:
            return False

        if queued.status in [QueueStatus.COMPLETED, QueueStatus.FAILED]:
            return False

        queued.status = QueueStatus.CANCELLED
        queued.completed_at = datetime.now()

        # Remove from appropriate collection
        if strategy_id in self._active:
            del self._active[strategy_id]

        self._failed[strategy_id] = queued

        logger.info(f"Cancelled strategy {strategy_id}")

        return True

    def update_progress(
        self,
        strategy_id: str,
        trades: int,
        pnl: float,
        sharpe: float,
    ) -> None:
        """Update progress metrics for an active strategy."""
        queued = self._by_id.get(strategy_id)
        if queued:
            queued.current_trades = trades
            queued.current_pnl = pnl
            queued.current_sharpe = sharpe

    def get_status(self, strategy_id: str) -> Optional[QueuedStrategy]:
        """Get queue status for a strategy."""
        return self._by_id.get(strategy_id)

    def get_waiting(self) -> List[QueuedStrategy]:
        """Get all waiting strategies (sorted by priority)."""
        return sorted(
            [q for q in self._by_id.values() if q.status == QueueStatus.WAITING],
            key=lambda x: x.sort_key,
        )

    def get_active(self) -> List[QueuedStrategy]:
        """Get all active strategies."""
        return list(self._active.values())

    def get_completed(self, limit: int = 100) -> List[QueuedStrategy]:
        """Get recently completed strategies."""
        completed = sorted(
            self._completed.values(),
            key=lambda x: x.completed_at or datetime.min,
            reverse=True,
        )
        return completed[:limit]

    def _process_expired(self) -> None:
        """Process and remove expired waiting strategies."""
        now = datetime.now()
        to_expire = []

        for queued in self._waiting_heap:
            if queued.status == QueueStatus.WAITING and queued.is_expired:
                to_expire.append(queued)

        for queued in to_expire:
            self._handle_expiration(queued)

    def _handle_expiration(self, queued: QueuedStrategy) -> None:
        """Handle an expired strategy."""
        queued.status = QueueStatus.EXPIRED
        queued.completed_at = datetime.now()

        self._failed[queued.strategy_id] = queued
        self.total_expired += 1

        if self.on_expire_callback:
            self.on_expire_callback(queued)

        logger.warning(f"Strategy {queued.strategy_id} expired after {queued.wait_hours:.1f}h")

    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            'waiting': self.waiting_count,
            'active': self.active_count,
            'completed': len(self._completed),
            'failed': len(self._failed),
            'available_slots': self.available_slots,
            'capacity_used_pct': (self.active_count / self.max_concurrent) * 100,
            'total_enqueued': self.total_enqueued,
            'total_deployed': self.total_deployed,
            'total_completed': self.total_completed,
            'total_passed': self.total_passed,
            'total_failed': self.total_failed,
            'total_expired': self.total_expired,
            'pass_rate': self.total_passed / self.total_completed if self.total_completed > 0 else 0,
        }

    def get_queue_snapshot(self) -> Dict[str, Any]:
        """Get complete queue snapshot for persistence."""
        return {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'max_concurrent': self.max_concurrent,
                'target_duration_days': self.target_duration_days,
                'max_wait_hours': self.max_wait_hours,
            },
            'waiting': [q.to_dict() for q in self.get_waiting()],
            'active': [q.to_dict() for q in self.get_active()],
            'statistics': self.get_statistics(),
        }
