"""
Forward Testing Module for EXPLORER PRIME

Provides live paper trading capabilities for strategy validation after HIFA approval.
Migrated and enhanced from Hinance paper trading system.

Key Components:
- ForwardTestingBridge: Main interface for deploying and monitoring shadow strategies
- ExecutionEngine: Realistic order execution with market microstructure simulation
- TransferGate: Gate 8 validation based on forward testing results
- DeploymentQueue: Queue management for concurrent shadow strategies
- ShadowMonitor: Real-time monitoring and alerting

Usage:
    from forward_testing import ForwardTestingBridge, TransferGate

    # Deploy strategy after HIFA validation
    bridge = ForwardTestingBridge()
    result = await bridge.deploy(strategy, capital=10000)

    # Monitor performance
    performance = await bridge.get_performance(strategy_id)

    # Check if ready for production
    gate = TransferGate()
    passed = gate.evaluate(performance)
"""

from .bridge import (
    ForwardTestingBridge,
    DeploymentResult,
    ShadowPerformance,
)

from .transfer_gate import (
    TransferGate,
    TransferGateResult,
    TransferGateConfig,
)

from .deployment_queue import (
    DeploymentQueue,
    QueuedStrategy,
    QueueStatus,
)

from .shadow_monitor import (
    ShadowMonitor,
    MonitorAlert,
    AlertLevel,
)

from .models import (
    OrderSide,
    OrderType,
    OrderStatus,
    VolatilityRegime,
    ExecutionResult,
    Position,
    Trade,
)

from .execution import (
    ExecutionEngine,
)

from .analytics import (
    PerformanceAnalyzer,
    TradeRecord,
)

__all__ = [
    # Bridge
    'ForwardTestingBridge',
    'DeploymentResult',
    'ShadowPerformance',

    # Transfer Gate
    'TransferGate',
    'TransferGateResult',
    'TransferGateConfig',

    # Queue
    'DeploymentQueue',
    'QueuedStrategy',
    'QueueStatus',

    # Monitor
    'ShadowMonitor',
    'MonitorAlert',
    'AlertLevel',

    # Models
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'VolatilityRegime',
    'ExecutionResult',
    'Position',
    'Trade',

    # Execution
    'ExecutionEngine',

    # Analytics
    'PerformanceAnalyzer',
    'TradeRecord',
]

__version__ = '2.0.0'
