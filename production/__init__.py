"""
Production Package for EXPLORER PRIME v2.0

Provides edge decay detection and strategy retirement management.

Key Components:
- EdgeDecayDetector: Kalman filter for Sharpe drift estimation
- RetirementManager: State machine for strategy lifecycle
"""

from .edge_decay import (
    EdgeDecayDetector,
)

from .retirement_manager import (
    StrategyState,
    RetirementAction,
    RetirementManager,
)

__all__ = [
    'EdgeDecayDetector',
    'StrategyState',
    'RetirementAction',
    'RetirementManager',
]
