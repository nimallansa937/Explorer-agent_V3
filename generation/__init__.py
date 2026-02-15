"""
Generation Package for EXPLORER PRIME v2.0

Provides Thompson sampling engine allocation and exploration budget management.

Key Components:
- EngineAllocator: Thompson sampling with gap-driven prior shifts
- ExplorationBudgetManager: 20% budget for experimental features
"""

from .engine_allocator import (
    EngineAllocator,
    ExplorationBudgetManager,
    FeatureExplorationStats,
    GAP_AFFINITY,
)

__all__ = [
    'EngineAllocator',
    'ExplorationBudgetManager',
    'FeatureExplorationStats',
    'GAP_AFFINITY',
]
