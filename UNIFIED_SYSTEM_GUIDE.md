# EXPLORER PRIME: Unified System Architecture Guide

**Version:** 1.0
**Date:** February 13, 2026
**Status:** Implementation Blueprint
**Scope:** Explorer Agent v3.0 + LSM + HIFA v2.0 + Hinance Forward Testing

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Unified Architecture Vision](#3-unified-architecture-vision)
4. [Component Integration Map](#4-component-integration-map)
5. [Data Flow Specification](#5-data-flow-specification)
6. [Strategy Format Unification](#6-strategy-format-unification)
7. [Hinance Migration Plan](#7-hinance-migration-plan)
8. [Implementation Phases](#8-implementation-phases)
9. [API Contracts](#9-api-contracts)
10. [File Structure](#10-file-structure)
11. [Success Metrics](#11-success-metrics)
12. [Risk Analysis](#12-risk-analysis)

---

## 1. Executive Summary

### 1.1 What We're Building

A **unified quantitative strategy development pipeline** that:

1. **Generates** strategies using multiple engines (Evolutionary, GenAI, Pattern, Recombine, LSM)
2. **Validates** strategies through rigorous statistical testing (HIFA v2.0 - 7 gates)
3. **Forward Tests** strategies on live market data (Hinance Paper Trading)
4. **Deploys** only strategies that pass all validation layers

### 1.2 Why This Matters

| Current Problem | Impact | Solution |
|-----------------|--------|----------|
| Strategies go to production after historical backtest only | ~50% false positive rate | Add forward testing layer |
| No transfer ratio measurement | Unknown live performance | Hinance calculates shadow_sharpe / backtest_sharpe |
| Systems scattered across repositories | Integration complexity | Unify in EXPLORER PRIME |
| Different strategy formats | Incompatibility | Single `UnifiedStrategy` format |

### 1.3 Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| False Positive Rate | ~50% | ~20% |
| Time to Production | 1 day | 2-4 weeks |
| Transfer Ratio Known | No | Yes |
| Regime Failures Caught | 0% | 30%+ |

---

## 2. Current State Analysis

### 2.1 EXPLORER PRIME (Current Repository)

```
EXPLORER PRIME/
├── explorer_agent_v3/          ✅ COMPLETE (49 files)
│   ├── agent.py                # Main agent orchestration
│   ├── core/                   # VFE scorer, Kalman updater, posteriors
│   ├── oscillators/            # Fast/Medium/Slow/Regime oscillators
│   ├── workspace/              # Global workspace, threads
│   ├── emt/                    # Empirical Merkle Tree storage
│   ├── c1/                     # Feature expansion, web mining
│   ├── meta/                   # Meta-cognition, exhaustion
│   └── session/                # Identity, consolidation
│
├── lsm/                        ✅ COMPLETE (38 files)
│   ├── tokenization/           # Strategy grammar, tokenizer
│   ├── corpus/                 # Training data builder, labeler
│   ├── model/                  # Transformer architecture
│   ├── training/               # MSM, contrastive learning
│   ├── generation/             # Strategy generator, diversity
│   ├── descartes/              # Z3 verification, backtest
│   └── engine/                 # LSM engine orchestrator
│
├── hifa_v2/                    ✅ COMPLETE (33 tests passing)
│   ├── gates/                  # 7 validation gates
│   ├── statistical/            # DSR, PBO, FDR
│   ├── cv/                     # CPCV, purging, embargo
│   ├── regime/                 # VIX classifier
│   ├── data/                   # Dollar bars, frac diff
│   ├── meta/                   # Meta-labeling, bet sizing
│   ├── synthetic/              # Block bootstrap, WFE, alpha lifecycle
│   ├── clustering/             # ONC, HRP, factor similarity
│   └── validation/             # Ablation, benchmarks, production
│
└── forward_testing/            ❌ MISSING (to be created)
    └── hinance/                # Paper trading (to migrate)
```

### 2.2 HIMARI OPUS 2 (Source for Hinance)

```
HIMARI OPUS 2/
├── HINNANCE PAPER TRADING/
│   └── hinance/                📦 TO MIGRATE
│       ├── src/
│       │   ├── execution_engine.py      # Realistic execution
│       │   ├── shadow_bridge.py         # Layer 1 integration
│       │   ├── strategy_interface.py    # Strategy execution
│       │   ├── feature_extractor.py     # 60-dim features
│       │   ├── performance_analytics.py # Transfer ratio
│       │   ├── signal_router.py         # Signal routing
│       │   └── position_manager.py      # Position tracking
│       ├── config/
│       └── migrations/
│
├── LAYER 1 EXPLORER AGENT/     🔄 OLD VERSION (reference only)
│   └── src/
│       ├── engines/            # Old generation engines
│       └── validation/         # Old HIFA implementation
│
└── HIMARI SIGNAL LAYER/        📊 OPTIONAL (sentiment/on-chain)
    ├── signal_processor.py
    └── onchain/
```

### 2.3 Gap Analysis

| Component | Explorer v3.0 | LSM | HIFA v2.0 | Hinance | Status |
|-----------|---------------|-----|-----------|---------|--------|
| Strategy Generation | ✅ | ✅ | - | - | Complete |
| Historical Validation | - | - | ✅ | - | Complete |
| Forward Testing | - | - | - | ✅ | Needs Migration |
| Transfer Ratio | - | - | - | ✅ | Needs Migration |
| Live Market Data | - | - | - | ✅ | Needs Migration |
| **Integration Layer** | ❌ | ❌ | ❌ | ❌ | **TO BUILD** |

---

## 3. Unified Architecture Vision

### 3.1 Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              EXPLORER PRIME v1.0                                     │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                        LAYER 1: STRATEGY GENERATION                            │ │
│  │                                                                                 │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐│ │
│  │  │ Engine 1    │ │ Engine 2    │ │ Engine 3    │ │ Engine 4    │ │ Engine 5  ││ │
│  │  │ Evolutionary│ │ GenAI       │ │ Pattern     │ │ Recombine   │ │ LSM       ││ │
│  │  │ Search      │ │ Generation  │ │ Discovery   │ │ Crossover   │ │ Language  ││ │
│  │  │ (40%)       │ │ (25%)       │ │ (15%)       │ │ (10%)       │ │ (10%)     ││ │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └─────┬─────┘│ │
│  │         └────────────────┴───────────────┴───────────────┴───────────────┘      │ │
│  │                                          │                                       │ │
│  │                                          ▼                                       │ │
│  │                          ┌───────────────────────────────┐                      │ │
│  │                          │     UnifiedStrategy Format    │                      │ │
│  │                          │  (genome, features, metadata) │                      │ │
│  │                          └───────────────┬───────────────┘                      │ │
│  └──────────────────────────────────────────┼──────────────────────────────────────┘ │
│                                             │                                        │
│                                             ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                     LAYER 2: HIFA v2.0 VALIDATION (Historical)                 │ │
│  │                                                                                 │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │ │
│  │  │ Gate 1 │→│ Gate 2 │→│ Gate 3 │→│ Gate 4 │→│ Gate 5 │→│ Gate 6 │→│ Gate 7 │ │ │
│  │  │Syntax  │ │Complex │ │Quick   │ │DSR+PBO │ │CPCV    │ │HRP     │ │Regime  │ │ │
│  │  │Check   │ │+BIC    │ │Test+MBL│ │+FDR    │ │15 paths│ │Cluster │ │VIX     │ │ │
│  │  │        │ │        │ │        │ │        │ │Purge   │ │Factor  │ │3-state │ │ │
│  │  │ 95%    │ │ 80%    │ │ 70%    │ │ 40%    │ │ 60%    │ │ 70%    │ │ 60%    │ │ │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ │ │
│  │                                                                                 │ │
│  │  Cumulative Pass Rate: ~2-4% of candidates                                     │ │
│  │  Output: HIFAResult with backtest_sharpe, regime_tier, statistical_scores      │ │
│  └──────────────────────────────────────────┬─────────────────────────────────────┘ │
│                                             │                                        │
│                                             ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                    LAYER 3: HINANCE FORWARD TESTING (Live)                     │ │
│  │                                                                                 │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │ │
│  │  │ Shadow Bridge   │  │ Paper Trading   │  │ Performance     │                 │ │
│  │  │                 │  │ Engine          │  │ Analytics       │                 │ │
│  │  │ • Deploy strat  │→ │ • Live Binance  │→ │ • Transfer Ratio│                 │ │
│  │  │ • Feature sync  │  │ • Execution sim │  │ • Sharpe (live) │                 │ │
│  │  │ • Signal route  │  │ • Slippage/fees │  │ • Max DD        │                 │ │
│  │  └─────────────────┘  └─────────────────┘  └────────┬────────┘                 │ │
│  │                                                      │                          │ │
│  │  Duration: 2-4 weeks          Capacity: 50 strategies concurrent               │ │
│  │  Pass Criteria: Transfer Ratio > 0.5, DD Ratio < 1.5                           │ │
│  └──────────────────────────────────────────┬─────────────────────────────────────┘ │
│                                             │                                        │
│                                             ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                         LAYER 4: EMT PRODUCTION STORAGE                        │ │
│  │                                                                                 │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │ │
│  │  │                    Empirical Merkle Tree (EMT)                           │  │ │
│  │  │                                                                          │  │ │
│  │  │  Only strategies with:                                                   │  │ │
│  │  │  ✓ HIFA v2.0 passed (7 gates)                                           │  │ │
│  │  │  ✓ Forward test passed (transfer ratio > 0.5)                           │  │ │
│  │  │  ✓ Minimum shadow period (2+ weeks)                                     │  │ │
│  │  │  ✓ No regime failures                                                   │  │ │
│  │  │                                                                          │  │ │
│  │  │  Stored: genome, backtest_metrics, shadow_metrics, transfer_ratio       │  │ │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          FEEDBACK & MONITORING                                 │ │
│  │                                                                                 │ │
│  │  Production Monitor ──→ Drift Detection ──→ Retirement Trigger                │ │
│  │         │                                           │                          │ │
│  │         └───────────── LSM Retraining ◄─────────────┘                          │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Design Principles

1. **Single Source of Truth**: All components in EXPLORER PRIME
2. **Unified Strategy Format**: One format flows through entire pipeline
3. **Progressive Filtering**: Cheap tests first, expensive tests for survivors
4. **Forward Validation Required**: No production without live testing
5. **Feedback Loop**: Production results improve generation

---

## 4. Component Integration Map

### 4.1 Module Responsibilities

| Module | Responsibility | Input | Output |
|--------|----------------|-------|--------|
| `explorer_agent_v3` | Strategy generation orchestration | Market data, regime | Candidate strategies |
| `lsm` | Language-based strategy generation | Training corpus, conditions | Generated strategies |
| `hifa_v2` | Historical validation (7 gates) | Strategy + returns | HIFAResult |
| `forward_testing` | Live paper trading | HIFAResult | ForwardTestResult |
| `emt` | Production storage | Validated strategies | Persistent storage |

### 4.2 Integration Points

```python
# Integration Point 1: Generation → Validation
explorer_agent_v3.generate() → UnifiedStrategy
                                    ↓
                            hifa_v2.validate(UnifiedStrategy)
                                    ↓
                              HIFAResult

# Integration Point 2: Validation → Forward Testing
hifa_v2.validate() → HIFAResult (if passed)
                           ↓
                   forward_testing.deploy(HIFAResult)
                           ↓
                     ForwardTestResult

# Integration Point 3: Forward Testing → Production
forward_testing.complete() → ForwardTestResult (if passed)
                                    ↓
                              emt.store(ProductionStrategy)
```

---

## 5. Data Flow Specification

### 5.1 End-to-End Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW DIAGRAM                                    │
│                                                                                   │
│  Market Data                                                                      │
│      │                                                                            │
│      ▼                                                                            │
│  ┌─────────────────┐                                                              │
│  │ Data Layer      │                                                              │
│  │ • Dollar Bars   │                                                              │
│  │ • Frac Diff     │                                                              │
│  │ • Triple Barrier│                                                              │
│  └────────┬────────┘                                                              │
│           │                                                                       │
│           ▼                                                                       │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐             │
│  │ Generation      │     │ Historical      │     │ Forward         │             │
│  │                 │     │ Validation      │     │ Testing         │             │
│  │ Input:          │     │                 │     │                 │             │
│  │ • ProcessedData │────▶│ Input:          │────▶│ Input:          │             │
│  │ • RegimeState   │     │ • UnifiedStrat  │     │ • HIFAResult    │             │
│  │ • Conditions    │     │ • Returns       │     │ • BacktestSharpe│             │
│  │                 │     │                 │     │                 │             │
│  │ Output:         │     │ Output:         │     │ Output:         │             │
│  │ • UnifiedStrat  │     │ • HIFAResult    │     │ • ForwardResult │             │
│  │ • Genome        │     │ • GateResults   │     │ • TransferRatio │             │
│  │ • Features      │     │ • BacktestMetric│     │ • ShadowMetrics │             │
│  └─────────────────┘     └─────────────────┘     └────────┬────────┘             │
│                                                           │                       │
│                                                           ▼                       │
│                                                  ┌─────────────────┐              │
│                                                  │ EMT Storage     │              │
│                                                  │                 │              │
│                                                  │ • ProductionStrat│             │
│                                                  │ • All metrics   │              │
│                                                  │ • Audit trail   │              │
│                                                  └─────────────────┘              │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Message Formats

```python
# Stage 1: Generation Output
@dataclass
class UnifiedStrategy:
    strategy_id: str
    genome: StrategyGenome           # Tree-structured strategy logic
    source_engine: str               # "evolutionary", "lsm", "pattern", etc.
    generation_timestamp: datetime
    conditioning: Dict[str, Any]     # Regime, volatility conditions
    metadata: Dict[str, Any]

# Stage 2: HIFA Output
@dataclass
class HIFAResult:
    strategy_id: str
    passed: bool
    final_gate: int                  # Last gate passed (1-7)
    gate_results: Dict[int, GateResult]
    backtest_sharpe: float
    backtest_max_dd: float
    regime_tier: str                 # "all_weather" or "regime_specific"
    statistical_scores: StatisticalScores  # DSR, PBO, FDR

# Stage 3: Forward Testing Output
@dataclass
class ForwardTestResult:
    strategy_id: str
    passed: bool
    shadow_sharpe: float
    shadow_max_dd: float
    transfer_ratio: float            # shadow_sharpe / backtest_sharpe
    trade_count: int
    win_rate: float
    avg_slippage: float
    shadow_duration_days: int
    rejection_reason: Optional[str]

# Stage 4: Production Storage
@dataclass
class ProductionStrategy:
    strategy_id: str
    genome: StrategyGenome
    backtest_metrics: BacktestMetrics
    shadow_metrics: ShadowMetrics
    transfer_ratio: float
    production_start: datetime
    status: str                      # "active", "monitoring", "retired"
```

---

## 6. Strategy Format Unification

### 6.1 The Problem

Currently, different components use different strategy formats:

| Component | Current Format | Issue |
|-----------|----------------|-------|
| Explorer v3.0 | `StrategyPosterior` | Complex, includes beliefs |
| LSM | `StrategyToken` sequence | Tokenized, needs decoding |
| Old Hinance | `genome_json` string | JSON, incompatible schema |
| HIFA v2.0 | Expects returns only | No genome awareness |

### 6.2 Unified Strategy Format

```python
# File: shared/unified_strategy.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

@dataclass
class StrategyGenome:
    """
    Universal strategy representation.

    Can be converted to/from:
    - Explorer v3.0 StrategyPosterior
    - LSM token sequences
    - Hinance genome_json
    - Executable Python code
    """

    # Core logic (tree structure)
    entry_conditions: List[Dict[str, Any]]    # When to enter
    exit_conditions: List[Dict[str, Any]]     # When to exit
    position_sizing: Dict[str, Any]           # How much to trade

    # Parameters
    parameters: Dict[str, float]              # Tunable parameters
    lookback_periods: Dict[str, int]          # Data windows

    # Indicators used
    indicators: List[str]                     # ["RSI", "MACD", "BB", ...]

    # Complexity metrics
    n_parameters: int = field(init=False)
    n_rules: int = field(init=False)
    tree_depth: int = field(init=False)

    def __post_init__(self):
        self.n_parameters = len(self.parameters)
        self.n_rules = len(self.entry_conditions) + len(self.exit_conditions)
        self.tree_depth = self._calculate_depth()

    def _calculate_depth(self) -> int:
        """Calculate max depth of condition tree."""
        def depth(node):
            if isinstance(node, dict) and 'children' in node:
                return 1 + max(depth(c) for c in node['children'])
            return 1
        return max(
            max(depth(c) for c in self.entry_conditions) if self.entry_conditions else 0,
            max(depth(c) for c in self.exit_conditions) if self.exit_conditions else 0,
        )

    def to_json(self) -> str:
        """Serialize to JSON for Hinance compatibility."""
        return json.dumps({
            "entry_conditions": self.entry_conditions,
            "exit_conditions": self.exit_conditions,
            "position_sizing": self.position_sizing,
            "parameters": self.parameters,
            "lookback_periods": self.lookback_periods,
            "indicators": self.indicators,
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'StrategyGenome':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(**data)

    def to_executable(self) -> str:
        """Generate executable Python code."""
        # Implementation generates tradeable code
        pass


@dataclass
class UnifiedStrategy:
    """
    Complete strategy package that flows through entire pipeline.
    """

    # Identity
    strategy_id: str
    version: str = "1.0"

    # Core content
    genome: StrategyGenome

    # Provenance
    source_engine: str                        # Which engine created it
    generation_timestamp: datetime = field(default_factory=datetime.now)
    parent_strategies: List[str] = field(default_factory=list)  # For recombination

    # Conditioning (what market state it was designed for)
    target_regime: Optional[str] = None       # "normal", "elevated", "crisis"
    target_volatility: Optional[str] = None   # "low", "medium", "high"
    target_asset: str = "BTCUSDT"

    # Validation state (updated as it passes gates)
    hifa_result: Optional['HIFAResult'] = None
    forward_result: Optional['ForwardTestResult'] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_hifa_validated(self) -> bool:
        return self.hifa_result is not None and self.hifa_result.passed

    @property
    def is_forward_validated(self) -> bool:
        return self.forward_result is not None and self.forward_result.passed

    @property
    def is_production_ready(self) -> bool:
        return self.is_hifa_validated and self.is_forward_validated
```

### 6.3 Conversion Adapters

```python
# File: shared/adapters.py

class StrategyAdapter:
    """Convert between different strategy formats."""

    @staticmethod
    def from_explorer_posterior(posterior: 'StrategyPosterior') -> UnifiedStrategy:
        """Convert Explorer v3.0 StrategyPosterior to UnifiedStrategy."""
        genome = StrategyGenome(
            entry_conditions=posterior.entry_rules,
            exit_conditions=posterior.exit_rules,
            position_sizing=posterior.sizing_config,
            parameters=posterior.params,
            lookback_periods=posterior.lookbacks,
            indicators=posterior.indicators_used,
        )
        return UnifiedStrategy(
            strategy_id=posterior.id,
            genome=genome,
            source_engine="explorer_v3",
            metadata={"vfe_score": posterior.vfe_score}
        )

    @staticmethod
    def from_lsm_tokens(tokens: List[int], tokenizer: 'StrategyTokenizer') -> UnifiedStrategy:
        """Convert LSM token sequence to UnifiedStrategy."""
        decoded = tokenizer.decode(tokens)
        genome = StrategyGenome(
            entry_conditions=decoded["entry"],
            exit_conditions=decoded["exit"],
            position_sizing=decoded["sizing"],
            parameters=decoded["params"],
            lookback_periods=decoded["lookbacks"],
            indicators=decoded["indicators"],
        )
        return UnifiedStrategy(
            strategy_id=f"lsm_{hash(tuple(tokens))}",
            genome=genome,
            source_engine="lsm",
        )

    @staticmethod
    def to_hinance_format(strategy: UnifiedStrategy) -> Dict[str, Any]:
        """Convert UnifiedStrategy to Hinance deployment format."""
        return {
            "strategy_id": strategy.strategy_id,
            "genome_json": strategy.genome.to_json(),
            "backtest_sharpe": strategy.hifa_result.backtest_sharpe if strategy.hifa_result else 0,
            "metadata": {
                "source_engine": strategy.source_engine,
                "target_regime": strategy.target_regime,
                "hifa_gates_passed": strategy.hifa_result.final_gate if strategy.hifa_result else 0,
            }
        }
```

---

## 7. Hinance Migration Plan

### 7.1 Files to Migrate

From `HIMARI OPUS 2/HINNANCE PAPER TRADING/hinance/` to `EXPLORER PRIME/forward_testing/`:

| Source File | Target Location | Modifications Needed |
|-------------|-----------------|----------------------|
| `src/execution_engine.py` | `forward_testing/execution/engine.py` | Minor imports |
| `src/shadow_bridge.py` | `forward_testing/bridge.py` | Update to UnifiedStrategy |
| `src/strategy_interface.py` | `forward_testing/strategy_executor.py` | Use StrategyGenome |
| `src/feature_extractor.py` | `shared/features.py` | Shared with HIFA |
| `src/performance_analytics.py` | `forward_testing/analytics.py` | Add transfer ratio |
| `src/signal_router.py` | `forward_testing/router.py` | Minor updates |
| `src/position_manager.py` | `forward_testing/positions.py` | Minor updates |
| `src/orderbook_manager.py` | `forward_testing/orderbook.py` | Keep as-is |
| `src/websocket_handler.py` | `forward_testing/websocket.py` | Keep as-is |
| `config/config.yaml` | `forward_testing/config/` | Update paths |

### 7.2 Migration Steps

```bash
# Step 1: Create directory structure
mkdir -p forward_testing/{execution,config,tests}

# Step 2: Copy core files
cp hinance/src/execution_engine.py forward_testing/execution/engine.py
cp hinance/src/shadow_bridge.py forward_testing/bridge.py
# ... etc

# Step 3: Update imports in all files
# - Change `from strategy_interface import` to `from ..shared import`
# - Change `from storage_manager import` to `from .storage import`

# Step 4: Create __init__.py with exports

# Step 5: Run tests
pytest forward_testing/tests/
```

### 7.3 New Files to Create

| File | Purpose |
|------|---------|
| `forward_testing/__init__.py` | Package exports |
| `forward_testing/transfer_gate.py` | Gate 8: Transfer ratio validation |
| `forward_testing/deployment_queue.py` | Queue management for 50 strategy limit |
| `forward_testing/shadow_monitor.py` | Monitor running shadow strategies |
| `shared/features.py` | Unified 60-dim feature schema |
| `shared/unified_strategy.py` | Strategy format definition |
| `shared/adapters.py` | Format conversion utilities |

---

## 8. Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Goal:** Create shared infrastructure

| Task | Deliverable | Priority |
|------|-------------|----------|
| Create `shared/` directory | Shared code location | P0 |
| Implement `UnifiedStrategy` | Strategy format | P0 |
| Implement `StrategyAdapter` | Format conversions | P0 |
| Create `shared/features.py` | 60-dim feature schema | P0 |
| Write unit tests | Test coverage | P1 |

**Files Created:**
```
shared/
├── __init__.py
├── unified_strategy.py
├── adapters.py
├── features.py
└── tests/
    └── test_strategy_format.py
```

### Phase 2: Hinance Migration (Week 3-4)

**Goal:** Move Hinance to EXPLORER PRIME

| Task | Deliverable | Priority |
|------|-------------|----------|
| Create `forward_testing/` structure | Directory setup | P0 |
| Migrate execution engine | `execution/engine.py` | P0 |
| Migrate shadow bridge | `bridge.py` | P0 |
| Update all imports | Compatible code | P0 |
| Create deployment queue | `deployment_queue.py` | P1 |
| Run integration tests | Working migration | P0 |

**Files Created:**
```
forward_testing/
├── __init__.py
├── bridge.py
├── deployment_queue.py
├── shadow_monitor.py
├── transfer_gate.py
├── execution/
│   ├── __init__.py
│   ├── engine.py
│   └── orderbook.py
├── config/
│   └── config.yaml
└── tests/
    └── test_forward_testing.py
```

### Phase 3: Integration Layer (Week 5-6)

**Goal:** Connect all components

| Task | Deliverable | Priority |
|------|-------------|----------|
| Explorer v3.0 → HIFA connector | `hifa_v2/integration.py` update | P0 |
| HIFA → Forward Testing connector | `forward_testing/bridge.py` | P0 |
| Forward Testing → EMT connector | `emt/production.py` | P1 |
| Create unified orchestrator | `orchestrator.py` | P0 |
| End-to-end test | Full pipeline test | P0 |

**New Orchestrator:**
```python
# File: orchestrator.py

class UnifiedOrchestrator:
    """
    Master orchestrator connecting all pipeline stages.
    """

    def __init__(self):
        self.explorer = EnhancedExplorerAgent()
        self.lsm = LSMEngine()
        self.hifa = HIFAv2Pipeline()
        self.forward_tester = ForwardTestingBridge()
        self.emt = EMTStorage()

    async def run_full_pipeline(
        self,
        market_data: pd.DataFrame,
        n_candidates: int = 1000,
    ) -> List[ProductionStrategy]:
        """
        Run complete pipeline: Generate → Validate → Forward Test → Store
        """
        # Stage 1: Generate candidates
        candidates = await self._generate_candidates(market_data, n_candidates)

        # Stage 2: HIFA validation
        hifa_passed = await self._run_hifa_validation(candidates)

        # Stage 3: Forward testing
        forward_passed = await self._run_forward_testing(hifa_passed)

        # Stage 4: Store production-ready
        production = await self._store_production(forward_passed)

        return production
```

### Phase 4: Testing & Documentation (Week 7-8)

**Goal:** Ensure reliability and usability

| Task | Deliverable | Priority |
|------|-------------|----------|
| Integration tests | Full coverage | P0 |
| Performance benchmarks | Timing data | P1 |
| API documentation | Docstrings, examples | P1 |
| User guide | How to use | P2 |
| Deployment scripts | Easy setup | P2 |

---

## 9. API Contracts

### 9.1 Generation API

```python
# Explorer Agent v3.0
class EnhancedExplorerAgent:
    def generate_candidates(
        self,
        market_data: pd.DataFrame,
        n_candidates: int = 100,
        regime: Optional[str] = None,
    ) -> List[UnifiedStrategy]:
        """Generate strategy candidates."""
        pass

# LSM Engine
class LSMEngine:
    def generate(
        self,
        conditions: Dict[str, Any],
        n_samples: int = 10,
    ) -> List[UnifiedStrategy]:
        """Generate strategies via language model."""
        pass
```

### 9.2 Validation API

```python
# HIFA v2.0
class HIFAv2Pipeline:
    def validate(
        self,
        strategy: UnifiedStrategy,
        returns: pd.Series,
        existing_strategies: Optional[List[UnifiedStrategy]] = None,
    ) -> HIFAResult:
        """Run 7-gate validation."""
        pass

    def validate_batch(
        self,
        strategies: List[UnifiedStrategy],
        returns_dict: Dict[str, pd.Series],
    ) -> List[HIFAResult]:
        """Batch validation for efficiency."""
        pass
```

### 9.3 Forward Testing API

```python
# Forward Testing Bridge
class ForwardTestingBridge:
    async def deploy(
        self,
        strategy: UnifiedStrategy,
        capital: float = 10000.0,
    ) -> DeploymentResult:
        """Deploy strategy to shadow environment."""
        pass

    async def get_performance(
        self,
        strategy_id: str,
    ) -> ShadowPerformance:
        """Get current shadow performance."""
        pass

    async def complete_test(
        self,
        strategy_id: str,
    ) -> ForwardTestResult:
        """Complete forward test and calculate transfer ratio."""
        pass
```

### 9.4 Storage API

```python
# EMT Storage
class EMTStorage:
    def store_production(
        self,
        strategy: UnifiedStrategy,
    ) -> str:
        """Store production-ready strategy."""
        pass

    def get_strategy(
        self,
        strategy_id: str,
    ) -> Optional[ProductionStrategy]:
        """Retrieve stored strategy."""
        pass

    def retire_strategy(
        self,
        strategy_id: str,
        reason: str,
    ) -> bool:
        """Mark strategy as retired."""
        pass
```

---

## 10. File Structure

### 10.1 Final Directory Structure

```
EXPLORER PRIME/
├── README.md
├── UNIFIED_SYSTEM_GUIDE.md          # This document
├── requirements.txt
├── setup.py
│
├── shared/                          # NEW: Shared components
│   ├── __init__.py
│   ├── unified_strategy.py          # Strategy format
│   ├── adapters.py                  # Format converters
│   ├── features.py                  # 60-dim feature schema
│   ├── constants.py                 # Shared constants
│   └── tests/
│       └── test_shared.py
│
├── explorer_agent_v3/               # Strategy generation
│   ├── agent.py                     # Main agent
│   ├── config.py
│   ├── core/
│   ├── oscillators/
│   ├── workspace/
│   ├── emt/
│   ├── c1/
│   ├── meta/
│   ├── session/
│   └── tests/
│
├── lsm/                             # Language Strategy Model
│   ├── tokenization/
│   ├── corpus/
│   ├── model/
│   ├── training/
│   ├── generation/
│   ├── descartes/
│   ├── engine/
│   └── tests/
│
├── hifa_v2/                         # Historical validation
│   ├── pipeline.py
│   ├── gates/
│   ├── statistical/
│   ├── cv/
│   ├── regime/
│   ├── data/
│   ├── meta/
│   ├── synthetic/
│   ├── clustering/
│   ├── validation/
│   ├── integration.py               # UPDATED: UnifiedStrategy support
│   └── tests/
│
├── forward_testing/                 # NEW: Hinance migration
│   ├── __init__.py
│   ├── bridge.py                    # Shadow bridge
│   ├── transfer_gate.py             # Gate 8
│   ├── deployment_queue.py          # Queue management
│   ├── shadow_monitor.py            # Monitoring
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── engine.py                # Execution engine
│   │   ├── orderbook.py             # Order book manager
│   │   └── positions.py             # Position manager
│   ├── analytics/
│   │   ├── __init__.py
│   │   └── performance.py           # Performance analytics
│   ├── config/
│   │   └── config.yaml
│   └── tests/
│       └── test_forward_testing.py
│
├── orchestrator.py                  # NEW: Master orchestrator
│
└── tests/
    ├── test_integration.py          # Full pipeline tests
    └── test_end_to_end.py
```

### 10.2 Import Structure

```python
# From any module, imports work like:

# Shared components
from shared import UnifiedStrategy, StrategyGenome, StrategyAdapter
from shared.features import FEATURE_SCHEMA, extract_features

# Generation
from explorer_agent_v3 import EnhancedExplorerAgent
from lsm import LSMEngine

# Validation
from hifa_v2 import HIFAv2Pipeline, HIFAResult

# Forward testing
from forward_testing import ForwardTestingBridge, ForwardTestResult

# Orchestration
from orchestrator import UnifiedOrchestrator
```

---

## 11. Success Metrics

### 11.1 Key Performance Indicators

| KPI | Target | Measurement Method |
|-----|--------|-------------------|
| **False Positive Rate** | <25% | Strategies failing in production / total deployed |
| **Transfer Ratio Accuracy** | >0.7 correlation | Predicted vs actual live performance |
| **Pipeline Throughput** | 100 strategies/day | End-to-end processing rate |
| **Time to Production** | <3 weeks | From generation to deployment |
| **System Uptime** | >99% | Forward testing availability |

### 11.2 Validation Checkpoints

| Checkpoint | Pass Criteria |
|------------|---------------|
| HIFA Gate 1-3 | ~50% of candidates |
| HIFA Gate 4 (Statistical) | ~40% of Gate 3 survivors |
| HIFA Gate 5 (CPCV) | ~60% of Gate 4 survivors |
| HIFA Gate 6-7 | ~40% of Gate 5 survivors |
| **Total HIFA** | **2-4% of initial candidates** |
| Forward Testing | ~50% of HIFA survivors |
| **Total Production** | **1-2% of initial candidates** |

### 11.3 Expected Improvement

| Metric | Before Unification | After Unification |
|--------|-------------------|-------------------|
| Components in sync | No | Yes |
| Strategy format | 3 different | 1 unified |
| Forward testing | None | 2-4 weeks mandatory |
| Transfer ratio | Unknown | Calculated for all |
| False positives | ~50% | ~20% |

---

## 12. Risk Analysis

### 12.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Hinance migration breaks execution | Medium | High | Comprehensive testing, staged rollout |
| Feature drift between HIFA and Hinance | Medium | Medium | Shared feature schema validation |
| Performance degradation with unified format | Low | Medium | Profiling, optimization |
| Binance API changes | Low | High | Abstraction layer, fallback exchanges |

### 12.2 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Forward testing backlog (>50 strategies) | Medium | Medium | Priority queue, capacity monitoring |
| Extended shadow period delays production | Medium | Low | Parallel testing, risk-adjusted duration |
| Strategy format conversion errors | Low | High | Extensive unit tests, validation |

### 12.3 Rollback Plan

If unification fails:

1. **HIFA v2.0**: Continues standalone (already working)
2. **Explorer v3.0**: Continues standalone (already working)
3. **LSM**: Continues standalone (already working)
4. **Hinance**: Remains in HIMARI OPUS 2 (original location)

---

## Appendix A: Quick Start Guide

### A.1 Installation

```bash
# Clone and setup
cd "C:\Users\chari\OneDrive\Documents\EXPLORER PRIME"

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

### A.2 Basic Usage

```python
from orchestrator import UnifiedOrchestrator
import pandas as pd

# Initialize
orchestrator = UnifiedOrchestrator()

# Load market data
market_data = pd.read_csv("data/btcusdt_1h.csv")

# Run full pipeline
production_strategies = await orchestrator.run_full_pipeline(
    market_data=market_data,
    n_candidates=1000,
)

print(f"Production-ready strategies: {len(production_strategies)}")
```

### A.3 Component-Level Usage

```python
# Just generation
from explorer_agent_v3 import EnhancedExplorerAgent
agent = EnhancedExplorerAgent()
candidates = agent.generate_candidates(market_data, n=100)

# Just validation
from hifa_v2 import HIFAv2Pipeline
hifa = HIFAv2Pipeline()
result = hifa.validate(candidates[0], returns)

# Just forward testing
from forward_testing import ForwardTestingBridge
bridge = ForwardTestingBridge()
await bridge.deploy(validated_strategy, capital=10000)
```

---

## Appendix B: Configuration Reference

### B.1 HIFA v2.0 Configuration

```yaml
# hifa_v2/config.yaml
gates:
  gate1_enabled: true
  gate2_enabled: true
  gate3_enabled: true
  gate4:
    dsr_threshold: 0.95
    pbo_threshold: 0.50
    t_stat_threshold: 3.0
  gate5:
    n_groups: 6
    k_test: 2
    purge_days: 20
    embargo_days: 60
  gate6:
    similarity_threshold: 0.70
  gate7:
    vix_normal: 20
    vix_elevated: 30
    min_regime_sharpe: 0.0
```

### B.2 Forward Testing Configuration

```yaml
# forward_testing/config/config.yaml
shadow:
  min_duration_days: 14
  max_duration_days: 28
  min_trades: 20

transfer_ratio:
  min_threshold: 0.5
  max_dd_ratio: 1.5

capacity:
  max_concurrent_strategies: 50

execution:
  slippage_min_pct: 0.01
  slippage_max_pct: 0.10
  fee_taker: 0.001
  fee_maker: 0.0
  latency_min_ms: 50
  latency_max_ms: 200
```

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **DSR** | Deflated Sharpe Ratio - Sharpe adjusted for multiple testing |
| **PBO** | Probability of Backtest Overfitting |
| **CPCV** | Combinatorial Purged Cross-Validation |
| **Transfer Ratio** | shadow_sharpe / backtest_sharpe |
| **EMT** | Empirical Merkle Tree - versioned strategy storage |
| **VFE** | Variational Free Energy - strategy quality score |
| **LSM** | Language Strategy Model - transformer-based generation |
| **Shadow Trading** | Paper trading with simulated execution |

---

*Document Version: 1.0 | Last Updated: February 13, 2026*
*Total Pages: ~40 | Implementation Timeline: 8 weeks*
