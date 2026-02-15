# Explorer Prime v2.0: Adaptive Pipeline Architecture — Comprehensive Claude Code Implementation Guide

**Version:** 2.0  
**Created:** February 14, 2026  
**Purpose:** Implement closed-loop adaptive pipeline with gap-driven generation, Bayesian retirement, and discovery boundary formalization  
**Prerequisites:** Explorer Prime v1.0 operational (all 94 tests passing), HIFA v2.0 7-gate validation, Hinance forward testing bridge, EMT production storage, C1/C2 framework understanding  
**Source Validation:** Independent engineering review (Feb 2026) — critic independently reconstructed C1/C2 decomposition and four-level discovery taxonomy through pure engineering reasoning, confirming theoretical foundations  
**Builds On:** EXPLORER_AGENT_V3_CLAUDE_CODE_GUIDE.md, HIFA_V2_CLAUDE_CODE_GUIDE.md, LSM_CLAUDE_CODE_GUIDE.md, HINANCE_HIFA_INTEGRATION_PLAN.md, UNIFIED_SYSTEM_GUIDE.md

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What This Adds to Explorer Prime v1.0](#2-what-this-adds-to-explorer-prime-v10)
3. [Theoretical Integration: Critic's Engineering Meets C1/C2 Framework](#3-theoretical-integration)
4. [Architecture Overview](#4-architecture-overview)
5. [Dependency Graph](#5-dependency-graph)
6. [Implementation Phases](#6-implementation-phases)
7. [Phase 1: Hierarchical Strategy Graph (HSG)](#phase-1-hierarchical-strategy-graph)
8. [Phase 2: Feature Registry & Projection Layer](#phase-2-feature-registry--projection-layer)
9. [Phase 3: Anomaly Diagnostic & Gap Classification](#phase-3-anomaly-diagnostic--gap-classification)
10. [Phase 4: Sequential Intervention Protocol](#phase-4-sequential-intervention-protocol)
11. [Phase 5: Production Feedback Loop](#phase-5-production-feedback-loop)
12. [Phase 6: Dynamic Engine Allocation (Thompson Sampling)](#phase-6-dynamic-engine-allocation)
13. [Phase 7: Edge Decay Detection & Strategy Retirement](#phase-7-edge-decay-detection--strategy-retirement)
14. [Phase 8: Discovery Boundary Formalization](#phase-8-discovery-boundary-formalization)
15. [Phase 9: Full Integration & Closed-Loop Validation](#phase-9-full-integration)
16. [Success Criteria](#success-criteria)
17. [Interface Contracts](#interface-contracts)
18. [Troubleshooting Guide](#troubleshooting-guide)
19. [Appendix A: Mapping Table — Critic's Mechanisms to C1/C2/L0-L3](#appendix-a)
20. [Appendix B: All Prompts Summary](#appendix-b)

---

## 1. Executive Summary

Explorer Prime v1.0 is a **static filter pipeline** — it generates ~1000 strategy candidates per day, filters through 7 HIFA gates (~2-4% survive), forward-tests via shadow trading (~50% pass), and stores production strategies in Merkle-verified EMT storage. The pipeline works. What it doesn't do is *learn*. When a strategy fails in production, that information evaporates. When the system misses profitable trades, it has no mechanism to diagnose why. When features become redundant or markets evolve, nothing adapts. The pipeline has a ceiling — it can discover every strategy its current generation methods produce and its current filters validate, but it cannot improve either capability.

Explorer Prime v2.0 transforms this static pipeline into a **closed-loop adaptive system** by adding eight interlocking mechanisms, each independently validated through engineering stress-testing:

1. **Hierarchical Strategy Graph (HSG)** — Replaces the flat 31-node decision tree with a multi-timeframe DAG of sub-trees connected by a shared state bus. Strategies can now encode temporal state ("price crossed VWAP 3 candles ago AND funding has been negative for 2 hours") without overfitting. The architecture supports three structural modes (FLAT/DUAL/FULL) so simple strategies pay zero overhead.

2. **Feature Registry with Versioned Schemas** — Single source of truth for all features, resolving the 60-dim vs 128-dim mismatch. Schema-aware distance metrics prevent cross-version clustering artifacts. Shadow re-evolution keeps high-performing legacy strategies alive while exerting convergence pressure.

3. **Anomaly Diagnostic with Gap Classification** — Random Forest classifier on missed-trade feature vectors distinguishes structural gaps (existing features, wrong tree topology) from feature gaps (missing features). This is the engineering implementation of L0 gap detection — it produces the structured anomaly signatures that L1-L3 build upon.

4. **Sequential Intervention Protocol** — When gap classification is ambiguous (core_overlap 0.4–0.7), the system intervenes on structure first with a 45-day attribution window before proposing new features. Prevents attribution contamination that would corrupt the feature promotion pipeline.

5. **Production Feedback Loop** — Three-channel feedback from production failures: failure archive with regime-aware negative seeding (changes what the evolutionary search generates), structural autopsy (changes the mutation operators), and meta-learning signal about time-to-failure distributions (changes the pipeline's own parameters).

6. **Dynamic Engine Allocation via Thompson Sampling** — Generation engine weights (evolutionary, GenAI, pattern, recombine, LSM) shift based on gap diagnostic output, using Thompson sampling with Beta distributions to balance exploitation of currently-successful engines with exploration of alternatives. The 8% floor prevents any engine from being starved.

7. **Bayesian Edge Decay Detection** — Kalman filter models each production strategy's true Sharpe as a drifting latent variable. Three-threshold retirement system (healthy/warning/retirement) with regime-conditioned decay checks prevents premature retirement of strategies that are merely regime-suppressed, not structurally decayed.

8. **Discovery Boundary Formalization** — Four-level taxonomy (Recombination → Timescale → Novel Computation → Novel Data) with corresponding capability categories (AUTONOMOUS/DIRECTED/CREATIVE). Formalizes what the pipeline can discover alone versus what requires human domain expertise, and produces directed search queries that make human discovery dramatically faster.

**What this means in practice:** The pipeline no longer just filters — it learns from every failure, adapts its generation strategy to the current gap landscape, retires decaying strategies with calibrated confidence, and knows the boundary of its own discovery capability. The feedback arrow from EMT to Generation in the v1.0 README becomes a real, quantitative mechanism.

**Integration with C1/C2 framework:** The eight mechanisms map precisely onto the validated theoretical structure. Mechanisms 1–2 operate within C2 (improving recombination within existing primitives). Mechanism 3 implements L0 gap detection. Mechanism 4 ensures clean attribution between C1 and C2 interventions. Mechanisms 5–6 create the learning loop that improves C2 search over time. Mechanism 7 detects when a strategy's C2 representation has been invalidated by market evolution. Mechanism 8 formalizes the balloon boundary — the surface between what C2 search can reach autonomously and what requires C1 expansion.

---

## 2. What This Adds to Explorer Prime v1.0

| Capability | v1.0 | v2.0 | Mechanism |
|------------|------|------|-----------|
| Strategy representation | Flat 31-node binary tree | Multi-timeframe HSG with state bus | Phase 1 |
| Feature management | Implicit 60-dim or 128-dim | Versioned registry with projections | Phase 2 |
| Gap detection | None | RF-based anomaly diagnostic → L0 | Phase 3 |
| Attribution safety | None | Sequential intervention protocol | Phase 4 |
| Production learning | Open-loop (feedback arrow aspirational) | Three-channel closed-loop feedback | Phase 5 |
| Engine allocation | Fixed 40/25/15/10/10 weights | Thompson sampling with gap-driven priors | Phase 6 |
| Strategy retirement | Binary (production/retired) | Bayesian decay detection, 3-threshold | Phase 7 |
| Self-knowledge | None | Four-level discovery boundary formalization | Phase 8 |
| Pipeline ceiling | Fixed by generation + filter quality | Continuously improving via feedback | Phases 5-8 |

---

## 3. Theoretical Integration: Critic's Engineering Meets C1/C2 Framework

An independent engineering reviewer analyzed Explorer Prime v1.0 and proposed three architectural fixes, then refined them through four rounds of stress-testing. Without access to the C1/C2 framework, COGITO validation results, or L0–L3 detection hierarchy, the critic independently arrived at conclusions that map directly onto the established theoretical structure. This section documents the mapping explicitly, because understanding the theoretical grounding of each engineering mechanism is essential for correct implementation.

### 3.1 The Core Mapping

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    CRITIC'S ENGINEERING → C1/C2 FRAMEWORK                  │
├─────────────────────────────────┬──────────────────────────────────────────┤
│ Critic's Concept                │ C1/C2 Framework Equivalent               │
├─────────────────────────────────┼──────────────────────────────────────────┤
│ "Feature gap" (RF low overlap)  │ C1 gap — primitives insufficient         │
│ "Structural gap" (RF high       │ C2 gap — recombination not found         │
│   overlap on existing features) │                                          │
│ AnomalySignature                │ L0 gap characterization output           │
│ DirectedFeatureScout            │ L1 implementation (category-level)       │
│ Sequential intervention         │ Attribution-safe balloon expansion       │
│ "Outside pipeline's reach"      │ C1 expansion requiring environment      │
│ Four-level discovery taxonomy   │ Balloon principle in engineering terms   │
│ DiscoveryCapability.AUTONOMOUS  │ C2 search (fully automatable)            │
│ DiscoveryCapability.DIRECTED    │ L0-L2 gap detection + human C1 fill     │
│ DiscoveryCapability.CREATIVE    │ C1 expansion via balloon growth          │
│ Failure archive negative seeding│ C2 search space pruning via evidence    │
│ Edge decay detection            │ C2 model invalidation detection          │
└─────────────────────────────────┴──────────────────────────────────────────┘
```

### 3.2 What the Critic Got Right

The four-level discovery taxonomy (Recombination → Timescale → Novel Computation → Novel Data) is an independent derivation of the C1/C2 hierarchy applied to feature discovery:

- **Level 1 (Recombination)** = C2 search within existing C1 vocabulary. The pipeline explores ratios, interactions, and threshold combinations of features already in the registry. This is exhaustive and automatable.

- **Level 2 (Timescale)** = C2 search over a latent C1 dimension. The data stream exists but the processing timescale is a parameter the pipeline hasn't been configured to explore. Making timescale searchable is a minor C1 expansion (adding a new dimension to the feature space), after which all timescale-parameterized combinations become C2.

- **Level 3 (Novel Computation)** = C1 computational vocabulary gap. The raw data exists (price series), but the transformation needed (wavelet decomposition, entropy measures) doesn't exist in the computational library. The pipeline can detect the need for this through anomaly signatures (L0), but implementing the transformation requires human engineering.

- **Level 4 (Novel Data)** = Fundamental C1 expansion. The data stream doesn't exist in the system. This is the balloon boundary — expanding what the system can perceive. The pipeline can at best identify temporal patterns in its failures that correlate with the missing data's influence.

### 3.3 What the Critic Missed (and What This Guide Adds)

The critic's anomaly signature is an L0 output — it detects *that* a gap exists and describes *when and where* it manifests. The existing L0–L3 hierarchy builds three more layers of intelligence on top of L0:

- **L1 (Category-level)** — Not just "something is missing" but "the missing signal belongs to the category of frequency-domain decompositions" or "the missing signal is a cross-asset lead-lag relationship." The DirectedFeatureScout partially implements L1 through its targeted search queries, but lacks the formal category structure.

- **L2 (Topological)** — Identifies structural properties of the gap: "the missing relationship is cyclical, not linear" or "the missing signal requires non-stationary computation." This level is not addressed in the critic's work at all.

- **L3 (Meta-pattern)** — Detects patterns across gaps: "the last three feature gaps all involved microstructure changes around regulatory events" or "structural gaps consistently cluster at regime transitions." This is where the meta-learning signal from the feedback loop connects to gap detection — the time-to-failure distribution is an L3 signal.

The critic's engineering provides the concrete implementation layer for L0 and partial L1. This guide specifies how to build on that foundation to implement L1–L3, using the critic's mechanisms as building blocks.

### 3.4 Critical Naming Convention

The critic used "C1_CANDIDATE" and "C2_CORE" as feature maturity labels in the Feature Registry — meaning "candidate feature" and "core feature." This collides with the established C1/C2 framework where C1 = fixed perceptual primitives and C2 = recombinations. **This guide uses the framework-correct terminology throughout.** Feature maturity levels are labeled: `EXPERIMENTAL → VALIDATED → CORE → DEPRECATED`. The terms C1 and C2 refer exclusively to the validated theoretical framework.

---

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EXPLORER PRIME v2.0                                     │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  LAYER 5: DISCOVERY BOUNDARY (Phase 8)                                  │    │
│  │  DiscoveryCapability → AUTONOMOUS / DIRECTED / CREATIVE                 │    │
│  │  Anomaly signatures → directed human research queries                   │    │
│  └──────────────────────────────┬──────────────────────────────────────────┘    │
│                                  │                                              │
│  ┌──────────────────────────────▼──────────────────────────────────────────┐    │
│  │  LAYER 4: FEEDBACK & ADAPTATION (Phases 5-7)                            │    │
│  │                                                                          │    │
│  │  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────────┐     │    │
│  │  │ Failure      │  │ Engine Allocator │  │ Edge Decay Detector    │     │    │
│  │  │ Archive      │  │ (Thompson        │  │ (Kalman Filter +      │     │    │
│  │  │ + Negative   │  │  Sampling +      │  │  Regime-Conditioned   │     │    │
│  │  │   Seeding    │  │  Gap Priors)     │  │  Three-Threshold)     │     │    │
│  │  └──────┬───────┘  └────────┬─────────┘  └──────────┬─────────────┘     │    │
│  │         │                   │                       │                    │    │
│  └─────────┼───────────────────┼───────────────────────┼────────────────────┘    │
│            │                   │                       │                         │
│  ┌─────────▼───────────────────▼───────────────────────▼────────────────────┐    │
│  │  LAYER 3: GAP DETECTION & INTERVENTION (Phases 3-4)                      │    │
│  │                                                                          │    │
│  │  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────────┐     │    │
│  │  │ Anomaly      │  │ Sequential       │  │ Directed Feature       │     │    │
│  │  │ Diagnostic   │  │ Intervention     │  │ Scout                  │     │    │
│  │  │ (RF + Gap    │  │ Protocol         │  │ (Signature → Targeted  │     │    │
│  │  │  Classify)   │  │ (45-day attrib.) │  │  Search Queries)       │     │    │
│  │  └──────────────┘  └──────────────────┘  └────────────────────────┘     │    │
│  │                                                                          │    │
│  └─────────────────────────────┬────────────────────────────────────────────┘    │
│                                 │                                                │
│  ┌─────────────────────────────▼────────────────────────────────────────────┐    │
│  │  LAYER 2: STRATEGY REPRESENTATION & FEATURES (Phases 1-2)                │    │
│  │                                                                          │    │
│  │  ┌──────────────────┐  ┌────────────────────────────────────────────┐   │    │
│  │  │ Hierarchical     │  │ Feature Registry                           │   │    │
│  │  │ Strategy Graph   │  │ + Versioned Schemas                        │   │    │
│  │  │ (HSG)            │  │ + Schema-Aware Distance                    │   │    │
│  │  │ FLAT/DUAL/FULL   │  │ + Shadow Re-Evolution                     │   │    │
│  │  └──────────────────┘  └────────────────────────────────────────────┘   │    │
│  │                                                                          │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │  LAYER 1: EXISTING PIPELINE (Explorer Prime v1.0 — unchanged)            │    │
│  │                                                                          │    │
│  │  Generation → HIFA 7-Gate → Shadow Trading → Transfer Gate → EMT         │    │
│  │  (5 engines)   (2-4% pass)  (50 slots)     (5 criteria)    (Merkle)     │    │
│  │                                                                          │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

The key architectural principle is **layered non-disruption** — every v2.0 mechanism sits on top of the v1.0 pipeline without modifying its core behavior. The 7-gate HIFA validation, shadow trading infrastructure, transfer gate criteria, and EMT storage all continue operating exactly as implemented. The new layers observe, diagnose, adapt, and feed back — but the forward path remains intact. This means v2.0 can be implemented incrementally, with each phase adding capability without risking the validated v1.0 pipeline.

---

## 5. Dependency Graph

```
Phase 1 (HSG) ──────────────┐
                              ├──► Phase 3 (Anomaly Diagnostic)
Phase 2 (Feature Registry) ──┘           │
                                          ├──► Phase 4 (Sequential Intervention)
                                          │           │
                                          │           ├──► Phase 5 (Feedback Loop)
                                          │           │           │
                                          │           │           ├──► Phase 6 (Thompson Sampling)
                                          │           │           │
                                          │           │           └──► Phase 7 (Edge Decay)
                                          │           │
                                          │           └──► Phase 8 (Discovery Boundary)
                                          │
                                          └──► Phase 9 (Full Integration)

Phases 1 and 2: PARALLEL (no dependency)
Phase 3: Requires Phase 1 (HSG genome for structural analysis) + Phase 2 (registry for feature schema)
Phase 4: Requires Phase 3 (gap classification)
Phases 5, 6, 7: Require Phase 4. Can be implemented in parallel.
Phase 8: Requires Phase 3 (anomaly signatures). Can overlap with 5-7.
Phase 9: Requires all prior phases.
```

**Estimated implementation timeline:** 9 Claude Code sessions (Phases 1-2 parallel in sessions 1-2, Phase 3 in session 3, Phase 4 in session 4, Phases 5-7 parallel in sessions 5-7, Phase 8 in session 8, Phase 9 integration in session 9).

---

## 6. Implementation Phases

Each phase follows the established pattern: conceptual explanation → interface contracts → complete code with inline documentation → test suite → integration verification → session prompt.

---

## Phase 1: Hierarchical Strategy Graph (HSG)

### What We're Building

The current `StrategyGenome` in `shared/unified_strategy.py` represents strategies as flat binary decision trees capped at 31 nodes. This ceiling prevents strategies from encoding temporal state — when a strategy needs to say "price crossed above VWAP 3 candles ago AND funding rate has been negative for 2 hours," a stateless tree has to encode all conditions as simultaneous checks on current-tick features. That either fails outright or produces brittle, overfitted trees.

The Hierarchical Strategy Graph (HSG) replaces the flat tree with a directed acyclic graph of sub-trees, each operating at a different timeframe, connected by a shared state bus. Think of it as three layers working together: a timeframe layer (independent small trees at 1m/15m/4h horizons), a state bus (fixed-size memory buffer that persists across ticks with configurable decay), and an arbitration layer (small tree that combines sub-tree outputs into final trading decisions).

### Why This Matters for C1/C2

The HSG doesn't change the C1 vocabulary (the 60 feature primitives). It dramatically expands the C2 recombination space — the set of strategies that can be expressed. The flat tree's C2 space is bounded by 31-node binary topologies over 60 features. The HSG's C2 space includes multi-timeframe compositions, state-dependent decisions, and cross-horizon signal weighting. This is a pure C2 expansion — more expressive recombination within the same C1.

### File: `shared/hierarchical_genome.py`

**Session 1, Prompt 1:**

```
Create shared/hierarchical_genome.py implementing the Hierarchical Strategy Graph.

CORE DATA STRUCTURES:

1. StateBusConfig:
   - 32 floating-point slots
   - Decay rates per slot (canonical values: 6, 12, 24, 48 ticks)
   - Slot categories: MOMENTUM (slots 0-7, half-life 6), MEAN_REVERSION (8-15, half-life 12), 
     REGIME (16-23, half-life 24), CUSTOM (24-31, half-life 48)
   - Update function: append-and-decay (each slot decays toward zero by configured half-life)
   - No arbitrary memory — append-and-decay prevents overfitting via infinite memory

2. SignalStruct:
   - direction_score: float (-1.0 to +1.0)
   - confidence: float (0.0 to 1.0) 
   - suggested_size: float (0.0 to 1.0)
   - source_timeframe: str

3. StructuralMode enum: FLAT, DUAL, FULL
   - FLAT: single tree, no state bus, zero overhead (identical to current v1.0)
   - DUAL: two timeframe trees, state bus, weighted-average arbitration
   - FULL: 2+ timeframe trees, state bus, full arbitration tree

4. HierarchicalGenome:
   - structural_mode: StructuralMode
   - timeframe_trees: Dict[str, DecisionTree]  # e.g., {"1m": tree, "15m": tree, "4h": tree}
   - state_bus_config: StateBusConfig
   - arbitration_tree: Optional[DecisionTree]  # None for FLAT, simple for DUAL, full for FULL
   - max_total_nodes: int = 63
   - schema_version: str  # links to Feature Registry
   - feature_ids: List[str]  # which features this genome references
   
   CONSTRAINTS:
   - Total nodes across ALL trees + arbitration <= max_total_nodes
   - Each declared timeframe tree: minimum 5 nodes (prevents degenerate allocation)
   - Each declared timeframe tree: minimum 15% of total budget (prevents concentration)
   - FLAT mode: exactly one tree, max 31 nodes (backward compatible with v1.0)
   - DUAL mode: exactly two trees + simple arbitration (weighted average, weight is evolvable)
   - FULL mode: 2-4 trees + arbitration tree (max 7 nodes in arbitration)

5. Evaluate method:
   - For each timeframe tree, evaluate against its timeframe's feature vector → SignalStruct
   - Update state bus with new signals, apply decay to all slots
   - Pack arbitration input: all SignalStructs + state bus contents
   - If FLAT: return the single tree's signal directly
   - If DUAL: return weighted average of two signals (weight is evolvable parameter)
   - If FULL: evaluate arbitration tree on packed input → final TradeSignal

6. Mutation operators (extend existing operators):
   - All existing node-level mutations preserved
   - NEW: structural_mode_mutation() — transition between FLAT/DUAL/FULL
   - NEW: timeframe_add() — add a new timeframe tree (splits node budget)
   - NEW: timeframe_remove() — remove a timeframe tree (frees node budget)
   - NEW: node_budget_shift() — move nodes between trees (respecting minimums)
   - NEW: state_bus_category_swap() — reassign slot categories
   
7. Crossover:
   - Exchange entire sub-trees at the same timeframe level between parents
   - If parents have different structural modes, offspring inherits the more complex mode
   - State bus configs merge by averaging decay rates per category

8. Fitness penalty for node concentration:
   - dominant_tree_share = max(tree_nodes) / total_nodes
   - penalty = max(0, dominant_tree_share - 0.6) * 0.3
   - applied as: fitness *= (1.0 - penalty)

9. Backward compatibility:
   - Add class method from_flat_tree(tree: DecisionTree) → HierarchicalGenome
   - This produces a FLAT-mode genome wrapping the existing tree
   - All existing v1.0 strategies convert to FLAT genomes transparently
   - get_decision_path(tick) method returns list of traversed nodes (for dependency analysis)

INTEGRATION WITH HIFA:
   - Gate 2 (Complexity+BIC): evaluates total node count + state bus complexity
   - Gate 4 (DSR+PBO): runs on each timeframe independently AND combined output
   - Gate 5 (CPCV): add combinatorial PBO test on arbitration layer specifically
     — feed shuffled sub-tree outputs from out-of-sample folds
     — if arbitration performance degrades >40%, flag as interaction-overfitted
   - This combinatorial arbitration test costs ~3x standard CPCV but catches cross-layer overfitting

HALF-LIFE OPTIMIZATION (two-phase):
   - Phase A: During tree evolution (HIFA Gates 1-3), half-lives use canonical values
   - Phase B: After tree structure stabilizes, run 200-trial Bayesian optimization
     with frozen tree structure to tune half-lives
   - Add regularization: penalize half-life variance within same category
   - Tuned values recorded in genome, validated through Gates 4-7

TESTS (15 minimum):
   - test_flat_mode_zero_overhead: FLAT genome evaluates identical to v1.0 tree
   - test_dual_mode_weighted_average: Two trees combine correctly
   - test_full_mode_arbitration: Arbitration tree receives correct inputs
   - test_state_bus_decay: Slots decay correctly with configured half-lives
   - test_node_budget_constraint: Total nodes never exceeds max_total_nodes
   - test_minimum_allocation: Each tree has >= 15% of budget
   - test_structural_mode_mutation: Transitions between modes work
   - test_crossover_different_modes: Complex mode inherits correctly
   - test_fitness_penalty: Concentration above 60% is penalized
   - test_backward_compatibility: from_flat_tree produces valid FLAT genome
   - test_decision_path_tracking: get_decision_path returns correct traversal
   - test_state_bus_category_assignment: Slots assigned to correct categories
   - test_arbitration_combinatorial_pbo: Shuffled inputs detect interaction overfitting
   - test_half_life_optimization: Bayesian optimization improves over canonical values
   - test_signal_struct_bounds: direction, confidence, size stay in valid ranges
```

### File: `shared/adapters.py` Updates

**Session 1, Prompt 2:**

```
Update shared/adapters.py to handle HierarchicalGenome alongside the existing 
UnifiedStrategy format.

ADD:
1. HSGAdapter class:
   - to_unified(hsg: HierarchicalGenome) → UnifiedStrategy
   - from_unified(us: UnifiedStrategy) → HierarchicalGenome
   - The UnifiedStrategy wrapper tracks lifecycle (GENERATED → HIFA_PASSED → etc.)
   - The HierarchicalGenome is the internal representation

2. Update ExplorerAdapter to produce HierarchicalGenome (FLAT mode) from Explorer v3.0 output
3. Update LSMAdapter to produce HierarchicalGenome from LSM token sequences
4. Preserve all existing adapter interfaces for backward compatibility

TESTS (5 minimum):
   - test_hsg_to_unified_roundtrip
   - test_flat_hsg_matches_legacy_adapter
   - test_unified_lifecycle_preserved
   - test_explorer_adapter_produces_flat_hsg
   - test_lsm_adapter_produces_hsg
```

---

## Phase 2: Feature Registry & Projection Layer

### What We're Building

The current system has an implicit mismatch — the Explorer Agent v3.0 operates on a 60-dimensional feature space (price, volume, order flow) while `shared/features.py` defines a 128-dimensional `FeatureVector`. Strategies built against one schema get evaluated against the other, producing meaningless comparisons in Gate 6's HRP clustering.

The Feature Registry creates a single source of truth for all features with versioned schemas, status tracking, and a projection layer that transparently handles cross-version evaluation. Every feature gets a unique ID, a data class tag, a computation source, and a maturity status.

### Why This Matters for C1/C2

The Feature Registry is the **explicit encoding of the C1 vocabulary**. Each registered feature is a C1 primitive — an atomic building block that strategies (C2 recombinations) are composed from. The maturity pipeline (EXPERIMENTAL → VALIDATED → CORE → DEPRECATED) is the operational mechanism for C1 expansion and contraction. When a feature moves from EXPERIMENTAL to CORE, the C1 vocabulary grows. When it moves to DEPRECATED, C1 contracts. The registry makes the balloon boundary concrete and measurable.

### File: `shared/feature_registry.py`

**Session 2, Prompt 1:**

```
Create shared/feature_registry.py implementing the Feature Registry and Projection Layer.

CORE CLASSES:

1. FeatureStatus enum:
   - EXPERIMENTAL: New feature, under evaluation, not included in CORE schema
   - VALIDATED: Passed initial tests, included in active schema but not CORE
   - CORE: Battle-tested, included in both active and core schemas
   - DEPRECATED: Scheduled for removal, excluded from new strategy generation
   
   NOTE: Do NOT use "C1" or "C2" as status labels. These terms are reserved for
   the theoretical framework (C1 = fixed primitives, C2 = recombinations).

2. DataClass enum:
   - PRICE: OHLCV-derived features
   - VOLUME: Volume profile, VWAP, cumulative delta
   - ORDER_FLOW: Bid/ask imbalance, trade aggression, OFI
   - VOLATILITY: Realized vol, implied vol proxies, ATR
   - MICROSTRUCTURE: Spread, depth, impact estimates
   - REGIME: HMM states, VIX-derived, correlation regimes
   - CROSS_ASSET: Lead-lag, correlation, relative strength
   - DERIVED: Features computed from other features (ratios, differences)

3. FeatureDefinition:
   - feature_id: str (unique, immutable once registered)
   - name: str (human-readable)
   - data_class: DataClass
   - compute_fn: Callable (function that computes this feature from raw data)
   - status: FeatureStatus
   - added_version: str
   - deprecated_version: Optional[str]
   - correlation_with_core: Optional[float] (max correlation with any CORE feature)
   - metadata: Dict (arbitrary key-value for domain info)

4. FeatureRegistry:
   - _features: Dict[str, FeatureDefinition]
   - _schema_version: str (increments on any CORE status change)
   - _version_history: List[Tuple[str, datetime, str]]  # (version, timestamp, description)
   
   Methods:
   - register(feature_id, name, data_class, compute_fn, status) → FeatureDefinition
   - get_active_schema() → List[str]  # CORE + VALIDATED features, ordered
   - get_core_schema() → List[str]  # CORE only, ordered
   - get_schema_at_version(version: str) → List[str]  # historical schema
   - promote(feature_id, new_status) → None  # changes status, may increment version
   - deprecate(feature_id) → None  # sets DEPRECATED, records in version history
   - get_features_by_class(data_class: DataClass) → List[FeatureDefinition]
   
   Pre-populated with 60 CORE features matching Explorer v3.0's feature space:
   - 20 PRICE features (returns at multiple horizons, MACD, RSI, Bollinger, etc.)
   - 10 VOLUME features (VWAP, OBV, volume profile, cumulative delta, etc.)
   - 10 ORDER_FLOW features (OFI, bid/ask imbalance, trade aggression, etc.)
   - 8 VOLATILITY features (realized vol windows, ATR, Garman-Klass, etc.)
   - 6 MICROSTRUCTURE features (spread, depth, impact, etc.)
   - 6 REGIME features (HMM state, VIX level, correlation regime, etc.)

5. FeatureProjector:
   - __init__(registry: FeatureRegistry)
   
   Methods:
   - project(full_vector: np.ndarray, target_schema: List[str]) → np.ndarray
     Extracts relevant features from full vector for a given schema.
   
   - pad_to_full(partial_vector: np.ndarray, source_schema: List[str]) → np.ndarray
     FOR NON-CLUSTERING PURPOSES ONLY. Expands partial vector with zeros.
     NOTE: Do NOT use for Gate 6 HRP clustering — use schema_aware_distance instead.
   
   - schema_aware_distance(vec_a, vec_b, schema_a, schema_b) → float
     Computes cosine distance only over shared feature dimensions.
     Jaccard-weighted: normalizes by intersection/union of feature sets.
     Returns 1.0 (max distance) if fewer than 5 shared features.
     USE THIS for all Gate 6 HRP clustering comparisons.
     
     Implementation:
     shared = set(schema_a) & set(schema_b)
     if len(shared) < 5: return 1.0
     indices_a = [schema_a.index(f) for f in shared]
     indices_b = [schema_b.index(f) for f in shared]
     cos_dist = cosine_distance(vec_a[indices_a], vec_b[indices_b])
     coverage = len(shared) / len(set(schema_a) | set(schema_b))
     return cos_dist / coverage

6. FeatureMaturityPipeline:
   - MIN_STRATEGIES: int = 500  # strategies generated using feature for significance
   - MIN_MARGINAL_PASS_RATE: float = 1.0  # relative to CORE-only base rate
   - MAX_CORE_CORRELATION: float = 0.7  # functional, not raw
   - MIN_SHADOW_ENTRIES: int = 3  # strategies reaching shadow trading
   - MIN_SHADOW_PASSES: int = 1  # strategies passing transfer gate
   - TRIAL_DURATION_DAYS: int = 90
   - EXTENSION_DURATION_DAYS: int = 45  # one extension allowed
   
   Methods:
   - start_trial(feature_id: str) → TrialRecord
   - update_trial(feature_id, strategies_generated, hifa_passed, shadow_results) → None
   - evaluate_trial(feature_id) → MaturityDecision (PROMOTE / DEPRECATE / EXTEND)
   - compute_functional_correlation(feature_id, existing_core_features) → float
     Uses strategy-level performance vector correlation, not raw feature correlation.
     Measures: do strategies using this feature BEHAVE differently from those using
     each core feature? Correlation of Sharpe contribution vectors.
   
   Evaluation logic:
   - If strategies_generated < MIN_STRATEGIES and first_extension → EXTEND
   - If strategies_generated < MIN_STRATEGIES and already extended → DEPRECATE
   - If marginal_pass_rate < MIN_MARGINAL_PASS_RATE → DEPRECATE
   - If functional_correlation > MAX_CORE_CORRELATION → DEPRECATE (redundant)
   - If shadow_entries < MIN_SHADOW_ENTRIES → DEPRECATE
   - If shadow_passes < MIN_SHADOW_PASSES → DEPRECATE
   - If ALL checks pass → PROMOTE to CORE

7. ShadowReEvolution:
   - When schema version increments, scan production for strategies on old schema
   - For each high-performing legacy strategy:
     a. Spawn parallel evolutionary search seeded with its genome
     b. Target new schema version
     c. 60-day window to match or exceed original performance
     d. If matched: swap in new-schema variant
     e. If not matched: keep original, tag as LEGACY
     f. LEGACY strategies capped at 5% of portfolio risk budget collectively

TESTS (20 minimum):
   - test_registry_pre_populated_60_core: Verify 60 CORE features on init
   - test_active_schema_includes_validated: CORE + VALIDATED features
   - test_core_schema_excludes_validated: CORE only
   - test_promote_increments_version: Status change bumps schema version
   - test_deprecate_records_history: Deprecated feature tracked
   - test_projector_extract: Correct subset extracted from full vector
   - test_projector_schema_aware_distance: Shared-dimension distance works
   - test_projector_min_shared_features: Returns 1.0 if <5 shared
   - test_projector_coverage_normalization: Jaccard weighting correct
   - test_maturity_promote_all_pass: All criteria met → PROMOTE
   - test_maturity_deprecate_low_samples: <500 strategies → EXTEND then DEPRECATE
   - test_maturity_deprecate_redundant: Correlation >0.7 → DEPRECATE
   - test_maturity_extend_once_only: Second extension → DEPRECATE
   - test_functional_correlation_uses_performance: Not raw feature correlation
   - test_shadow_reevolution_spawns_search: Legacy strategies trigger re-evolution
   - test_legacy_risk_cap: LEGACY strategies capped at 5%
   - test_schema_version_history: Full history retrievable
   - test_feature_by_class: Correct features returned per DataClass
   - test_backward_compatible_60dim: 60-dim projections match v1.0 behavior
   - test_cross_version_distance: v1.0 vs v2.0 strategies compared correctly
```

---

## Phase 3: Anomaly Diagnostic & Gap Classification

### What We're Building

When existing strategies consistently miss profitable trades, the system currently has no way to diagnose *why*. The Anomaly Diagnostic uses a Random Forest classifier on missed-trade feature vectors to determine whether the miss was caused by a structural gap (existing features are sufficient but tree topologies can't express the needed logic) or a feature gap (the signal lives in data the system doesn't currently measure).

### Why This Matters: This Is the L0 Implementation

The Anomaly Diagnostic is the concrete engineering implementation of L0 gap detection. The validated L0 mechanisms — Ljung-Box tests, ensemble disagreement, compression failure — detect *that* a gap exists. The Anomaly Diagnostic produces a structured **AnomalySignature** that characterizes *where and when* the gap manifests. This signature is the output format for L0 — it transforms a binary gap-detected signal into a multi-dimensional gap profile that L1–L3 and the intervention protocol can act on.

The gap classification (structural vs. feature, based on `core_overlap` from the RF classifier) is the first step toward L1 category-level detection. Full L1 implementation extends this with formal category structure — not just "feature gap exists" but "the missing signal belongs to the frequency-domain decomposition category."

### Files: `diagnostics/anomaly_diagnostic.py`, `diagnostics/anomaly_signature.py`

**Session 3, Prompt 1:**

```
Create the diagnostics/ package with anomaly_diagnostic.py and anomaly_signature.py.

FILE 1: diagnostics/anomaly_signature.py

AnomalySignature class — the structured output of L0 gap detection:
   - anomaly_id: str (UUID)
   - discovery_date: datetime
   - missed_trades: List[TradeOpportunity]  # the raw missed-trade data
   
   Observable characteristics:
   - temporal_clustering: TemporalProfile
     - hour_distribution: Dict[int, float]  # probability mass per hour
     - day_of_week_distribution: Dict[int, float]
     - has_strong_pattern: bool (entropy < 0.7 of uniform = strong)
     - peak_hours: List[int]
   
   - regime_distribution: RegimeProfile
     - regime_counts: Dict[str, int]  # BULL/BEAR/RANGE
     - concentrated_at_transitions: bool (>40% of misses within 2 hours of regime change)
     - transition_types: List[Tuple[str, str]]  # (from_regime, to_regime) patterns
   
   - asset_concentration: AssetProfile
     - asset_counts: Dict[str, int]
     - dominant_asset: Optional[str] (if any asset > 50%)
     - skew_coefficient: float
   
   - volatility_context: VolatilityProfile
     - vol_at_miss: List[float]  # realized vol at each missed trade
     - vol_regime_at_miss: List[str]  # LOW/MEDIUM/HIGH/EXTREME
     - concentrated_in_regime: Optional[str] (if >60% in one vol regime)
   
   - preceding_market_pattern: PrecedingPattern
     - feature_vectors_30min_before: np.ndarray  # shape (n_misses, n_features)
     - umap_embedding: np.ndarray  # 2D embedding for cluster visualization
     - n_subclusters: int  # HDBSCAN on UMAP embedding
     - subcluster_labels: np.ndarray
   
   - lead_lag_structure: LeadLagProfile
     - cross_asset_correlations: Dict[str, float]  # corr with other assets before miss
     - leading_instruments: List[str]  # assets that moved before the miss
     - has_significant_leads: bool (any lead corr > 0.3 with p < 0.05)


FILE 2: diagnostics/anomaly_diagnostic.py

AnomalyDiagnostic class:
   - __init__(feature_registry: FeatureRegistry)
   
   Methods:
   
   1. collect_missed_trades(strategies: List[UnifiedStrategy], 
                            market_data: MarketData,
                            lookback_days: int = 30) → List[TradeOpportunity]
      - For each strategy, identify time periods where the strategy was flat (no position)
        but a profitable trade existed (forward return > 2x transaction cost within horizon)
      - Return the feature vectors at those missed entry points
   
   2. build_signature(missed_trades: List[TradeOpportunity]) → AnomalySignature
      - Extract all observable characteristics listed in AnomalySignature
      - Run UMAP + HDBSCAN on missed-trade feature vectors for subcluster detection
      - Compute lead-lag structure using Granger causality tests
      - Each subcluster in the UMAP embedding may indicate a different gap type
   
   3. classify_gap(missed_trades: List[TradeOpportunity]) → GapClassification
      - Build control set: sample equal number of "normal" conditions (strategy was active)
      - Train Random Forest (200 trees) on missed_trades (label=1) vs control (label=0)
      - Extract feature importances
      - Identify top 5 most important features
      - Compute core_overlap: fraction of top-5 features that are in CORE schema
      
      Returns GapClassification:
        - core_overlap: float (0.0 to 1.0)
        - gap_type: GapType enum:
          - STRUCTURAL if core_overlap >= 0.7
          - FEATURE if core_overlap <= 0.4
          - AMBIGUOUS if 0.4 < core_overlap < 0.7
        - top_features: List[Tuple[str, float]]  # (feature_id, importance)
        - rf_classifier: trained classifier (for potential tree-seed extraction)
        - baseline_auc: float  # AUC of the classifier (measures how distinguishable misses are)
        - missing_features: List[str]  # top features NOT in CORE schema
   
   4. diagnose(strategies, market_data, lookback_days=30) → DiagnosisResult
      - Orchestrates: collect → build_signature → classify_gap
      - Returns DiagnosisResult containing AnomalySignature + GapClassification
      - If GapType.STRUCTURAL: include RF decision boundaries as tree topology seeds
      - If GapType.FEATURE: include missing feature IDs for Feature Scout
      - If GapType.AMBIGUOUS: include both, tagged for sequential intervention

GapType enum: STRUCTURAL, FEATURE, AMBIGUOUS

DiagnosisResult:
   - signature: AnomalySignature
   - classification: GapClassification
   - recommended_action: str  # "structural_seeds" / "feature_scout" / "sequential_intervention"
   - tree_topology_seeds: Optional[List]  # RF splits converted to tree templates
   - missing_feature_candidates: Optional[List[str]]
   - confidence: float  # how distinguishable the misses are (RF AUC)

INTEGRATION WITH L0-L3:
   - L0: The AnomalySignature IS the L0 output — structured gap characterization
   - L1: GapClassification's top_features + DataClass mapping provides category-level info
     (e.g., top features all ORDER_FLOW → "missing signal is in order flow category")
   - L2-L3: Will be implemented in future phases, building on this foundation

TESTS (15 minimum):
   - test_missed_trade_collection: Identifies profitable gaps correctly
   - test_signature_temporal_clustering: Entropy-based pattern detection works
   - test_signature_regime_distribution: Transition concentration detected
   - test_signature_umap_subclusters: HDBSCAN finds meaningful subclusters
   - test_signature_lead_lag: Granger causality identifies leading instruments
   - test_classify_gap_structural: High core_overlap → STRUCTURAL
   - test_classify_gap_feature: Low core_overlap → FEATURE
   - test_classify_gap_ambiguous: Mid core_overlap → AMBIGUOUS
   - test_rf_feature_importance: Top features identified correctly
   - test_diagnosis_structural_includes_seeds: Tree topology seeds provided
   - test_diagnosis_feature_includes_candidates: Missing features listed
   - test_diagnosis_ambiguous_includes_both: Both seeds and candidates
   - test_control_set_balanced: Equal sizes for missed vs normal
   - test_auc_measures_distinguishability: Higher AUC = more distinguishable gap
   - test_l0_integration: AnomalySignature matches L0 output interface
```

---

## Phase 4: Sequential Intervention Protocol

### What We're Building

When the Anomaly Diagnostic classifies a gap as AMBIGUOUS (core_overlap between 0.4 and 0.7), intervening on both structure and features simultaneously creates a lethal attribution problem — you can never determine which intervention caused any subsequent improvement. The Sequential Intervention Protocol enforces structure-first intervention with a 45-day attribution window, then conditionally routes to feature investigation only if the structural fix was insufficient.

### Why This Matters

This protocol ensures clean separation between C1 interventions (adding new features = expanding primitives) and C2 interventions (adding new tree topologies = improving recombination). Without this separation, the Feature Maturity Pipeline evaluates new features on contaminated evidence — marginal pass rate improvements might reflect structural improvements, not feature value. Over time, this inflates the feature space with possibly-redundant dimensions, degrading the evolutionary search's efficiency permanently.

### File: `diagnostics/intervention_protocol.py`

**Session 4, Prompt 1:**

```
Create diagnostics/intervention_protocol.py implementing the Sequential Intervention Protocol.

CORE CLASSES:

1. InterventionType enum:
   - STRUCTURAL_SEEDS: inject tree topology seeds into evolutionary search
   - FEATURE_SCOUT: propose new features for maturity pipeline
   - CONDITIONAL_FEATURE_SCOUT: feature investigation contingent on structural results

2. InterventionPlan:
   - anomaly_id: str (links to AnomalySignature)
   - phase_1: InterventionType
   - phase_2: Optional[InterventionType]
   - attribution_window_days: int (0 for non-ambiguous, 45 for ambiguous)
   - confounded_tag: bool (if True, feature promotion threshold rises to 1.5x)
   - created_at: datetime
   - phase_1_started: Optional[datetime]
   - phase_1_completed: Optional[datetime]
   - phase_2_started: Optional[datetime]
   - status: PlanStatus (PENDING / PHASE_1_ACTIVE / AWAITING_ATTRIBUTION / 
                          PHASE_2_ACTIVE / COMPLETED / CANCELLED)

3. InterventionRouter:
   - AMBIGUOUS_LOW: float = 0.4
   - AMBIGUOUS_HIGH: float = 0.7
   - STRUCTURAL_WINDOW_DAYS: int = 45
   - REGIME_STABILITY_REQUIRED: bool = True  # pause window during regime transitions
   
   Methods:
   
   route(diagnosis: DiagnosisResult) → InterventionPlan:
     - core_overlap >= AMBIGUOUS_HIGH → structural seeds only, no window
     - core_overlap <= AMBIGUOUS_LOW → feature scout directly, no window
     - AMBIGUOUS_LOW < core_overlap < AMBIGUOUS_HIGH → sequential protocol:
       Phase 1: structural seeds
       Phase 2: conditional feature scout (only if phase 1 resolves < 70%)
       45-day attribution window between phases
   
   evaluate_phase_1(plan: InterventionPlan, 
                     original_anomaly_rate: float,
                     current_anomaly_rate: float) → Phase1Result:
     - resolution_rate = 1.0 - (current_anomaly_rate / original_anomaly_rate)
     - If resolution_rate > 0.70 → CLOSE (structural fix sufficient)
     - If resolution_rate < 0.30 → PROCEED_TO_FEATURE (primarily feature gap)
     - If 0.30 <= resolution_rate <= 0.70 → PROCEED_CONFOUNDED 
       (mixed, but investigate features with stricter threshold)
     - Returns Phase1Result with resolution_rate, decision, remaining_anomaly_signature
   
   check_regime_stability(plan: InterventionPlan, 
                          regime_at_start: str,
                          current_regime: str) → bool:
     - If current_regime != regime_at_start:
       PAUSE the attribution window (don't count these days)
       Resume only when original regime returns
     - Returns whether the window should be active (True) or paused (False)
     - Log regime transitions for post-hoc analysis

4. DirectedFeatureScout:
   - __init__(feature_registry: FeatureRegistry)
   
   Methods:
   
   search_from_signature(signature: AnomalySignature) → List[FeatureProposal]:
     - Uses anomaly signature to generate TARGETED search queries, not generic sweeps
     
     Three search channels:
     
     a. Event-driven search (triggered by temporal_clustering.has_strong_pattern):
        - If misses cluster at specific hours → scan event calendars for those hours
        - Propose event-driven features (economic release indicators, session boundaries)
     
     b. Regime-transition search (triggered by regime_distribution.concentrated_at_transitions):
        - Search literature for regime transition predictors
        - Convert transition_types to search terms
        - Propose transition-detection features
     
     c. Cross-asset search (triggered by lead_lag_structure.has_significant_leads):
        - Identify leading instruments not currently in registry data sources
        - Propose cross-asset features from untapped data streams
     
     Returns List[FeatureProposal] with:
       - feature_id: proposed ID
       - source_channel: which search channel found it
       - relevance_score: how well it matches the anomaly signature
       - compute_specification: description of how to compute the feature
   
   validate_fill(proposed_feature: FeatureProposal,
                  anomaly_set: AnomalySet,
                  baseline_auc: float) → bool:
     - Add proposed feature to missed-trade classification task
     - Re-train RF with augmented feature vector
     - If AUC improves by >= 0.08, the feature plausibly fills this specific gap
     - NOTE: This is a preliminary filter. Real validation happens through
       the Feature Maturity Pipeline (Phase 2) via strategy generation + HIFA
     - Critic caveat: classification AUC ≠ predictive trading utility
       This filter catches clearly irrelevant features, not marginal ones

5. StructuralSeedInjector:
   - extract_tree_seeds(rf_classifier: RandomForestClassifier) → List[TreeSeed]
     - Convert RF decision boundaries into HierarchicalGenome tree templates
     - These become initialization seeds for the evolutionary search
     - Tag seeds with source_anomaly_id for attribution tracking
   
   - inject_seeds(seeds: List[TreeSeed], generation_engine: EvolutionaryEngine) → None
     - Add seeds to the next generation's initial population
     - Seeds replace random initialization for 20% of the population

TESTS (12 minimum):
   - test_route_structural: High overlap → structural only
   - test_route_feature: Low overlap → feature scout only
   - test_route_ambiguous_sequential: Mid overlap → sequential protocol
   - test_phase_1_sufficient: >70% resolution → CLOSE
   - test_phase_1_insufficient: <30% resolution → PROCEED_TO_FEATURE
   - test_phase_1_mixed: 30-70% → PROCEED_CONFOUNDED with tag
   - test_confounded_tag_raises_threshold: Promotion requires 1.5x not 1.0x
   - test_regime_stability_pauses_window: Regime change pauses attribution clock
   - test_directed_scout_temporal: Temporal clustering → event search
   - test_directed_scout_regime: Transition clustering → regime search
   - test_directed_scout_crossasset: Lead-lag → cross-asset search
   - test_validate_fill_auc_threshold: AUC improvement >= 0.08 required
```

---

## Phase 5: Production Feedback Loop

### What We're Building

The feedback arrow from EMT production storage back to strategy generation becomes a real, three-channel mechanism. When a production strategy fails, three distinct pieces of information propagate backward, each targeting a different layer of the generation system: a failure signature for negative seeding (changes what gets generated), a structural autopsy (changes mutation operators), and a meta-learning signal (changes pipeline parameters).

### Why This Matters

Without this, the pipeline is open-loop — it can filter bad strategies increasingly well but can never generate better candidates. The feedback loop breaks through the pipeline's ceiling by making generation quality a function of production experience. Every failure teaches the system something: where not to search (failure archive), how to mutate differently (structural autopsy), and how fast edges decay (meta-learning).

### File: `feedback/failure_archive.py`, `feedback/structural_autopsy.py`, `feedback/meta_learning.py`

**Session 5, Prompt 1:**

```
Create the feedback/ package with three modules implementing the closed-loop feedback system.

FILE 1: feedback/failure_archive.py

FailureRecord:
   - strategy_id: str
   - genome: HierarchicalGenome
   - failure_date: datetime
   - failure_regime: str (BULL/BEAR/RANGE)
   - decay_type: GapType (STRUCTURAL/FEATURE/AMBIGUOUS)
   - anomaly_signature: AnomalySignature
   - feature_activation_vector: np.ndarray  # which features the strategy used
   - trade_signal_history: np.ndarray  # behavioral signature over backtest period
   - time_to_failure_days: int  # days from production entry to retirement trigger

FailureArchive:
   - _records: List[FailureRecord]
   - _max_records: int = 10000  # ring buffer, oldest evicted
   
   Methods:
   
   add(record: FailureRecord) → None
   
   penalty(candidate_genome: HierarchicalGenome, 
           current_regime: str) → float:
     - For each record in archive:
       a. Compute BEHAVIORAL similarity (correlation of trade signal histories)
          NOT structural similarity (feature activation vectors are too coarse)
          — Two strategies with identical feature sets but different tree topologies
            trade differently. Behavioral correlation captures this.
       b. If behavioral_similarity < 0.7: skip (too dissimilar)
       c. Time decay: exp(-days_since_failure / 120)  # 120-day half-life
       d. Regime match: regime_similarity(current_regime, failure_regime)
          regime_factor = 0.3 + 0.7 * regime_match  # floor at 0.3
       e. penalty = similarity * time_factor * regime_factor
     - Return max(penalties across all records)  # 0.0 to ~1.0
     - Applied to fitness: fitness *= (1.0 - 0.5 * penalty)
       (cap at 50% penalty to avoid completely killing similar strategies)
   
   get_failure_distribution() → FailureDistribution:
     - time_to_failure histogram
     - regime distribution at failure
     - gap_type distribution
     - Used by meta-learning module


FILE 2: feedback/structural_autopsy.py

StructuralAutopsy:
   - __init__(anomaly_diagnostic: AnomalyDiagnostic)
   
   Methods:
   
   analyze(strategy: UnifiedStrategy, 
           decay_period_trades: List[Trade]) → AutopsyResult:
     - Run the same AnomalyDiagnostic on the strategy's LOSING trades
       during its decay period (not all trades — specifically the losers)
     - Classify: structural decay or feature decay
     - If structural: extract anti-patterns from the tree topology
       (the tree structure that USED to work but no longer does)
     - If feature: queue anomaly signature for Feature Scout investigation
     
     Returns AutopsyResult:
       - gap_type: GapType
       - signature: AnomalySignature (of the decay period)
       - anti_templates: Optional[List[TreeTopology]]  # for structural decay
       - feature_investigation_priority: str  # "HIGH" for production failures

AntiTemplateInjector:
   - Stores anti-templates: tree topologies that recently failed
   - During evolutionary search, penalizes new candidates whose tree topology
     is too similar to an anti-template (cosine similarity on topology encoding > 0.8)
   - Anti-templates decay with 90-day half-life
   - This is the "how to mutate differently" channel


FILE 3: feedback/meta_learning.py

MetaLearningSignal:
   - __init__(failure_archive: FailureArchive)
   
   Methods:
   
   compute_characteristic_decay_timescale() → float:
     - Analyze time-to-failure distribution from failure archive
     - Compute median time-to-failure → market's characteristic non-stationarity timescale
     - If median is 50 days, the market's edge half-life is ~50 days
     - This DIRECTLY calibrates:
       a. Shadow trading minimum duration (currently fixed at 14 days — probably too short)
          Recommended: min(14, median_ttf * 0.3) days
       b. Edge decay detector's drift_var parameter (Phase 7)
          Calibrate so that genuine Sharpe decline from 1.0→0.0 over median_ttf days
          produces decay_probability > 0.7 within 0.7 * median_ttf days
       c. Failure archive time decay half-life (should match ~2x median_ttf)
   
   detect_bimodal_failure() → Optional[BimodalAnalysis]:
     - Test time-to-failure distribution for bimodality (Hartigan's dip test)
     - If bimodal: two distinct failure modes exist
       - Short-lived failures (median ~15 days): likely overfit strategies that 
         HIFA gates should have caught → tighten validation
       - Long-lived failures (median ~90 days): genuine edge decay from market evolution
         → improve retirement detection
     - Returns analysis with recommended gate adjustments
   
   get_pipeline_calibration() → PipelineCalibration:
     - Aggregates all meta-learning signals into recommended parameter changes
     - shadow_min_duration: int
     - drift_var: float (for edge decay detector)
     - archive_half_life: float
     - bimodal_gate_adjustments: Optional[Dict]
     - These are RECOMMENDATIONS — applied only after human review
       (meta-learning changes pipeline parameters, which is high-impact)

TESTS (18 minimum):
   - test_failure_archive_behavioral_similarity: Uses trade signals, not feature vectors
   - test_failure_archive_time_decay: Penalty decays with 120-day half-life
   - test_failure_archive_regime_amplifier: Same-regime penalties higher
   - test_failure_archive_penalty_cap: Never exceeds 50% fitness reduction
   - test_failure_archive_ring_buffer: Oldest records evicted at capacity
   - test_autopsy_structural: Losing trades in existing features → anti-templates
   - test_autopsy_feature: Losing trades need new features → queue investigation
   - test_anti_template_penalizes_similar: Similar topologies penalized
   - test_anti_template_decay: 90-day half-life on anti-templates
   - test_meta_decay_timescale: Correct median computed from distribution
   - test_meta_calibrates_shadow_duration: Shadow duration adjusts proportionally
   - test_meta_calibrates_drift_var: Drift variance matches decay timescale
   - test_meta_bimodal_detection: Hartigan dip test identifies bimodality
   - test_meta_bimodal_recommendations: Correct gate adjustments for each mode
   - test_pipeline_calibration_aggregates: All signals combined into one recommendation
   - test_feedback_loop_structural: Failure → autopsy → anti-template → generation avoids
   - test_feedback_loop_feature: Failure → autopsy → feature scout → investigation queued
   - test_full_loop_closes: Production failure information reaches generation layer
```

---

## Phase 6: Dynamic Engine Allocation (Thompson Sampling)

### What We're Building

The fixed 40/25/15/10/10 allocation across generation engines (evolutionary, GenAI, pattern, recombine, LSM) made sense before gap diagnostics existed. Now that the system can classify gaps, different gap types should favor different engines. But hard-switching weights based on the current gap diagnostic would overfit the meta-level allocation — the system would chase whichever engine produced the last success. Thompson sampling with gap-driven prior shifts provides the correct balance between exploitation and exploration at the meta-level.

### File: `generation/engine_allocator.py`

**Session 6, Prompt 1:**

```
Create generation/engine_allocator.py implementing Thompson sampling engine allocation.

CONSTANTS:

GAP_AFFINITY = {
    GapType.STRUCTURAL: {
        "evolutionary": 0.55, "genai": 0.15, "pattern": 0.10, 
        "recombine": 0.15, "lsm": 0.05
    },
    GapType.FEATURE: {
        "evolutionary": 0.20, "genai": 0.35, "pattern": 0.15, 
        "recombine": 0.05, "lsm": 0.25
    },
    GapType.PATTERN: {
        "evolutionary": 0.25, "genai": 0.10, "pattern": 0.40, 
        "recombine": 0.15, "lsm": 0.10
    },
    GapType.AMBIGUOUS: {
        "evolutionary": 0.35, "genai": 0.25, "pattern": 0.15, 
        "recombine": 0.15, "lsm": 0.10
    },
    GapType.UNKNOWN: {
        "evolutionary": 0.30, "genai": 0.25, "pattern": 0.15, 
        "recombine": 0.10, "lsm": 0.20
    },
}

CORE CLASSES:

1. EngineAllocator:
   - betas: Dict[str, List[float]]  # Beta(alpha, beta) per engine
     Initialize all at [2.0, 2.0] (weak uniform prior)
   
   - decay_rates: Dict[str, float]  # per-engine decay (NOT uniform)
     "evolutionary": 0.990  # 70-cycle half-life (highly regime-sensitive)
     "genai": 0.995         # 138-cycle half-life (moderate sensitivity)
     "pattern": 0.993       # 99-cycle half-life
     "recombine": 0.997     # 231-cycle half-life (recombination less regime-dependent)
     "lsm": 0.998           # 346-cycle half-life (least regime-sensitive)
   
   - exploration_floor: float = 0.08  # no engine below 8%
   - prior_bonus: float = 1.5  # strength of gap diagnostic influence
   
   Methods:
   
   allocate(gap_type: GapType, n_strategies: int = 1000) → Dict[str, int]:
     - Get affinity vector for current gap type
     - For each engine:
       a. Get current Beta params (alpha, beta)
       b. Add gap affinity * prior_bonus to alpha (prior shift)
       c. Sample from Beta(alpha_shifted, beta)
       d. Apply floor: max(sample, exploration_floor)
     - Normalize samples to sum to 1.0
     - Allocate integer strategy counts: int(weight * n_strategies)
     - Handle rounding to ensure total = n_strategies
   
   update(engine: str, n_generated: int, n_hifa_passed: int) → None:
     - Update Beta distribution: alpha += n_hifa_passed, beta += (n_generated - n_hifa_passed)
     - Apply per-engine decay: alpha *= decay_rate, beta *= decay_rate
     - This prevents permanent lock-in from historical success
   
   get_current_weights() → Dict[str, float]:
     - Return expected value of each engine's Beta distribution (alpha / (alpha + beta))
     - For monitoring/dashboard display
   
   get_allocation_history() → List[Dict]:
     - Return time series of allocations for analysis
   
   IMPORTANT DESIGN NOTES:
   - Gap diagnostic as PRIOR SHIFT, not deterministic override
     When diagnostic says "structural gap," it doesn't force 55% to evolutionary
     It makes evolutionary MORE LIKELY to be sampled heavily, but not guaranteed
     This hedges against diagnostic errors (misclassified gap types)
   
   - Per-engine decay rates reflect regime sensitivity
     Evolutionary search success is highly regime-dependent → faster decay (0.990)
     LSM success is less regime-dependent → slower decay (0.998)
     Uniform decay treats all engines as equally regime-sensitive, which they aren't
   
   - The 8% floor prevents starvation
     Even in a strong structural-gap period, LSM gets at least 80 strategies/day
     This maintains exploration of all generation methods

2. ExplorationBudgetManager:
   - exploration_fraction: float = 0.20  # 20% of daily budget for experimental features
   
   Methods:
   
   allocate_exploration(total_strategies: int, 
                        experimental_features: List[str]) → int:
     - Return number of strategies that MUST include at least one experimental feature
     - Default: 0.20 * total_strategies = 200 out of 1000
     - These strategies go through normal HIFA pipeline
     - Track per-feature metrics for the Feature Maturity Pipeline (Phase 2)
   
   get_feature_exploration_stats() → Dict[str, FeatureExplorationStats]:
     - Per experimental feature: strategies generated, HIFA passed, shadow results
     - Fed into FeatureMaturityPipeline.update_trial()

TESTS (10 minimum):
   - test_allocate_respects_floor: No engine below 8%
   - test_allocate_sums_to_total: Allocations sum to n_strategies
   - test_gap_affinity_shifts_prior: Structural gap increases evolutionary sampling
   - test_update_adjusts_beta: HIFA results update distributions correctly
   - test_decay_prevents_lockin: Old success decays over time
   - test_per_engine_decay_rates: Evolutionary decays faster than LSM
   - test_thompson_stochastic: Multiple allocate() calls produce variance
   - test_exploration_budget_20_percent: 200 of 1000 go to experimental features
   - test_exploration_stats_tracked: Per-feature metrics maintained
   - test_history_recorded: Allocation history retrievable
```

---

## Phase 7: Edge Decay Detection & Strategy Retirement

### What We're Building

Production strategies currently have two states: production and retired. The system needs to detect whether a strategy's poor performance reflects normal variance, regime suppression, or genuine edge decay — and respond differently to each. A Kalman filter models each strategy's true Sharpe as a drifting latent variable, producing calibrated posterior probabilities of edge decay.

### File: `production/edge_decay.py`, `production/retirement_manager.py`

**Session 7, Prompt 1:**

```
Create the production/ package with edge_decay.py and retirement_manager.py.

FILE 1: production/edge_decay.py

EdgeDecayDetector:
   - mu: float  # estimated true Sharpe (initialized from backtest Sharpe)
   - sigma2: float  # uncertainty in estimate (initialized at 0.1)
   - drift_var: float  # daily Sharpe drift variance
     DEFAULT 0.001, but calibrated from MetaLearningSignal:
     Set so that Sharpe decline 1.0→0.0 over median_ttf days
     produces decay_probability > 0.7 within 0.7 * median_ttf days
   - obs_var: float = 0.5  # daily return observation variance
   - regime_sharpe_history: Dict[str, List[float]]  # per-regime Sharpe observations
   
   Methods:
   
   update(daily_return: float, current_regime: str, 
          annualization: int = 252) → Tuple[float, float]:
     - Kalman prediction step: sigma2 += drift_var
     - Kalman update step:
       expected_return = mu / sqrt(annualization)
       residual = daily_return - expected_return
       kalman_gain = sigma2 / (sigma2 + obs_var)
       mu += kalman_gain * residual * sqrt(annualization)
       sigma2 *= (1 - kalman_gain)
     - Update regime_sharpe_history[current_regime].append(mu)
     - Return (mu, sigma2)
   
   decay_probability(threshold_sharpe: float = 0.3) → float:
     - P(true_sharpe < threshold | observations)
     - z = (threshold - mu) / sqrt(sigma2)
     - return norm.cdf(z)
   
   regime_conditioned_decay(current_regime: str) → float:
     - Get regime_sharpe_history for current regime and other regimes
     - If strategy is strong in OTHER regimes (avg > 0.5) but weak in current (<0.2):
       This is regime suppression, not decay → return 0.15 (low decay probability)
     - Otherwise: return standard decay_probability()
     
     This prevents false retirement of strategies that are merely in an adverse regime.
   
   calibrate_drift_var(median_ttf: float) → None:
     - Set drift_var so that the Kalman filter's response matches empirical decay speed
     - Use binary search: find drift_var where Sharpe decline 1.0→0.0 over median_ttf
       simulated days produces decay_prob > 0.7 at 0.7 * median_ttf days
     - This connects retirement sensitivity to the pipeline's experience of edge decay speed


FILE 2: production/retirement_manager.py

RetirementManager:
   - detectors: Dict[str, EdgeDecayDetector]  # one per production strategy
   - healthy_threshold: float = 0.3  # below this = healthy
   - warning_threshold: float = 0.7  # above this = retirement candidate
   - confirmation_days: int = 30  # days at reduced allocation before retirement
   - warning_allocation_fraction: float = 0.50  # 50% allocation in warning zone
   - confirmation_allocation_fraction: float = 0.25  # 25% during confirmation
   
   Strategy states (extends existing PRODUCTION/RETIRED):
   - HEALTHY: decay_probability < 0.3, full allocation
   - WARNING: 0.3 <= decay_probability < 0.7, 50% allocation
   - CONFIRMING_RETIREMENT: decay_probability >= 0.7, 25% allocation, 30-day timer
   - RETIRED: confirmed decay, removed from production
   
   Methods:
   
   daily_update(strategy_id: str, daily_return: float, 
                current_regime: str) → RetirementAction:
     - Update the strategy's EdgeDecayDetector
     - Get regime-conditioned decay probability
     
     Decision logic:
     
     If decay_prob < healthy_threshold:
       → CONTINUE at full allocation
       If was in WARNING: restore to HEALTHY (strategy recovered)
       If was in CONFIRMING_RETIREMENT: cancel retirement, restore to WARNING
     
     If healthy_threshold <= decay_prob < warning_threshold:
       → WARNING: reduce allocation to 50%
       Run regime_conditioned_decay check:
         If regime_suppressed: keep at WARNING, don't escalate
         If not regime_suppressed: log as genuine concern
     
     If decay_prob >= warning_threshold:
       If not already CONFIRMING: enter CONFIRMING_RETIREMENT
         Start 30-day confirmation window at 25% allocation
       If already CONFIRMING:
         If 30 days elapsed: → RETIRE
         If decay_prob drops below 0.5 during confirmation: cancel, back to WARNING
     
     Returns RetirementAction:
       - strategy_id
       - new_state
       - allocation_fraction
       - decay_probability
       - regime_conditioned
       - days_in_confirmation (if applicable)
   
   on_retirement(strategy_id: str, 
                  anomaly_diagnostic: AnomalyDiagnostic,
                  failure_archive: FailureArchive,
                  feature_scout: DirectedFeatureScout,
                  structural_injector: AntiTemplateInjector) → None:
     
     THIS IS WHERE THE FULL LOOP CLOSES:
     
     1. Run structural autopsy on strategy's decay-period losses
     2. Store failure record in failure archive (for negative seeding)
     3. If decay was FEATURE type: queue anomaly signature for Feature Scout (HIGH priority)
     4. If decay was STRUCTURAL type: add anti-template to structural injector
     5. Update meta-learning signal with time-to-failure data
     
     Every retirement generates actionable intelligence that improves the next generation cycle.

INTEGRATION NOTES:
   - The 30-day confirmation at 25% allocation is the asymmetry correction:
     Cost of premature retirement (lost future alpha) > cost of delayed retirement
     (temporary drawdown at reduced allocation)
   - At 25% allocation over 30 days, worst case = 0.25 * 30/252 ≈ 3% of annual risk
     This is acceptable insurance against killing a recovering strategy.
   - drift_var should be calibrated from MetaLearningSignal.compute_characteristic_decay_timescale()
     NOT set as a fixed constant.

TESTS (15 minimum):
   - test_kalman_update_tracks_sharpe: Mu converges toward true Sharpe
   - test_decay_probability_increases_on_losses: Consistent losses raise probability
   - test_regime_conditioned_suppression: Strong elsewhere + weak here = suppressed
   - test_healthy_full_allocation: Low decay → full allocation
   - test_warning_reduced_allocation: Mid decay → 50% allocation
   - test_confirming_25_percent: High decay → 25% allocation
   - test_confirmation_30_days: Full 30 days required to retire
   - test_recovery_cancels_retirement: Decay drops below 0.5 → back to WARNING
   - test_recovery_restores_healthy: Decay drops below 0.3 → HEALTHY
   - test_on_retirement_stores_failure: Failure record added to archive
   - test_on_retirement_structural_antitemplate: Structural decay → anti-template
   - test_on_retirement_feature_scout: Feature decay → investigation queued
   - test_on_retirement_meta_learning: Time-to-failure recorded
   - test_drift_var_calibration: Binary search converges to correct value
   - test_full_lifecycle: Strategy goes HEALTHY → WARNING → CONFIRMING → RETIRED
```

---

## Phase 8: Discovery Boundary Formalization

### What We're Building

The system needs to know what it can and cannot discover autonomously. This isn't philosophical — it directly affects how the pipeline allocates research resources between automated exploration and human domain expertise. The four-level discovery taxonomy formalizes the boundary between C2 search (fully automatable) and C1 expansion (requiring varying degrees of human input).

### Why This Matters for the Balloon Principle

The Discovery Boundary is the formal encoding of the balloon surface. Level 1 (recombination) and Level 2 (timescale) are inside the balloon — the pipeline explores them autonomously. Level 3 (novel computation) is on the balloon surface — the pipeline can characterize the gap but can't fill it without human engineering. Level 4 (novel data) is outside the balloon — the pipeline can at best identify temporal correlations with the missing signal. The `DiscoveryCapability` classifier makes this boundary explicit and actionable.

### File: `diagnostics/discovery_boundary.py`

**Session 8, Prompt 1:**

```
Create diagnostics/discovery_boundary.py implementing the Discovery Boundary Formalization.

CORE CLASSES:

1. DiscoveryLevel enum:
   RECOMBINATION = 1    # New combinations of existing features
   TIMESCALE = 2        # Existing streams at different temporal resolution
   NOVEL_COMPUTATION = 3  # Existing data, new transformation
   NOVEL_DATA = 4       # Entirely new data modality

2. CapabilityCategory enum:
   AUTONOMOUS = "autonomous"    # Pipeline finds AND implements
   DIRECTED = "directed_human"  # Pipeline identifies gap shape, human implements
   CREATIVE = "creative_human"  # Human imagines, pipeline validates

3. MAPPING (Level → Capability):
   RECOMBINATION → AUTONOMOUS
   TIMESCALE → AUTONOMOUS (IF timescale sweep configured, else DIRECTED)
   NOVEL_COMPUTATION → DIRECTED
   NOVEL_DATA → CREATIVE

4. DiscoveryBoundary:
   - __init__(feature_registry: FeatureRegistry, 
              computational_library: ComputationalLibrary)
   
   ComputationalLibrary tracks what transforms exist:
   - rolling_window_stats (mean, std, skew, kurtosis)
   - momentum_indicators (RSI, MACD, Bollinger, etc.)
   - order_flow_metrics (OFI, imbalance, aggression)
   - volatility_estimators (realized, Garman-Klass, Yang-Zhang)
   - regime_detectors (HMM, threshold-based)
   - Each transform is a Callable with input/output type signatures
   
   Methods:
   
   classify_feature(proposed_feature: FeatureProposal) → DiscoveryClassification:
     - Check: does it use existing data streams? 
       No → Level 4 (NOVEL_DATA) → CREATIVE
     - Check: does it use existing transforms from computational_library?
       Yes + existing streams → Level 1 (RECOMBINATION) → AUTONOMOUS
     - Check: does it use existing streams at a different timescale?
       Yes + timescale_sweep_configured → Level 2 (TIMESCALE) → AUTONOMOUS
       Yes + timescale_sweep_NOT_configured → Level 2 → DIRECTED
     - Check: does it use existing streams with novel computation?
       Yes → Level 3 (NOVEL_COMPUTATION) → DIRECTED
     
     Returns DiscoveryClassification:
       - level: DiscoveryLevel
       - capability: CapabilityCategory
       - rationale: str
       - autonomous_actions: List[str]  # what the pipeline can do automatically
       - human_actions: List[str]  # what requires human input
       - gap_signature: Optional[AnomalySignature]  # the gap that motivated this
   
   characterize_boundary_surface() → BoundarySurface:
     - Enumerate what the pipeline can currently explore autonomously:
       a. All pairwise combinations of CORE features (Level 1)
       b. All CORE features at all configured timescales (Level 2)
       c. All transforms in computational_library applied to all data streams
     - Compute: total_autonomous_space_size
     - Compare against: estimated total useful feature space (from literature)
     - Returns coverage estimate: autonomous_space / estimated_total
     - This quantifies how much of the relevant feature space the pipeline can search
   
   expand_boundary(addition_type: str, addition: Any) → BoundaryExpansion:
     - When a human adds a new transform to computational_library:
       Everything computable from existing streams + new transform becomes Level 1
     - When a human adds a new data stream:
       Everything computable from new stream + existing transforms becomes Level 1
     - Returns: description of what became AUTONOMOUS as a result
     - This is the "inflating the balloon" operation
   
   generate_human_research_brief(signature: AnomalySignature,
                                  classification: GapClassification) → ResearchBrief:
     - For gaps classified as DIRECTED or CREATIVE:
       Produce a structured document for human researchers containing:
       a. Quantitative description of the gap (when, where, how severe)
       b. What the pipeline has already tried (features used, structures tested)
       c. What kind of signal is missing (temporal profile, regime context)
       d. Suggested search directions (from DirectedFeatureScout)
       e. Priority level (based on estimated alpha opportunity)
     - This is the pipeline's most valuable non-strategy output:
       it tells humans exactly where to push the boundary outward

   INTEGRATION WITH L0-L3:
   - L0 (AnomalySignature) → feeds classify_feature and generate_human_research_brief
   - L1 (Category-level): GapClassification.top_features → DataClass mapping determines
     which category the missing feature likely belongs to
   - L2 (Topological): AnomalySignature.preceding_market_pattern.subcluster structure
     indicates the structural properties of the gap
   - L3 (Meta-pattern): MetaLearningSignal.detect_bimodal_failure() identifies 
     patterns across gaps, which BoundarySurface analysis quantifies

TESTS (10 minimum):
   - test_classify_recombination: Existing streams + existing transforms → Level 1
   - test_classify_timescale: Existing stream at new timescale → Level 2
   - test_classify_novel_computation: Existing stream + new transform → Level 3
   - test_classify_novel_data: New data stream → Level 4
   - test_capability_mapping: Each level maps to correct capability
   - test_boundary_surface_size: Computes autonomous space correctly
   - test_expand_boundary_transform: New transform expands Level 1 space
   - test_expand_boundary_data: New data stream expands Level 1 space
   - test_research_brief_contains_signature: Brief includes gap description
   - test_research_brief_contains_suggestions: Brief includes search directions
```

---

## Phase 9: Full Integration & Closed-Loop Validation

### What We're Building

Phase 9 connects all eight mechanisms into a single orchestrated pipeline, verifies the closed-loop feedback path, and runs end-to-end integration tests with synthetic market data to confirm the system learns from production failures.

### File: `orchestrator_v2.py`

**Session 9, Prompt 1:**

```
Create orchestrator_v2.py extending the existing orchestrator.py with v2.0 mechanisms.

UnifiedOrchestratorV2 (extends UnifiedOrchestrator):
   - All v1.0 functionality preserved
   
   New components:
   - feature_registry: FeatureRegistry
   - anomaly_diagnostic: AnomalyDiagnostic
   - intervention_router: InterventionRouter
   - failure_archive: FailureArchive
   - engine_allocator: EngineAllocator
   - retirement_manager: RetirementManager
   - discovery_boundary: DiscoveryBoundary
   - meta_learning: MetaLearningSignal
   
   New pipeline stages:
   
   1. Pre-generation diagnostic (weekly):
      - Run anomaly_diagnostic on production strategies
      - Classify gaps
      - Route to intervention protocol if needed
      - Feed gap type into engine_allocator
   
   2. Adaptive generation:
      - engine_allocator.allocate(gap_type) replaces fixed weights
      - 20% exploration budget for experimental features
      - failure_archive.penalty applied to candidate fitness
   
   3. Enhanced HIFA validation:
      - All v1.0 gates preserved
      - Gate 2: HSG complexity check (total nodes + state bus)
      - Gate 5: Combinatorial PBO on arbitration layer
      - Gate 6: schema_aware_distance for cross-version clustering
   
   4. Production monitoring (daily):
      - retirement_manager.daily_update for all production strategies
      - on_retirement triggers full feedback loop
   
   5. Meta-learning (monthly):
      - meta_learning.compute_characteristic_decay_timescale
      - Calibrate drift_var, shadow_min_duration, archive_half_life
      - Generate pipeline_calibration recommendations
   
   6. Discovery boundary assessment (quarterly):
      - discovery_boundary.characterize_boundary_surface
      - Generate human_research_briefs for DIRECTED/CREATIVE gaps
      - Report on boundary expansion since last assessment

   New method:
   
   async run_adaptive_pipeline(market_data, n_candidates=1000, regime=None) → PipelineResult:
     - Same interface as run_full_pipeline but uses adaptive mechanisms
     - Returns extended PipelineResult with:
       - gap_diagnostic: DiagnosisResult
       - engine_allocation: Dict[str, int]
       - retirement_actions: List[RetirementAction]
       - feedback_summary: FeedbackSummary
       - discovery_boundary_status: BoundarySurface

INTEGRATION TESTS (25 minimum):

Tests verifying the CLOSED LOOP:

   - test_production_failure_reaches_generation:
     1. Create strategy, push through pipeline to production
     2. Simulate performance decay
     3. Verify retirement triggers
     4. Verify failure record appears in failure archive
     5. Verify next generation cycle applies negative seeding penalty
   
   - test_structural_gap_produces_tree_seeds:
     1. Create synthetic missed-trade pattern explainable by existing features
     2. Run anomaly diagnostic → verify STRUCTURAL classification
     3. Verify tree topology seeds extracted
     4. Verify seeds injected into evolutionary search
   
   - test_feature_gap_produces_feature_proposals:
     1. Create synthetic missed-trade pattern requiring unknown features
     2. Run anomaly diagnostic → verify FEATURE classification
     3. Verify Feature Scout generates proposals
     4. Verify proposals enter Feature Maturity Pipeline
   
   - test_ambiguous_gap_sequential_protocol:
     1. Create synthetic gap at core_overlap = 0.55
     2. Verify Phase 1 structural intervention starts
     3. Simulate 45-day attribution window
     4. Verify Phase 2 conditional feature scout triggers
     5. Verify STRUCTURALLY_CONFOUNDED tag applied
   
   - test_engine_allocation_adapts_to_gap:
     1. Set gap type to STRUCTURAL
     2. Run allocation multiple times
     3. Verify evolutionary engine receives higher allocation on average
     4. Verify all engines stay above 8% floor
   
   - test_retirement_recovery:
     1. Push strategy to WARNING state
     2. Simulate regime change + recovery
     3. Verify strategy restored to HEALTHY
   
   - test_regime_conditioned_retirement:
     1. Create strategy strong in BULL, weak in BEAR
     2. Simulate BEAR regime
     3. Verify regime_conditioned_decay returns suppressed (not decayed)
   
   - test_meta_learning_calibrates_parameters:
     1. Populate failure archive with synthetic failure records
     2. Run meta-learning
     3. Verify drift_var calibration matches failure distribution
     4. Verify shadow_min_duration recommendation is reasonable
   
   - test_discovery_boundary_expands:
     1. Check boundary surface size
     2. Add new transform to computational library
     3. Check boundary surface size again
     4. Verify expansion occurred
   
   - test_backward_compatible:
     1. Run v1.0 pipeline (run_full_pipeline)
     2. Run v2.0 pipeline (run_adaptive_pipeline) with UNKNOWN gap type
     3. Verify v2.0 produces valid results when no gaps detected
     4. Forward path behavior matches v1.0

Additional tests for each component interaction (15 more covering edge cases,
error handling, concurrent operations, and data integrity across the pipeline).
```

---

## Success Criteria

| Phase | Criterion | Measurement |
|-------|-----------|-------------|
| 1 (HSG) | FLAT mode identical to v1.0 | Zero performance difference on existing strategies |
| 1 (HSG) | HSG strategies pass HIFA | At least 1% of HSG candidates survive 7 gates |
| 2 (Registry) | Cross-version clustering valid | No artificial clustering by schema version in Gate 6 |
| 2 (Registry) | Feature maturity pipeline functional | First experimental feature trial completes within 90 days |
| 3 (Diagnostic) | Gap classification accurate | >75% agreement with manually labeled gap types (on synthetic data) |
| 4 (Intervention) | Attribution clean | Structural-only interventions show measurable effect before feature investigation starts |
| 5 (Feedback) | Loop closes | Production failure information demonstrably changes next-cycle generation |
| 5 (Feedback) | Negative seeding works | Failure-similar candidates receive measurable fitness penalty |
| 6 (Thompson) | Allocation adapts | Engine weights shift meaningfully toward gap-appropriate engines |
| 6 (Thompson) | No engine starved | All engines maintain ≥8% allocation across 100 cycles |
| 7 (Decay) | Regime suppression detected | Strategies strong elsewhere but weak in current regime NOT retired |
| 7 (Decay) | Genuine decay detected | Strategies with collapsing Sharpe retired within 1.5x median_ttf |
| 8 (Boundary) | Taxonomy classifies correctly | 100% accuracy on synthetic feature proposals across all 4 levels |
| 8 (Boundary) | Research briefs actionable | Human reviewers rate briefs as "useful for directing research" |
| 9 (Integration) | Closed loop verified | End-to-end test: production failure → generation improvement in ≤2 cycles |
| 9 (Integration) | All tests pass | ≥150 tests across all phases, all green |

---

## Interface Contracts

### Between Phases

```python
# Phase 1 → Phase 3: HSG provides genome for structural analysis
HierarchicalGenome.get_decision_path(tick: int) → List[NodeID]
HierarchicalGenome.feature_ids → List[str]
HierarchicalGenome.structural_mode → StructuralMode

# Phase 2 → Phase 3: Registry provides schema for gap classification
FeatureRegistry.get_core_schema() → List[str]
FeatureRegistry.get_active_schema() → List[str]
FeatureProjector.schema_aware_distance(vec_a, vec_b, schema_a, schema_b) → float

# Phase 3 → Phase 4: Diagnostic provides classification for intervention routing
DiagnosisResult.classification.core_overlap → float
DiagnosisResult.classification.gap_type → GapType
DiagnosisResult.signature → AnomalySignature

# Phase 3 → Phase 5: Anomaly signatures feed failure archive
AnomalySignature (used by StructuralAutopsy and FailureArchive)

# Phase 5 → Phase 6: Gap type from feedback drives engine allocation
GapType → EngineAllocator.allocate(gap_type, n_strategies)

# Phase 5 → Phase 7: Meta-learning calibrates decay detection
MetaLearningSignal.compute_characteristic_decay_timescale() → float
EdgeDecayDetector.calibrate_drift_var(median_ttf: float)

# Phase 7 → Phase 5: Retirement triggers feedback
RetirementManager.on_retirement() → triggers FailureArchive.add() + StructuralAutopsy + FeatureScout

# Phase 3 → Phase 8: Anomaly signatures feed discovery boundary
AnomalySignature → DiscoveryBoundary.generate_human_research_brief()
```

### With v1.0 Components (unchanged interfaces)

```python
# Generation engines: same interface, different allocation
Engine.generate(n_candidates: int) → List[UnifiedStrategy]

# HIFA gates: same validation, enhanced inputs
HIFAGate.validate(strategy: UnifiedStrategy) → bool

# Shadow trading: same deployment, enhanced monitoring
DeploymentQueue.deploy(strategy: UnifiedStrategy) → ShadowSlot
TransferGate.evaluate(shadow_result: ShadowResult) → bool

# EMT: same storage, new retirement trigger
EMTProduction.add(strategy: UnifiedStrategy) → MerkleNode
EMTProduction.retire(strategy_id: str) → AuditRecord
```

---

## Troubleshooting Guide

### Phase 1: HSG

**Problem:** FULL-mode strategies consistently outperform FLAT in backtest but fail shadow trading.  
**Diagnosis:** Cross-timeframe overfitting — the arbitration layer is exploiting spurious correlations between sub-tree outputs.  
**Fix:** Verify the combinatorial PBO test on the arbitration layer is active. If degradation > 40% with shuffled inputs, the strategy is interaction-overfitted. Tighten the degradation threshold to 30%.

**Problem:** Evolution converges to FLAT mode, never exploring DUAL or FULL.  
**Diagnosis:** The fitness penalty for node concentration isn't strong enough to overcome the simpler-is-better gradient.  
**Fix:** Increase the concentration penalty coefficient from 0.3 to 0.5. Alternatively, reserve 10% of the population for structural mode diversity — force 10% of each generation to be non-FLAT.

### Phase 2: Feature Registry

**Problem:** All legacy (v1.0) strategies cluster together in Gate 6, separate from v2.0 strategies.  
**Diagnosis:** Schema-aware distance isn't active in Gate 6, or the minimum shared features threshold is too high.  
**Fix:** Verify Gate 6 calls `FeatureProjector.schema_aware_distance()` not standard cosine distance. Check that `min_shared` is set to 5, not higher.

### Phase 3: Anomaly Diagnostic

**Problem:** RF classifier shows AUC near 0.5 — can't distinguish missed trades from normal conditions.  
**Diagnosis:** Either the missed-trade definition is too loose (capturing noise) or the feature set genuinely contains no signal for these misses.  
**Fix:** First, tighten missed-trade definition (require forward return > 3x cost, not 2x). If AUC remains near 0.5, this IS a genuine signal — it means the gap is at Level 3 or 4, and no combination of existing features can explain the misses.

### Phase 5: Feedback Loop

**Problem:** Failure archive grows but generation quality doesn't improve.  
**Diagnosis:** Behavioral similarity threshold (0.7) may be too high — penalties never trigger because new candidates aren't similar enough to failed strategies.  
**Fix:** Lower threshold to 0.5 and verify that penalties are actually being applied (log penalty values during generation).

### Phase 7: Retirement

**Problem:** Too many strategies entering WARNING state simultaneously during volatility events.  
**Diagnosis:** The drift_var is too high, making the Kalman filter overreact to short-term volatility.  
**Fix:** Re-run calibration from MetaLearningSignal. If the failure archive is too small for reliable calibration, use the default 0.001 and increase gradually.

---

## Appendix A: Mapping Table — Critic's Mechanisms to C1/C2/L0-L3

| Critic's Mechanism | C1/C2 Layer | L0-L3 Level | Phase | Role |
|-------------------|-------------|-------------|-------|------|
| Hierarchical Strategy Graph | C2 expansion | — | 1 | Expands recombination expressiveness |
| Feature Registry | C1 vocabulary management | — | 2 | Makes primitive set explicit and versioned |
| Feature Maturity Pipeline | C1 expansion/contraction | — | 2 | Quantitative balloon growth/shrinkage |
| Schema-Aware Distance | — | — | 2 | Prevents cross-C1-version artifacts |
| Shadow Re-Evolution | C2 search under new C1 | — | 2 | Maintains strategies across C1 changes |
| AnomalySignature | L0 output | L0 | 3 | Structured gap characterization |
| RF Gap Classification | L0→L1 bridge | L0-L1 | 3 | Distinguishes C1 gap from C2 gap |
| DirectedFeatureScout | L1 implementation | L1 | 4 | Category-level gap → targeted search |
| Sequential Intervention | — | — | 4 | Clean C1/C2 attribution |
| Failure Archive | C2 search pruning | — | 5 | Avoids re-exploring failed C2 regions |
| Structural Autopsy | L0 applied to decay | L0 | 5 | Gap detection on retiring strategies |
| Anti-Template Injector | C2 mutation guidance | — | 5 | Directs C2 search away from failures |
| Meta-Learning Signal | — | L3 | 5 | Patterns across gaps (meta-pattern detection) |
| Thompson Sampling Allocator | — | — | 6 | Adapts C2 search method to gap type |
| Edge Decay Detector | C2 invalidation | — | 7 | Detects when C2 model no longer fits market |
| Regime-Conditioned Decay | — | — | 7 | Prevents false C2 invalidation |
| Discovery Level 1 (Recombination) | C2 search | — | 8 | Inside the balloon |
| Discovery Level 2 (Timescale) | C1 minor expansion | — | 8 | Balloon surface — latent C1 dimension |
| Discovery Level 3 (Novel Computation) | C1 computation gap | L0-L2 | 8 | Balloon surface — directed human action |
| Discovery Level 4 (Novel Data) | C1 fundamental expansion | L3+ | 8 | Outside the balloon — creative human action |
| Human Research Brief | — | All | 8 | Pipeline's most valuable non-strategy output |

---

## Appendix B: All Prompts Summary

| Session | Prompt | File(s) | Lines (est.) | Tests |
|---------|--------|---------|--------------|-------|
| 1.1 | HSG core | shared/hierarchical_genome.py | ~400 | 15 |
| 1.2 | HSG adapters | shared/adapters.py (update) | ~150 | 5 |
| 2.1 | Feature Registry | shared/feature_registry.py | ~500 | 20 |
| 3.1 | Anomaly Diagnostic | diagnostics/anomaly_diagnostic.py, anomaly_signature.py | ~450 | 15 |
| 4.1 | Intervention Protocol | diagnostics/intervention_protocol.py | ~400 | 12 |
| 5.1 | Feedback Loop | feedback/failure_archive.py, structural_autopsy.py, meta_learning.py | ~500 | 18 |
| 6.1 | Engine Allocator | generation/engine_allocator.py | ~250 | 10 |
| 7.1 | Edge Decay & Retirement | production/edge_decay.py, retirement_manager.py | ~400 | 15 |
| 8.1 | Discovery Boundary | diagnostics/discovery_boundary.py | ~350 | 10 |
| 9.1 | Integration | orchestrator_v2.py | ~300 | 25 |
| **Total** | **10 prompts** | **~15 files** | **~3,700** | **145+** |

### Updated Project Structure

```
EXPLORER PRIME/
├── shared/                         # Extended in Phase 1-2
│   ├── unified_strategy.py         # v1.0 (unchanged)
│   ├── hierarchical_genome.py      # NEW: HSG representation
│   ├── feature_registry.py         # NEW: Versioned feature management
│   ├── adapters.py                 # UPDATED: HSG adapters added
│   ├── features.py                 # v1.0 (unchanged, registry wraps this)
│   ├── constants.py                # v1.0 (unchanged)
│   └── tests/
│       ├── test_shared.py          # v1.0 (35 tests)
│       ├── test_hsg.py             # NEW (20 tests)
│       └── test_registry.py        # NEW (20 tests)
│
├── diagnostics/                    # NEW package (Phases 3-4, 8)
│   ├── __init__.py
│   ├── anomaly_signature.py        # L0 output format
│   ├── anomaly_diagnostic.py       # RF-based gap classification
│   ├── intervention_protocol.py    # Sequential intervention routing
│   ├── discovery_boundary.py       # Four-level discovery taxonomy
│   └── tests/
│       ├── test_diagnostic.py      # (15 tests)
│       ├── test_intervention.py    # (12 tests)
│       └── test_boundary.py        # (10 tests)
│
├── feedback/                       # NEW package (Phase 5)
│   ├── __init__.py
│   ├── failure_archive.py          # Behavioral negative seeding
│   ├── structural_autopsy.py       # Decay cause analysis
│   ├── meta_learning.py            # Pipeline self-calibration
│   └── tests/
│       └── test_feedback.py        # (18 tests)
│
├── generation/                     # NEW package (Phase 6)
│   ├── __init__.py
│   ├── engine_allocator.py         # Thompson sampling allocation
│   └── tests/
│       └── test_allocator.py       # (10 tests)
│
├── production/                     # NEW package (Phase 7)
│   ├── __init__.py
│   ├── edge_decay.py               # Kalman filter decay detection
│   ├── retirement_manager.py       # Three-threshold retirement
│   └── tests/
│       └── test_retirement.py      # (15 tests)
│
├── forward_testing/                # v1.0 (unchanged)
│   └── ...
│
├── emt/                            # v1.0 (unchanged)
│   └── ...
│
├── dashboard/                      # v1.0 (unchanged, future: add decay dashboards)
│   └── ...
│
├── tests/
│   ├── test_integration.py         # v1.0 (21 tests)
│   └── test_integration_v2.py      # NEW (25 tests)
│
├── orchestrator.py                 # v1.0 (unchanged)
├── orchestrator_v2.py              # NEW: Adaptive pipeline coordinator
├── UNIFIED_SYSTEM_GUIDE.md         # v1.0 reference
└── README.md                       # UPDATE with v2.0 architecture
```

---

*This guide is standalone and self-contained. Each phase prompt can be executed in a Claude Code session without requiring additional architectural documents. The prompts include all interface contracts, data structures, integration points, and test specifications needed for complete implementation.*
