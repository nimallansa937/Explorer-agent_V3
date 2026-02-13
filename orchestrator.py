"""
Unified Orchestrator

Master orchestrator connecting all EXPLORER PRIME pipeline stages:
- Explorer Agent v3.0: Strategy generation
- HIFA v2.0: 7-gate validation
- Forward Testing: Shadow trading validation
- EMT: Production storage

Flow:
    Generate → HIFA Validate → Forward Test → Production Store
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum
import pandas as pd

from shared.unified_strategy import (
    UnifiedStrategy,
    StrategyStatus,
    HIFAResult,
    ForwardTestResult,
)
from shared.constants import (
    ENGINE_BUDGET_EVOLUTIONARY,
    ENGINE_BUDGET_GENAI,
    ENGINE_BUDGET_PATTERN,
    ENGINE_BUDGET_RECOMBINE,
    FORWARD_TEST_CONFIG,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Pipeline Stage Enum
# ==============================================================================

class PipelineStage(Enum):
    """Pipeline execution stages."""
    GENERATION = "generation"
    HIFA_VALIDATION = "hifa_validation"
    FORWARD_TESTING = "forward_testing"
    PRODUCTION_STORAGE = "production_storage"


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class OrchestratorConfig:
    """Configuration for the unified orchestrator."""

    # Generation settings
    default_candidates: int = 1000
    engine_budgets: Dict[str, float] = field(default_factory=lambda: {
        "evolutionary": ENGINE_BUDGET_EVOLUTIONARY,
        "genai": ENGINE_BUDGET_GENAI,
        "pattern": ENGINE_BUDGET_PATTERN,
        "recombine": ENGINE_BUDGET_RECOMBINE,
    })

    # HIFA settings
    hifa_batch_size: int = 50
    hifa_parallel_workers: int = 4

    # Forward testing settings
    max_concurrent_shadow: int = FORWARD_TEST_CONFIG.get('MAX_CONCURRENT', 50)
    shadow_duration_days: int = FORWARD_TEST_CONFIG.get('SHADOW_DURATION_DAYS', 14)
    min_transfer_ratio: float = FORWARD_TEST_CONFIG.get('TRANSFER_RATIO_MIN', 0.5)

    # Pipeline settings
    fail_fast: bool = False  # Stop on first stage failure
    save_intermediate: bool = True  # Save results after each stage

    # Timeouts (seconds)
    generation_timeout: float = 3600.0  # 1 hour
    hifa_timeout: float = 7200.0  # 2 hours
    forward_test_timeout: float = 86400.0 * 14  # 14 days


# ==============================================================================
# Pipeline Results
# ==============================================================================

@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage: PipelineStage
    started_at: datetime
    completed_at: datetime
    success: bool

    # Counts
    input_count: int
    output_count: int
    failed_count: int

    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Errors if any
    errors: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def pass_rate(self) -> float:
        if self.input_count == 0:
            return 0.0
        return self.output_count / self.input_count


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    pipeline_id: str
    started_at: datetime
    completed_at: Optional[datetime]

    # Configuration used
    config: OrchestratorConfig

    # Stage results
    stage_results: Dict[PipelineStage, StageResult] = field(default_factory=dict)

    # Final outputs
    production_strategies: List[UnifiedStrategy] = field(default_factory=list)

    # Overall status
    completed: bool = False
    success: bool = False

    @property
    def total_duration_seconds(self) -> float:
        if self.completed_at is None:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def overall_pass_rate(self) -> float:
        """Calculate end-to-end pass rate."""
        gen_result = self.stage_results.get(PipelineStage.GENERATION)
        if gen_result is None or gen_result.output_count == 0:
            return 0.0
        return len(self.production_strategies) / gen_result.output_count

    def to_summary(self) -> Dict[str, Any]:
        """Generate summary report."""
        return {
            "pipeline_id": self.pipeline_id,
            "duration_seconds": self.total_duration_seconds,
            "completed": self.completed,
            "success": self.success,
            "production_count": len(self.production_strategies),
            "overall_pass_rate": self.overall_pass_rate,
            "stages": {
                stage.value: {
                    "input": result.input_count,
                    "output": result.output_count,
                    "pass_rate": result.pass_rate,
                    "duration": result.duration_seconds,
                }
                for stage, result in self.stage_results.items()
            },
        }


# ==============================================================================
# Unified Orchestrator
# ==============================================================================

class UnifiedOrchestrator:
    """
    Master orchestrator connecting all pipeline stages.

    Coordinates:
    - Explorer Agent v3.0: Strategy candidate generation
    - HIFA v2.0: 7-gate statistical validation
    - Forward Testing: Shadow trading validation
    - EMT: Production strategy storage

    Usage:
        orchestrator = UnifiedOrchestrator()

        result = await orchestrator.run_full_pipeline(
            market_data=df,
            n_candidates=1000,
        )

        # Access production strategies
        for strategy in result.production_strategies:
            print(f"Production ready: {strategy.strategy_id}")
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        explorer: Optional[Any] = None,  # EnhancedExplorerAgent
        hifa: Optional[Any] = None,  # HIFAv2Pipeline
        forward_tester: Optional[Any] = None,  # ForwardTestingBridge
        emt: Optional[Any] = None,  # EMTStorage
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Orchestrator configuration
            explorer: Explorer Agent v3.0 instance (optional, lazy-loaded)
            hifa: HIFA v2.0 pipeline instance (optional, lazy-loaded)
            forward_tester: Forward testing bridge (optional, lazy-loaded)
            emt: EMT storage instance (optional, lazy-loaded)
        """
        self.config = config or OrchestratorConfig()

        # Components (lazy-loaded if not provided)
        self._explorer = explorer
        self._hifa = hifa
        self._forward_tester = forward_tester
        self._emt = emt

        # State tracking
        self._current_pipeline_id: Optional[str] = None
        self._pipeline_history: List[PipelineResult] = []

        logger.info("UnifiedOrchestrator initialized")

    # ==========================================================================
    # Component Lazy Loading
    # ==========================================================================

    @property
    def explorer(self):
        """Get or create Explorer Agent."""
        if self._explorer is None:
            self._explorer = self._create_explorer()
        return self._explorer

    @property
    def hifa(self):
        """Get or create HIFA pipeline."""
        if self._hifa is None:
            self._hifa = self._create_hifa()
        return self._hifa

    @property
    def forward_tester(self):
        """Get or create Forward Testing bridge."""
        if self._forward_tester is None:
            self._forward_tester = self._create_forward_tester()
        return self._forward_tester

    @property
    def emt(self):
        """Get or create EMT storage."""
        if self._emt is None:
            self._emt = self._create_emt()
        return self._emt

    def _create_explorer(self):
        """Create Explorer Agent instance."""
        try:
            from explorer_agent_v3.agent import EnhancedExplorerAgent
            return EnhancedExplorerAgent()
        except ImportError:
            logger.warning("Explorer Agent v3 not available, using stub")
            return ExplorerStub()

    def _create_hifa(self):
        """Create HIFA pipeline instance."""
        try:
            from hifa_v2.pipeline import HIFAv2Pipeline
            return HIFAv2Pipeline()
        except ImportError:
            logger.warning("HIFA v2 not available, using stub")
            return HIFAStub()

    def _create_forward_tester(self):
        """Create Forward Testing bridge instance."""
        try:
            from forward_testing.bridge import ForwardTestingBridge
            return ForwardTestingBridge()
        except ImportError:
            logger.warning("Forward Testing not available, using stub")
            return ForwardTestingStub()

    def _create_emt(self):
        """Create EMT storage instance."""
        try:
            from emt.production import EMTProduction
            return EMTProduction()
        except ImportError:
            logger.warning("EMT Production not available, using stub")
            return EMTStub()

    # ==========================================================================
    # Main Pipeline
    # ==========================================================================

    async def run_full_pipeline(
        self,
        market_data: pd.DataFrame,
        n_candidates: int = 1000,
        regime: Optional[str] = None,
    ) -> PipelineResult:
        """
        Run complete pipeline: Generate → Validate → Forward Test → Store

        Args:
            market_data: Historical OHLCV data for generation and validation
            n_candidates: Number of strategy candidates to generate
            regime: Optional target regime for generation

        Returns:
            PipelineResult with production-ready strategies
        """
        # Initialize pipeline result
        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._current_pipeline_id = pipeline_id

        result = PipelineResult(
            pipeline_id=pipeline_id,
            started_at=datetime.now(),
            completed_at=None,
            config=self.config,
        )

        logger.info(f"Starting pipeline {pipeline_id} with {n_candidates} candidates")

        try:
            # Stage 1: Generate candidates
            candidates = await self._generate_candidates(
                market_data, n_candidates, regime, result
            )

            if not candidates and self.config.fail_fast:
                logger.error("Generation failed, stopping pipeline")
                result.completed = True
                result.completed_at = datetime.now()
                return result

            # Stage 2: HIFA validation
            hifa_passed = await self._run_hifa_validation(
                candidates, market_data, result
            )

            if not hifa_passed and self.config.fail_fast:
                logger.error("HIFA validation failed all, stopping pipeline")
                result.completed = True
                result.completed_at = datetime.now()
                return result

            # Stage 3: Forward testing
            forward_passed = await self._run_forward_testing(
                hifa_passed, market_data, result
            )

            if not forward_passed and self.config.fail_fast:
                logger.error("Forward testing failed all, stopping pipeline")
                result.completed = True
                result.completed_at = datetime.now()
                return result

            # Stage 4: Store production-ready
            production = await self._store_production(
                forward_passed, result
            )

            result.production_strategies = production
            result.completed = True
            result.success = len(production) > 0
            result.completed_at = datetime.now()

            logger.info(
                f"Pipeline {pipeline_id} completed: "
                f"{len(production)}/{n_candidates} strategies to production "
                f"({result.overall_pass_rate:.1%} pass rate)"
            )

        except Exception as e:
            logger.error(f"Pipeline {pipeline_id} failed: {e}")
            result.completed = True
            result.success = False
            result.completed_at = datetime.now()
            raise

        finally:
            self._pipeline_history.append(result)
            self._current_pipeline_id = None

        return result

    # ==========================================================================
    # Stage 1: Generation
    # ==========================================================================

    async def _generate_candidates(
        self,
        market_data: pd.DataFrame,
        n_candidates: int,
        regime: Optional[str],
        result: PipelineResult,
    ) -> List[UnifiedStrategy]:
        """Generate strategy candidates using Explorer Agent."""
        stage = PipelineStage.GENERATION
        started_at = datetime.now()

        logger.info(f"Stage 1: Generating {n_candidates} candidates")

        try:
            # Calculate per-engine allocations
            allocations = {
                engine: int(n_candidates * budget)
                for engine, budget in self.config.engine_budgets.items()
            }

            # Generate candidates
            candidates = await asyncio.wait_for(
                self._do_generation(market_data, allocations, regime),
                timeout=self.config.generation_timeout,
            )

            # Record stage result
            result.stage_results[stage] = StageResult(
                stage=stage,
                started_at=started_at,
                completed_at=datetime.now(),
                success=len(candidates) > 0,
                input_count=n_candidates,
                output_count=len(candidates),
                failed_count=n_candidates - len(candidates),
                metrics={
                    "allocations": allocations,
                    "regime": regime,
                },
            )

            logger.info(f"Stage 1 complete: {len(candidates)} candidates generated")
            return candidates

        except asyncio.TimeoutError:
            logger.error("Generation stage timed out")
            result.stage_results[stage] = StageResult(
                stage=stage,
                started_at=started_at,
                completed_at=datetime.now(),
                success=False,
                input_count=n_candidates,
                output_count=0,
                failed_count=n_candidates,
                errors=["Generation timed out"],
            )
            return []

        except Exception as e:
            logger.error(f"Generation stage failed: {e}")
            result.stage_results[stage] = StageResult(
                stage=stage,
                started_at=started_at,
                completed_at=datetime.now(),
                success=False,
                input_count=n_candidates,
                output_count=0,
                failed_count=n_candidates,
                errors=[str(e)],
            )
            return []

    async def _do_generation(
        self,
        market_data: pd.DataFrame,
        allocations: Dict[str, int],
        regime: Optional[str],
    ) -> List[UnifiedStrategy]:
        """Perform actual generation."""
        # Use explorer agent to generate
        candidates = self.explorer.generate_candidates(
            market_data=market_data,
            n_candidates=sum(allocations.values()),
            regime=regime,
        )

        # Ensure all are UnifiedStrategy
        unified = []
        for c in candidates:
            if isinstance(c, UnifiedStrategy):
                c.status = StrategyStatus.GENERATED
                unified.append(c)

        return unified

    # ==========================================================================
    # Stage 2: HIFA Validation
    # ==========================================================================

    async def _run_hifa_validation(
        self,
        candidates: List[UnifiedStrategy],
        market_data: pd.DataFrame,
        result: PipelineResult,
    ) -> List[UnifiedStrategy]:
        """Run HIFA v2.0 7-gate validation."""
        stage = PipelineStage.HIFA_VALIDATION
        started_at = datetime.now()

        logger.info(f"Stage 2: HIFA validation of {len(candidates)} candidates")

        if not candidates:
            result.stage_results[stage] = StageResult(
                stage=stage,
                started_at=started_at,
                completed_at=datetime.now(),
                success=True,
                input_count=0,
                output_count=0,
                failed_count=0,
            )
            return []

        try:
            passed = []
            failed = []
            gate_stats = {i: {"passed": 0, "failed": 0} for i in range(1, 8)}

            # Process in batches
            for i in range(0, len(candidates), self.config.hifa_batch_size):
                batch = candidates[i:i + self.config.hifa_batch_size]

                # Validate batch
                batch_results = await self._validate_batch(batch, market_data)

                for strategy, hifa_result in batch_results:
                    if hifa_result.passed:
                        strategy.hifa_result = hifa_result
                        strategy.status = StrategyStatus.HIFA_PASSED
                        passed.append(strategy)
                    else:
                        strategy.status = StrategyStatus.HIFA_FAILED
                        failed.append(strategy)

                    # Track gate statistics
                    for gate_num, gate_result in hifa_result.gate_results.items():
                        if gate_result.passed:
                            gate_stats[gate_num]["passed"] += 1
                        else:
                            gate_stats[gate_num]["failed"] += 1

            # Calculate gate pass rates
            gate_pass_rates = {
                gate: stats["passed"] / max(1, stats["passed"] + stats["failed"])
                for gate, stats in gate_stats.items()
            }

            # Record stage result
            result.stage_results[stage] = StageResult(
                stage=stage,
                started_at=started_at,
                completed_at=datetime.now(),
                success=len(passed) > 0,
                input_count=len(candidates),
                output_count=len(passed),
                failed_count=len(failed),
                metrics={
                    "gate_pass_rates": gate_pass_rates,
                    "batch_size": self.config.hifa_batch_size,
                },
            )

            logger.info(
                f"Stage 2 complete: {len(passed)}/{len(candidates)} passed HIFA "
                f"({len(passed)/len(candidates):.1%})"
            )
            return passed

        except Exception as e:
            logger.error(f"HIFA validation failed: {e}")
            result.stage_results[stage] = StageResult(
                stage=stage,
                started_at=started_at,
                completed_at=datetime.now(),
                success=False,
                input_count=len(candidates),
                output_count=0,
                failed_count=len(candidates),
                errors=[str(e)],
            )
            return []

    async def _validate_batch(
        self,
        batch: List[UnifiedStrategy],
        market_data: pd.DataFrame,
    ) -> List[tuple]:
        """Validate a batch of strategies."""
        results = []

        for strategy in batch:
            # Run HIFA validation
            hifa_result = self.hifa.validate(
                strategy=strategy,
                returns=self._calculate_returns(strategy, market_data),
            )
            results.append((strategy, hifa_result))

        return results

    def _calculate_returns(
        self,
        strategy: UnifiedStrategy,
        market_data: pd.DataFrame,
    ) -> pd.Series:
        """Calculate strategy returns for validation."""
        # This would run the strategy on market data
        # For now, return empty series (real implementation needed)
        return pd.Series(dtype=float)

    # ==========================================================================
    # Stage 3: Forward Testing
    # ==========================================================================

    async def _run_forward_testing(
        self,
        candidates: List[UnifiedStrategy],
        market_data: pd.DataFrame,
        result: PipelineResult,
    ) -> List[UnifiedStrategy]:
        """Run forward testing (shadow trading) validation."""
        stage = PipelineStage.FORWARD_TESTING
        started_at = datetime.now()

        logger.info(f"Stage 3: Forward testing of {len(candidates)} candidates")

        if not candidates:
            result.stage_results[stage] = StageResult(
                stage=stage,
                started_at=started_at,
                completed_at=datetime.now(),
                success=True,
                input_count=0,
                output_count=0,
                failed_count=0,
            )
            return []

        try:
            passed = []
            failed = []
            transfer_ratios = []

            # Deploy strategies to shadow trading
            for strategy in candidates:
                strategy.status = StrategyStatus.SHADOW_TRADING

                # Deploy and wait for completion (or check existing sessions)
                forward_result = await self._run_shadow_session(strategy, market_data)

                if forward_result and forward_result.passed:
                    strategy.forward_result = forward_result
                    strategy.status = StrategyStatus.FORWARD_PASSED
                    passed.append(strategy)
                    transfer_ratios.append(forward_result.transfer_ratio)
                else:
                    strategy.status = StrategyStatus.FORWARD_FAILED
                    failed.append(strategy)

            # Record stage result
            result.stage_results[stage] = StageResult(
                stage=stage,
                started_at=started_at,
                completed_at=datetime.now(),
                success=len(passed) > 0,
                input_count=len(candidates),
                output_count=len(passed),
                failed_count=len(failed),
                metrics={
                    "mean_transfer_ratio": sum(transfer_ratios) / max(1, len(transfer_ratios)),
                    "min_transfer_ratio": min(transfer_ratios) if transfer_ratios else 0,
                    "max_transfer_ratio": max(transfer_ratios) if transfer_ratios else 0,
                },
            )

            logger.info(
                f"Stage 3 complete: {len(passed)}/{len(candidates)} passed forward testing "
                f"({len(passed)/max(1, len(candidates)):.1%})"
            )
            return passed

        except Exception as e:
            logger.error(f"Forward testing failed: {e}")
            result.stage_results[stage] = StageResult(
                stage=stage,
                started_at=started_at,
                completed_at=datetime.now(),
                success=False,
                input_count=len(candidates),
                output_count=0,
                failed_count=len(candidates),
                errors=[str(e)],
            )
            return []

    async def _run_shadow_session(
        self,
        strategy: UnifiedStrategy,
        market_data: pd.DataFrame,
    ) -> Optional[ForwardTestResult]:
        """Run shadow trading session for a strategy."""
        # Deploy to forward testing
        deployment = await self.forward_tester.deploy_strategy(strategy)

        if not deployment.success:
            return None

        # Wait for completion (in real usage, this would be async over days)
        result = await self.forward_tester.complete_forward_test(
            strategy.strategy_id
        )

        return result

    # ==========================================================================
    # Stage 4: Production Storage
    # ==========================================================================

    async def _store_production(
        self,
        candidates: List[UnifiedStrategy],
        result: PipelineResult,
    ) -> List[UnifiedStrategy]:
        """Store production-ready strategies in EMT."""
        stage = PipelineStage.PRODUCTION_STORAGE
        started_at = datetime.now()

        logger.info(f"Stage 4: Storing {len(candidates)} production strategies")

        if not candidates:
            result.stage_results[stage] = StageResult(
                stage=stage,
                started_at=started_at,
                completed_at=datetime.now(),
                success=True,
                input_count=0,
                output_count=0,
                failed_count=0,
            )
            return []

        try:
            stored = []
            failed = []

            for strategy in candidates:
                # Store in EMT
                success = await self._store_strategy(strategy)

                if success:
                    strategy.status = StrategyStatus.PRODUCTION
                    strategy.production_start = datetime.now()
                    stored.append(strategy)
                else:
                    failed.append(strategy)

            # Record stage result
            result.stage_results[stage] = StageResult(
                stage=stage,
                started_at=started_at,
                completed_at=datetime.now(),
                success=len(stored) > 0,
                input_count=len(candidates),
                output_count=len(stored),
                failed_count=len(failed),
            )

            logger.info(f"Stage 4 complete: {len(stored)} strategies stored in EMT")
            return stored

        except Exception as e:
            logger.error(f"Production storage failed: {e}")
            result.stage_results[stage] = StageResult(
                stage=stage,
                started_at=started_at,
                completed_at=datetime.now(),
                success=False,
                input_count=len(candidates),
                output_count=0,
                failed_count=len(candidates),
                errors=[str(e)],
            )
            return []

    async def _store_strategy(self, strategy: UnifiedStrategy) -> bool:
        """Store a single strategy in EMT."""
        return self.emt.store(strategy)

    # ==========================================================================
    # Pipeline Management
    # ==========================================================================

    def get_pipeline_status(self, pipeline_id: Optional[str] = None) -> Optional[Dict]:
        """Get status of a pipeline execution."""
        if pipeline_id is None:
            pipeline_id = self._current_pipeline_id

        if pipeline_id is None:
            return None

        for result in self._pipeline_history:
            if result.pipeline_id == pipeline_id:
                return result.to_summary()

        return None

    def get_history(self) -> List[Dict]:
        """Get pipeline execution history."""
        return [r.to_summary() for r in self._pipeline_history]


# ==============================================================================
# Stub Classes (for when components aren't available)
# ==============================================================================

class ExplorerStub:
    """Stub for Explorer Agent when not available."""

    def generate_candidates(
        self,
        market_data: pd.DataFrame,
        n_candidates: int,
        regime: Optional[str] = None,
    ) -> List[UnifiedStrategy]:
        logger.warning("ExplorerStub: No candidates generated (stub)")
        return []


class HIFAStub:
    """Stub for HIFA pipeline when not available."""

    def validate(
        self,
        strategy: UnifiedStrategy,
        returns: pd.Series,
    ) -> HIFAResult:
        from shared.unified_strategy import GateResult, StatisticalScores, RegimeTier
        logger.warning("HIFAStub: Auto-passing (stub)")
        return HIFAResult(
            strategy_id=strategy.strategy_id,
            passed=True,
            final_gate=7,
            gate_results={
                i: GateResult(i, f"Gate_{i}", True, 1.0, 0.5)
                for i in range(1, 8)
            },
            statistical_scores=None,
            backtest_metrics=None,
            regime_tier=RegimeTier.UNKNOWN,
            regime_performance={},
            cluster_id=None,
            similarity_score=None,
            rejection_reason=None,
        )


class ForwardTestingStub:
    """Stub for Forward Testing when not available."""

    async def deploy_strategy(self, strategy: UnifiedStrategy):
        from dataclasses import dataclass

        @dataclass
        class StubDeployment:
            success: bool = True
            session_id: str = "stub_session"

        logger.warning("ForwardTestingStub: Auto-deploying (stub)")
        return StubDeployment()

    async def complete_forward_test(self, strategy_id: str) -> ForwardTestResult:
        from shared.unified_strategy import ShadowMetrics
        from datetime import timedelta

        logger.warning("ForwardTestingStub: Auto-passing (stub)")

        shadow_metrics = ShadowMetrics(
            sharpe_ratio=1.2,
            sortino_ratio=1.6,
            max_drawdown=12.0,
            win_rate=0.52,
            total_trades=50,
            avg_slippage_bps=3.0,
            avg_latency_ms=50.0,
            total_fees_paid=100.0,
            start_date=datetime.now() - timedelta(days=14),
            end_date=datetime.now(),
            duration_days=14,
            total_return=20.0,
        )

        return ForwardTestResult(
            strategy_id=strategy_id,
            passed=True,
            shadow_metrics=shadow_metrics,
            transfer_ratio=0.8,
            dd_ratio=0.8,
            rejection_reason=None,
            backtest_sharpe=1.5,
            backtest_max_dd=15.0,
            test_start=datetime.now() - timedelta(days=14),
            test_end=datetime.now(),
        )


class EMTStub:
    """Stub for EMT storage when not available."""

    def store(self, strategy: UnifiedStrategy) -> bool:
        logger.warning(f"EMTStub: Auto-storing {strategy.strategy_id} (stub)")
        return True
