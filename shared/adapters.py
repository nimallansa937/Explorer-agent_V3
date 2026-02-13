"""
Strategy Format Adapters for EXPLORER PRIME

Provides conversion utilities between different strategy formats:
- Explorer v3.0 StrategyPosterior
- LSM token sequences
- Hinance genome_json
- HIFA validation input

Each adapter ensures seamless integration between pipeline components.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime
import json
import hashlib

from .unified_strategy import (
    StrategyGenome,
    UnifiedStrategy,
    SourceEngine,
    StrategyStatus,
)

# Type hints for external classes (avoid circular imports)
if TYPE_CHECKING:
    from explorer_agent_v3.core.posteriors import StrategyPosterior
    from lsm.tokenization.tokenizer import StrategyTokenizer


# ==============================================================================
# Base Adapter
# ==============================================================================

class StrategyAdapter:
    """
    Factory class for strategy format conversions.

    Usage:
        # From Explorer v3.0
        unified = StrategyAdapter.from_explorer_posterior(posterior)

        # From LSM tokens
        unified = StrategyAdapter.from_lsm_tokens(tokens, tokenizer)

        # To Hinance format
        hinance_dict = StrategyAdapter.to_hinance_format(unified)

        # To HIFA input
        hifa_input = StrategyAdapter.to_hifa_input(unified, returns)
    """

    @staticmethod
    def from_explorer_posterior(posterior: Any) -> UnifiedStrategy:
        """
        Convert Explorer v3.0 StrategyPosterior to UnifiedStrategy.

        Args:
            posterior: StrategyPosterior from explorer_agent_v3

        Returns:
            UnifiedStrategy with genome extracted from posterior
        """
        return ExplorerAdapter.convert(posterior)

    @staticmethod
    def from_lsm_tokens(
        tokens: List[int],
        tokenizer: Any,
        generation_condition: Optional[Dict[str, Any]] = None,
    ) -> UnifiedStrategy:
        """
        Convert LSM token sequence to UnifiedStrategy.

        Args:
            tokens: Token sequence from LSM generation
            tokenizer: StrategyTokenizer for decoding
            generation_condition: Optional conditioning used during generation

        Returns:
            UnifiedStrategy with decoded genome
        """
        return LSMAdapter.convert(tokens, tokenizer, generation_condition)

    @staticmethod
    def from_hinance_json(genome_json: str, metadata: Dict[str, Any] = None) -> UnifiedStrategy:
        """
        Convert Hinance genome_json to UnifiedStrategy.

        Args:
            genome_json: JSON string from Hinance storage
            metadata: Optional metadata dict

        Returns:
            UnifiedStrategy with parsed genome
        """
        return HinanceAdapter.from_json(genome_json, metadata)

    @staticmethod
    def to_hinance_format(strategy: UnifiedStrategy) -> Dict[str, Any]:
        """
        Convert UnifiedStrategy to Hinance deployment format.

        Args:
            strategy: UnifiedStrategy to convert

        Returns:
            Dict compatible with Hinance shadow bridge
        """
        return HinanceAdapter.to_deployment_format(strategy)

    @staticmethod
    def to_hifa_input(
        strategy: UnifiedStrategy,
        returns: Any,  # pd.Series
    ) -> Dict[str, Any]:
        """
        Prepare strategy for HIFA v2.0 validation.

        Args:
            strategy: UnifiedStrategy to validate
            returns: pd.Series of strategy returns

        Returns:
            Dict with all required HIFA inputs
        """
        return {
            "strategy_id": strategy.strategy_id,
            "genome": strategy.genome,
            "returns": returns,
            "metadata": {
                "source_engine": strategy.source_engine.value,
                "target_regime": strategy.target_regime,
                "target_asset": strategy.target_asset,
                "n_parameters": strategy.genome.n_parameters if strategy.genome else 0,
                "n_rules": strategy.genome.n_rules if strategy.genome else 0,
                "indicators": strategy.genome.indicators if strategy.genome else [],
            }
        }

    @staticmethod
    def to_emt_format(strategy: UnifiedStrategy) -> Dict[str, Any]:
        """
        Convert UnifiedStrategy to EMT storage format.

        Args:
            strategy: Production-ready UnifiedStrategy

        Returns:
            Dict for EMT persistent storage
        """
        if not strategy.is_production_ready:
            raise ValueError("Strategy must be production-ready for EMT storage")

        return {
            "strategy_id": strategy.strategy_id,
            "version": strategy.version,
            "genome_json": strategy.genome.to_json() if strategy.genome else None,
            "genome_hash": strategy.genome_hash,
            "source_engine": strategy.source_engine.value,
            "generation_timestamp": strategy.generation_timestamp.isoformat(),
            "target_asset": strategy.target_asset,
            "target_regime": strategy.target_regime,

            # HIFA results
            "hifa_passed": True,
            "hifa_final_gate": strategy.hifa_result.final_gate,
            "backtest_sharpe": strategy.hifa_result.backtest_sharpe,
            "backtest_max_dd": strategy.hifa_result.backtest_max_dd,
            "regime_tier": strategy.hifa_result.regime_tier.value,

            # Forward test results
            "forward_passed": True,
            "transfer_ratio": strategy.forward_result.transfer_ratio,
            "shadow_sharpe": strategy.forward_result.shadow_sharpe,
            "shadow_max_dd": strategy.forward_result.shadow_max_dd,
            "shadow_duration_days": strategy.forward_result.shadow_metrics.duration_days,

            # Metadata
            "tags": strategy.tags,
            "metadata": strategy.metadata,
        }


# ==============================================================================
# Explorer v3.0 Adapter
# ==============================================================================

class ExplorerAdapter:
    """Adapter for Explorer Agent v3.0 strategy format."""

    @staticmethod
    def convert(posterior: Any) -> UnifiedStrategy:
        """
        Convert StrategyPosterior to UnifiedStrategy.

        Handles the complex belief/probability structure from Explorer v3.0.
        """
        # Extract core strategy logic
        genome = StrategyGenome(
            entry_conditions=ExplorerAdapter._extract_entry_rules(posterior),
            exit_conditions=ExplorerAdapter._extract_exit_rules(posterior),
            position_sizing=ExplorerAdapter._extract_sizing(posterior),
            parameters=getattr(posterior, 'params', {}),
            lookback_periods=getattr(posterior, 'lookbacks', {}),
            indicators=ExplorerAdapter._extract_indicators(posterior),
            stop_loss_pct=getattr(posterior, 'stop_loss', None),
            take_profit_pct=getattr(posterior, 'take_profit', None),
        )

        # Create unified strategy
        strategy = UnifiedStrategy(
            strategy_id=getattr(posterior, 'id', f"exp_{hashlib.md5(str(posterior).encode()).hexdigest()[:12]}"),
            genome=genome,
            source_engine=SourceEngine.EVOLUTIONARY,
            generation_timestamp=getattr(posterior, 'timestamp', datetime.now()),
            target_regime=getattr(posterior, 'target_regime', None),
            target_asset=getattr(posterior, 'asset', 'BTCUSDT'),
            metadata={
                "vfe_score": getattr(posterior, 'vfe_score', None),
                "posterior_mean": getattr(posterior, 'mean', None),
                "posterior_variance": getattr(posterior, 'variance', None),
                "oscillator_alignment": getattr(posterior, 'oscillator_alignment', None),
            }
        )

        return strategy

    @staticmethod
    def _extract_entry_rules(posterior: Any) -> List[Dict[str, Any]]:
        """Extract entry rules from posterior."""
        if hasattr(posterior, 'entry_rules'):
            return posterior.entry_rules
        if hasattr(posterior, 'rules') and 'entry' in posterior.rules:
            return posterior.rules['entry']
        return []

    @staticmethod
    def _extract_exit_rules(posterior: Any) -> List[Dict[str, Any]]:
        """Extract exit rules from posterior."""
        if hasattr(posterior, 'exit_rules'):
            return posterior.exit_rules
        if hasattr(posterior, 'rules') and 'exit' in posterior.rules:
            return posterior.rules['exit']
        return []

    @staticmethod
    def _extract_sizing(posterior: Any) -> Dict[str, Any]:
        """Extract position sizing configuration."""
        if hasattr(posterior, 'sizing_config'):
            return posterior.sizing_config
        if hasattr(posterior, 'position_sizing'):
            return posterior.position_sizing
        return {"method": "fixed", "size": 1.0}

    @staticmethod
    def _extract_indicators(posterior: Any) -> List[str]:
        """Extract list of indicators used."""
        if hasattr(posterior, 'indicators_used'):
            return list(posterior.indicators_used)
        if hasattr(posterior, 'indicators'):
            return list(posterior.indicators)
        return []


# ==============================================================================
# LSM Adapter
# ==============================================================================

class LSMAdapter:
    """Adapter for Language Strategy Model output."""

    @staticmethod
    def convert(
        tokens: List[int],
        tokenizer: Any,
        generation_condition: Optional[Dict[str, Any]] = None,
    ) -> UnifiedStrategy:
        """
        Convert LSM token sequence to UnifiedStrategy.

        Decodes the token sequence and reconstructs strategy structure.
        """
        # Decode tokens to strategy structure
        decoded = LSMAdapter._decode_tokens(tokens, tokenizer)

        # Build genome
        genome = StrategyGenome(
            entry_conditions=decoded.get("entry", []),
            exit_conditions=decoded.get("exit", []),
            position_sizing=decoded.get("sizing", {"method": "fixed", "size": 1.0}),
            parameters=decoded.get("params", {}),
            lookback_periods=decoded.get("lookbacks", {}),
            indicators=decoded.get("indicators", []),
        )

        # Generate strategy ID from token hash
        token_hash = hashlib.sha256(str(tokens).encode()).hexdigest()[:12]

        # Create unified strategy
        strategy = UnifiedStrategy(
            strategy_id=f"lsm_{token_hash}",
            genome=genome,
            source_engine=SourceEngine.LSM,
            generation_timestamp=datetime.now(),
            target_regime=generation_condition.get("regime") if generation_condition else None,
            target_asset=generation_condition.get("asset", "BTCUSDT") if generation_condition else "BTCUSDT",
            metadata={
                "token_count": len(tokens),
                "generation_condition": generation_condition,
                "decoded_length": len(str(decoded)),
            }
        )

        return strategy

    @staticmethod
    def _decode_tokens(tokens: List[int], tokenizer: Any) -> Dict[str, Any]:
        """Decode token sequence to strategy structure."""
        if tokenizer is None:
            # Fallback: return empty structure
            return {
                "entry": [],
                "exit": [],
                "sizing": {"method": "fixed", "size": 1.0},
                "params": {},
                "lookbacks": {},
                "indicators": [],
            }

        # Use tokenizer's decode method
        if hasattr(tokenizer, 'decode'):
            return tokenizer.decode(tokens)
        elif hasattr(tokenizer, 'decode_to_strategy'):
            return tokenizer.decode_to_strategy(tokens)
        else:
            raise ValueError("Tokenizer must have decode() or decode_to_strategy() method")


# ==============================================================================
# Hinance Adapter
# ==============================================================================

class HinanceAdapter:
    """Adapter for Hinance paper trading format."""

    @staticmethod
    def from_json(
        genome_json: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UnifiedStrategy:
        """
        Convert Hinance genome_json to UnifiedStrategy.

        Parses the legacy Hinance JSON format.
        """
        data = json.loads(genome_json)

        # Build genome from JSON
        genome = StrategyGenome(
            entry_conditions=data.get("entry_conditions", data.get("entry", [])),
            exit_conditions=data.get("exit_conditions", data.get("exit", [])),
            position_sizing=data.get("position_sizing", data.get("sizing", {"method": "fixed", "size": 1.0})),
            parameters=data.get("parameters", data.get("params", {})),
            lookback_periods=data.get("lookback_periods", data.get("lookbacks", {})),
            indicators=data.get("indicators", []),
            stop_loss_pct=data.get("stop_loss_pct", data.get("stop_loss")),
            take_profit_pct=data.get("take_profit_pct", data.get("take_profit")),
        )

        # Extract metadata
        metadata = metadata or {}
        strategy_id = metadata.get("strategy_id", f"hinance_{hashlib.md5(genome_json.encode()).hexdigest()[:12]}")

        strategy = UnifiedStrategy(
            strategy_id=strategy_id,
            genome=genome,
            source_engine=SourceEngine(metadata.get("source_engine", "external")),
            target_asset=metadata.get("asset", "BTCUSDT"),
            target_regime=metadata.get("target_regime"),
            metadata=metadata,
        )

        return strategy

    @staticmethod
    def to_deployment_format(strategy: UnifiedStrategy) -> Dict[str, Any]:
        """
        Convert UnifiedStrategy to Hinance deployment format.

        Creates the format expected by Hinance shadow bridge.
        """
        if strategy.genome is None:
            raise ValueError("Strategy must have genome for Hinance deployment")

        # Validate HIFA result exists
        if not strategy.is_hifa_validated:
            raise ValueError("Strategy must pass HIFA validation before Hinance deployment")

        return {
            # Identity
            "strategy_id": strategy.strategy_id,
            "version": strategy.version,

            # Strategy logic (JSON for Hinance compatibility)
            "genome_json": strategy.genome.to_json(),

            # Backtest reference metrics
            "backtest_sharpe": strategy.hifa_result.backtest_sharpe,
            "backtest_max_dd": strategy.hifa_result.backtest_max_dd,
            "backtest_trades": strategy.hifa_result.backtest_metrics.total_trades if strategy.hifa_result.backtest_metrics else 0,

            # Validation metadata
            "hifa_gates_passed": strategy.hifa_result.final_gate,
            "regime_tier": strategy.hifa_result.regime_tier.value,
            "cluster_id": strategy.hifa_result.cluster_id,

            # Configuration
            "target_asset": strategy.target_asset,
            "target_regime": strategy.target_regime,

            # Source tracking
            "source_engine": strategy.source_engine.value,
            "generation_timestamp": strategy.generation_timestamp.isoformat(),

            # Risk parameters
            "stop_loss_pct": strategy.genome.stop_loss_pct,
            "take_profit_pct": strategy.genome.take_profit_pct,
            "max_position_size": strategy.genome.max_position_size,

            # Metadata
            "metadata": {
                **strategy.metadata,
                "n_parameters": strategy.genome.n_parameters,
                "n_rules": strategy.genome.n_rules,
                "n_indicators": len(strategy.genome.indicators),
            }
        }

    @staticmethod
    def to_execution_format(strategy: UnifiedStrategy) -> Dict[str, Any]:
        """
        Convert to format for actual trade execution.

        Minimal format with only execution-relevant fields.
        """
        if strategy.genome is None:
            raise ValueError("Strategy must have genome for execution")

        return {
            "strategy_id": strategy.strategy_id,
            "entry_conditions": strategy.genome.entry_conditions,
            "exit_conditions": strategy.genome.exit_conditions,
            "position_sizing": strategy.genome.position_sizing,
            "parameters": strategy.genome.parameters,
            "lookback_periods": strategy.genome.lookback_periods,
            "indicators": strategy.genome.indicators,
            "stop_loss": strategy.genome.stop_loss_pct,
            "take_profit": strategy.genome.take_profit_pct,
            "max_position": strategy.genome.max_position_size,
        }


# ==============================================================================
# Batch Conversion Utilities
# ==============================================================================

class BatchAdapter:
    """Utilities for batch strategy conversion."""

    @staticmethod
    def convert_batch(
        items: List[Any],
        source_type: str,
        **kwargs,
    ) -> List[UnifiedStrategy]:
        """
        Convert a batch of strategies to UnifiedStrategy format.

        Args:
            items: List of source strategies
            source_type: "explorer", "lsm", or "hinance"
            **kwargs: Additional arguments for specific adapters

        Returns:
            List of UnifiedStrategy objects
        """
        results = []

        for item in items:
            try:
                if source_type == "explorer":
                    strategy = ExplorerAdapter.convert(item)
                elif source_type == "lsm":
                    tokenizer = kwargs.get("tokenizer")
                    condition = kwargs.get("generation_condition")
                    strategy = LSMAdapter.convert(item, tokenizer, condition)
                elif source_type == "hinance":
                    metadata = kwargs.get("metadata", {})
                    strategy = HinanceAdapter.from_json(item, metadata)
                else:
                    raise ValueError(f"Unknown source type: {source_type}")

                results.append(strategy)

            except Exception as e:
                # Log error but continue with other items
                print(f"Warning: Failed to convert strategy: {e}")
                continue

        return results

    @staticmethod
    def to_hinance_batch(strategies: List[UnifiedStrategy]) -> List[Dict[str, Any]]:
        """Convert batch of strategies to Hinance format."""
        results = []

        for strategy in strategies:
            try:
                hinance_format = HinanceAdapter.to_deployment_format(strategy)
                results.append(hinance_format)
            except Exception as e:
                print(f"Warning: Failed to convert {strategy.strategy_id}: {e}")
                continue

        return results
