"""
Execution Engine for Forward Testing

Realistic order execution with market microstructure simulation.
Migrated from Hinance V2 with enhancements for EXPLORER PRIME integration.

Features:
- Square-root market impact model (η=0.314)
- Bimodal latency distribution (90% fast, 10% slow)
- Regime-dependent spread multipliers
- Time-of-day spread adjustments
- Order rejection model with rate limits
- Partial fill simulation
"""

import asyncio
import logging
import math
import random
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List
from collections import deque
from dataclasses import dataclass, field

from ..models import (
    OrderSide, OrderType, OrderStatus, VolatilityRegime,
    ExecutionResult, Trade, Position, MarketState,
    ExecutionConfig, LatencyConfig, SpreadConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStats:
    """Execution statistics tracking."""
    total_executions: int = 0
    total_volume_usd: float = 0.0
    total_fees_paid: float = 0.0
    total_slippage_paid: float = 0.0
    rejected_orders: int = 0
    partial_fills: int = 0


class ExecutionEngine:
    """
    Realistic execution engine with market microstructure simulation.

    Implements:
    - Square-root market impact model
    - Regime-dependent spreads and latency
    - Bimodal latency distribution
    - Order rejection model
    - Rate limiting
    """

    # Market impact constant (empirically calibrated)
    MARKET_IMPACT_ETA = 0.314

    # Regime spread multipliers
    REGIME_SPREAD_MULT = {
        VolatilityRegime.LOW_VOL: 0.8,
        VolatilityRegime.NORMAL: 1.0,
        VolatilityRegime.HIGH_VOL: 2.5,
        VolatilityRegime.CRISIS: 5.0,
        VolatilityRegime.CASCADE: 15.0,
    }

    # Regime latency multipliers
    REGIME_LATENCY_MULT = {
        VolatilityRegime.LOW_VOL: 0.9,
        VolatilityRegime.NORMAL: 1.0,
        VolatilityRegime.HIGH_VOL: 1.5,
        VolatilityRegime.CRISIS: 3.0,
        VolatilityRegime.CASCADE: 5.0,
    }

    # Regime rejection rates
    REGIME_REJECTION_RATES = {
        VolatilityRegime.LOW_VOL: 0.0005,
        VolatilityRegime.NORMAL: 0.001,
        VolatilityRegime.HIGH_VOL: 0.01,
        VolatilityRegime.CRISIS: 0.05,
        VolatilityRegime.CASCADE: 0.15,
    }

    # Time-of-day spread multipliers (UTC)
    TIME_SPREAD_MULT = {
        "asia": 1.0,      # 00:00-08:00
        "eu": 0.8,        # 08:00-14:00
        "us": 0.7,        # 14:00-21:00
        "dead": 1.3,      # 21:00-00:00
        "weekend": 1.8,   # Saturday-Sunday
    }

    def __init__(
        self,
        fee_taker: float = 0.0004,
        fee_maker: float = 0.0002,
        fee_liquidation: float = 0.005,
        simulate_latency: bool = True,
    ):
        """
        Initialize execution engine.

        Args:
            fee_taker: Taker fee rate (default 0.04%)
            fee_maker: Maker fee rate (default 0.02%)
            fee_liquidation: Liquidation fee rate (default 0.5%)
            simulate_latency: Whether to simulate network latency
        """
        # Fee configuration
        self.fee_taker = fee_taker
        self.fee_maker = fee_maker
        self.fee_liquidation = fee_liquidation
        self.simulate_latency = simulate_latency

        # Current regime (updated externally)
        self.current_regime = VolatilityRegime.NORMAL

        # Rate limiting
        self.order_timestamps: deque = deque(maxlen=1200)  # Last minute
        self.second_timestamps: deque = deque(maxlen=50)   # Last second
        self.rate_limit_per_minute = 1200
        self.rate_limit_per_second = 50

        # Market data cache
        self.market_states: Dict[str, MarketState] = {}
        self.adv_cache: Dict[str, float] = {}  # Average daily volume
        self.volatility_cache: Dict[str, float] = {}  # 20-day realized vol

        # Statistics
        self.stats = ExecutionStats()

        logger.info(
            f"ExecutionEngine initialized: "
            f"fee_taker={fee_taker:.4f}, "
            f"fee_maker={fee_maker:.4f}"
        )

    # ==========================================================================
    # Market Impact Model
    # ==========================================================================

    def calculate_market_impact(
        self,
        order_size_usd: float,
        adv: float,
        volatility: float
    ) -> float:
        """
        Calculate market impact using square-root model.

        Formula: impact_bps = η × σ_daily × sqrt(Q / ADV) × 10000

        Args:
            order_size_usd: Order size in USD
            adv: 20-day average daily volume in USD
            volatility: 20-day realized volatility (daily returns std)

        Returns:
            Market impact in basis points
        """
        if adv <= 0:
            return 0.5  # Minimum impact

        q_adv_ratio = order_size_usd / adv

        # Floor and cap
        if q_adv_ratio < 0.001:
            return 0.5  # Minimum impact
        if q_adv_ratio > 0.10:
            return 500.0  # Maximum impact cap (5%)

        impact_bps = self.MARKET_IMPACT_ETA * volatility * math.sqrt(q_adv_ratio) * 10000
        return max(0.5, min(impact_bps, 500.0))

    # ==========================================================================
    # Spread Model
    # ==========================================================================

    def _get_time_session(self) -> str:
        """Get current trading session based on UTC time."""
        now = datetime.now(timezone.utc)

        # Weekend check
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return "weekend"

        hour = now.hour
        if 0 <= hour < 8:
            return "asia"
        elif 8 <= hour < 14:
            return "eu"
        elif 14 <= hour < 21:
            return "us"
        else:
            return "dead"

    def calculate_effective_spread(
        self,
        base_spread: float,
        regime: Optional[VolatilityRegime] = None
    ) -> float:
        """
        Calculate effective spread with regime and time adjustments.

        Args:
            base_spread: Base spread from live orderbook (as decimal)
            regime: Current volatility regime (uses self.current_regime if None)

        Returns:
            Effective spread as decimal (e.g., 0.001 = 0.1%)
        """
        if regime is None:
            regime = self.current_regime

        # Get multipliers
        regime_mult = self.REGIME_SPREAD_MULT.get(regime, 1.0)
        time_session = self._get_time_session()
        time_mult = self.TIME_SPREAD_MULT.get(time_session, 1.0)

        # Calculate effective spread
        spread = base_spread * regime_mult * time_mult

        # Apply floor and cap (1 bps to 200 bps)
        return max(0.0001, min(spread, 0.02))

    # ==========================================================================
    # Slippage Calculation
    # ==========================================================================

    def calculate_total_slippage(
        self,
        effective_spread: float,
        market_impact_bps: float,
        side: OrderSide
    ) -> Tuple[float, float, float, float]:
        """
        Calculate total slippage with all components.

        Components:
        1. Spread cost: effective_spread / 2
        2. Market impact: impact_bps / 10000
        3. Timing noise: uniform(-0.01%, +0.02%)

        Args:
            effective_spread: Effective spread as decimal
            market_impact_bps: Market impact in basis points
            side: Order side

        Returns:
            Tuple of (total_slippage, spread_cost, impact_cost, timing_noise)
        """
        # Spread cost (half spread)
        spread_cost = effective_spread / 2

        # Market impact
        impact_cost = market_impact_bps / 10000

        # Timing noise (slightly biased against trader)
        timing_noise = random.uniform(-0.0001, 0.0002)

        # Total slippage
        total_slippage = spread_cost + impact_cost + timing_noise

        return total_slippage, spread_cost, impact_cost, timing_noise

    # ==========================================================================
    # Latency Model
    # ==========================================================================

    async def simulate_network_latency(
        self,
        regime: Optional[VolatilityRegime] = None
    ) -> float:
        """
        Simulate bimodal network latency.

        Distribution:
        - 90% probability: triangular(30, 50, 80) ms
        - 10% probability: triangular(200, 500, 2000) ms

        Args:
            regime: Current volatility regime

        Returns:
            Latency in milliseconds
        """
        if not self.simulate_latency:
            return 50.0  # Fixed latency for testing

        if regime is None:
            regime = self.current_regime

        # Bimodal distribution
        if random.random() < 0.90:
            # Fast path (90%)
            latency = random.triangular(30, 80, 50)
        else:
            # Slow path (10%)
            latency = random.triangular(200, 2000, 500)

        # Regime adjustment
        regime_mult = self.REGIME_LATENCY_MULT.get(regime, 1.0)
        latency *= regime_mult

        # CASCADE adds random spike
        if regime == VolatilityRegime.CASCADE:
            latency += random.uniform(0, 5000)

        # Actually wait (scaled down for simulation - 1/10th)
        await asyncio.sleep(latency / 10000)

        return latency

    # ==========================================================================
    # Order Rejection Model
    # ==========================================================================

    def check_rate_limit(self) -> Optional[str]:
        """Check if rate limit would be exceeded."""
        now = datetime.now()

        # Clean old timestamps
        minute_ago = now.timestamp() - 60
        second_ago = now.timestamp() - 1

        while self.order_timestamps and self.order_timestamps[0] < minute_ago:
            self.order_timestamps.popleft()
        while self.second_timestamps and self.second_timestamps[0] < second_ago:
            self.second_timestamps.popleft()

        if len(self.order_timestamps) >= self.rate_limit_per_minute:
            return "RATE_LIMIT_MINUTE"
        if len(self.second_timestamps) >= self.rate_limit_per_second:
            return "RATE_LIMIT_SECOND"

        return None

    def check_order_rejection(
        self,
        symbol: str,
        order_size_usd: float,
        available_margin: float,
        mark_price: float,
        order_price: Optional[float] = None,
        price_change_1min: float = 0.0
    ) -> Optional[str]:
        """
        Check if order should be rejected.

        Args:
            symbol: Trading symbol
            order_size_usd: Order size in USD
            available_margin: Available margin in account
            mark_price: Current mark price
            order_price: Order price (for price protection check)
            price_change_1min: 1-minute price change percentage

        Returns:
            Rejection reason string, or None if valid
        """
        # Rate limit check
        rate_limit = self.check_rate_limit()
        if rate_limit:
            return rate_limit

        # Margin check (assume 10x leverage for rough check)
        required_margin = order_size_usd / 10
        if required_margin > available_margin:
            return "INSUFFICIENT_MARGIN"

        # Price protection (order price diverges >5% from mark)
        if order_price is not None and mark_price > 0:
            divergence = abs(order_price - mark_price) / mark_price
            if divergence > 0.05:
                return "PRICE_PROTECTION"

        # Volatility circuit breaker (1-min change > 5%)
        if abs(price_change_1min) > 0.05:
            return "VOLATILITY_CIRCUIT"

        # Regime-based random rejection (system overload)
        rejection_rate = self.REGIME_REJECTION_RATES.get(self.current_regime, 0.001)
        if random.random() < rejection_rate:
            return "SYSTEM_OVERLOAD"

        return None

    # ==========================================================================
    # Fee Calculation
    # ==========================================================================

    def calculate_fee(
        self,
        notional: float,
        is_maker: bool = False,
        is_liquidation: bool = False
    ) -> float:
        """
        Calculate trading fee.

        Args:
            notional: Trade notional value
            is_maker: Whether this is a maker order
            is_liquidation: Whether this is a liquidation

        Returns:
            Fee amount in USD
        """
        if is_liquidation:
            rate = self.fee_liquidation
        elif is_maker:
            rate = self.fee_maker
        else:
            rate = self.fee_taker

        return notional * rate

    # ==========================================================================
    # Main Execution Methods
    # ==========================================================================

    async def execute_market_order(
        self,
        strategy_id: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        current_price: float,
        available_margin: float = float('inf'),
    ) -> ExecutionResult:
        """
        Execute a market order with realistic simulation.

        Args:
            strategy_id: Strategy identifier
            symbol: Trading pair (e.g., 'BTCUSDT')
            side: OrderSide.BUY or OrderSide.SELL
            quantity: Order quantity in base asset
            current_price: Current market price
            available_margin: Available margin for rejection check

        Returns:
            ExecutionResult with full metrics
        """
        result = ExecutionResult(
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            regime=self.current_regime,
        )

        # Step 1: Simulate latency
        result.latency_ms = await self.simulate_network_latency()

        # Step 2: Get market data
        market_state = self.market_states.get(symbol)
        if market_state:
            best_bid = market_state.bid
            best_ask = market_state.ask
            mid_price = market_state.mid_price
            base_spread = market_state.spread
        else:
            # Fallback: use current_price
            best_bid = current_price * 0.9999
            best_ask = current_price * 1.0001
            mid_price = current_price
            base_spread = 0.001  # 10 bps default

        # Step 3: Calculate order notional
        entry_price = best_ask if side == OrderSide.BUY else best_bid
        order_size_usd = quantity * entry_price

        # Step 4: Check for rejection
        rejection = self.check_order_rejection(
            symbol=symbol,
            order_size_usd=order_size_usd,
            available_margin=available_margin,
            mark_price=mid_price,
        )
        if rejection:
            result.rejection_reason = rejection
            self.stats.rejected_orders += 1
            return result

        # Record order for rate limiting
        now = datetime.now().timestamp()
        self.order_timestamps.append(now)
        self.second_timestamps.append(now)

        # Step 5: Calculate market impact
        adv = self.adv_cache.get(symbol, order_size_usd * 1000)  # Default ADV
        volatility = self.volatility_cache.get(symbol, 0.02)  # Default 2% daily vol
        result.market_impact_bps = self.calculate_market_impact(
            order_size_usd, adv, volatility
        )

        # Step 6: Calculate effective spread
        effective_spread = self.calculate_effective_spread(base_spread)

        # Step 7: Calculate total slippage
        total_slip, spread_cost, impact_cost, timing_noise = self.calculate_total_slippage(
            effective_spread, result.market_impact_bps, side
        )
        result.slippage_total_bps = total_slip * 10000
        result.spread_cost_bps = spread_cost * 10000
        result.timing_noise_bps = timing_noise * 10000

        # Step 8: Calculate fill price with slippage
        result.filled_qty = quantity
        result.avg_fill_price = entry_price

        # Apply market impact to fill price
        if side == OrderSide.BUY:
            result.avg_fill_price *= (1 + total_slip)
        else:
            result.avg_fill_price *= (1 - total_slip)

        # Step 9: Calculate fees
        notional = result.filled_qty * result.avg_fill_price
        result.fee = self.calculate_fee(notional, is_maker=False)
        result.fee_rate = self.fee_taker

        # Step 10: Calculate effective price
        if side == OrderSide.BUY:
            result.effective_price = (notional + result.fee) / result.filled_qty
        else:
            result.effective_price = (notional - result.fee) / result.filled_qty

        result.liquidity_available = True
        result.timestamp = datetime.now()

        # Update statistics
        self.stats.total_executions += 1
        self.stats.total_volume_usd += notional
        self.stats.total_fees_paid += result.fee
        self.stats.total_slippage_paid += notional * total_slip

        logger.debug(
            f"Executed {side.value} {symbol}: "
            f"{result.filled_qty:.6f} @ ${result.avg_fill_price:.2f} "
            f"(impact: {result.market_impact_bps:.1f}bps, "
            f"fee: ${result.fee:.2f}, "
            f"latency: {result.latency_ms:.0f}ms)"
        )

        return result

    async def execute_limit_order(
        self,
        strategy_id: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        limit_price: float,
        available_margin: float = float('inf'),
    ) -> ExecutionResult:
        """
        Execute a limit order with fill probability model.

        Args:
            strategy_id: Strategy identifier
            symbol: Trading pair
            side: OrderSide.BUY or OrderSide.SELL
            quantity: Order quantity
            limit_price: Limit price
            available_margin: Available margin

        Returns:
            ExecutionResult
        """
        result = ExecutionResult(
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            regime=self.current_regime,
        )

        # Simulate latency
        result.latency_ms = await self.simulate_network_latency()

        # Get market data
        market_state = self.market_states.get(symbol)
        if market_state:
            best_bid = market_state.bid
            best_ask = market_state.ask
        else:
            best_bid = limit_price * 0.9999
            best_ask = limit_price * 1.0001

        # Check for rejection
        order_size_usd = quantity * limit_price
        rejection = self.check_order_rejection(
            symbol=symbol,
            order_size_usd=order_size_usd,
            available_margin=available_margin,
            mark_price=(best_bid + best_ask) / 2,
            order_price=limit_price,
        )
        if rejection:
            result.rejection_reason = rejection
            self.stats.rejected_orders += 1
            return result

        # Record for rate limiting
        now = datetime.now().timestamp()
        self.order_timestamps.append(now)
        self.second_timestamps.append(now)

        # Check if limit crosses spread (immediate fill as taker)
        is_taker = False
        fill_price = limit_price

        if side == OrderSide.BUY and limit_price >= best_ask:
            is_taker = True
            fill_price = best_ask
        elif side == OrderSide.SELL and limit_price <= best_bid:
            is_taker = True
            fill_price = best_bid

        # For simulation, assume eventual fill at limit price
        result.avg_fill_price = fill_price
        result.filled_qty = quantity
        result.liquidity_available = True

        # Calculate fee (maker or taker)
        notional = result.filled_qty * result.avg_fill_price
        result.fee = self.calculate_fee(notional, is_maker=not is_taker)
        result.fee_rate = self.fee_taker if is_taker else self.fee_maker

        # Effective price
        if side == OrderSide.BUY:
            result.effective_price = (notional + result.fee) / result.filled_qty
        else:
            result.effective_price = (notional - result.fee) / result.filled_qty

        result.timestamp = datetime.now()

        # Update statistics
        self.stats.total_executions += 1
        self.stats.total_volume_usd += notional
        self.stats.total_fees_paid += result.fee

        logger.debug(
            f"Limit {side.value} {symbol}: "
            f"{result.filled_qty:.6f} @ ${result.avg_fill_price:.2f} "
            f"(fee: ${result.fee:.2f}, type: {'taker' if is_taker else 'maker'})"
        )

        return result

    # ==========================================================================
    # State Management
    # ==========================================================================

    def set_regime(self, regime: VolatilityRegime):
        """Set current volatility regime."""
        if regime != self.current_regime:
            logger.info(f"Regime change: {self.current_regime.value} -> {regime.value}")
            self.current_regime = regime

    def update_market_state(self, symbol: str, state: MarketState):
        """Update market state for a symbol."""
        self.market_states[symbol] = state

    def update_market_data(
        self,
        symbol: str,
        adv: float,
        volatility: float
    ):
        """Update cached market data for a symbol."""
        self.adv_cache[symbol] = adv
        self.volatility_cache[symbol] = volatility

    def get_execution_stats(self) -> Dict:
        """Get execution quality statistics."""
        return {
            "total_executions": self.stats.total_executions,
            "total_volume_usd": self.stats.total_volume_usd,
            "total_fees_paid": self.stats.total_fees_paid,
            "total_slippage_paid": self.stats.total_slippage_paid,
            "rejected_orders": self.stats.rejected_orders,
            "partial_fills": self.stats.partial_fills,
            "current_regime": self.current_regime.value,
            "avg_fee_per_trade": (
                self.stats.total_fees_paid / self.stats.total_executions
                if self.stats.total_executions > 0 else 0
            ),
            "avg_slippage_per_trade": (
                self.stats.total_slippage_paid / self.stats.total_executions
                if self.stats.total_executions > 0 else 0
            ),
        }

    def reset_stats(self):
        """Reset execution statistics."""
        self.stats = ExecutionStats()
