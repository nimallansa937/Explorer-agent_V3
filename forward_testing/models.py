"""
Forward Testing Data Models

Enhanced data structures for realistic paper trading simulation.
Adapted from Hinance V2 with integration to shared module.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import uuid


# ==============================================================================
# Enumerations
# ==============================================================================

class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TRAILING_STOP = "TRAILING_STOP"


class OrderStatus(Enum):
    """Order status enumeration."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(Enum):
    """Time in force options."""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"


class MarginType(Enum):
    """Margin mode."""
    CROSS = "CROSS"
    ISOLATED = "ISOLATED"


class VolatilityRegime(Enum):
    """Volatility regime states for spread/latency adjustments."""
    LOW_VOL = "LOW_VOL"      # <30% annualized
    NORMAL = "NORMAL"        # 30-50% annualized
    HIGH_VOL = "HIGH_VOL"    # 50-100% annualized
    CRISIS = "CRISIS"        # 100-200% annualized
    CASCADE = "CASCADE"      # >200% annualized

    @classmethod
    def from_vix(cls, vix_level: float) -> 'VolatilityRegime':
        """Determine regime from VIX-analog level."""
        if vix_level < 15:
            return cls.LOW_VOL
        elif vix_level < 25:
            return cls.NORMAL
        elif vix_level < 40:
            return cls.HIGH_VOL
        elif vix_level < 60:
            return cls.CRISIS
        else:
            return cls.CASCADE


class LiquidationRisk(Enum):
    """Account liquidation risk level."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ==============================================================================
# Order & Trade Structures
# ==============================================================================

@dataclass
class Order:
    """Order data structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    status: OrderStatus = OrderStatus.NEW
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    fee: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Trade:
    """Trade/Execution record."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    strategy_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    price: float = 0.0
    quantity: float = 0.0
    fee: float = 0.0
    fee_asset: str = "USDT"
    realized_pnl: float = 0.0
    slippage_bps: float = 0.0
    latency_ms: float = 0.0
    market_impact_bps: float = 0.0
    spread_cost_bps: float = 0.0
    regime: VolatilityRegime = VolatilityRegime.NORMAL
    executed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'order_id': self.order_id,
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'price': self.price,
            'quantity': self.quantity,
            'fee': self.fee,
            'realized_pnl': self.realized_pnl,
            'slippage_bps': self.slippage_bps,
            'latency_ms': self.latency_ms,
            'market_impact_bps': self.market_impact_bps,
            'regime': self.regime.value,
            'executed_at': self.executed_at.isoformat(),
        }


# ==============================================================================
# Position & Account Structures
# ==============================================================================

@dataclass
class Position:
    """Position data structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str = ""
    symbol: str = ""
    side: PositionSide = PositionSide.LONG
    size: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    liquidation_price: float = 0.0
    margin: float = 0.0
    leverage: float = 1.0
    margin_type: MarginType = MarginType.CROSS
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L based on mark price."""
        if self.side == PositionSide.LONG:
            return (self.mark_price - self.entry_price) * self.size
        else:
            return (self.entry_price - self.mark_price) * self.size

    def get_notional(self) -> float:
        """Get position notional value."""
        return self.size * self.mark_price

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'size': self.size,
            'entry_price': self.entry_price,
            'mark_price': self.mark_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'opened_at': self.opened_at.isoformat(),
        }


@dataclass
class Account:
    """Shadow account for paper trading."""
    wallet_balance: float = 100000.0
    unrealized_pnl: float = 0.0
    margin_balance: float = 0.0
    available_balance: float = 100000.0
    position_margin: float = 0.0
    open_order_margin: float = 0.0
    margin_ratio: float = 0.0
    liquidation_risk: LiquidationRisk = LiquidationRisk.LOW

    def update_margin_balance(self):
        """Update margin balance = wallet + unrealized P&L."""
        self.margin_balance = self.wallet_balance + self.unrealized_pnl

    def update_available_balance(self):
        """Update available balance."""
        self.available_balance = (
            self.wallet_balance +
            self.unrealized_pnl -
            self.position_margin -
            self.open_order_margin
        )

    def update_liquidation_risk(self):
        """Update liquidation risk level based on margin ratio."""
        if self.margin_ratio < 0.5:
            self.liquidation_risk = LiquidationRisk.LOW
        elif self.margin_ratio < 0.7:
            self.liquidation_risk = LiquidationRisk.MEDIUM
        elif self.margin_ratio < 0.9:
            self.liquidation_risk = LiquidationRisk.HIGH
        else:
            self.liquidation_risk = LiquidationRisk.CRITICAL


# ==============================================================================
# Execution Result
# ==============================================================================

@dataclass
class ExecutionResult:
    """Enhanced execution result with full metrics."""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    avg_fill_price: float = 0.0
    effective_price: float = 0.0

    # Slippage breakdown
    slippage_total_bps: float = 0.0
    spread_cost_bps: float = 0.0
    market_impact_bps: float = 0.0
    timing_noise_bps: float = 0.0

    # Fees
    fee: float = 0.0
    fee_rate: float = 0.0

    # Execution quality
    latency_ms: float = 0.0
    liquidity_available: bool = True
    partial_fill: bool = False
    filled_qty: float = 0.0

    # Context
    regime: VolatilityRegime = VolatilityRegime.NORMAL
    rejection_reason: Optional[str] = None

    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.rejection_reason is None and self.filled_qty > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'filled_qty': self.filled_qty,
            'avg_fill_price': self.avg_fill_price,
            'effective_price': self.effective_price,
            'slippage_total_bps': self.slippage_total_bps,
            'market_impact_bps': self.market_impact_bps,
            'fee': self.fee,
            'latency_ms': self.latency_ms,
            'regime': self.regime.value,
            'rejection_reason': self.rejection_reason,
            'success': self.success,
            'timestamp': self.timestamp.isoformat(),
        }


# ==============================================================================
# Market State
# ==============================================================================

@dataclass
class MarketState:
    """Current market state for a symbol."""
    symbol: str = ""
    last_price: float = 0.0
    mark_price: float = 0.0
    index_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread_bps: float = 0.0
    volume_24h: float = 0.0
    open_interest: float = 0.0
    funding_rate: float = 0.0
    next_funding_time: datetime = field(default_factory=datetime.now)
    volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL
    realized_vol_1h: float = 0.0
    realized_vol_24h: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def mid_price(self) -> float:
        """Get mid price."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last_price

    @property
    def spread(self) -> float:
        """Get spread as decimal."""
        if self.bid > 0:
            return (self.ask - self.bid) / self.bid
        return 0.001  # Default 10 bps


# ==============================================================================
# Configuration Types
# ==============================================================================

@dataclass
class ExecutionConfig:
    """Execution engine configuration."""
    market_impact_eta: float = 0.314
    min_impact_bps: float = 0.5
    max_impact_bps: float = 500.0
    fee_taker: float = 0.0004  # 4 bps
    fee_maker: float = 0.0002  # 2 bps
    fee_liquidation: float = 0.005  # 50 bps


@dataclass
class LatencyConfig:
    """Latency simulation configuration."""
    fast_latency_min_ms: float = 30.0
    fast_latency_mode_ms: float = 50.0
    fast_latency_max_ms: float = 80.0
    slow_latency_min_ms: float = 200.0
    slow_latency_mode_ms: float = 500.0
    slow_latency_max_ms: float = 2000.0
    slow_latency_probability: float = 0.10


@dataclass
class SpreadConfig:
    """Spread simulation configuration."""
    min_spread_bps: float = 1.0
    max_spread_bps: float = 200.0

    # Regime multipliers
    regime_mult_low_vol: float = 0.8
    regime_mult_normal: float = 1.0
    regime_mult_high_vol: float = 2.5
    regime_mult_crisis: float = 5.0
    regime_mult_cascade: float = 15.0

    # Time multipliers (UTC sessions)
    time_mult_asia: float = 1.0      # 00:00-08:00
    time_mult_eu: float = 0.8        # 08:00-14:00
    time_mult_us: float = 0.7        # 14:00-21:00
    time_mult_dead: float = 1.3      # 21:00-00:00
    time_mult_weekend: float = 1.8   # Saturday-Sunday

    def get_regime_multiplier(self, regime: VolatilityRegime) -> float:
        """Get spread multiplier for regime."""
        multipliers = {
            VolatilityRegime.LOW_VOL: self.regime_mult_low_vol,
            VolatilityRegime.NORMAL: self.regime_mult_normal,
            VolatilityRegime.HIGH_VOL: self.regime_mult_high_vol,
            VolatilityRegime.CRISIS: self.regime_mult_crisis,
            VolatilityRegime.CASCADE: self.regime_mult_cascade,
        }
        return multipliers.get(regime, 1.0)
