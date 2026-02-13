"""
Dashboard Data Connector

Connects the dashboard to the shadow trading infrastructure.
Provides real-time data feeds for monitoring.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import random

# Import from forward testing module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forward_testing.models import VolatilityRegime
from forward_testing.shadow_monitor import (
    ShadowMonitor,
    MonitorAlert,
    AlertLevel,
    AlertType,
    MonitorConfig,
    StrategyHealth,
)
from forward_testing.deployment_queue import (
    DeploymentQueue,
    QueueStatus,
    QueuePriority,
)
from forward_testing.transfer_gate import TransferGate, TransferGateConfig


# ==============================================================================
# Data Models for Dashboard
# ==============================================================================

@dataclass
class StrategySnapshot:
    """Point-in-time snapshot of a strategy's performance."""
    strategy_id: str
    timestamp: datetime
    equity: float
    pnl: float
    pnl_pct: float
    drawdown_pct: float
    total_trades: int
    winning_trades: int
    current_position: Optional[str]  # "LONG", "SHORT", or None
    position_size: float
    unrealized_pnl: float
    sharpe_estimate: float
    transfer_ratio: float
    regime: VolatilityRegime
    health_status: str  # "healthy", "warning", "critical"


@dataclass
class MarketSnapshot:
    """Point-in-time market data snapshot."""
    timestamp: datetime
    symbol: str
    price: float
    bid: float
    ask: float
    spread_bps: float
    volume_24h: float
    volatility: float
    regime: VolatilityRegime
    funding_rate: float


@dataclass
class DashboardState:
    """Complete dashboard state."""
    last_updated: datetime
    active_strategies: int
    total_equity: float
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    active_alerts: int
    critical_alerts: int
    current_regime: VolatilityRegime
    queue_waiting: int
    queue_active: int


# ==============================================================================
# Dashboard Connector
# ==============================================================================

class DashboardConnector:
    """
    Connects the dashboard to shadow trading infrastructure.

    Provides:
    - Real-time strategy snapshots
    - Market data feeds
    - Alert streams
    - Queue status
    - Historical data access
    """

    def __init__(
        self,
        monitor: Optional[ShadowMonitor] = None,
        queue: Optional[DeploymentQueue] = None,
        gate: Optional[TransferGate] = None,
    ):
        """Initialize connector with optional existing components."""
        self.monitor = monitor or ShadowMonitor()
        self.queue = queue or DeploymentQueue()
        self.gate = gate or TransferGate()

        # Simulated data for demo mode
        self._demo_mode = True
        self._demo_strategies: Dict[str, Dict] = {}
        self._demo_market: Dict[str, MarketSnapshot] = {}
        self._history: List[DashboardState] = []

        # Initialize demo data
        self._init_demo_data()

    def _init_demo_data(self):
        """Initialize demo data for testing."""
        # Create demo strategies
        demo_ids = [f"STRAT_{i:03d}" for i in range(1, 6)]

        for sid in demo_ids:
            self._demo_strategies[sid] = {
                "strategy_id": sid,
                "initial_equity": 100000.0,
                "current_equity": 100000.0 + random.uniform(-5000, 10000),
                "total_trades": random.randint(20, 100),
                "winning_trades": 0,
                "position": random.choice([None, "LONG", "SHORT"]),
                "position_size": random.uniform(0, 0.2),
                "unrealized_pnl": random.uniform(-500, 1000),
                "sharpe": random.uniform(0.5, 2.0),
                "max_dd": random.uniform(5.0, 15.0),
                "current_dd": random.uniform(0, 10.0),
                "backtest_sharpe": random.uniform(1.2, 2.0),
                "health": random.choice(["healthy", "healthy", "healthy", "warning", "critical"]),
                "last_trade": datetime.now() - timedelta(minutes=random.randint(5, 120)),
            }
            # Calculate winning trades
            win_rate = random.uniform(0.45, 0.65)
            self._demo_strategies[sid]["winning_trades"] = int(
                self._demo_strategies[sid]["total_trades"] * win_rate
            )
            # Register with monitor
            self.monitor.register_strategy(sid, initial_equity=100000.0)

        # Create demo market data
        self._demo_market["BTCUSDT"] = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            price=67500.0,
            bid=67498.0,
            ask=67502.0,
            spread_bps=0.6,
            volume_24h=2.5e9,
            volatility=0.45,
            regime=VolatilityRegime.NORMAL,
            funding_rate=0.0001,
        )

    # ==========================================================================
    # Strategy Data
    # ==========================================================================

    def get_strategy_snapshot(self, strategy_id: str) -> Optional[StrategySnapshot]:
        """Get current snapshot for a strategy."""
        if self._demo_mode and strategy_id in self._demo_strategies:
            data = self._demo_strategies[strategy_id]
            pnl = data["current_equity"] - data["initial_equity"]
            return StrategySnapshot(
                strategy_id=strategy_id,
                timestamp=datetime.now(),
                equity=data["current_equity"],
                pnl=pnl,
                pnl_pct=(pnl / data["initial_equity"]) * 100,
                drawdown_pct=data["current_dd"],
                total_trades=data["total_trades"],
                winning_trades=data["winning_trades"],
                current_position=data["position"],
                position_size=data["position_size"],
                unrealized_pnl=data["unrealized_pnl"],
                sharpe_estimate=data["sharpe"],
                transfer_ratio=data["sharpe"] / data["backtest_sharpe"],
                regime=self._demo_market.get("BTCUSDT", MarketSnapshot(
                    timestamp=datetime.now(),
                    symbol="BTCUSDT",
                    price=0, bid=0, ask=0, spread_bps=0,
                    volume_24h=0, volatility=0,
                    regime=VolatilityRegime.NORMAL,
                    funding_rate=0,
                )).regime,
                health_status=data["health"],
            )

        # Real data from monitor
        health = self.monitor.get_health(strategy_id)
        if health is None:
            return None

        return StrategySnapshot(
            strategy_id=strategy_id,
            timestamp=datetime.now(),
            equity=health.current_equity,
            pnl=health.current_equity - health.initial_equity,
            pnl_pct=((health.current_equity / health.initial_equity) - 1) * 100,
            drawdown_pct=health.current_drawdown_pct,
            total_trades=health.total_trades,
            winning_trades=health.winning_trades,
            current_position=None,
            position_size=0.0,
            unrealized_pnl=0.0,
            sharpe_estimate=0.0,
            transfer_ratio=0.0,
            regime=self.monitor.current_regime,
            health_status="healthy" if health.is_healthy else "warning",
        )

    def get_all_strategy_snapshots(self) -> List[StrategySnapshot]:
        """Get snapshots for all active strategies."""
        snapshots = []

        if self._demo_mode:
            for sid in self._demo_strategies:
                snapshot = self.get_strategy_snapshot(sid)
                if snapshot:
                    snapshots.append(snapshot)
        else:
            for sid in self.monitor.strategies:
                snapshot = self.get_strategy_snapshot(sid)
                if snapshot:
                    snapshots.append(snapshot)

        return snapshots

    # ==========================================================================
    # Market Data
    # ==========================================================================

    def get_market_snapshot(self, symbol: str = "BTCUSDT") -> MarketSnapshot:
        """Get current market snapshot."""
        if self._demo_mode and symbol in self._demo_market:
            # Add some variation
            market = self._demo_market[symbol]
            noise = random.uniform(-50, 50)
            return MarketSnapshot(
                timestamp=datetime.now(),
                symbol=symbol,
                price=market.price + noise,
                bid=market.bid + noise,
                ask=market.ask + noise,
                spread_bps=market.spread_bps + random.uniform(-0.1, 0.1),
                volume_24h=market.volume_24h * random.uniform(0.95, 1.05),
                volatility=market.volatility + random.uniform(-0.05, 0.05),
                regime=market.regime,
                funding_rate=market.funding_rate + random.uniform(-0.00005, 0.00005),
            )

        # Return default
        return MarketSnapshot(
            timestamp=datetime.now(),
            symbol=symbol,
            price=67500.0,
            bid=67498.0,
            ask=67502.0,
            spread_bps=0.6,
            volume_24h=2.5e9,
            volatility=0.45,
            regime=VolatilityRegime.NORMAL,
            funding_rate=0.0001,
        )

    # ==========================================================================
    # Alerts
    # ==========================================================================

    def get_active_alerts(self) -> List[MonitorAlert]:
        """Get all active (unacknowledged) alerts."""
        alerts = []
        for sid in self.monitor.strategies:
            strategy_alerts = self.monitor.get_alerts(sid, unacknowledged_only=True)
            alerts.extend(strategy_alerts)

        # Sort by level (critical first) then by time
        alerts.sort(key=lambda a: (
            0 if a.level == AlertLevel.EMERGENCY else
            1 if a.level == AlertLevel.CRITICAL else
            2 if a.level == AlertLevel.WARNING else 3,
            -a.timestamp.timestamp()
        ))

        return alerts

    def get_alerts_by_strategy(self, strategy_id: str) -> List[MonitorAlert]:
        """Get alerts for a specific strategy."""
        return self.monitor.get_alerts(strategy_id)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        return self.monitor.acknowledge_alert(alert_id)

    # ==========================================================================
    # Queue Status
    # ==========================================================================

    def get_queue_status(self) -> Dict[str, Any]:
        """Get deployment queue status."""
        return {
            "waiting": self.queue.waiting_count,
            "active": self.queue.active_count,
            "total_capacity": self.queue.max_concurrent,
            "capacity_remaining": self.queue.max_concurrent - self.queue.active_count,
            "is_full": self.queue.is_full,
            "total_completed": self.queue.total_completed,
            "total_passed": self.queue.total_passed,
            "total_failed": self.queue.total_failed,
        }

    # ==========================================================================
    # Dashboard State
    # ==========================================================================

    def get_dashboard_state(self) -> DashboardState:
        """Get complete dashboard state."""
        snapshots = self.get_all_strategy_snapshots()
        alerts = self.get_active_alerts()
        queue = self.get_queue_status()
        market = self.get_market_snapshot()

        total_equity = sum(s.equity for s in snapshots)
        total_pnl = sum(s.pnl for s in snapshots)
        initial_equity = total_equity - total_pnl
        max_dd = max((s.drawdown_pct for s in snapshots), default=0.0)

        critical = sum(1 for a in alerts if a.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY])

        state = DashboardState(
            last_updated=datetime.now(),
            active_strategies=len(snapshots),
            total_equity=total_equity,
            total_pnl=total_pnl,
            total_pnl_pct=(total_pnl / initial_equity * 100) if initial_equity > 0 else 0,
            max_drawdown=max_dd,
            active_alerts=len(alerts),
            critical_alerts=critical,
            current_regime=market.regime,
            queue_waiting=queue["waiting"],
            queue_active=queue["active"],
        )

        # Store in history
        self._history.append(state)
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        return state

    # ==========================================================================
    # Historical Data
    # ==========================================================================

    def get_equity_history(
        self,
        strategy_id: str,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """Get equity curve history for a strategy."""
        # In demo mode, generate synthetic history
        if self._demo_mode:
            now = datetime.now()
            history = []
            equity = 100000.0

            for i in range(hours * 12):  # 5-min intervals
                timestamp = now - timedelta(minutes=(hours * 60) - (i * 5))
                change = random.gauss(0.0002, 0.001)  # ~2 bps mean, 10 bps std
                equity *= (1 + change)
                history.append({
                    "timestamp": timestamp,
                    "equity": equity,
                    "drawdown": random.uniform(0, 5),
                })

            return history

        # Real implementation would fetch from storage
        return []

    def get_trade_history(
        self,
        strategy_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get recent trade history for a strategy."""
        if self._demo_mode:
            trades = []
            now = datetime.now()

            for i in range(min(limit, 20)):
                is_win = random.random() > 0.4
                pnl = random.uniform(200, 1000) if is_win else -random.uniform(100, 500)
                trades.append({
                    "trade_id": f"T{strategy_id}_{i:04d}",
                    "timestamp": now - timedelta(hours=i * 2),
                    "side": random.choice(["LONG", "SHORT"]),
                    "entry_price": 67000 + random.uniform(-1000, 1000),
                    "exit_price": 67000 + random.uniform(-1000, 1000),
                    "size": random.uniform(0.05, 0.2),
                    "pnl": pnl,
                    "pnl_pct": pnl / 5000 * 100,
                    "is_win": is_win,
                    "duration_mins": random.randint(15, 480),
                })

            return trades

        return []

    # ==========================================================================
    # Control Operations
    # ==========================================================================

    def pause_strategy(self, strategy_id: str) -> bool:
        """Pause a strategy's shadow trading."""
        # Implementation would interact with the bridge
        return True

    def resume_strategy(self, strategy_id: str) -> bool:
        """Resume a paused strategy."""
        return True

    def set_regime(self, regime: VolatilityRegime):
        """Manually set the market regime (for testing)."""
        self.monitor.on_regime_change(regime)
        if self._demo_mode:
            for symbol in self._demo_market:
                self._demo_market[symbol].regime = regime

    # ==========================================================================
    # Simulation Updates (Demo Mode)
    # ==========================================================================

    def simulate_tick(self):
        """Simulate a market tick (for demo mode)."""
        if not self._demo_mode:
            return

        for sid, data in self._demo_strategies.items():
            # Random equity change
            change = random.gauss(0.0001, 0.002)
            data["current_equity"] *= (1 + change)

            # Update drawdown
            peak = max(data["initial_equity"], data["current_equity"])
            data["current_dd"] = max(0, (peak - data["current_equity"]) / peak * 100)

            # Random trade
            if random.random() < 0.05:  # 5% chance per tick
                data["total_trades"] += 1
                if random.random() < 0.55:  # 55% win rate
                    data["winning_trades"] += 1

            # Position changes
            if random.random() < 0.02:
                data["position"] = random.choice([None, "LONG", "SHORT"])
                data["position_size"] = random.uniform(0, 0.2) if data["position"] else 0

            # Update unrealized
            if data["position"]:
                data["unrealized_pnl"] = random.uniform(-500, 1000)
            else:
                data["unrealized_pnl"] = 0

        # Update market
        for symbol, market in self._demo_market.items():
            noise = random.gauss(0, 50)
            market.price += noise
            market.bid = market.price - market.spread_bps * market.price / 20000
            market.ask = market.price + market.spread_bps * market.price / 20000
