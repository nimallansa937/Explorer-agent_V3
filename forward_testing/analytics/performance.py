"""
Performance Analytics for Forward Testing

Calculates comprehensive performance metrics for shadow trading including:
- Return metrics (Sharpe, Sortino, Calmar)
- Risk metrics (VaR, CVaR, max drawdown)
- Transfer metrics (shadow vs backtest comparison)
- Trade statistics

Migrated from Hinance V2 with enhancements for transfer ratio calculation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import math
import statistics
from collections import defaultdict

from ..models import Trade, Position, VolatilityRegime, OrderSide


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class TradeRecord:
    """Record of a completed trade (entry + exit)."""
    trade_id: str = ""
    strategy_id: str = ""
    symbol: str = ""
    side: str = "LONG"  # LONG or SHORT

    # Entry details
    entry_time: datetime = field(default_factory=datetime.now)
    entry_price: float = 0.0
    entry_size: float = 0.0
    entry_fee: float = 0.0
    entry_slippage_bps: float = 0.0

    # Exit details
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    exit_fee: float = 0.0
    exit_slippage_bps: float = 0.0
    exit_reason: str = ""  # signal, stop_loss, take_profit, liquidation

    # P&L
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0

    # Context
    regime_at_entry: VolatilityRegime = VolatilityRegime.NORMAL
    regime_at_exit: VolatilityRegime = VolatilityRegime.NORMAL

    @property
    def duration(self) -> timedelta:
        """Trade duration."""
        if self.exit_time:
            return self.exit_time - self.entry_time
        return timedelta(0)

    @property
    def duration_hours(self) -> float:
        """Trade duration in hours."""
        return self.duration.total_seconds() / 3600

    @property
    def total_fees(self) -> float:
        """Total fees for the trade."""
        return self.entry_fee + self.exit_fee

    @property
    def total_slippage_bps(self) -> float:
        """Total slippage in basis points."""
        return self.entry_slippage_bps + self.exit_slippage_bps

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_time': self.entry_time.isoformat(),
            'entry_price': self.entry_price,
            'entry_size': self.entry_size,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'gross_pnl': self.gross_pnl,
            'net_pnl': self.net_pnl,
            'return_pct': self.return_pct,
            'total_fees': self.total_fees,
            'duration_hours': self.duration_hours,
        }


@dataclass
class DrawdownMetrics:
    """Drawdown analysis metrics."""
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: float = 0.0
    current_drawdown_pct: float = 0.0
    drawdown_start: Optional[datetime] = None
    drawdown_end: Optional[datetime] = None
    recovery_time_days: float = 0.0
    underwater_pct: float = 0.0  # % of time in drawdown

    # Drawdown distribution
    avg_drawdown_pct: float = 0.0
    drawdown_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_drawdown_pct': self.max_drawdown_pct,
            'max_drawdown_duration_days': self.max_drawdown_duration_days,
            'current_drawdown_pct': self.current_drawdown_pct,
            'recovery_time_days': self.recovery_time_days,
            'underwater_pct': self.underwater_pct,
            'avg_drawdown_pct': self.avg_drawdown_pct,
            'drawdown_count': self.drawdown_count,
        }


@dataclass
class RiskMetrics:
    """Risk analysis metrics."""
    volatility_annualized: float = 0.0
    downside_volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    var_99: float = 0.0  # Value at Risk 99%
    cvar_95: float = 0.0  # Conditional VaR 95%
    cvar_99: float = 0.0  # Conditional VaR 99%

    # Tail risk
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0  # 95th percentile / 5th percentile

    # Regime analysis
    regime_returns: Dict[str, float] = field(default_factory=dict)
    worst_regime: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'volatility_annualized': self.volatility_annualized,
            'downside_volatility': self.downside_volatility,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'tail_ratio': self.tail_ratio,
            'regime_returns': self.regime_returns,
            'worst_regime': self.worst_regime,
        }


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Basic returns
    total_return_pct: float = 0.0
    total_pnl: float = 0.0
    annualized_return: float = 0.0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0

    # Win/Loss statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L statistics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    payoff_ratio: float = 0.0

    # Trade statistics
    avg_trade_duration_hours: float = 0.0
    trades_per_day: float = 0.0

    # Execution quality
    total_fees: float = 0.0
    total_slippage_bps: float = 0.0
    avg_slippage_bps: float = 0.0

    # Time period
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_days: float = 0.0

    # Sub-metrics
    drawdown: DrawdownMetrics = field(default_factory=DrawdownMetrics)
    risk: RiskMetrics = field(default_factory=RiskMetrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_return_pct': self.total_return_pct,
            'total_pnl': self.total_pnl,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'omega_ratio': self.omega_ratio,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'avg_trade_duration_hours': self.avg_trade_duration_hours,
            'trades_per_day': self.trades_per_day,
            'total_fees': self.total_fees,
            'avg_slippage_bps': self.avg_slippage_bps,
            'duration_days': self.duration_days,
            'drawdown': self.drawdown.to_dict(),
            'risk': self.risk.to_dict(),
        }


@dataclass
class TransferMetrics:
    """Metrics comparing shadow performance to backtest."""
    # Core transfer metrics
    transfer_ratio: float = 0.0  # shadow_sharpe / backtest_sharpe
    return_degradation: float = 0.0  # 1 - (shadow_return / backtest_return)
    drawdown_expansion: float = 0.0  # shadow_max_dd / backtest_max_dd

    # Backtest reference
    backtest_sharpe: float = 0.0
    backtest_return: float = 0.0
    backtest_max_dd: float = 0.0
    backtest_trades: int = 0

    # Shadow actual
    shadow_sharpe: float = 0.0
    shadow_return: float = 0.0
    shadow_max_dd: float = 0.0
    shadow_trades: int = 0

    # Execution impact
    slippage_drag_pct: float = 0.0  # Return lost to slippage
    fee_drag_pct: float = 0.0  # Return lost to fees
    total_execution_drag: float = 0.0

    # Trade comparison
    trade_count_ratio: float = 0.0  # shadow_trades / backtest_trades
    win_rate_diff: float = 0.0  # shadow_win_rate - backtest_win_rate

    # Pass/fail status
    transfer_ratio_passed: bool = False
    drawdown_ratio_passed: bool = False
    minimum_trades_passed: bool = False
    overall_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'transfer_ratio': self.transfer_ratio,
            'return_degradation': self.return_degradation,
            'drawdown_expansion': self.drawdown_expansion,
            'backtest_sharpe': self.backtest_sharpe,
            'backtest_return': self.backtest_return,
            'backtest_max_dd': self.backtest_max_dd,
            'shadow_sharpe': self.shadow_sharpe,
            'shadow_return': self.shadow_return,
            'shadow_max_dd': self.shadow_max_dd,
            'slippage_drag_pct': self.slippage_drag_pct,
            'fee_drag_pct': self.fee_drag_pct,
            'total_execution_drag': self.total_execution_drag,
            'transfer_ratio_passed': self.transfer_ratio_passed,
            'drawdown_ratio_passed': self.drawdown_ratio_passed,
            'overall_passed': self.overall_passed,
        }


# ==============================================================================
# Performance Analyzer
# ==============================================================================

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for shadow trading.

    Features:
    - Risk-adjusted return metrics (Sharpe, Sortino, Calmar)
    - Drawdown analysis with underwater tracking
    - VaR/CVaR risk metrics
    - Transfer ratio calculation vs backtest
    - Execution quality metrics
    """

    # Annualization factor (crypto = 365 days)
    TRADING_DAYS_PER_YEAR = 365
    RISK_FREE_RATE = 0.04  # 4% risk-free rate assumption

    # Transfer ratio thresholds
    TRANSFER_RATIO_MIN = 0.5
    DRAWDOWN_RATIO_MAX = 1.5
    MIN_SHADOW_TRADES = 30

    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.04,
    ):
        """
        Initialize analyzer.

        Args:
            initial_capital: Starting capital for return calculations
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        # Trade records
        self.trades: List[TradeRecord] = []

        # Equity curve (timestamp -> equity)
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.current_equity = initial_capital

        # Daily returns for ratio calculations
        self.daily_returns: List[float] = []
        self.daily_timestamps: List[datetime] = []

        # Regime tracking
        self.regime_pnl: Dict[VolatilityRegime, float] = defaultdict(float)
        self.regime_trades: Dict[VolatilityRegime, int] = defaultdict(int)

    def add_trade(self, trade: TradeRecord) -> None:
        """Add a completed trade record."""
        self.trades.append(trade)

        # Update equity
        self.current_equity += trade.net_pnl
        self.equity_curve.append((trade.exit_time or datetime.now(), self.current_equity))

        # Track by regime
        self.regime_pnl[trade.regime_at_exit] += trade.net_pnl
        self.regime_trades[trade.regime_at_exit] += 1

    def add_daily_return(self, timestamp: datetime, return_pct: float) -> None:
        """Add a daily return observation."""
        self.daily_returns.append(return_pct)
        self.daily_timestamps.append(timestamp)

    def update_equity(self, timestamp: datetime, equity: float) -> None:
        """Update current equity value."""
        self.current_equity = equity
        self.equity_curve.append((timestamp, equity))

    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        metrics = PerformanceMetrics()

        if not self.trades:
            return metrics

        # Time period
        metrics.start_time = self.trades[0].entry_time
        metrics.end_time = self.trades[-1].exit_time or datetime.now()
        metrics.duration_days = (metrics.end_time - metrics.start_time).total_seconds() / 86400

        # Basic returns
        metrics.total_pnl = sum(t.net_pnl for t in self.trades)
        metrics.total_return_pct = (metrics.total_pnl / self.initial_capital) * 100

        if metrics.duration_days > 0:
            metrics.annualized_return = (
                ((1 + metrics.total_return_pct / 100) ** (365 / metrics.duration_days) - 1) * 100
            )

        # Trade statistics
        metrics.total_trades = len(self.trades)
        metrics.winning_trades = sum(1 for t in self.trades if t.net_pnl > 0)
        metrics.losing_trades = sum(1 for t in self.trades if t.net_pnl < 0)
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0

        # P&L statistics
        wins = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        losses = [t.net_pnl for t in self.trades if t.net_pnl < 0]

        metrics.avg_win = statistics.mean(wins) if wins else 0
        metrics.avg_loss = abs(statistics.mean(losses)) if losses else 0

        total_wins = sum(wins)
        total_losses = abs(sum(losses))

        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        metrics.payoff_ratio = metrics.avg_win / metrics.avg_loss if metrics.avg_loss > 0 else float('inf')
        metrics.expectancy = (metrics.win_rate * metrics.avg_win) - ((1 - metrics.win_rate) * metrics.avg_loss)

        # Trade duration
        durations = [t.duration_hours for t in self.trades if t.exit_time]
        metrics.avg_trade_duration_hours = statistics.mean(durations) if durations else 0
        metrics.trades_per_day = metrics.total_trades / metrics.duration_days if metrics.duration_days > 0 else 0

        # Execution quality
        metrics.total_fees = sum(t.total_fees for t in self.trades)
        metrics.total_slippage_bps = sum(t.total_slippage_bps for t in self.trades)
        metrics.avg_slippage_bps = metrics.total_slippage_bps / metrics.total_trades if metrics.total_trades > 0 else 0

        # Risk-adjusted returns
        if self.daily_returns:
            metrics.sharpe_ratio = self._calculate_sharpe()
            metrics.sortino_ratio = self._calculate_sortino()

        # Drawdown analysis
        metrics.drawdown = self._calculate_drawdown()

        # Calmar ratio
        if metrics.drawdown.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.drawdown.max_drawdown_pct

        # Omega ratio
        metrics.omega_ratio = self._calculate_omega()

        # Risk metrics
        metrics.risk = self._calculate_risk_metrics()

        return metrics

    def calculate_transfer_metrics(
        self,
        backtest_sharpe: float,
        backtest_return: float,
        backtest_max_dd: float,
        backtest_trades: int,
        backtest_win_rate: float = 0.0,
    ) -> TransferMetrics:
        """
        Calculate transfer metrics comparing shadow to backtest.

        Args:
            backtest_sharpe: Sharpe ratio from historical backtest
            backtest_return: Total return % from backtest
            backtest_max_dd: Max drawdown % from backtest
            backtest_trades: Number of trades in backtest
            backtest_win_rate: Win rate from backtest

        Returns:
            TransferMetrics with comparison analysis
        """
        metrics = TransferMetrics()

        # Store backtest reference
        metrics.backtest_sharpe = backtest_sharpe
        metrics.backtest_return = backtest_return
        metrics.backtest_max_dd = backtest_max_dd
        metrics.backtest_trades = backtest_trades

        # Calculate shadow metrics
        shadow_metrics = self.calculate_metrics()

        metrics.shadow_sharpe = shadow_metrics.sharpe_ratio
        metrics.shadow_return = shadow_metrics.total_return_pct
        metrics.shadow_max_dd = shadow_metrics.drawdown.max_drawdown_pct
        metrics.shadow_trades = shadow_metrics.total_trades

        # Transfer ratio (key metric!)
        if backtest_sharpe > 0:
            metrics.transfer_ratio = shadow_metrics.sharpe_ratio / backtest_sharpe
        else:
            metrics.transfer_ratio = 0.0

        # Return degradation
        if backtest_return != 0:
            metrics.return_degradation = 1 - (shadow_metrics.total_return_pct / backtest_return)

        # Drawdown expansion
        if backtest_max_dd > 0:
            metrics.drawdown_expansion = shadow_metrics.drawdown.max_drawdown_pct / backtest_max_dd

        # Execution impact
        total_notional = sum(t.entry_price * t.entry_size for t in self.trades)
        if total_notional > 0:
            metrics.slippage_drag_pct = (shadow_metrics.total_slippage_bps / 10000) * 100
            metrics.fee_drag_pct = (shadow_metrics.total_fees / self.initial_capital) * 100
            metrics.total_execution_drag = metrics.slippage_drag_pct + metrics.fee_drag_pct

        # Trade comparison
        if backtest_trades > 0:
            metrics.trade_count_ratio = shadow_metrics.total_trades / backtest_trades

        metrics.win_rate_diff = shadow_metrics.win_rate - backtest_win_rate

        # Pass/fail evaluation
        metrics.transfer_ratio_passed = metrics.transfer_ratio >= self.TRANSFER_RATIO_MIN
        metrics.drawdown_ratio_passed = metrics.drawdown_expansion <= self.DRAWDOWN_RATIO_MAX
        metrics.minimum_trades_passed = shadow_metrics.total_trades >= self.MIN_SHADOW_TRADES

        metrics.overall_passed = (
            metrics.transfer_ratio_passed and
            metrics.drawdown_ratio_passed and
            metrics.minimum_trades_passed
        )

        return metrics

    def _calculate_sharpe(self) -> float:
        """Calculate annualized Sharpe ratio."""
        if not self.daily_returns or len(self.daily_returns) < 2:
            return 0.0

        mean_return = statistics.mean(self.daily_returns)
        std_return = statistics.stdev(self.daily_returns)

        if std_return == 0:
            return 0.0

        # Daily risk-free rate
        daily_rf = self.risk_free_rate / self.TRADING_DAYS_PER_YEAR

        # Annualized Sharpe
        sharpe = ((mean_return - daily_rf) / std_return) * math.sqrt(self.TRADING_DAYS_PER_YEAR)

        return sharpe

    def _calculate_sortino(self) -> float:
        """Calculate annualized Sortino ratio."""
        if not self.daily_returns or len(self.daily_returns) < 2:
            return 0.0

        mean_return = statistics.mean(self.daily_returns)

        # Downside returns only
        downside_returns = [r for r in self.daily_returns if r < 0]

        if not downside_returns:
            return float('inf')

        downside_std = statistics.stdev(downside_returns) if len(downside_returns) > 1 else abs(downside_returns[0])

        if downside_std == 0:
            return float('inf')

        # Daily risk-free rate
        daily_rf = self.risk_free_rate / self.TRADING_DAYS_PER_YEAR

        # Annualized Sortino
        sortino = ((mean_return - daily_rf) / downside_std) * math.sqrt(self.TRADING_DAYS_PER_YEAR)

        return sortino

    def _calculate_omega(self, threshold: float = 0.0) -> float:
        """Calculate Omega ratio."""
        if not self.daily_returns:
            return 0.0

        gains = sum(r - threshold for r in self.daily_returns if r > threshold)
        losses = sum(threshold - r for r in self.daily_returns if r <= threshold)

        if losses == 0:
            return float('inf')

        return gains / losses

    def _calculate_drawdown(self) -> DrawdownMetrics:
        """Calculate drawdown metrics from equity curve."""
        metrics = DrawdownMetrics()

        if not self.equity_curve:
            return metrics

        # Track peak and drawdowns
        peak = self.initial_capital
        drawdowns = []
        current_dd_start = None
        underwater_periods = 0
        total_periods = len(self.equity_curve)

        max_dd = 0.0
        max_dd_start = None
        max_dd_end = None

        for timestamp, equity in self.equity_curve:
            if equity > peak:
                # New peak
                if current_dd_start:
                    # Recovered from drawdown
                    drawdowns.append(max_dd)
                    current_dd_start = None
                peak = equity
            else:
                # In drawdown
                dd = (peak - equity) / peak

                if current_dd_start is None:
                    current_dd_start = timestamp

                underwater_periods += 1

                if dd > max_dd:
                    max_dd = dd
                    max_dd_start = current_dd_start
                    max_dd_end = timestamp

        # Check if still in drawdown
        current_equity = self.equity_curve[-1][1] if self.equity_curve else self.initial_capital
        metrics.current_drawdown_pct = ((peak - current_equity) / peak) * 100 if peak > 0 else 0

        metrics.max_drawdown_pct = max_dd * 100
        metrics.drawdown_start = max_dd_start
        metrics.drawdown_end = max_dd_end

        if max_dd_start and max_dd_end:
            metrics.max_drawdown_duration_days = (max_dd_end - max_dd_start).total_seconds() / 86400

        metrics.underwater_pct = (underwater_periods / total_periods * 100) if total_periods > 0 else 0

        if drawdowns:
            metrics.avg_drawdown_pct = statistics.mean(drawdowns) * 100
            metrics.drawdown_count = len(drawdowns)

        return metrics

    def _calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate VaR, CVaR, and other risk metrics."""
        metrics = RiskMetrics()

        if not self.daily_returns or len(self.daily_returns) < 10:
            return metrics

        # Volatility
        metrics.volatility_annualized = statistics.stdev(self.daily_returns) * math.sqrt(self.TRADING_DAYS_PER_YEAR)

        # Downside volatility
        downside = [r for r in self.daily_returns if r < 0]
        if downside:
            metrics.downside_volatility = statistics.stdev(downside) * math.sqrt(self.TRADING_DAYS_PER_YEAR) if len(downside) > 1 else 0

        # Sort returns for VaR calculation
        sorted_returns = sorted(self.daily_returns)
        n = len(sorted_returns)

        # VaR (historical method)
        var_95_idx = int(n * 0.05)
        var_99_idx = int(n * 0.01)

        metrics.var_95 = abs(sorted_returns[var_95_idx]) * 100 if var_95_idx < n else 0
        metrics.var_99 = abs(sorted_returns[var_99_idx]) * 100 if var_99_idx < n else 0

        # CVaR (Expected Shortfall)
        if var_95_idx > 0:
            metrics.cvar_95 = abs(statistics.mean(sorted_returns[:var_95_idx])) * 100
        if var_99_idx > 0:
            metrics.cvar_99 = abs(statistics.mean(sorted_returns[:var_99_idx])) * 100

        # Higher moments
        mean_return = statistics.mean(self.daily_returns)
        std_return = statistics.stdev(self.daily_returns)

        if std_return > 0:
            # Skewness
            metrics.skewness = sum((r - mean_return) ** 3 for r in self.daily_returns) / (n * std_return ** 3)

            # Kurtosis (excess)
            metrics.kurtosis = sum((r - mean_return) ** 4 for r in self.daily_returns) / (n * std_return ** 4) - 3

        # Tail ratio
        p95_idx = int(n * 0.95)
        p5_idx = int(n * 0.05)

        if sorted_returns[p5_idx] != 0:
            metrics.tail_ratio = abs(sorted_returns[p95_idx] / sorted_returns[p5_idx])

        # Regime analysis
        for regime, pnl in self.regime_pnl.items():
            trade_count = self.regime_trades[regime]
            if trade_count > 0:
                metrics.regime_returns[regime.value] = pnl / self.initial_capital * 100

        if metrics.regime_returns:
            metrics.worst_regime = min(metrics.regime_returns, key=metrics.regime_returns.get)

        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current performance."""
        metrics = self.calculate_metrics()

        return {
            'total_trades': metrics.total_trades,
            'total_pnl': metrics.total_pnl,
            'total_return_pct': metrics.total_return_pct,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown_pct': metrics.drawdown.max_drawdown_pct,
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'duration_days': metrics.duration_days,
            'current_equity': self.current_equity,
        }

    def reset(self, initial_capital: Optional[float] = None) -> None:
        """Reset analyzer state."""
        if initial_capital:
            self.initial_capital = initial_capital

        self.trades = []
        self.equity_curve = []
        self.current_equity = self.initial_capital
        self.daily_returns = []
        self.daily_timestamps = []
        self.regime_pnl = defaultdict(float)
        self.regime_trades = defaultdict(int)
