"""
Shadow Trading Dashboard

Real-time monitoring interface for shadow trading strategies.
Built with Streamlit for interactive visualization.

Run with: streamlit run dashboard/shadow_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from typing import List, Dict, Any, Optional

# Import dashboard connector
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.data_connector import (
    DashboardConnector,
    StrategySnapshot,
    MarketSnapshot,
    DashboardState,
)
from forward_testing.models import VolatilityRegime
from forward_testing.shadow_monitor import AlertLevel, AlertType


# ==============================================================================
# Page Configuration
# ==============================================================================

st.set_page_config(
    page_title="Shadow Trading Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    .alert-critical {
        background-color: #FF4B4B;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-warning {
        background-color: #FFA500;
        color: black;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-info {
        background-color: #4B9AFF;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .healthy {
        color: #00FF00;
    }
    .warning {
        color: #FFA500;
    }
    .critical {
        color: #FF4B4B;
    }
    .stMetric {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Session State Initialization
# ==============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "connector" not in st.session_state:
        st.session_state.connector = DashboardConnector()
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = True
    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 5
    if "selected_strategy" not in st.session_state:
        st.session_state.selected_strategy = None
    if "last_update" not in st.session_state:
        st.session_state.last_update = datetime.now()


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_health_color(status: str) -> str:
    """Get color based on health status."""
    colors = {
        "healthy": "#00FF00",
        "warning": "#FFA500",
        "critical": "#FF4B4B",
    }
    return colors.get(status, "#FFFFFF")


def get_regime_emoji(regime: VolatilityRegime) -> str:
    """Get emoji for volatility regime."""
    emojis = {
        VolatilityRegime.LOW: "🟢",
        VolatilityRegime.NORMAL: "🟡",
        VolatilityRegime.HIGH: "🟠",
        VolatilityRegime.EXTREME: "🔴",
    }
    return emojis.get(regime, "⚪")


def format_currency(value: float) -> str:
    """Format value as currency."""
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.2f}K"
    else:
        return f"${value:.2f}"


def format_pct(value: float) -> str:
    """Format value as percentage."""
    return f"{value:+.2f}%"


# ==============================================================================
# Dashboard Components
# ==============================================================================

def render_header(state: DashboardState, market: MarketSnapshot):
    """Render dashboard header with key metrics."""
    st.title("🔮 Shadow Trading Monitor")

    # Top metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            "Active Strategies",
            state.active_strategies,
            delta=None,
        )

    with col2:
        st.metric(
            "Total Equity",
            format_currency(state.total_equity),
            delta=format_pct(state.total_pnl_pct),
        )

    with col3:
        st.metric(
            "Total P&L",
            format_currency(state.total_pnl),
            delta=format_pct(state.total_pnl_pct),
        )

    with col4:
        st.metric(
            "Max Drawdown",
            format_pct(-state.max_drawdown),
            delta=None,
            delta_color="inverse",
        )

    with col5:
        regime_emoji = get_regime_emoji(state.current_regime)
        st.metric(
            "Market Regime",
            f"{regime_emoji} {state.current_regime.value}",
        )

    with col6:
        alert_color = "🔴" if state.critical_alerts > 0 else "🟡" if state.active_alerts > 0 else "🟢"
        st.metric(
            "Active Alerts",
            f"{alert_color} {state.active_alerts}",
            delta=f"{state.critical_alerts} critical" if state.critical_alerts > 0 else None,
            delta_color="inverse",
        )

    # Market data strip
    st.markdown("---")
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)

    with mcol1:
        st.caption(f"**{market.symbol}**")
        st.write(f"${market.price:,.2f}")

    with mcol2:
        st.caption("Spread")
        st.write(f"{market.spread_bps:.2f} bps")

    with mcol3:
        st.caption("24h Volume")
        st.write(f"${market.volume_24h/1e9:.2f}B")

    with mcol4:
        st.caption("Volatility")
        st.write(f"{market.volatility*100:.1f}%")

    with mcol5:
        st.caption("Funding Rate")
        fr_color = "green" if market.funding_rate >= 0 else "red"
        st.markdown(f"<span style='color:{fr_color}'>{market.funding_rate*100:.4f}%</span>", unsafe_allow_html=True)


def render_strategy_cards(snapshots: List[StrategySnapshot]):
    """Render strategy performance cards."""
    st.subheader("📈 Strategy Performance")

    # Create grid of strategy cards
    cols = st.columns(3)

    for i, snapshot in enumerate(snapshots):
        with cols[i % 3]:
            health_color = get_health_color(snapshot.health_status)
            position_text = f"{snapshot.current_position} ({snapshot.position_size:.1%})" if snapshot.current_position else "No Position"

            with st.container():
                st.markdown(f"""
                <div style='background-color:#262730; padding:15px; border-radius:10px; border-left:4px solid {health_color}; margin-bottom:10px;'>
                    <h4 style='margin:0; color:white;'>{snapshot.strategy_id}</h4>
                    <p style='margin:5px 0; color:gray;'>Health: <span style='color:{health_color}'>{snapshot.health_status.upper()}</span></p>
                </div>
                """, unsafe_allow_html=True)

                # Metrics
                m1, m2 = st.columns(2)
                with m1:
                    pnl_color = "green" if snapshot.pnl >= 0 else "red"
                    st.markdown(f"**Equity:** {format_currency(snapshot.equity)}")
                    st.markdown(f"**P&L:** <span style='color:{pnl_color}'>{format_currency(snapshot.pnl)} ({format_pct(snapshot.pnl_pct)})</span>", unsafe_allow_html=True)

                with m2:
                    st.markdown(f"**Drawdown:** {format_pct(-snapshot.drawdown_pct)}")
                    st.markdown(f"**Position:** {position_text}")

                # Performance metrics
                p1, p2, p3 = st.columns(3)
                with p1:
                    win_rate = snapshot.winning_trades / snapshot.total_trades * 100 if snapshot.total_trades > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                with p2:
                    st.metric("Sharpe Est.", f"{snapshot.sharpe_estimate:.2f}")
                with p3:
                    ratio_color = "green" if snapshot.transfer_ratio >= 0.5 else "red"
                    st.metric("Transfer Ratio", f"{snapshot.transfer_ratio:.2f}")

                # Action buttons
                if st.button(f"View Details", key=f"details_{snapshot.strategy_id}"):
                    st.session_state.selected_strategy = snapshot.strategy_id


def render_equity_chart(connector: DashboardConnector, strategy_id: Optional[str] = None):
    """Render equity curve chart."""
    st.subheader("📊 Equity Curves")

    if strategy_id:
        # Single strategy equity curve
        history = connector.get_equity_history(strategy_id, hours=24)
        if history:
            df = pd.DataFrame(history)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.1,
                              row_heights=[0.7, 0.3])

            # Equity curve
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df["equity"],
                          mode="lines", name="Equity",
                          line=dict(color="#00FF00", width=2)),
                row=1, col=1
            )

            # Drawdown
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=-df["drawdown"],
                          mode="lines", name="Drawdown",
                          fill="tozeroy", fillcolor="rgba(255,0,0,0.3)",
                          line=dict(color="#FF4B4B", width=1)),
                row=2, col=1
            )

            fig.update_layout(
                title=f"Equity Curve - {strategy_id}",
                height=400,
                template="plotly_dark",
                showlegend=True,
            )
            fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
            fig.update_yaxes(title_text="DD (%)", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)
    else:
        # Aggregate equity chart
        snapshots = connector.get_all_strategy_snapshots()
        if snapshots:
            # Create comparative chart
            fig = go.Figure()

            for snapshot in snapshots[:5]:  # Limit to 5 for clarity
                history = connector.get_equity_history(snapshot.strategy_id, hours=12)
                if history:
                    df = pd.DataFrame(history)
                    # Normalize to percentage returns
                    df["returns"] = (df["equity"] / df["equity"].iloc[0] - 1) * 100
                    fig.add_trace(
                        go.Scatter(x=df["timestamp"], y=df["returns"],
                                  mode="lines", name=snapshot.strategy_id)
                    )

            fig.update_layout(
                title="Strategy Returns Comparison (24h)",
                xaxis_title="Time",
                yaxis_title="Return (%)",
                height=350,
                template="plotly_dark",
            )

            st.plotly_chart(fig, use_container_width=True)


def render_alerts_panel(connector: DashboardConnector):
    """Render alerts panel."""
    st.subheader("🚨 Active Alerts")

    alerts = connector.get_active_alerts()

    if not alerts:
        st.success("No active alerts")
        return

    for alert in alerts[:10]:  # Show top 10
        # Determine alert styling
        if alert.level == AlertLevel.EMERGENCY:
            alert_class = "alert-critical"
            icon = "🚨"
        elif alert.level == AlertLevel.CRITICAL:
            alert_class = "alert-critical"
            icon = "❌"
        elif alert.level == AlertLevel.WARNING:
            alert_class = "alert-warning"
            icon = "⚠️"
        else:
            alert_class = "alert-info"
            icon = "ℹ️"

        time_ago = datetime.now() - alert.timestamp
        time_str = f"{int(time_ago.total_seconds() / 60)}m ago" if time_ago.total_seconds() < 3600 else f"{int(time_ago.total_seconds() / 3600)}h ago"

        st.markdown(f"""
        <div class='{alert_class}'>
            <strong>{icon} {alert.level.value.upper()}</strong> | {alert.strategy_id} | {time_str}<br/>
            <span style='font-size:0.9em;'>{alert.alert_type.value}: {alert.message}</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Acknowledge", key=f"ack_{alert.alert_id}"):
                connector.acknowledge_alert(alert.alert_id)
                st.rerun()


def render_queue_status(connector: DashboardConnector):
    """Render deployment queue status."""
    st.subheader("📋 Deployment Queue")

    status = connector.get_queue_status()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Waiting", status["waiting"])
    with col2:
        st.metric("Active", f"{status['active']}/{status['total_capacity']}")
    with col3:
        st.metric("Completed", status["total_completed"])

    # Progress bar for capacity
    capacity_pct = status["active"] / status["total_capacity"] if status["total_capacity"] > 0 else 0
    st.progress(capacity_pct)

    # Success rate
    if status["total_completed"] > 0:
        success_rate = status["total_passed"] / status["total_completed"] * 100
        st.caption(f"Success Rate: {success_rate:.1f}% ({status['total_passed']} passed, {status['total_failed']} failed)")


def render_trade_history(connector: DashboardConnector, strategy_id: str):
    """Render trade history table."""
    st.subheader(f"📜 Recent Trades - {strategy_id}")

    trades = connector.get_trade_history(strategy_id, limit=20)

    if not trades:
        st.info("No trade history available")
        return

    df = pd.DataFrame(trades)

    # Format columns
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    df["entry_price"] = df["entry_price"].apply(lambda x: f"${x:,.2f}")
    df["exit_price"] = df["exit_price"].apply(lambda x: f"${x:,.2f}")
    df["size"] = df["size"].apply(lambda x: f"{x:.4f}")
    df["pnl"] = df["pnl"].apply(lambda x: f"${x:+,.2f}")
    df["pnl_pct"] = df["pnl_pct"].apply(lambda x: f"{x:+.2f}%")
    df["duration"] = df["duration_mins"].apply(lambda x: f"{x}m")
    df["result"] = df["is_win"].apply(lambda x: "✅ Win" if x else "❌ Loss")

    # Display table
    st.dataframe(
        df[["timestamp", "side", "entry_price", "exit_price", "size", "pnl", "pnl_pct", "duration", "result"]],
        use_container_width=True,
        hide_index=True,
    )


def render_strategy_detail(connector: DashboardConnector, strategy_id: str):
    """Render detailed view for a single strategy."""
    snapshot = connector.get_strategy_snapshot(strategy_id)

    if not snapshot:
        st.error(f"Strategy {strategy_id} not found")
        return

    # Back button
    if st.button("← Back to Overview"):
        st.session_state.selected_strategy = None
        st.rerun()

    st.title(f"Strategy Details: {strategy_id}")

    # Health status banner
    health_color = get_health_color(snapshot.health_status)
    st.markdown(f"""
    <div style='background-color:{health_color}33; padding:15px; border-radius:10px; border:2px solid {health_color}; margin-bottom:20px;'>
        <h3 style='margin:0; color:{health_color};'>Status: {snapshot.health_status.upper()}</h3>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Equity", format_currency(snapshot.equity))
        st.metric("Total P&L", format_currency(snapshot.pnl), delta=format_pct(snapshot.pnl_pct))

    with col2:
        st.metric("Current Drawdown", format_pct(-snapshot.drawdown_pct))
        st.metric("Position", snapshot.current_position or "None")

    with col3:
        win_rate = snapshot.winning_trades / snapshot.total_trades * 100 if snapshot.total_trades > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
        st.metric("Total Trades", snapshot.total_trades)

    with col4:
        st.metric("Sharpe Estimate", f"{snapshot.sharpe_estimate:.2f}")
        st.metric("Transfer Ratio", f"{snapshot.transfer_ratio:.2f}")

    # Equity chart
    render_equity_chart(connector, strategy_id)

    # Trade history
    render_trade_history(connector, strategy_id)

    # Strategy alerts
    st.subheader("🚨 Strategy Alerts")
    alerts = connector.get_alerts_by_strategy(strategy_id)
    if alerts:
        for alert in alerts[:5]:
            st.warning(f"{alert.level.value}: {alert.message}")
    else:
        st.success("No alerts for this strategy")


def render_sidebar():
    """Render sidebar controls."""
    with st.sidebar:
        st.header("⚙️ Controls")

        # Auto-refresh toggle
        st.session_state.auto_refresh = st.checkbox(
            "Auto Refresh",
            value=st.session_state.auto_refresh,
        )

        # Refresh interval
        st.session_state.refresh_interval = st.slider(
            "Refresh Interval (seconds)",
            min_value=1,
            max_value=30,
            value=st.session_state.refresh_interval,
        )

        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.session_state.connector.simulate_tick()
            st.rerun()

        st.markdown("---")

        # Regime override (for testing)
        st.subheader("🧪 Testing Controls")
        regime = st.selectbox(
            "Set Market Regime",
            options=[r.value for r in VolatilityRegime],
            index=1,  # NORMAL
        )

        if st.button("Apply Regime"):
            st.session_state.connector.set_regime(VolatilityRegime(regime))
            st.rerun()

        st.markdown("---")

        # Strategy selector
        st.subheader("📊 Strategy Selector")
        snapshots = st.session_state.connector.get_all_strategy_snapshots()
        strategy_options = ["Overview"] + [s.strategy_id for s in snapshots]

        selected = st.selectbox(
            "View Strategy",
            options=strategy_options,
            index=0 if st.session_state.selected_strategy is None else strategy_options.index(st.session_state.selected_strategy) if st.session_state.selected_strategy in strategy_options else 0,
        )

        if selected != "Overview":
            st.session_state.selected_strategy = selected
        else:
            st.session_state.selected_strategy = None

        st.markdown("---")

        # Last update time
        st.caption(f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")


# ==============================================================================
# Main Dashboard
# ==============================================================================

def main():
    """Main dashboard entry point."""
    # Initialize session state
    init_session_state()

    # Get connector
    connector = st.session_state.connector

    # Simulate market tick for demo
    connector.simulate_tick()

    # Render sidebar
    render_sidebar()

    # Get current data
    state = connector.get_dashboard_state()
    market = connector.get_market_snapshot()

    # Update last update time
    st.session_state.last_update = datetime.now()

    # Check if viewing specific strategy
    if st.session_state.selected_strategy:
        render_strategy_detail(connector, st.session_state.selected_strategy)
    else:
        # Render main dashboard
        render_header(state, market)

        st.markdown("---")

        # Main content area
        left_col, right_col = st.columns([2, 1])

        with left_col:
            # Strategy cards
            snapshots = connector.get_all_strategy_snapshots()
            render_strategy_cards(snapshots)

            # Equity chart
            render_equity_chart(connector)

        with right_col:
            # Alerts panel
            render_alerts_panel(connector)

            st.markdown("---")

            # Queue status
            render_queue_status(connector)

    # Auto-refresh
    if st.session_state.auto_refresh:
        time.sleep(st.session_state.refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
