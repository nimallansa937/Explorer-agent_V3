"""
Shadow Trading Dashboard Module

Provides real-time monitoring interface for:
- Shadow trading sessions
- Strategy performance metrics
- Alert management
- Transfer gate status
"""

from .data_connector import DashboardConnector

__all__ = ["DashboardConnector"]
