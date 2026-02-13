"""
Dashboard Launcher

Simple launcher script for the Shadow Trading Dashboard.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard."""
    # Get the dashboard path
    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "shadow_dashboard.py"
    )

    # Check if streamlit is installed
    try:
        import streamlit
        print(f"Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("Streamlit not found. Installing requirements...")
        requirements_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "requirements.txt"
        )
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path])

    # Launch dashboard
    print("\n" + "="*60)
    print("  SHADOW TRADING DASHBOARD")
    print("="*60)
    print(f"\nLaunching dashboard from: {dashboard_path}")
    print("\nPress Ctrl+C to stop the server\n")

    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        dashboard_path,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ])


if __name__ == "__main__":
    main()
