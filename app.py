"""
app.py
Entry point for launching the EUR/USD Financial Market Forecasting and Analytics Dashboard.
"""

from src.dashboard import Dashboard

if __name__ == "__main__":
    app = Dashboard()
    app.run()
