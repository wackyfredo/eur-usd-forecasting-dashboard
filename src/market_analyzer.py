"""
market_analyzer.py
Performs financial calculations: daily returns, moving averages, and volatility metrics.
"""

from typing import Dict
import numpy as np
import pandas as pd


class MarketAnalyzer:
    """
    Computes statistical and technical market indicators on cleaned EUR/USD data.
    """

    TRADING_DAYS_PER_YEAR: int = 252

    def __init__(self, df: pd.DataFrame, short_window: int = 7, long_window: int = 30, vol_window: int = 21) -> None:
        if df.empty or "Close" not in df.columns:
            raise ValueError("MarketAnalyzer requires a non-empty DataFrame with a 'Close' column.")
        self._df = df.copy()
        self._short_window = short_window
        self._long_window = long_window
        self._vol_window = vol_window
        self._compute_indicators()

    def _compute_indicators(self) -> None:
        """Calculates daily percentage returns, SMA, EMA, and rolling annualized volatility."""
        self._df["Daily_Return_Pct"] = self._df["Close"].pct_change() * 100.0
        self._df[f"SMA_{self._short_window}"] = self._df["Close"].rolling(window=self._short_window).mean()
        self._df[f"SMA_{self._long_window}"] = self._df["Close"].rolling(window=self._long_window).mean()
        self._df[f"EMA_{self._short_window}"] = self._df["Close"].ewm(span=self._short_window, adjust=False).mean()

        rolling_std = self._df["Daily_Return_Pct"].rolling(window=self._vol_window).std()
        self._df["Rolling_Volatility_Pct"] = rolling_std * np.sqrt(self.TRADING_DAYS_PER_YEAR)

    @property
    def analyzed_data(self) -> pd.DataFrame:
        """Returns the enriched DataFrame containing all computed indicators."""
        return self._df.copy()

    def get_summary_metrics(self) -> Dict[str, float]:
        """Returns key summary statistics for dashboard KPI cards."""
        returns = self._df["Daily_Return_Pct"].dropna()
        latest_price = float(self._df["Close"].iloc[-1])
        start_price = float(self._df["Close"].iloc[0])
        period_change_pct = ((latest_price - start_price) / start_price) * 100.0
        annualized_vol = float(returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)) if not returns.empty else 0.0

        return {
            "latest_rate": latest_price,
            "period_change_pct": period_change_pct,
            "period_high": float(self._df["Close"].max()),
            "period_low": float(self._df["Close"].min()),
            "mean_daily_return_pct": float(returns.mean()) if not returns.empty else 0.0,
            "annualized_volatility_pct": annualized_vol,
        }
