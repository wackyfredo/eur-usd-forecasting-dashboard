"""
test_analytics.py
Automated unit test suite verifying MarketData validation, MarketAnalyzer indicators,
and polymorphic BaseForecastModel implementations.
"""

import io
import unittest
import pandas as pd
import numpy as np

from src.market_data import MarketData, MarketDataError, InvalidDataFormatError
from src.market_analyzer import MarketAnalyzer
from src.forecast_models import (
    NaiveBaselineModel,
    LinearRegressionForecastModel,
    RandomForestForecastModel,
)


class TestEURUSDSystem(unittest.TestCase):
    """Unit tests covering data ingestion, financial indicators, and forecasting models."""

    def setUp(self) -> None:
        """Creates a synthetic 60-day FRED-style CSV buffer with holiday missing entries."""
        dates = pd.bdate_range(start="2025-01-01", periods=60)
        prices = np.linspace(1.0500, 1.1100, 60)
        lines = ["observation_date,DEXUSEU"]
        for i, (dt, price) in enumerate(zip(dates, prices)):
            if i in {5, 15}:
                lines.append(f"{dt.strftime('%Y-%m-%d')},.")
            else:
                lines.append(f"{dt.strftime('%Y-%m-%d')},{price:.4f}")
        self.valid_csv = io.StringIO("\n".join(lines))

    def test_market_data_loads_and_cleans_fred_holidays(self) -> None:
        """Verifies FRED column renaming and forward-filling of '.' holiday values."""
        md = MarketData(self.valid_csv)
        df = md.data
        self.assertEqual(len(df), 60)
        self.assertEqual(md.missing_holidays_filled, 2)
        self.assertListEqual(list(df.columns), ["Date", "Close"])
        self.assertFalse(df["Close"].isna().any())

    def test_invalid_columns_raises_custom_exception(self) -> None:
        """Verifies InvalidDataFormatError is raised when required columns are missing."""
        bad_csv = io.StringIO("Timestamp,Volume\n2025-01-01,1000\n")
        with self.assertRaises(InvalidDataFormatError):
            MarketData(bad_csv)

    def test_empty_date_filter_raises_market_data_error(self) -> None:
        """Verifies MarketDataError is raised when filtering outside available dates."""
        md = MarketData(self.valid_csv)
        with self.assertRaises(MarketDataError):
            md.filter_by_date(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31"))

    def test_market_analyzer_indicators_and_metrics(self) -> None:
        """Verifies SMA, EMA, daily returns, and summary KPI calculations."""
        md = MarketData(self.valid_csv)
        analyzer = MarketAnalyzer(md.data, short_window=5, long_window=10, vol_window=10)
        analyzed = analyzer.analyzed_data
        metrics = analyzer.get_summary_metrics()

        self.assertIn("SMA_5", analyzed.columns)
        self.assertIn("SMA_10", analyzed.columns)
        self.assertIn("EMA_5", analyzed.columns)
        self.assertIn("Rolling_Volatility_Pct", analyzed.columns)
        self.assertGreater(metrics["latest_rate"], 1.10)
        self.assertGreater(metrics["annualized_volatility_pct"], 0.0)

    def test_polymorphic_forecasting_models(self) -> None:
        """Verifies all BaseForecastModel subclasses compute MAE and generate future horizons."""
        md = MarketData(self.valid_csv)
        df = md.data
        models = [
            NaiveBaselineModel(lag_days=3),
            LinearRegressionForecastModel(lag_days=3),
            RandomForestForecastModel(lag_days=3, n_estimators=20),
        ]
        for model in models:
            mae = model.train_and_evaluate(df, train_ratio=0.8)
            self.assertGreaterEqual(mae, 0.0)
            forecast_df = model.forecast_future(df, horizon_days=7)
            self.assertEqual(len(forecast_df), 7)
            self.assertListEqual(list(forecast_df.columns), ["Date", "Forecast_Close"])


if __name__ == "__main__":
    unittest.main()
