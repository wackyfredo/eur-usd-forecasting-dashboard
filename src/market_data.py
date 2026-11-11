"""
market_data.py
Handles loading, validation, and preprocessing of historical EUR/USD exchange-rate data.
Demonstrates Encapsulation and Custom Exception Handling.
"""

from pathlib import Path
from typing import Union, IO
import pandas as pd


class MarketDataError(Exception):
    """Base custom exception for MarketData processing errors."""


class InvalidDataFormatError(MarketDataError):
    """Raised when required columns or valid numeric values are missing."""


class MarketData:
    """
    Encapsulates historical EUR/USD exchange-rate data with validation and cleaning methods.
    """

    def __init__(self, source: Union[str, Path, IO]) -> None:
        self._source = source
        self._raw_df: pd.DataFrame = pd.DataFrame()
        self._clean_df: pd.DataFrame = pd.DataFrame()
        self._missing_count: int = 0
        self._load_and_clean()

    def _load_and_clean(self) -> None:
        """Loads CSV data, normalizes FRED/standard column names, and cleans missing entries."""
        try:
            df = pd.read_csv(self._source)
        except FileNotFoundError as exc:
            raise MarketDataError(f"Dataset file not found: {self._source}") from exc
        except Exception as exc:
            raise MarketDataError(f"Failed to read CSV source: {exc}") from exc

        if df.empty:
            raise InvalidDataFormatError("The provided CSV file is empty.")

        self._raw_df = df.copy()

        col_map = {}
        for col in df.columns:
            clean_col = col.strip().lower()
            if clean_col in {"observation_date", "date"}:
                col_map[col] = "Date"
            elif clean_col in {"dexuseu", "close", "price", "rate", "eur_usd"}:
                col_map[col] = "Close"

        df = df.rename(columns=col_map)

        if "Date" not in df.columns or "Close" not in df.columns:
            raise InvalidDataFormatError(
                f"CSV must contain Date and Close/DEXUSEU columns. Found: {list(self._raw_df.columns)}"
            )

        df = df[["Date", "Close"]].copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

        df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"])

        self._missing_count = int(df["Close"].isna().sum())
        df["Close"] = df["Close"].ffill().bfill()

        if df["Close"].isna().all() or len(df) < 30:
            raise InvalidDataFormatError("Insufficient valid numeric exchange-rate records (minimum 30 required).")

        self._clean_df = df.reset_index(drop=True)

    @property
    def data(self) -> pd.DataFrame:
        """Returns a copy of the cleaned DataFrame to preserve encapsulation."""
        return self._clean_df.copy()

    @property
    def missing_holidays_filled(self) -> int:
        """Returns the count of non-trading holiday entries cleaned during ingestion."""
        return self._missing_count

    def filter_by_date(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Returns a filtered copy of the dataset within the specified date range."""
        mask = (self._clean_df["Date"] >= pd.to_datetime(start_date)) & (
            self._clean_df["Date"] <= pd.to_datetime(end_date)
        )
        filtered = self._clean_df.loc[mask].copy()
        if filtered.empty:
            raise MarketDataError("No exchange-rate records found in the selected date range.")
        return filtered.reset_index(drop=True)
