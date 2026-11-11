"""
forecast_models.py
Implements forecasting models using OOP Abstraction, Inheritance, and Polymorphism.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


class BaseForecastModel(ABC):
    """
    Abstract Base Class defining the contract for all exchange-rate forecasting models.
    """

    def __init__(self, name: str, lag_days: int = 5) -> None:
        self.name = name
        self.lag_days = lag_days
        self._is_trained: bool = False
        self._mae_score: float = float("nan")
        self._test_dates: pd.Series = pd.Series(dtype="datetime64[ns]")
        self._y_test: np.ndarray = np.array([])
        self._y_pred_test: np.ndarray = np.array([])

    def _create_lag_features(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """Builds supervised lag features X (t-1 ... t-lag) and target y (t)."""
        df_lags = pd.DataFrame({"target": series.values})
        for lag in range(1, self.lag_days + 1):
            df_lags[f"lag_{lag}"] = series.shift(lag).values

        df_lags = df_lags.dropna().reset_index(drop=True)
        valid_dates = pd.Series(series.index[self.lag_days :]).reset_index(drop=True)

        X = df_lags[[f"lag_{lag}" for lag in range(1, self.lag_days + 1)]].values
        y = df_lags["target"].values
        return X, y, valid_dates

    @abstractmethod
    def _fit_estimator(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the underlying forecasting estimator."""

    @abstractmethod
    def _predict_step(self, feature_row: np.ndarray) -> float:
        """Predicts a single next-day value from a 1D array of lagged features."""

    def train_and_evaluate(self, df: pd.DataFrame, train_ratio: float = 0.8) -> float:
        """
        Splits data chronologically (80/20), trains the model, and computes test Mean Absolute Error (MAE).
        """
        indexed_series = pd.Series(df["Close"].values, index=df["Date"].values)
        X, y, dates = self._create_lag_features(indexed_series)

        split_idx = int(len(X) * train_ratio)
        if split_idx < 10 or (len(X) - split_idx) < 5:
            raise ValueError("Not enough observations to perform chronological train/test split.")

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self._fit_estimator(X_train, y_train)

        y_pred = np.array([self._predict_step(row) for row in X_test])
        self._mae_score = float(mean_absolute_error(y_test, y_pred))
        self._test_dates = dates.iloc[split_idx:].reset_index(drop=True)
        self._y_test = y_test
        self._y_pred_test = y_pred
        self._is_trained = True
        return self._mae_score

    def forecast_future(self, df: pd.DataFrame, horizon_days: int = 14) -> pd.DataFrame:
        """Generates recursive multi-step short-term forecasts for future business days."""
        if not self._is_trained:
            raise RuntimeError(f"Model '{self.name}' must be trained before calling forecast_future().")

        recent_window = list(df["Close"].iloc[-self.lag_days :].values)
        last_date = pd.to_datetime(df["Date"].iloc[-1])
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days)

        predictions = []
        for _ in range(horizon_days):
            features = np.array(recent_window[::-1], dtype=float)
            next_val = self._predict_step(features)
            predictions.append(next_val)
            recent_window.pop(0)
            recent_window.append(next_val)

        return pd.DataFrame({"Date": future_dates, "Forecast_Close": predictions})

    @property
    def mae(self) -> float:
        """Returns the out-of-sample Mean Absolute Error."""
        return self._mae_score

    @property
    def test_results(self) -> pd.DataFrame:
        """Returns actual vs. predicted values on the held-out test set."""
        return pd.DataFrame(
            {
                "Date": self._test_dates,
                "Actual": self._y_test,
                "Predicted": self._y_pred_test,
            }
        )


class NaiveBaselineModel(BaseForecastModel):
    """
    Baseline forecasting model predicting tomorrow's rate equals today's closing rate (Random Walk).
    """

    def __init__(self, lag_days: int = 5) -> None:
        super().__init__(name="Naive Baseline (Random Walk)", lag_days=lag_days)

    def _fit_estimator(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    def _predict_step(self, feature_row: np.ndarray) -> float:
        return float(feature_row[0])


class LinearRegressionForecastModel(BaseForecastModel):
    """
    Autoregressive Linear Regression model trained on lagged exchange rates.
    """

    def __init__(self, lag_days: int = 5) -> None:
        super().__init__(name="Linear Regression (AR)", lag_days=lag_days)
        self._model = LinearRegression()

    def _fit_estimator(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def _predict_step(self, feature_row: np.ndarray) -> float:
        return float(self._model.predict(feature_row.reshape(1, -1))[0])


class RandomForestForecastModel(BaseForecastModel):
    """
    Non-linear Random Forest Regressor trained on lagged exchange rates.
    """

    def __init__(self, lag_days: int = 5, n_estimators: int = 100, random_state: int = 42) -> None:
        super().__init__(name="Random Forest Regressor", lag_days=lag_days)
        self._model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=8, random_state=random_state
        )

    def _fit_estimator(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def _predict_step(self, feature_row: np.ndarray) -> float:
        return float(self._model.predict(feature_row.reshape(1, -1))[0])
