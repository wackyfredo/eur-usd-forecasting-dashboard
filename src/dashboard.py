"""
dashboard.py
Encapsulates the Streamlit user interface and Plotly visualizations.
"""

from pathlib import Path
from typing import List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.market_data import MarketData, MarketDataError
from src.market_analyzer import MarketAnalyzer
from src.forecast_models import (
    BaseForecastModel,
    NaiveBaselineModel,
    LinearRegressionForecastModel,
    RandomForestForecastModel,
)


class Dashboard:
    """
    Interactive Financial Market Forecasting and Analytics Dashboard.
    """

    DEFAULT_DATA_PATH = Path("data/DEXUSEU.csv")

    def __init__(self) -> None:
        st.set_page_config(
            page_title="EUR/USD Forecasting & Analytics Dashboard",
            page_icon="📈",
            layout="wide",
        )

    def run(self) -> None:
        """Main execution loop rendering sidebar controls, KPIs, charts, and forecasts."""
        st.title("EUR/USD Financial Market Forecasting & Analytics Dashboard")
        st.caption("CSC8101: Object-Oriented Programming with Python | Uganda Christian University")

        st.sidebar.header("1. Data Source")
        uploaded_file = st.sidebar.file_uploader("Upload EUR/USD CSV (Optional)", type=["csv"])

        try:
            source = uploaded_file if uploaded_file is not None else self.DEFAULT_DATA_PATH
            market_data = MarketData(source)
        except MarketDataError as err:
            st.error(f"Data Loading Error: {err}")
            st.info("Ensure 'data/DEXUSEU.csv' is saved in your project folder or upload a valid CSV above.")
            return

        full_df = market_data.data
        min_date = full_df["Date"].min().date()
        max_date = full_df["Date"].max().date()

        st.sidebar.success(
            f"Loaded {len(full_df):,} trading days ({market_data.missing_holidays_filled} holiday entries cleaned)."
        )

        st.sidebar.header("2. Analysis Parameters")
        date_range = st.sidebar.date_input(
            "Select Analysis Period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if len(date_range) != 2:
            st.warning("Please select both a start date and an end date.")
            return

        short_ma = st.sidebar.slider("Short Moving Average (Days)", min_value=3, max_value=30, value=7)
        long_ma = st.sidebar.slider("Long Moving Average (Days)", min_value=15, max_value=120, value=30)
        vol_window = st.sidebar.slider("Rolling Volatility Window (Days)", min_value=10, max_value=60, value=21)

        st.sidebar.header("3. Forecasting Settings")
        lag_days = st.sidebar.slider("Autoregressive Lag Window (Days)", min_value=3, max_value=15, value=5)
        horizon_days = st.sidebar.slider("Forecast Horizon (Business Days)", min_value=5, max_value=30, value=14)

        try:
            filtered_df = market_data.filter_by_date(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
            analyzer = MarketAnalyzer(filtered_df, short_window=short_ma, long_window=long_ma, vol_window=vol_window)
        except Exception as err:
            st.error(f"Analysis Error: {err}")
            return

        analyzed_df = analyzer.analyzed_data
        metrics = analyzer.get_summary_metrics()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest EUR/USD Rate", f"${metrics['latest_rate']:.4f}", f"{metrics['period_change_pct']:+.2f}%")
        col2.metric("Period High / Low", f"${metrics['period_high']:.4f} / ${metrics['period_low']:.4f}")
        col3.metric("Mean Daily Return", f"{metrics['mean_daily_return_pct']:+.4f}%")
        col4.metric("Annualized Volatility", f"{metrics['annualized_volatility_pct']:.2f}%")

        tab_trends, tab_volatility, tab_forecast, tab_data = st.tabs(
            ["Historical Trends & MAs", "Returns & Volatility", "Short-Term Forecasting", "Processed Dataset"]
        )

        with tab_trends:
            self._render_price_chart(analyzed_df, short_ma, long_ma)

        with tab_volatility:
            self._render_volatility_charts(analyzed_df, vol_window)

        with tab_forecast:
            self._render_forecast_section(filtered_df, lag_days, horizon_days)

        with tab_data:
            st.subheader("Cleaned & Enriched Market Data")
            st.dataframe(analyzed_df.tail(100), use_container_width=True)
            csv_export = analyzed_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Analyzed CSV", csv_export, "eur_usd_analyzed.csv", "text/csv")

    def _render_price_chart(self, df: pd.DataFrame, short_ma: int, long_ma: int) -> None:
        """Plots historical EUR/USD exchange rate with SMA overlays."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="EUR/USD Spot", line=dict(color="#1f77b4", width=1.8)))
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df[f"SMA_{short_ma}"],
                name=f"SMA ({short_ma}d)",
                line=dict(color="#ff7f0e", width=1.4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df[f"SMA_{long_ma}"],
                name=f"SMA ({long_ma}d)",
                line=dict(color="#2ca02c", width=1.4),
            )
        )
        fig.update_layout(
            title="Historical EUR/USD Exchange Rate & Moving Averages",
            xaxis_title="Date",
            yaxis_title="USD per 1 EUR",
            hovermode="x unified",
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_volatility_charts(self, df: pd.DataFrame, vol_window: int) -> None:
        """Plots daily percentage returns and rolling annualized volatility."""
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Daily Percentage Returns (%)", f"{vol_window}-Day Rolling Annualized Volatility (%)"),
        )
        fig.add_trace(
            go.Bar(x=df["Date"], y=df["Daily_Return_Pct"], name="Daily Return (%)", marker_color="#4a90e2"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Rolling_Volatility_Pct"],
                name="Annualized Volatility (%)",
                line=dict(color="#d62728", width=1.8),
            ),
            row=2,
            col=1,
        )
        fig.update_layout(height=550, hovermode="x unified", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def _render_forecast_section(self, df: pd.DataFrame, lag_days: int, horizon_days: int) -> None:
        """Trains all BaseForecastModel subclasses polymorphically, compares MAE, and plots forecasts."""
        models: List[BaseForecastModel] = [
            NaiveBaselineModel(lag_days=lag_days),
            LinearRegressionForecastModel(lag_days=lag_days),
            RandomForestForecastModel(lag_days=lag_days),
        ]

        eval_rows = []
        trained_models = {}

        try:
            for model in models:
                mae_val = model.train_and_evaluate(df, train_ratio=0.8)
                eval_rows.append({"Model": model.name, "Test MAE (USD)": round(mae_val, 6)})
                trained_models[model.name] = model
        except ValueError as err:
            st.warning(f"Forecasting requires a wider date window: {err}")
            return

        st.subheader("Model Performance Comparison (Chronological 80/20 Split)")
        eval_df = pd.DataFrame(eval_rows).sort_values("Test MAE (USD)")
        st.table(eval_df)

        selected_name = st.selectbox("Select Model for Short-Term Projection", list(trained_models.keys()), index=1)
        active_model = trained_models[selected_name]
        future_df = active_model.forecast_future(df, horizon_days=horizon_days)

        recent_df = df.tail(90)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=recent_df["Date"],
                y=recent_df["Close"],
                name="Recent Historical Rate (90d)",
                line=dict(color="#1f77b4", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_df["Date"],
                y=future_df["Forecast_Close"],
                name=f"{active_model.name} Forecast ({horizon_days}d)",
                line=dict(color="#d62728", width=2.5, dash="dash"),
                mode="lines+markers",
            )
        )
        fig.update_layout(
            title=f"Short-Term EUR/USD Forecast ({horizon_days} Business Days Ahead)",
            xaxis_title="Date",
            yaxis_title="Projected USD per 1 EUR",
            hovermode="x unified",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Disclaimer: Forecasts are statistical estimates generated for educational and analytical evaluation only."
        )
