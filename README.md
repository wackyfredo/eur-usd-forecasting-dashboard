## EUR/USD Financial Market Forecasting & Analytics Dashboard

**Course:** CSC8101 – Object-Oriented Programming with Python  
**Institution:** Uganda Christian University  

## Overview
An interactive Python web application built with Object-Oriented Programming (OOP) principles to analyze 5 years of daily EUR/USD spot exchange-rate data (`2021-11-08` to `2026-11-06`) from the Federal Reserve Economic Data (FRED) repository and generate short-term forecasts.

## Key Features
- **Automated Data Ingestion & Validation (`MarketData`)**: Parses FRED (`DEXUSEU.csv`) and standard CSV files, cleans non-trading bank holiday placeholders, and enforces custom exception handling.
- **Financial & Volatility Analytics (`MarketAnalyzer`)**: Computes daily percentage returns, Simple Moving Averages (SMA), Exponential Moving Averages (EMA), and rolling annualized volatility.
- **Polymorphic Forecasting Engine (`BaseForecastModel`)**: Compares a Naive Random Walk baseline against Autoregressive Linear Regression and Random Forest models using chronological 80/20 Mean Absolute Error (MAE).
- **Interactive Dashboard (`Dashboard`)**: Built with Streamlit and Plotly for dynamic date filtering, parameter tuning, and CSV export.

## Project Structure
```text
eur-usd-forecasting-dashboard/
├── data/
│   └── DEXUSEU.csv
├── src/
│   ├── __init__.py
│   ├── market_data.py
│   ├── market_analyzer.py
│   ├── forecast_models.py
│   └── dashboard.py
├── tests/
│   └── test_analytics.py
├── app.py
├── requirements.txt
└── README.md
```

## Setup & Execution
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
streamlit run app.py
```
 eur-usd-forecasting-dashboard