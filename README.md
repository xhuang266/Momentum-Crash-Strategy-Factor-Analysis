# Momentum Crash Strategy & Factor Analysis

## Overview
This project implements a complete quantitative research pipeline for analyzing and mitigating **Momentum Crashes**. 

It evolves from a simple hypothesis testing framework (OLS) to a market-neutral baseline strategy, and finally to an advanced machine-learning-based risk management system using **XGBoost Classifier**. It also includes a volume forecasting module using **ARIMA**.

## Project Features

  1. **XGBoost Classifier:** Switched from Regression (predicting returns) to Classification (predicting crash events).
  2. **Look-ahead Bias Free:** Target variable `y` is based on future 5-day rolling drawdown, strictly separated from input features `X`.
  3. **Feature Engineering:** Added rolling features (MA 5/20, Sum 5/20) to capture trend dynamics.
  4. **Class Imbalance Handling:** Implemented `scale_pos_weight` to address the scarcity of crash events.

## Project Structure

* **`src/part0_data.py`**: Data downloading (yfinance) and initial feature engineering (Log VIX, Realized Volatility).
* **`src/part1_hypothesis.py`**: OLS regression to validate the relationship between Momentum Returns, Volatility, and VIX.
* **`src/part2_baseline.py`**: Construction of a Market-Neutral Baseline Strategy using 252-day Rolling Beta.
* **`src/part3_backtest.py`**: Definition of the backtesting engine (including transaction costs) and drawdown calculation logic.
* **`src/part4_original.py`**: Sensitivity analysis for the original Bollinger Band strategy to find robust parameters.
* **`src/part5_xgboost.py`**: Implementation of the XGBoost Classifier for crash prediction using Walk-Forward Validation.
* **`src/part6_comparison.py`**: Final performance comparison between Baseline, Bollinger Band, and XGBoost strategies.
* **`src/part7_arima.py`**: ARIMA time-series modeling for volume forecasting.

## How to Run

1. Install dependencies:
   ```bash

   pip install -r requirements.txt
