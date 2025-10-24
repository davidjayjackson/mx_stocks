# Stock Price Forecasting: ARIMA, ETS, and Naive Comparison
# -----------------------------------------------------------

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Download stock data
symbol = "AAPL"  # Change ticker here
data = yf.download(symbol, start="2020-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d"))
close = data["Close"]

# Step 2: Filter data from April 2025 onward for plotting later
recent_data = close[close.index >= "2025-04-01"]

# Step 3: Split into train/test
train_size = int(len(close) * 0.8)
train, test = close.iloc[:train_size], close.iloc[train_size:]

# Step 4: Define helper functions
def forecast_arima(train, test):
    model = ARIMA(train, order=(5, 1, 0))
    fit = model.fit()
    forecast = fit.forecast(steps=len(test))
    return forecast

def forecast_ets(train, test):
    model = ExponentialSmoothing(train, trend="add", seasonal=None)
    fit = model.fit()
    forecast = fit.forecast(steps=len(test))
    return forecast

def forecast_naive(train, test):
    forecast = np.repeat(train.iloc[-1], len(test))
    return pd.Series(forecast, index=test.index)

# Step 5: Generate forecasts
pred_arima = forecast_arima(train, test)
pred_ets = forecast_ets(train, test)
pred_naive = forecast_naive(train, test)

# Step 6: Compute accuracy metrics
def evaluate_forecast(true, pred):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return rmse, mae, mape

results = pd.DataFrame({
    "Model": ["ARIMA", "ETS", "Naive"],
    "RMSE": [evaluate_forecast(test, pred_arima)[0],
             evaluate_forecast(test, pred_ets)[0],
             evaluate_forecast(test, pred_naive)[0]],
    "MAE": [evaluate_forecast(test, pred_arima)[1],
            evaluate_forecast(test, pred_ets)[1],
            evaluate_forecast(test, pred_naive)[1]],
    "MAPE": [evaluate_forecast(test, pred_arima)[2],
             evaluate_forecast(test, pred_ets)[2],
             evaluate_forecast(test, pred_naive)[2]]
})

# Step 7: Choose best model
best_model = results.loc[results["RMSE"].idxmin(), "Model"]
print("Best model based on RMSE:", best_model)
print(results)

# Step 8: Refit best model and forecast next 30 days
horizon = 30
if best_model == "ARIMA":
    final_model = ARIMA(close, order=(5, 1, 0)).fit()
    final_forecast = final_model.forecast(steps=horizon)
elif best_model == "ETS":
    final_model = ExponentialSmoothing(close, trend="add").fit()
    final_forecast = final_model.forecast(steps=horizon)
else:
    final_forecast = pd.Series(np.repeat(close.iloc[-1], horizon),
                               index=pd.date_range(close.index[-1], periods=horizon+1, freq="B")[1:])

# Step 9: Plot results — recent history + forecast
plt.figure(figsize=(12,6))
plt.plot(recent_data, label="Historical (since Apr 2025)", color="black")
plt.plot(final_forecast, label=f"{best_model} Forecast (next 30 days)", color="blue")
plt.title(f"{symbol} - {best_model} Forecast (since Apr 2025)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()
